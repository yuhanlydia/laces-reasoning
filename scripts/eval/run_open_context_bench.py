#!/usr/bin/env python3
"""Aligned open-ended context evaluation for the 30k dynamic LACES checkpoint."""
from __future__ import annotations

import argparse
import json
import re
import string
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch


QA_DATASETS = {
    "hotpotqa", "2wikimqa", "musique", "narrativeqa", "multifieldqa_en"
}
SUMMARY_DATASETS = {"gov_report", "qmsum", "multi_news"}
BABILONG_TASK_LABELS = {
    "qa1": ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"],
    "qa2": ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"],
    "qa3": ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"],
    "qa6": ["no", "yes"],
    "qa9": ["no", "yes"],
}


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def normalized_exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def qa_f1(prediction: str, ground_truth: str) -> float:
    pred = normalize_answer(prediction).split()
    gold = normalize_answer(ground_truth).split()
    common = Counter(pred) & Counter(gold)
    overlap = sum(common.values())
    if overlap == 0 or not pred or not gold:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def rouge_l_f1(prediction: str, ground_truth: str) -> float:
    """LongBench's official ``rouge==1.0.1`` ROUGE-L F1."""
    from rouge import Rouge

    try:
        return float(Rouge().get_scores([prediction], [ground_truth], avg=True)["rouge-l"]["f"])
    except (ValueError, RecursionError):
        return 0.0


def babilong_score(task: str, prediction: str, target: str, question: str) -> float:
    """BABILong's official label-aware ``compare_answers`` for selected tasks."""
    output = prediction.lower().split(".")[0]
    output = output.split("<context>")[0].split("<example>")[0].split("question")[0]
    labels = {label.lower() for label in BABILONG_TASK_LABELS[task]}
    labels_in_question = {label for label in labels if label in question.lower()}
    labels_in_output = {label for label in labels if label in output} - labels_in_question
    target = target.lower()
    if "," in target and len(target) > 3:
        subtargets = target.split(",")
        return float(all(item in labels_in_output for item in subtargets) and
                     len(labels_in_output) == len(subtargets))
    return float(target in labels_in_output and len(labels_in_output) == 1)


def ruler_score(task: str, prediction: str, references: list[str]) -> float:
    """Per-sample equivalent of RULER's official substring metrics."""
    prediction = prediction.lower()
    hits = [float(reference.lower() in prediction) for reference in references]
    if not hits:
        return 0.0
    return sum(hits) / len(hits) if task == "vt" else max(hits)


def retrieval_score(prediction: str, ground_truth: str) -> float:
    match = re.search(r"Paragraph (\d+)", ground_truth)
    if not match:
        return 0.0
    numbers = re.findall(r"\d+", prediction)
    return 0.0 if not numbers else sum(x == match.group(1) for x in numbers) / len(numbers)


def load_longbench_rows(path: Path, dataset: str, prompts: dict[str, str]) -> list[dict[str, Any]]:
    template = prompts[dataset]
    rows = []
    for index, line in enumerate(path.read_text().splitlines()):
        item = json.loads(line)
        rows.append({
            "id": item.get("_id", index),
            "prompt": template.format(**item),
            "answers": list(item.get("answers") or [item.get("answer", "")]),
        })
    return rows


def load_babilong_rows(path: Path) -> list[dict[str, Any]]:
    items = json.loads(path.read_text())
    return [{
        "id": index,
        "prompt": f"{item['input'].rstrip()}\nQuestion: {item['question'].strip()}\nAnswer:",
        "question": item["question"].strip(),
        "answers": [str(item["target"])],
    } for index, item in enumerate(items)]


def load_ruler_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for index, line in enumerate(path.read_text().splitlines()):
        item = json.loads(line)
        rows.append({"id": item.get("index", index), "prompt": item["input"],
                     "answers": list(item.get("outputs") or [])})
    return rows


def metric_for(suite: str, dataset: str):
    if dataset in SUMMARY_DATASETS:
        return rouge_l_f1
    if dataset == "passage_retrieval_en":
        return retrieval_score
    return qa_f1


def truncate_ids(tokenizer, prompt: str, max_input_tokens: int, device: str) -> torch.Tensor:
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if ids.numel() > max_input_tokens:
        left = max_input_tokens // 2
        ids = torch.cat([ids[:left], ids[-(max_input_tokens - left):]])
    if ids.numel() < 2:
        raise ValueError("Prompt must contain at least two tokens")
    return ids.unsqueeze(0).to(device)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("longbench", "babilong", "ruler"), required=True)
    parser.add_argument("--datasets", required=True, help="Comma-separated dataset names or relative paths")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--longbench_config_dir", default="/root/benchmarks/LongBench/LongBench/config")
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--arm", choices=("dynamic", "raw"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--max_input_tokens", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--plan_steps", type=int, default=100)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--blend", type=float, default=0.7)
    parser.add_argument("--diffusion_sampler", choices=("ddpm", "ddim"), default="ddpm")
    parser.add_argument("--expected_step", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


@torch.no_grad()
def main(argv=None):
    args = parse_args(argv)
    from models.laces_latent_refiner import LACESRuntime
    from scripts.eval.relay_utils import load_relay_model

    data_dir = Path(args.data_dir)
    prompt_map = {}
    max_gen_map = {}
    if args.suite == "longbench":
        config_dir = Path(args.longbench_config_dir)
        prompt_map = json.loads((config_dir / "dataset2prompt.json").read_text())
        max_gen_map = json.loads((config_dir / "dataset2maxlen.json").read_text())

    model, _rwkv, tokenizer, checkpoint, _cfg = load_relay_model(args.ckpt_dir, args.device)
    runtime = LACESRuntime(
        model, checkpoint, expected_step=args.expected_step, plan_steps=args.plan_steps,
        cfg_scale=args.cfg_scale, blend=args.blend,
        diffusion_sampler=args.diffusion_sampler, protocol="aligned",
    )
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)

    started = time.time()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_out = []
    aggregates = {}
    for dataset_spec in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        if args.suite == "longbench":
            dataset = dataset_spec
            rows = load_longbench_rows(data_dir / f"{dataset}.jsonl", dataset, prompt_map)
        elif args.suite == "babilong":
            dataset = dataset_spec.replace("/", "_")
            rows = load_babilong_rows(data_dir / f"{dataset_spec}.json")
        else:
            dataset = dataset_spec.replace("/", "_")
            rows = load_ruler_rows(data_dir / dataset_spec / "validation.jsonl")
        rows = rows[:args.max_samples]
        metric = metric_for(args.suite, dataset_spec)
        scores = []
        for index, row in enumerate(rows):
            prefix_ids = truncate_ids(tokenizer, row["prompt"], args.max_input_tokens, args.device)
            if args.arm == "dynamic":
                z, _hidden, _prefix = runtime.prepare(prefix_ids, seed=args.seed + index)
            else:
                z = torch.zeros(1, int(model.trajectory_horizon), int(model.latent_dim), device=args.device)
            generation_limit = min(
                args.max_new_tokens,
                int(max_gen_map.get(dataset_spec, args.max_new_tokens)),
                int(model.trajectory_horizon) * int(model.trajectory_chunk_size),
            )
            token_ids, logps = runtime.generate(
                prefix_ids, z, max_new_tokens=generation_limit, raw=args.arm == "raw",
                temperature=0.0, repetition_penalty=1.0,
                eos_id=getattr(tokenizer, "eos_token_id", None),
            )
            prediction = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            if args.suite == "babilong":
                task = dataset_spec.split("/")[0]
                score = babilong_score(task, prediction, row["answers"][0], row["question"])
            elif args.suite == "ruler":
                task = dataset_spec.split("/")[-1]
                score = ruler_score(task, prediction, row["answers"])
            else:
                score = max((metric(prediction, answer) for answer in row["answers"]), default=0.0)
            scores.append(score)
            rows_out.append({
                "suite": args.suite, "dataset": dataset, "id": row["id"], "arm": args.arm,
                "input_tokens": int(prefix_ids.shape[1]), "prediction": prediction,
                "prediction_token_ids": token_ids, "mean_logp": sum(logps) / len(logps) if logps else None,
                "answers": row["answers"], "score": score,
            })
            print(f"[{args.suite}/{dataset}/{args.arm}] {index + 1}/{len(rows)} score={score:.3f}", flush=True)
        aggregates[dataset] = {"n": len(scores), "score": sum(scores) / len(scores) if scores else 0.0}

    peak_mb = None
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        peak_mb = torch.cuda.max_memory_allocated(args.device) / 1024**2
    result = {
        "config": vars(args), "checkpoint_step": checkpoint.get("step"), "audit": runtime.audit,
        "aggregates": aggregates, "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_mb": peak_mb, "rows": rows_out,
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    runtime.close()
    print(json.dumps({"aggregates": aggregates, "peak_gpu_memory_mb": peak_mb}, indent=2), flush=True)


if __name__ == "__main__":
    main()
