#!/usr/bin/env python3
"""LongBench full evaluation for trajectory models.
Supports all 34 LongBench tasks. Context auto-truncated to fit model.
"""
from __future__ import annotations
import argparse, json, os, sys, re
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/tmp/LongBench/LongBench")
from relay_utils import load_relay_model
from sample_prefix_suffix_trajectory_cfg import (
    encode_prefix, sample_trajectory_cfg, apply_repetition_penalty,
)
from metrics import qa_f1_score, retrieval_score, count_score

DATA_DIR = "/tmp/LongBench/LongBench/data/data"
CONFIG_DIR = "/tmp/LongBench/LongBench/config"

# All tasks with their metrics
TASK_METRICS = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": qa_f1_score,
    "gov_report": qa_f1_score,
    "qmsum": qa_f1_score,
    "multi_news": qa_f1_score,
    "vcsum": qa_f1_score,
    "trec": qa_f1_score,
    "triviaqa": qa_f1_score,
    "samsum": qa_f1_score,
    "lsht": qa_f1_score,
    "passage_retrieval_en": retrieval_score,
    "passage_retrieval_zh": retrieval_score,
    "passage_count": count_score,
    "lcc": qa_f1_score,
    "repobench-p": qa_f1_score,
    # English variants:
    "narrativeqa_e": qa_f1_score,
    "qasper_e": qa_f1_score,
    "multifieldqa_en_e": qa_f1_score,
    "hotpotqa_e": qa_f1_score,
    "2wikimqa_e": qa_f1_score,
    "musique_e": qa_f1_score,
    "gov_report_e": qa_f1_score,
    "qmsum_e": qa_f1_score,
    "multi_news_e": qa_f1_score,
    "trec_e": qa_f1_score,
    "triviaqa_e": qa_f1_score,
    "samsum_e": qa_f1_score,
    "lcc_e": qa_f1_score,
    "repobench-p_e": qa_f1_score,
    "passage_retrieval_en_e": retrieval_score,
    "passage_count_e": count_score,
}

DATASET2PROMPT = json.load(open(f"{CONFIG_DIR}/dataset2prompt.json"))
DATASET2MAXGEN = json.load(open(f"{CONFIG_DIR}/dataset2maxlen.json"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--datasets", default=None, help="Comma-separated, default: all available")
    p.add_argument("--max_length", type=int, default=3800, help="Max context length")
    p.add_argument("--sample_steps", type=int, default=100)
    p.add_argument("--cfg_scale", type=float, default=3.0)
    p.add_argument("--max_samples", type=int, default=0, help="0=all")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trajectory_state_blend", type=float, default=None)
    return p.parse_args()


@torch.no_grad()
def evaluate_dataset(model, tokenizer, dataset_name, args):
    data_path = Path(DATA_DIR) / f"{dataset_name}.jsonl"
    if not data_path.exists():
        print(f"  [SKIP] {dataset_name}: data not found")
        return None

    prompt_template = DATASET2PROMPT.get(dataset_name, "")
    if not prompt_template:
        print(f"  [SKIP] {dataset_name}: no prompt template")
        return None

    max_gen = DATASET2MAXGEN.get(dataset_name, 128)
    metric = TASK_METRICS.get(dataset_name, qa_f1_score)
    base_ds = dataset_name.replace("_e", "") if dataset_name.endswith("_e") else dataset_name
    use_english = dataset_name.endswith("_e")

    samples = []
    with open(data_path) as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    if args.max_samples > 0:
        samples = samples[:args.max_samples]

    predictions = []
    ground_truths = []
    all_classes = None

    for i, sample in enumerate(samples):
        if (i + 1) % 10 == 0:
            print(f"    [{dataset_name}] {i+1}/{len(samples)}", flush=True)

        # Build prompt
        if use_english:
            context = sample.get("context", sample.get("input", ""))
            question = sample.get("input", sample.get("question", ""))
        else:
            context = sample.get("context", "")
            question = sample.get("input", sample.get("question", ""))

        prompt = prompt_template.format(**sample)

        # Truncate to fit context window
        tokenized = tokenizer(prompt, truncation=False, return_tensors="pt")
        token_ids = tokenized.input_ids[0]
        if len(token_ids) > args.max_length:
            # Middle truncation: keep start + end
            half = args.max_length // 2
            token_ids = torch.cat([token_ids[:half], token_ids[-half:]])
            prompt = tokenizer.decode(token_ids, skip_special_tokens=True)

        # Encode prefix and sample trajectory
        input_ids = token_ids.unsqueeze(0).to(args.device)
        attention_mask = torch.ones_like(input_ids)

        torch.manual_seed(args.seed + i)
        z_prefix, prefix_cache, prefix_logits = encode_prefix(model, input_ids, attention_mask)
        z_traj = sample_trajectory_cfg(model, z_prefix, args.sample_steps, args.cfg_scale, args.device, torch.bfloat16)

        # Generate using proven trajectory chunk-wise decode
        generated_ids = list(input_ids[0].tolist())
        past_kv = prefix_cache
        logits = prefix_logits
        chunk_size = int(model.trajectory_chunk_size)

        for h in range(z_traj.shape[1]):
            if len(generated_ids) - input_ids.shape[1] >= args.max_new_tokens:
                break
            states_h = model.predict_states(z_traj[:, h])
            past_kv = model.inject_into_cache(past_kv, states_h)
            for _ in range(chunk_size):
                if len(generated_ids) - input_ids.shape[1] >= args.max_new_tokens:
                    break
                logits = apply_repetition_penalty(logits.float(), generated_ids, 1.0)
                next_id = int(torch.argmax(logits).item())
                generated_ids.append(next_id)
                out = model.rwkv_model(
                    input_ids=torch.tensor([[next_id]], device=args.device),
                    past_key_values=past_kv, use_cache=True, return_dict=True,
                )
                past_kv = out.past_key_values
                logits = out.logits[0, -1]

        new_ids = generated_ids[input_ids.shape[1]:]
        prediction = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        ground_truths.append(sample.get("answers", sample.get("answer", "")))
        predictions.append(prediction)

    def score_one(pred, gts):
        gts_list = [gts] if isinstance(gts, str) else (list(gts) if gts else [""])
        best = 0.0
        for gt in gts_list:
            try:
                s = metric(pred, gt, all_classes=all_classes)
            except TypeError:
                s = metric(pred, gt)
            best = max(best, s)
        return best

    per_sample = [score_one(p, g) for p, g in zip(predictions, ground_truths)]
    score = 100.0 * sum(per_sample) / len(per_sample) if per_sample else 0.0
    print(f"  [DONE] {dataset_name}: score={score:.4f}", flush=True)
    return {"dataset": dataset_name, "score": score, "n_samples": len(samples), "predictions": predictions[:3]}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    model, _rwkv, tokenizer, ckpt, cfg = load_relay_model(args.ckpt_dir, args.device)
    model._prefix_suffix_trajectory_s2 = True
    model._training_stage = 2
    model._cfg_drop_prob = float(cfg.training.get("cfg_drop_prob", 0.0))
    if args.trajectory_state_blend is not None:
        model.trajectory_state_blend = float(args.trajectory_state_blend)
        if hasattr(model, "config"):
            model.config.trajectory_state_blend = float(args.trajectory_state_blend)

    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",")]
    else:
        datasets = sorted([f.stem for f in Path(DATA_DIR).glob("*.jsonl") if f.stem in TASK_METRICS])

    print(f"LongBench eval: {len(datasets)} tasks, max_length={args.max_length}, cfg={args.cfg_scale}")
    results = []
    for ds in datasets:
        print(f"\n--- {ds} ---")
        r = evaluate_dataset(model, tokenizer, ds, args)
        if r:
            results.append(r)

    output = {
        "ckpt_dir": args.ckpt_dir,
        "cfg_scale": args.cfg_scale,
        "max_length": args.max_length,
        "n_tasks": len(results),
        "results": {r["dataset"]: {"score": r["score"], "n": r["n_samples"]} for r in results},
        "avg_score": np.mean([r["score"] for r in results]) if results else 0,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n=== SUMMARY: avg={output['avg_score']:.3f} across {len(results)} tasks ===")
    for r in results:
        print(f"  {r['dataset']}: {r['score']:.4f}")


if __name__ == "__main__":
    main()
