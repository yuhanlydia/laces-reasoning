#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.eval.relay_utils import load_relay_model
from scripts.eval.run_cola_dlm_tasks_prefix_suffix_cfg import (
    build_prompt,
    get_ground_truth,
    iter_jsonl,
    sample_next_token,
)


DEFAULT_CKPT = "outputs_relay/test-v6-2.9B-s2-prefix-suffix-cfg/step_00150000"
DEFAULT_OUTPUT_ROOT = "/tmp/diffrwkv_traj_diag_cleanz"
REFERENCE_ACCURACY = {
    "trajectory": {"avg": 45.7},
    "single_z": {"avg": 56.3},
    "raw": {"avg": 52.7},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Zero-training clean-Z diagnostic: encode each 32-token suffix chunk with "
            "the trained single-z encoder and reuse its own predict_states bridge per chunk."
        )
    )
    p.add_argument("--ckpt_dir", default=DEFAULT_CKPT)
    p.add_argument("--task_data_dir", default="baseline/Cola-DLM/eval_output/tasks_default")
    p.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--json_output", default=str(Path(DEFAULT_OUTPUT_ROOT) / "reuse_singlez_projection_eval.json"))
    p.add_argument("--tasks", default="mmlu,obqa,race")
    p.add_argument("--max_samples", type=int, default=100)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--chunk_size", type=int, default=32)
    p.add_argument("--blends", default="0.4,1.0")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _set_pad_token(tokenizer: Any) -> int:
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is not None:
        return int(pad_id)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
        return int(eos_id)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is not None:
        tokenizer.pad_token = tokenizer.unk_token
        return int(unk_id)
    return 0


def _tokenize_without_specials(tokenizer: Any, text: str) -> list[int]:
    try:
        ids = tokenizer(text, add_special_tokens=False).input_ids
    except TypeError:
        ids = tokenizer(text).input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def chunk_suffix_ids(tokenizer: Any, suffix_text: str, chunk_size: int, pad_id: int) -> list[list[int]]:
    ids = _tokenize_without_specials(tokenizer, suffix_text)
    if not ids:
        eos_id = getattr(tokenizer, "eos_token_id", None)
        ids = [int(eos_id if eos_id is not None else pad_id)]
    chunks = [ids[i : i + chunk_size] for i in range(0, len(ids), chunk_size)]
    return chunks or [[pad_id]]


@torch.no_grad()
def encode_single_z_chunk(model: Any, chunk_ids: list[int], device: str) -> torch.Tensor:
    input_ids = torch.tensor([chunk_ids], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    out = model.rwkv_model(
        input_ids=input_ids,
        attention_mask=attention_mask.bool(),
        output_hidden_states=True,
        use_cache=True,
        return_dict=True,
    )
    pooled = model._pool_hidden(out.hidden_states[-1], attention_mask)
    z_h, _kl = model._encode_pooled(pooled)
    expected = (1, int(model.latent_dim))
    if tuple(z_h.shape) != expected:
        raise RuntimeError(f"single-z chunk z_h shape {tuple(z_h.shape)} != expected {expected}")
    return z_h


@torch.no_grad()
def encode_single_z_chunks(
    model: Any,
    tokenizer: Any,
    suffix_text: str,
    chunk_size: int,
    device: str,
    pad_id: int,
) -> tuple[list[torch.Tensor], list[int], list[float]]:
    chunks = chunk_suffix_ids(tokenizer, suffix_text, chunk_size, pad_id)
    z_chunks: list[torch.Tensor] = []
    norms: list[float] = []
    for chunk in chunks:
        z_h = encode_single_z_chunk(model, chunk, device)
        z_chunks.append(z_h)
        norms.append(float(z_h.detach().float().norm(dim=-1).mean().item()))
    return z_chunks, [len(c) for c in chunks], norms


@torch.no_grad()
def generate_with_reused_singlez_projection(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    z_chunks: list[torch.Tensor],
    blend: float,
    args: argparse.Namespace,
) -> tuple[str, int, int, list[int], list[float]]:
    attention_mask = torch.ones_like(input_ids)
    out = model.rwkv_model(input_ids=input_ids, attention_mask=attention_mask.bool(), use_cache=True, return_dict=True)
    cache = out.past_key_values
    logits = out.logits[0, -1]
    all_ids = list(input_ids[0].tolist())
    new_ids: list[int] = []
    injected_chunks: list[int] = []
    state_norms: list[float] = []
    eos_id = getattr(tokenizer, "eos_token_id", None)
    stop = False

    for h, z_h in enumerate(z_chunks):
        if stop or len(new_ids) >= int(args.max_new_tokens):
            break
        states_h = model.predict_states(z_h)
        state_norms.append(float(torch.stack([s.detach().float().norm() for s in states_h]).mean().item()))
        cache = model.blend_into_cache(cache, states_h, float(blend))
        injected_chunks.append(h)
        if h == 0:
            out = model.rwkv_model(input_ids=input_ids, past_key_values=cache, use_cache=True, return_dict=True)
            cache = out.past_key_values
            logits = out.logits[0, -1]
        for _ in range(int(args.chunk_size)):
            if len(new_ids) >= int(args.max_new_tokens):
                stop = True
                break
            next_id = sample_next_token(logits, all_ids, args)
            if eos_id is not None and next_id == eos_id:
                stop = True
                break
            new_ids.append(next_id)
            all_ids.append(next_id)
            out = model.rwkv_model(
                input_ids=torch.tensor([[next_id]], device=input_ids.device),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = out.past_key_values
            logits = out.logits[0, -1]

    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return text, int(input_ids.shape[1]), len(new_ids), injected_chunks, state_norms


def blend_alias(blend: float) -> str:
    return f"blend{str(blend).replace('.', 'p')}"


def load_acc_calc_module() -> Any:
    path = Path(__file__).resolve().parents[2] / "baseline" / "Cola-DLM" / "scripts" / "acc_calc.py"
    spec = importlib.util.spec_from_file_location("cola_acc_calc_reuse_singlez", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_accuracy_csv(summary_csv: Path) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    with summary_csv.open(encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        aliases = header[1:]
        for alias in aliases:
            table[alias] = {}
        for row in reader:
            if not row:
                continue
            task = row[0]
            key = "avg" if task == "tasks_average" else task.upper()
            for alias, value in zip(aliases, row[1:]):
                if value != "":
                    table[alias][key] = float(value)
    return table


def run_acc_calc(eval_root: Path, summary_csv: Path) -> dict[str, dict[str, float]]:
    script = Path(__file__).resolve().parents[2] / "baseline" / "Cola-DLM" / "scripts" / "acc_calc.py"
    subprocess.run([sys.executable, str(script), str(eval_root), str(summary_csv)], check=True)
    return parse_accuracy_csv(summary_csv)


def evaluate_blend(
    model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    pad_id: int,
    tasks: list[str],
    blend: float,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"blend": float(blend), "tasks": {}, "sanity": {}}
    for task_i, task in enumerate(tasks):
        input_path = Path(args.task_data_dir) / f"{task}.jsonl"
        if not input_path.exists():
            print(f"[SKIP] missing {input_path}", flush=True)
            continue
        output_path = output_dir / f"{task}.jsonl"
        n = 0
        first_z_shape = None
        first_z_norm = None
        first_state_norm = None
        with output_path.open("w", encoding="utf-8") as out_f:
            for sample_i, item in iter_jsonl(input_path, int(args.max_samples)):
                prompt = build_prompt(task, item)
                gt = get_ground_truth(item)
                torch.manual_seed(int(args.seed) + 100000 * task_i + sample_i)
                z_chunks, clean_chunk_tokens, z_norms = encode_single_z_chunks(
                    model, tokenizer, gt, int(args.chunk_size), str(args.device), pad_id
                )
                if first_z_shape is None:
                    first_z_shape = list(z_chunks[0].shape)
                    first_z_norm = z_norms[0]
                    print(
                        f"[SANITY] {task} {blend_alias(blend)} z_h.shape={first_z_shape} "
                        f"first_norm={first_z_norm:.6f} chunks={len(z_chunks)}",
                        flush=True,
                    )
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
                generated, prompt_tokens, generated_tokens, injected_chunks, state_norms = generate_with_reused_singlez_projection(
                    model, tokenizer, input_ids, z_chunks, blend, args
                )
                if first_state_norm is None and state_norms:
                    first_state_norm = state_norms[0]
                rec = dict(item)
                rec.update(
                    {
                        "id": item.get("id", sample_i),
                        "prompt": prompt,
                        "generate": generated,
                        "ground_truth": gt,
                        "choices": item.get("choices", []),
                        "prompt_tokens": prompt_tokens,
                        "generated_tokens": generated_tokens,
                        "method": "reuse_singlez_projection_per_chunk",
                        "bridge": "single_z_encoder_plus_predict_states_per_32tok_chunk",
                        "blend": float(blend),
                        "chunk_size": int(args.chunk_size),
                        "clean_chunk_tokens": clean_chunk_tokens,
                        "z_h_shapes": [list(z.shape) for z in z_chunks],
                        "z_h_norms": z_norms,
                        "state_norms": state_norms,
                        "injected_chunks": injected_chunks,
                    }
                )
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                if n % 20 == 0:
                    print(f"[{task} {blend_alias(blend)}] {n} samples", flush=True)
        summary["tasks"][task] = {"samples": n, "output": str(output_path)}
        summary["sanity"][task] = {
            "first_z_h_shape": first_z_shape,
            "first_z_h_norm": first_z_norm,
            "first_state_norm": first_state_norm,
        }
        print(f"[DONE] {task} {blend_alias(blend)}: {n} -> {output_path}", flush=True)
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def format_accuracy_table(accuracy_by_alias: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for alias, scores in sorted(accuracy_by_alias.items()):
        rows.append(
            {
                "method": f"reuse-single-z-per-chunk ({alias})",
                "MMLU": scores.get("MMLU"),
                "OBQA": scores.get("OBQA"),
                "RACE": scores.get("RACE"),
                "avg": scores.get("avg"),
            }
        )
    rows.extend(
        [
            {"method": "trajectory", "MMLU": None, "OBQA": None, "RACE": None, "avg": 45.7},
            {"method": "single-z", "MMLU": None, "OBQA": None, "RACE": None, "avg": 56.3},
            {"method": "raw", "MMLU": None, "OBQA": None, "RACE": None, "avg": 52.7},
        ]
    )
    return rows


def print_accuracy_table(rows: list[dict[str, Any]]) -> None:
    print("\naccuracy table (%):")
    print(f"{'method':48s} {'MMLU':>8s} {'OBQA':>8s} {'RACE':>8s} {'avg':>8s}")
    for row in rows:
        def fmt(value: Any) -> str:
            return "--" if value is None else f"{float(value):.2f}"

        print(
            f"{row['method'][:48]:48s} {fmt(row.get('MMLU')):>8s} {fmt(row.get('OBQA')):>8s} "
            f"{fmt(row.get('RACE')):>8s} {fmt(row.get('avg')):>8s}"
        )


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    tasks = [t.strip() for t in str(args.tasks).split(",") if t.strip()]
    blends = [float(x.strip()) for x in str(args.blends).split(",") if x.strip()]
    output_root = Path(args.output_root)
    eval_root = output_root / "reuse_singlez_projection_eval_output"
    eval_root.mkdir(parents=True, exist_ok=True)

    model, _rwkv, tokenizer, ckpt, cfg = load_relay_model(str(args.ckpt_dir), str(args.device))
    model_any = cast(Any, model)
    tokenizer_any = cast(Any, tokenizer)
    cfg_any = cast(Any, cfg)
    model_any._prefix_suffix_s2 = True
    model_any._training_stage = 2
    model_any._cfg_drop_prob = float(cfg_any.training.get("cfg_drop_prob", 0.0))
    pad_id = _set_pad_token(tokenizer_any)

    generation_summaries: dict[str, Any] = {}
    for blend in blends:
        alias = f"reuse_singlez_per_chunk_{blend_alias(blend)}"
        generation_summaries[alias] = evaluate_blend(
            model_any,
            tokenizer_any,
            args,
            pad_id,
            tasks,
            blend,
            eval_root / f"tasks_{alias}",
        )
        gc.collect()
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    summary_csv = output_root / "reuse_singlez_projection_accuracy_summary.csv"
    accuracy_by_alias = run_acc_calc(eval_root, summary_csv)
    rows = format_accuracy_table(accuracy_by_alias)
    print_accuracy_table(rows)

    result = {
        "experiment": "reuse_singlez_projection_per_chunk_clean_z",
        "critical_caveat_handled": (
            "Each 32-token suffix chunk is encoded with the single-z checkpoint's own encoder "
            "before calling the same checkpoint's predict_states; no trajectory-S0 latents are used."
        ),
        "ckpt_dir": str(args.ckpt_dir),
        "checkpoint_step": ckpt.get("step", -1),
        "tasks": tasks,
        "max_samples_per_task": int(args.max_samples),
        "max_new_tokens": int(args.max_new_tokens),
        "chunk_size": int(args.chunk_size),
        "decoding": {
            "temperature": float(args.temperature),
            "top_k": int(args.top_k),
            "top_p": float(args.top_p),
            "repetition_penalty": float(args.repetition_penalty),
        },
        "generation_summaries": generation_summaries,
        "accuracy_summary_csv": str(summary_csv),
        "accuracy_by_alias": accuracy_by_alias,
        "accuracy_table": rows,
        "references": REFERENCE_ACCURACY,
    }
    json_output = Path(args.json_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    with json_output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[RESULT] {json_output}", flush=True)


if __name__ == "__main__":
    main()
