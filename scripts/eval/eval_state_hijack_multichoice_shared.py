#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.eval.relay_utils import list_npz_files, load_relay_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--task_dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max_samples", type=int, default=2000)
    p.add_argument("--conditions", default="raw,ddim100")
    p.add_argument("--length_normalize", action="store_true", default=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bf16", choices=("bf16", "fp32"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--progress_every", type=int, default=50)
    return p.parse_args()


def parse_condition(name: str):
    name = name.strip().lower()
    if name == "raw":
        return name, None, None
    for method in ("ddpm", "ddim", "flow"):
        if name.startswith(method):
            return name, method, int(name[len(method):])
    raise ValueError(f"Unknown condition: {name}")


def save_json(path: str, payload: Mapping[str, Any]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


@torch.no_grad()
def sample_states(model, method: str, steps: int, device: str, dtype, seed: int):
    torch.manual_seed(seed)
    if getattr(model, "trajectory_enabled", False):
        previous = getattr(model, "_gen_type", "ddpm")
        if method == "flow":
            model._gen_type = "flow"
        try:
            z_traj = model.trajectory_sample(1, num_steps=steps, device=device, dtype=dtype)
        finally:
            model._gen_type = previous
        trajectory_states = model.predict_trajectory_states(z_traj)
        initial_chunk_states = [state[:, 0] for state in trajectory_states]
        return initial_chunk_states
    if method == "ddpm":
        z = model.ddpm_sample(1, num_steps=steps, device=device, dtype=dtype)
        return model.predict_states(z)
    if method == "ddim":
        z = model.ddim_sample(1, num_steps=steps, device=device, dtype=dtype)
        return model.predict_states(z)
    if method == "flow":
        previous = getattr(model, "_gen_type", "ddpm")
        model._gen_type = "flow"
        try:
            z = model.ddim_sample(1, num_steps=steps, device=device, dtype=dtype)
        finally:
            model._gen_type = previous
        return model.predict_states(z)
    raise ValueError(method)


def span_score(logits, ids, mask, choice_start, length_normalize):
    if choice_start <= 0:
        choice_start = 1
    pred_logits = logits[0, choice_start - 1:ids.shape[1] - 1].float()
    targets = ids[0, choice_start:ids.shape[1]]
    valid = mask[0, choice_start:ids.shape[1]].bool()
    if valid.sum() == 0:
        return float("inf")
    log_probs = F.log_softmax(pred_logits, dim=-1)
    nll = -log_probs.gather(1, targets.unsqueeze(-1)).squeeze(-1)
    nll = nll[valid]
    total = float(nll.sum().item())
    if length_normalize:
        return total / max(1, int(nll.numel()))
    return total


@torch.no_grad()
def score_raw(model, ids, mask, choice_start, length_normalize):
    out = model.rwkv_model(input_ids=ids, attention_mask=mask.bool(), return_dict=True)
    return span_score(out.logits, ids, mask, choice_start, length_normalize)


@torch.no_grad()
def score_injected(model, ids, mask, choice_start, states, length_normalize):
    out_pool = model.rwkv_model(
        input_ids=ids,
        attention_mask=mask.bool(),
        output_hidden_states=True,
        use_cache=True,
        return_dict=True,
    )
    cache = model.inject_into_cache(out_pool.past_key_values, states)
    out = model.rwkv_model(
        input_ids=ids,
        attention_mask=mask.bool(),
        past_key_values=cache,
        use_cache=False,
        return_dict=True,
    )
    return span_score(out.logits, ids, mask, choice_start, length_normalize)


def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model_raw, _rwkv, _tokenizer, ckpt, cfg_raw = load_relay_model(args.ckpt_dir, args.device)
    model = cast(Any, model_raw)
    cfg = cast(DictConfig, cfg_raw)
    files = list_npz_files(args.task_dir, args.max_samples)
    task_name = os.path.basename(args.task_dir.rstrip("/"))
    parsed = [parse_condition(x) for x in args.conditions.split(",") if x.strip()]
    results = {
        "ckpt_dir": args.ckpt_dir,
        "step": ckpt.get("step", -1),
        "checkpoint_gen_type": cfg.training.get("gen_type"),
        "task": task_name,
        "task_dir": args.task_dir,
        "n_files": len(files),
        "conditions": args.conditions,
        "shared_state_per_example": True,
    }
    save_json(args.output, results)
    for cond_i, (name, method, steps) in enumerate(parsed):
        correct = 0
        t0 = time.time()
        for file_i, fp in enumerate(files):
            d = np.load(fp)
            input_ids = d["input_ids"]
            attn_mask = d["attention_mask"]
            choice_start = d["choice_start"]
            label = int(d["label"][0])
            states = None
            if method is not None:
                if steps is None:
                    raise RuntimeError(f"condition {name} has no step count")
                states = sample_states(model, method, steps, args.device, dtype, args.seed + 100000 * (cond_i + 1) + file_i)
            scores = []
            for j in range(input_ids.shape[0]):
                ids = torch.from_numpy(input_ids[j:j + 1].astype(np.int64)).to(args.device)
                mask = torch.from_numpy(attn_mask[j:j + 1].astype(np.int64)).to(args.device)
                cs = int(choice_start[j])
                if states is None:
                    score = score_raw(model, ids, mask, cs, args.length_normalize)
                else:
                    score = score_injected(model, ids, mask, cs, states, args.length_normalize)
                scores.append(score)
            pred = int(np.argmin(scores))
            correct += int(pred == label)
            if args.progress_every and ((file_i + 1) % args.progress_every == 0 or file_i + 1 == len(files)):
                print(f"{name}: files={file_i + 1}/{len(files)} acc={correct / max(1, file_i + 1):.4f} elapsed={time.time() - t0:.1f}s", flush=True)
        results[f"acc_{name}"] = correct / max(1, len(files))
        results[f"correct_{name}"] = correct
        results[f"elapsed_{name}_s"] = round(time.time() - t0, 1)
        save_json(args.output, results)
        print(f"{name}: acc={results[f'acc_{name}']:.4f} correct={correct}/{len(files)}", flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
