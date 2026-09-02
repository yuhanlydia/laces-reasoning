#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.eval.relay_utils import load_relay_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=("bf16", "fp32"))
    p.add_argument("--conditions", default="raw,ddim100")
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


def save_json(path: str, payload: dict[str, Any]):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def load_ids_mask_target(fp: str, device: str):
    data = np.load(fp)
    ids_raw = data["input_ids"].astype(np.int64)
    mask_raw = data.get("attention_mask")
    if mask_raw is None:
        mask_raw = np.ones_like(ids_raw, dtype=np.int64)
    else:
        mask_raw = mask_raw.astype(np.int64)
    if ids_raw.ndim == 1:
        ids_raw = ids_raw[None, :]
        mask_raw = mask_raw[None, :]
    target_start = int(data["target_start"][0]) if "target_start" in data else max(1, ids_raw.shape[1] // 2)
    return torch.from_numpy(ids_raw).to(device), torch.from_numpy(mask_raw).to(device), target_start


@torch.no_grad()
def sample_z(model, method: str, steps: int, batch_size: int, device: str, dtype, seed: int):
    torch.manual_seed(seed)
    if method == "ddpm":
        return model.ddpm_sample(batch_size, num_steps=steps, device=device, dtype=dtype)
    if method == "ddim":
        return model.ddim_sample(batch_size, num_steps=steps, device=device, dtype=dtype)
    if method == "flow":
        previous = getattr(model, "_gen_type", "ddpm")
        model._gen_type = "flow"
        try:
            return model.ddim_sample(batch_size, num_steps=steps, device=device, dtype=dtype)
        finally:
            model._gen_type = previous
    raise ValueError(method)


@torch.no_grad()
def target_nll_from_logits(logits, ids, mask, target_start):
    pred_logits = logits[:, target_start - 1:ids.shape[1] - 1].contiguous().float()
    labels = ids[:, target_start:].contiguous()
    score_mask = mask[:, target_start:].contiguous().float()
    loss = F.cross_entropy(pred_logits.reshape(-1, pred_logits.size(-1)), labels.reshape(-1), reduction="none")
    loss = loss.reshape(labels.shape)
    return float((loss * score_mask).sum().item()), float(score_mask.sum().item())


@torch.no_grad()
def eval_condition(model, files, name, method, steps, device, dtype, seed_base, progress_every):
    total_nll = 0.0
    total_tok = 0.0
    n_docs = 0
    t0 = time.time()
    for idx, fp in enumerate(files):
        ids, mask, target_start = load_ids_mask_target(fp, device)
        if ids.shape[1] < 2 or target_start >= ids.shape[1]:
            continue
        if method is None:
            out = model.rwkv_model(input_ids=ids, attention_mask=mask.bool(), return_dict=True)
            nll, ntok = target_nll_from_logits(out.logits, ids, mask, target_start)
        else:
            prefix_ids = ids[:, :target_start]
            prefix_mask = mask[:, :target_start]
            suffix_context = ids[:, target_start - 1:]
            suffix_mask = mask[:, target_start - 1:]
            z = sample_z(model, method, steps, ids.shape[0], device, dtype, seed_base + idx)
            states = model.predict_states(z)
            out_prefix = model.rwkv_model(input_ids=prefix_ids, attention_mask=prefix_mask.bool(), use_cache=True, return_dict=True)
            cache = model.inject_into_cache(out_prefix.past_key_values, states)
            out = model.rwkv_model(input_ids=suffix_context, attention_mask=suffix_mask.bool(), past_key_values=cache, use_cache=False, return_dict=True)
            nll, ntok = target_nll_from_logits(out.logits, suffix_context, suffix_mask, 1)
        total_nll += nll
        total_tok += ntok
        n_docs += ids.shape[0]
        if progress_every and ((idx + 1) % progress_every == 0 or idx + 1 == len(files)):
            avg = total_nll / max(total_tok, 1.0)
            print(f"{name}: files={idx + 1}/{len(files)} docs={n_docs} tokens={int(total_tok)} ppl={math.exp(min(avg, 100.0)):.4f} elapsed={time.time() - t0:.1f}s", flush=True)
    avg_nll = total_nll / total_tok
    return {"name": name, "method": method or "raw", "steps": steps, "ppl": math.exp(min(avg_nll, 100.0)), "avg_nll": avg_nll, "total_tokens": int(total_tok), "n_docs": int(n_docs), "elapsed_s": round(time.time() - t0, 1)}


def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if args.max_samples is not None:
        files = files[:args.max_samples]
    Path(os.path.dirname(args.output) or ".").mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"status": "running", "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "ckpt_dir": args.ckpt_dir, "data_dir": args.data_dir, "output": args.output, "num_files": len(files), "max_samples": args.max_samples, "device": args.device, "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "conditions": []}
    save_json(args.output, summary)
    model, _rwkv, _tokenizer, ckpt, cfg = load_relay_model(args.ckpt_dir, args.device)
    cfg_any = cast(Any, cfg)
    summary["checkpoint_step"] = ckpt.get("step")
    summary["checkpoint_gen_type"] = cfg_any.training.get("gen_type")
    save_json(args.output, summary)
    try:
        for i, cond in enumerate(args.conditions.split(",")):
            name, method, steps = parse_condition(cond)
            summary["conditions"].append(eval_condition(model, files, name, method, steps, args.device, dtype, args.seed + 100000 * (i + 1), args.progress_every))
            save_json(args.output, summary)
        summary["status"] = "completed"
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = repr(exc)
        raise
    finally:
        summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save_json(args.output, summary)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
