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
from scripts.eval.sample_prefix_suffix_cfg import encode_prefix, sample_ddim_cfg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=("bf16", "fp32"))
    p.add_argument("--scales", default="0.5,1,5,10")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--progress_every", type=int, default=50)
    p.add_argument("--split", default="half", choices=("half", "target_start"))
    return p.parse_args()


def save_json(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def load_ids_mask_target(fp, device):
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
    ids = torch.from_numpy(ids_raw).to(device)
    mask = torch.from_numpy(mask_raw).to(device)
    target_start = None
    if "target_start" in data:
        target_start = int(data["target_start"][0])
    return ids, mask, target_start


def split_index(ids, mask, target_start, mode):
    length = int(mask[0].sum().item()) if mask is not None else ids.shape[1]
    if mode == "target_start" and target_start is not None:
        return max(1, min(target_start, length - 1))
    return max(1, min(length // 2, length - 1))


@torch.no_grad()
def score_raw_suffix(model, ids, mask, start):
    out = model.rwkv_model(input_ids=ids, attention_mask=mask.bool(), return_dict=True)
    logits = out.logits[:, start - 1:ids.shape[1] - 1].contiguous().float()
    labels = ids[:, start:].contiguous()
    score_mask = mask[:, start:].contiguous().float()
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="none")
    loss = loss.reshape(labels.shape)
    return float((loss * score_mask).sum().item()), float(score_mask.sum().item())


@torch.no_grad()
def score_cfg_suffix(model, ids, mask, start, scale, steps, device, dtype, seed):
    prefix_ids = ids[:, :start]
    prefix_mask = mask[:, :start]
    suffix_context = ids[:, start - 1:]
    suffix_mask = mask[:, start - 1:]
    torch.manual_seed(seed)
    cond = encode_prefix(model, prefix_ids, prefix_mask)
    z = sample_ddim_cfg(model, cond, steps, scale, device, dtype)
    states = model.predict_states(z)
    out_prefix = model.rwkv_model(
        input_ids=prefix_ids,
        attention_mask=prefix_mask.bool(),
        use_cache=True,
        return_dict=True,
    )
    cache = model.inject_into_cache(out_prefix.past_key_values, states)
    out = model.rwkv_model(
        input_ids=suffix_context,
        attention_mask=suffix_mask.bool(),
        past_key_values=cache,
        use_cache=False,
        return_dict=True,
    )
    logits = out.logits[:, :-1].contiguous().float()
    labels = suffix_context[:, 1:].contiguous()
    score_mask = suffix_mask[:, 1:].contiguous().float()
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="none")
    loss = loss.reshape(labels.shape)
    return float((loss * score_mask).sum().item()), float(score_mask.sum().item())


def eval_raw(model, files, args):
    total_nll = 0.0
    total_tok = 0.0
    n_docs = 0
    t0 = time.time()
    for idx, fp in enumerate(files):
        ids, mask, target_start = load_ids_mask_target(fp, args.device)
        if ids.shape[1] < 2:
            continue
        start = split_index(ids, mask, target_start, args.split)
        nll, ntok = score_raw_suffix(model, ids, mask, start)
        total_nll += nll
        total_tok += ntok
        n_docs += ids.shape[0]
        if args.progress_every and ((idx + 1) % args.progress_every == 0 or idx + 1 == len(files)):
            avg = total_nll / max(total_tok, 1.0)
            print(f"raw_suffix: files={idx + 1}/{len(files)} tokens={int(total_tok)} ppl={math.exp(min(avg, 100.0)):.4f} elapsed={time.time() - t0:.1f}s", flush=True)
    avg = total_nll / total_tok
    return {"name": "raw_suffix", "method": "raw_suffix", "steps": None, "cfg_scale": None, "ppl": math.exp(min(avg, 100.0)), "avg_nll": avg, "total_tokens": int(total_tok), "n_docs": int(n_docs), "elapsed_s": round(time.time() - t0, 1)}


def eval_scale(model, files, scale, args, dtype, seed_base):
    total_nll = 0.0
    total_tok = 0.0
    n_docs = 0
    t0 = time.time()
    name = f"cfg{scale:g}"
    for idx, fp in enumerate(files):
        ids, mask, target_start = load_ids_mask_target(fp, args.device)
        if ids.shape[1] < 2:
            continue
        start = split_index(ids, mask, target_start, args.split)
        nll, ntok = score_cfg_suffix(model, ids, mask, start, scale, args.steps, args.device, dtype, seed_base + idx)
        total_nll += nll
        total_tok += ntok
        n_docs += ids.shape[0]
        if args.progress_every and ((idx + 1) % args.progress_every == 0 or idx + 1 == len(files)):
            avg = total_nll / max(total_tok, 1.0)
            print(f"{name}: files={idx + 1}/{len(files)} tokens={int(total_tok)} ppl={math.exp(min(avg, 100.0)):.4f} elapsed={time.time() - t0:.1f}s", flush=True)
    avg = total_nll / total_tok
    return {"name": name, "method": "prefix_suffix_cfg", "steps": args.steps, "cfg_scale": scale, "ppl": math.exp(min(avg, 100.0)), "avg_nll": avg, "total_tokens": int(total_tok), "n_docs": int(n_docs), "elapsed_s": round(time.time() - t0, 1)}


def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    scales = [float(x.strip()) for x in args.scales.split(",") if x.strip()]
    files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if args.max_samples is not None:
        files = files[:args.max_samples]
    Path(os.path.dirname(args.output) or ".").mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"status": "running", "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "ckpt_dir": args.ckpt_dir, "data_dir": args.data_dir, "output": args.output, "num_files": len(files), "max_samples": args.max_samples, "device": args.device, "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "steps": args.steps, "split": args.split, "scales": scales, "conditions": []}
    save_json(args.output, summary)
    model, _rwkv, _tokenizer, ckpt, cfg = load_relay_model(args.ckpt_dir, args.device)
    model_any = cast(Any, model)
    cfg_any = cast(Any, cfg)
    model_any._prefix_suffix_s2 = True
    model_any._training_stage = 2
    model_any._cfg_drop_prob = float(cfg_any.training.get("cfg_drop_prob", 0.0))
    summary["checkpoint_step"] = ckpt.get("step")
    summary["checkpoint_gen_type"] = cfg_any.training.get("gen_type")
    save_json(args.output, summary)
    try:
        summary["conditions"].append(eval_raw(model, files, args))
        save_json(args.output, summary)
        for i, scale in enumerate(scales):
            summary["conditions"].append(eval_scale(model, files, scale, args, dtype, args.seed + 100000 * (i + 1)))
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
