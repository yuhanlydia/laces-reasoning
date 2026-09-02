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

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from relay_utils import load_relay_model  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=("bf16", "fp32"))
    p.add_argument("--conditions", default="raw,ddpm100,ddpm1000,ddim100")
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


def save_json(path: str, payload: dict):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def load_ids_and_mask(fp: str, device: str):
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
    return ids, mask


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
def eval_condition(model, files, name, method, steps, device, dtype, seed_base, progress_every):
    total_nll = 0.0
    total_tok = 0.0
    n_docs = 0
    t0 = time.time()

    for idx, fp in enumerate(files):
        ids, mask = load_ids_and_mask(fp, device)
        if ids.shape[1] < 2:
            continue

        if method is None:
            out = model.rwkv_model(input_ids=ids, attention_mask=mask.bool(), return_dict=True)
        else:
            out_pool = model.rwkv_model(
                input_ids=ids,
                attention_mask=mask.bool(),
                output_hidden_states=True,
                use_cache=True,
                return_dict=True,
            )
            cache = out_pool.past_key_values
            z = sample_z(model, method, steps, ids.shape[0], device, dtype, seed_base + idx)
            states = model.predict_states(z)
            cache = model.inject_into_cache(cache, states)
            out = model.rwkv_model(
                input_ids=ids,
                attention_mask=mask.bool(),
                past_key_values=cache,
                use_cache=False,
                return_dict=True,
            )

        logits = out.logits
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = ids[:, 1:].contiguous()
        shift_mask = mask[:, 1:].contiguous().float()
        loss_per_pos = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
        ).reshape(shift_labels.shape)

        n_tok = float(shift_mask.sum().item())
        nll = float((loss_per_pos * shift_mask).sum().item())
        total_nll += nll
        total_tok += n_tok
        n_docs += ids.shape[0]

        if progress_every and ((idx + 1) % progress_every == 0 or idx + 1 == len(files)):
            avg = total_nll / max(total_tok, 1.0)
            ppl = math.exp(min(avg, 100.0))
            print(
                f"{name}: files={idx + 1}/{len(files)} docs={n_docs} "
                f"tokens={int(total_tok)} ppl={ppl:.4f} elapsed={time.time() - t0:.1f}s",
                flush=True,
            )

    avg_nll = total_nll / total_tok
    return {
        "name": name,
        "method": method or "raw",
        "steps": steps,
        "ppl": math.exp(min(avg_nll, 100.0)),
        "avg_nll": avg_nll,
        "total_tokens": int(total_tok),
        "n_docs": int(n_docs),
        "elapsed_s": round(time.time() - t0, 1),
    }


def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if args.max_samples is not None:
        files = files[: args.max_samples]
    Path(os.path.dirname(args.output) or ".").mkdir(parents=True, exist_ok=True)

    summary = {
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ckpt_dir": args.ckpt_dir,
        "data_dir": args.data_dir,
        "output": args.output,
        "num_files": len(files),
        "max_samples": args.max_samples,
        "device": args.device,
        "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "conditions": [],
    }
    save_json(args.output, summary)

    print(f"Loading model from {args.ckpt_dir}", flush=True)
    t0 = time.time()
    model, _rwkv, _tokenizer, ckpt, cfg = load_relay_model(args.ckpt_dir, args.device)
    model._gen_type = "ddpm"
    model.eval()
    summary["checkpoint_step"] = ckpt.get("step")
    summary["checkpoint_gen_type"] = cfg.training.get("gen_type")
    summary["model_load_s"] = round(time.time() - t0, 1)
    save_json(args.output, summary)
    print(
        f"Loaded step={summary['checkpoint_step']} gen={summary['checkpoint_gen_type']} "
        f"files={len(files)} load_s={summary['model_load_s']}",
        flush=True,
    )

    try:
        for i, cond in enumerate(args.conditions.split(",")):
            name, method, steps = parse_condition(cond)
            print(f"\n=== {name} START ===", flush=True)
            result = eval_condition(
                model,
                files,
                name,
                method,
                steps,
                args.device,
                dtype,
                args.seed + 100000 * (i + 1),
                args.progress_every,
            )
            summary["conditions"].append(result)
            save_json(args.output, summary)
            print(f"=== {name} DONE === {result}", flush=True)
        summary["status"] = "completed"
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = repr(exc)
        raise
    finally:
        summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save_json(args.output, summary)
        print("\n=== FINAL SUMMARY ===", flush=True)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
