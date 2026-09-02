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
    p.add_argument("--conditions", default="raw,ddpm100,ddpm1000")
    p.add_argument("--trajectory_sampler", default=None, choices=(None, "rf_heun"))
    p.add_argument("--trajectory_state_blend", type=float, default=None,
                   help="Override blend; default uses the checkpoint's trained value.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--progress_every", type=int, default=50)
    return p.parse_args()


def parse_condition(name: str):
    name = name.strip().lower()
    if name == "raw":
        return name, None, None
    for method in ("ddpm", "ddim", "flow", "rf"):
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
def sample_trajectory(model, method, steps, batch_size, device, dtype, sampler, seed):
    torch.manual_seed(seed)
    previous = getattr(model, "_gen_type", "ddpm")
    model._gen_type = method
    try:
        return model.trajectory_sample(
            batch_size, num_steps=steps, device=device, dtype=dtype, sampler=sampler
        )
    finally:
        model._gen_type = previous


@torch.no_grad()
def chunk_ce(logits, labels, mask):
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = mask[:, 1:].contiguous().float()
    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
    ).reshape(shift_labels.shape)
    return float((loss * shift_mask).sum().item()), float(shift_mask.sum().item())


@torch.no_grad()
def eval_condition(model, files, name, method, steps, device, dtype, sampler, blend,
                   seed_base, progress_every):
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
            nll, n_tok = chunk_ce(out.logits, ids, mask)
        else:
            chunks, chunk_mask, H_eff, C = model._trajectory_view(ids, mask)
            B = chunks.shape[0]
            z_traj = sample_trajectory(model, method, steps, B, device, dtype, sampler, seed_base + idx)
            z_traj = z_traj[:, :H_eff]
            layer_states = model.predict_trajectory_states(z_traj)

            bootstrap = model.rwkv_model(input_ids=chunks[:, 0, :1], use_cache=True, return_dict=True)
            cache = bootstrap.past_key_values
            nll = 0.0
            n_tok = 0.0
            for h in range(H_eff):
                states_h = [ls[:, h] for ls in layer_states]
                cache = model.blend_into_cache(cache, states_h, blend)
                mask_h = chunk_mask[:, h].bool() if chunk_mask is not None else None
                out_h = model.rwkv_model(
                    input_ids=chunks[:, h],
                    attention_mask=mask_h,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
                cache = out_h.past_key_values
                m_h = chunk_mask[:, h].float() if chunk_mask is not None else torch.ones_like(chunks[:, h]).float()
                c_nll, c_tok = chunk_ce(out_h.logits, chunks[:, h], m_h)
                nll += c_nll
                n_tok += c_tok

        total_nll += nll
        total_tok += n_tok
        n_docs += ids.shape[0]

        if progress_every and ((idx + 1) % progress_every == 0 or idx + 1 == len(files)):
            avg = total_nll / max(total_tok, 1.0)
            print(
                f"{name}: files={idx + 1}/{len(files)} docs={n_docs} "
                f"tokens={int(total_tok)} ppl={math.exp(min(avg, 100.0)):.4f} "
                f"elapsed={time.time() - t0:.1f}s",
                flush=True,
            )

    avg_nll = total_nll / max(total_tok, 1.0)
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

    print(f"Loading model from {args.ckpt_dir}", flush=True)
    model, _rwkv, _tokenizer, ckpt, cfg = load_relay_model(args.ckpt_dir, args.device)
    model._gen_type = "ddpm"
    model.eval()

    if args.trajectory_state_blend is not None:
        model.trajectory_state_blend = float(args.trajectory_state_blend)
    blend = float(getattr(model, "trajectory_state_blend",
                          cfg.model.get("trajectory_state_blend", 1.0)))
    blend_source = "cli" if args.trajectory_state_blend is not None else "checkpoint"
    print(f"trajectory_state_blend = {blend} ({blend_source}); "
          f"s1_mode={getattr(model, 'trajectory_s1_mode', '?')} "
          f"chunk={getattr(model, 'trajectory_chunk_size', '?')} "
          f"horizon={getattr(model, 'trajectory_horizon', '?')}", flush=True)

    summary = {
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ckpt_dir": args.ckpt_dir,
        "data_dir": args.data_dir,
        "num_files": len(files),
        "max_samples": args.max_samples,
        "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "checkpoint_step": ckpt.get("step"),
        "checkpoint_gen_type": cfg.training.get("gen_type"),
        "trajectory_state_blend": blend,
        "trajectory_state_blend_source": blend_source,
        "trajectory_s1_mode": str(getattr(model, "trajectory_s1_mode", "?")),
        "conditions": [],
    }
    save_json(args.output, summary)

    try:
        for i, cond in enumerate(args.conditions.split(",")):
            name, method, steps = parse_condition(cond)
            print(f"\n=== {name} START ===", flush=True)
            result = eval_condition(
                model, files, name, method, steps, args.device, dtype,
                args.trajectory_sampler, blend,
                args.seed + 100000 * (i + 1), args.progress_every,
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
