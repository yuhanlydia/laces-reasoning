#!/usr/bin/env python3
"""Iterative refinement inference for trajectory diffusion.
No retraining needed — the model already learned p(Z | any_prefix).

  Round 1: z_prefix → S2 → sample 16 z_h → decode first REFINE_CHUNKS
  Round 2: re-encode (prompt + decoded) → z_prefix' → S2 → new 16 z_h → decode rest

Usage on 4090:
  CUDA_VISIBLE_DEVICES=0 python scripts/eval/sample_trajectory_iterative_refine.py \
    --ckpt_dir outputs_relay/.../step_00020000 \
    --prompt "The history of artificial intelligence" --output /tmp/refine.json \
    --refine_chunks 8 --steps 100 --cfg_scale 3 --max_new_tokens 512
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.eval.relay_utils import load_relay_model
from scripts.eval.sample_prefix_suffix_trajectory_cfg import (
    apply_repetition_penalty,
    apply_top_p,
    encode_prefix,
    sample_trajectory_cfg,
)


@torch.no_grad()
def decode_trajectory_chunks(model, tokenizer, input_ids, attention_mask, past_kv, logits, z_traj, args, start_h=0):
    """Decode chunks [start_h, horizon) from z_traj. Return (text, past_kv, logits)."""
    chunk_size = int(model.trajectory_chunk_size)
    s1_mode = str(model.config.get("trajectory_s1_mode", "independent"))
    horizon = z_traj.shape[1]
    generated = list(input_ids[0].tolist())

    if s1_mode in ("transformer", "rwkv", "birwkv"):
        layer_states = model.predict_trajectory_states(z_traj)
        blend = float(model.config.get("trajectory_state_blend", 0.7))
    else:
        layer_states = None
        blend = 1.0

    for h in range(start_h, horizon):
        if len(generated) - input_ids.shape[1] >= args.max_new_tokens:
            break
        if s1_mode in ("transformer", "rwkv", "birwkv"):
            states_h = [layer_state[:, h] for layer_state in layer_states]
            past_kv = model.blend_into_cache(past_kv, states_h, blend)
        else:
            states_h = model.predict_states(z_traj[:, h])
            past_kv = model.inject_into_cache(past_kv, states_h)
        for _ in range(chunk_size):
            if len(generated) - input_ids.shape[1] >= args.max_new_tokens:
                break
            logits_f = apply_repetition_penalty(logits.float(), generated, args.repetition_penalty)
            if args.temperature > 0:
                logits_f = logits_f / args.temperature
                probs = torch.softmax(logits_f, dim=-1)
                if args.top_k > 0:
                    topk_vals, topk_idx = torch.topk(probs, args.top_k)
                    probs = torch.zeros_like(probs).scatter(-1, topk_idx, topk_vals)
                    probs = probs / probs.sum().clamp(min=1e-12)
                probs = apply_top_p(probs, args.top_p)
                next_id = torch.multinomial(probs, 1).item()
            else:
                next_id = logits_f.argmax().item()
            generated.append(next_id)
            out = model.rwkv_model(
                input_ids=torch.tensor([[next_id]], device=input_ids.device),
                past_key_values=past_kv, use_cache=True, return_dict=True,
            )
            past_kv = out.past_key_values
            logits = out.logits[0, -1]

    return tokenizer.decode(generated), past_kv, logits


def parse_args():
    p = argparse.ArgumentParser(description="Iterative refinement trajectory sampling")
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--cfg_scale", type=float, default=3.0)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.2)
    p.add_argument("--trajectory_state_blend", type=float, default=None)
    p.add_argument("--trajectory_sampler", choices=("rf_heun",), default=None)
    p.add_argument("--cond_boundary_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--refine_chunks", type=int, default=8, help="Chunks to decode before re-encoding")
    p.add_argument("--no_refine", action="store_true", help="One-shot baseline")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.bfloat16

    model, _rwkv, tokenizer, ckpt, cfg = load_relay_model(args.ckpt_dir, args.device)
    model_any = cast(Any, model)
    tokenizer_any = cast(Any, tokenizer)
    cfg_any = cast(Any, cfg)
    if args.trajectory_state_blend is not None:
        model_any.config.trajectory_state_blend = float(args.trajectory_state_blend)
        model_any.trajectory_state_blend = float(args.trajectory_state_blend)
    model_any._trajectory_sampler = args.trajectory_sampler
    model_any._prefix_suffix_trajectory_s2 = True
    model_any._training_stage = 2
    model_any._cfg_drop_prob = float(cfg_any.training.get("cfg_drop_prob", 0.0))
    if args.cond_boundary_scale != 1.0 and hasattr(model_any.trajectory_dit, "cond_boundary_scale"):
        model_any.trajectory_dit.cond_boundary_scale = float(args.cond_boundary_scale)

    horizon = int(model_any.trajectory_horizon)
    input_ids = tokenizer_any(args.prompt, return_tensors="pt").input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)

    if args.no_refine:
        # ── one-shot baseline ──
        z_prefix, past_kv, logits = encode_prefix(model_any, input_ids, attention_mask)
        z_traj = sample_trajectory_cfg(model_any, z_prefix, args.steps, args.cfg_scale, device, dtype)
        text, _, _ = decode_trajectory_chunks(model_any, tokenizer_any, input_ids, attention_mask,
                                              past_kv, logits, z_traj, args, start_h=0)
        payload = {"text": text, "method": "one-shot", "prompt": args.prompt}
        print(f"[one-shot] {len(text.split())} words")
    else:
        # ── iterative refinement ──
        n_r1 = min(args.refine_chunks, horizon)
        # Round 1: full trajectory, decode first n_r1 chunks
        z_prefix_r1, past_kv_r1, logits_r1 = encode_prefix(model_any, input_ids, attention_mask)
        z_traj_r1 = sample_trajectory_cfg(model_any, z_prefix_r1, args.steps, args.cfg_scale,
                                          device, dtype)
        z_r1 = z_traj_r1[:, :n_r1, :]
        text_r1, past_after_r1, logits_after_r1 = decode_trajectory_chunks(
            model_any, tokenizer_any, input_ids, attention_mask,
            past_kv_r1, logits_r1, z_r1, args, start_h=0)
        print(f"[round1] {n_r1} chunks → {len(text_r1.split())} words")

        # Round 2: re-encode (prompt + r1 decoded) → new trajectory → decode remaining
        extended = args.prompt + " " + text_r1
        ext_ids = tokenizer_any(extended, return_tensors="pt").input_ids.to(device)
        ext_mask = torch.ones_like(ext_ids)
        z_prefix_r2, _, _ = encode_prefix(model_any, ext_ids, ext_mask)
        z_traj_r2 = sample_trajectory_cfg(model_any, z_prefix_r2, args.steps, args.cfg_scale,
                                          device, dtype)
        n_r2 = horizon - n_r1
        z_r2 = z_traj_r2[:, -n_r2:, :] if n_r2 > 0 else z_traj_r2[:, :0, :]
        text_r2, _, _ = decode_trajectory_chunks(
            model_any, tokenizer_any, input_ids, attention_mask,
            past_after_r1, logits_after_r1, z_r2, args, start_h=0)
        print(f"[round2] {n_r2} chunks → {len(text_r2.split())} words")

        full = text_r1 + text_r2
        payload = {"text": full, "text_r1": text_r1, "text_r2": text_r2,
                   "method": f"refine_{n_r1}chunks", "prompt": args.prompt,
                   "extended_prompt": extended}
        print(f"[refine] total {len(full.split())} words")
        print(f"  r1: {text_r1[:200]}...")
        print(f"  r2: {text_r2[:200]}...")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
