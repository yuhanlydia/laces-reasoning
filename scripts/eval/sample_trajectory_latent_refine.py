#!/usr/bin/env python3
"""Latent-space iterative refinement sampling (train-free, form-2).

After sampling Z (16 latents), instead of decoding immediately, we RESTART:
add noise back to an intermediate timestep t_restart, then re-denoise. Repeat
K times. Each restart lets the 16 chunk latents re-coordinate via the BiRWKV
bidirectional scan toward a more on-manifold joint plan (Restart Sampling,
arXiv:2306.14878 applied in latent space).

Then decode the refined Z chunk-by-chunk (same as normal generation).

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/eval/sample_trajectory_latent_refine.py \
    --ckpt_dir outputs_relay/.../step_00020000 \
    --prompt "The history of artificial intelligence" --output /tmp/lr.json \
    --restart_k 3 --restart_t 0.4 --steps 100 --cfg_scale 3 --max_new_tokens 512
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
    cosine_alpha_bar,
    encode_prefix,
    sample_trajectory_cfg,
)


@torch.no_grad()
def ddim_denoise_from(model, z, cond, cfg_scale, t_start, steps, device, dtype):
    """Denoise z from timestep t_start (in [0,1]) down to 0 via DDIM. Returns z0."""
    ts = torch.linspace(float(t_start), 0.0, steps + 1, device=device, dtype=dtype)
    uncond = torch.zeros_like(cond)
    for i in range(steps):
        t_cur, t_nxt = ts[i], ts[i + 1]
        ab_cur = cosine_alpha_bar(t_cur.unsqueeze(0)).to(dtype).clamp(min=1e-4)
        ab_nxt = cosine_alpha_bar(t_nxt.unsqueeze(0)).to(dtype).clamp(min=1e-4)
        t_batch = t_cur.expand(cond.shape[0])
        eps_cond = model.trajectory_dit(z, t_batch, cond=cond)
        if cfg_scale == 1.0:
            eps = eps_cond
        else:
            eps_uncond = model.trajectory_dit(z, t_batch, cond=uncond)
            eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
        z0_pred = (z - (1 - ab_cur).sqrt() * eps) / ab_cur.sqrt()
        z = ab_nxt.sqrt() * z0_pred + (1 - ab_nxt).sqrt() * eps
    return z


@torch.no_grad()
def sample_with_latent_refine(model, cond, steps, cfg_scale, device, dtype,
                              restart_k, restart_t, restart_steps):
    """Full sample, then K restart-refine passes in latent space."""
    # Initial full sample
    z = sample_trajectory_cfg(model, cond, steps, cfg_scale, device, dtype)
    # Restart refinement: noise back to restart_t, re-denoise
    for _ in range(restart_k):
        ab_r = cosine_alpha_bar(torch.tensor([restart_t], device=device, dtype=dtype)).clamp(min=1e-4)
        noise = torch.randn_like(z)
        z_noisy = ab_r.sqrt() * z + (1 - ab_r).sqrt() * noise   # forward-noise z to t_restart
        z = ddim_denoise_from(model, z_noisy, cond, cfg_scale, restart_t, restart_steps, device, dtype)
    return z


@torch.no_grad()
def decode_chunks(model, tokenizer, input_ids, past_kv, logits, z_traj, args, device):
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
    for h in range(horizon):
        if len(generated) - input_ids.shape[1] >= args.max_new_tokens:
            break
        if s1_mode in ("transformer", "rwkv", "birwkv"):
            states_h = [ls[:, h] for ls in layer_states]
            past_kv = model.blend_into_cache(past_kv, states_h, blend)
        else:
            states_h = model.predict_states(z_traj[:, h])
            past_kv = model.inject_into_cache(past_kv, states_h)
        for _ in range(chunk_size):
            if len(generated) - input_ids.shape[1] >= args.max_new_tokens:
                break
            lf = apply_repetition_penalty(logits.float(), generated, args.repetition_penalty)
            if args.temperature > 0:
                lf = lf / args.temperature
                probs = torch.softmax(lf, dim=-1)
                if args.top_k > 0:
                    tv, ti = torch.topk(probs, args.top_k)
                    probs = torch.zeros_like(probs).scatter(-1, ti, tv)
                    probs = probs / probs.sum().clamp(min=1e-12)
                probs = apply_top_p(probs, args.top_p)
                nid = torch.multinomial(probs, 1).item()
            else:
                nid = lf.argmax().item()
            generated.append(nid)
            out = model.rwkv_model(input_ids=torch.tensor([[nid]], device=device),
                                   past_key_values=past_kv, use_cache=True, return_dict=True)
            past_kv = out.past_key_values
            logits = out.logits[0, -1]
    return tokenizer.decode(generated)


def parse_args():
    p = argparse.ArgumentParser()
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
    p.add_argument("--cond_boundary_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--restart_k", type=int, default=0, help="Number of restart-refine passes (0=baseline)")
    p.add_argument("--restart_t", type=float, default=0.4, help="Timestep to noise back to (0-1)")
    p.add_argument("--restart_steps", type=int, default=50, help="Denoise steps per restart")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.bfloat16
    model, _rwkv, tokenizer, ckpt, cfg = load_relay_model(args.ckpt_dir, args.device)
    m = cast(Any, model); tk = cast(Any, tokenizer); cf = cast(Any, cfg)
    if args.trajectory_state_blend is not None:
        m.config.trajectory_state_blend = float(args.trajectory_state_blend)
        m.trajectory_state_blend = float(args.trajectory_state_blend)
    m._prefix_suffix_trajectory_s2 = True
    m._training_stage = 2
    m._cfg_drop_prob = float(cf.training.get("cfg_drop_prob", 0.0))
    if args.cond_boundary_scale != 1.0 and hasattr(m.trajectory_dit, "cond_boundary_scale"):
        m.trajectory_dit.cond_boundary_scale = float(args.cond_boundary_scale)

    input_ids = tk(args.prompt, return_tensors="pt").input_ids.to(device)
    attn = torch.ones_like(input_ids)
    z_prefix, past_kv, logits = encode_prefix(m, input_ids, attn)

    if args.restart_k == 0:
        z_traj = sample_trajectory_cfg(m, z_prefix, args.steps, args.cfg_scale, device, dtype)
        method = "baseline"
    else:
        z_traj = sample_with_latent_refine(m, z_prefix, args.steps, args.cfg_scale, device, dtype,
                                           args.restart_k, args.restart_t, args.restart_steps)
        method = f"latent-refine-k{args.restart_k}-t{args.restart_t}"

    text = decode_chunks(m, tk, input_ids, past_kv, logits, z_traj, args, device)
    payload = {"text": text, "method": method, "prompt": args.prompt,
               "restart_k": args.restart_k, "restart_t": args.restart_t}
    print(f"[{method}] {len(text.split())} words")
    print(text[:400])
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
