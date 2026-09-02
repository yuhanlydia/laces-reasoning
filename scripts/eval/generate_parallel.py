#!/usr/bin/env python3
"""Parallel chunk decoding for StateDiffRWKV trajectory models.

SERIAL baseline (existing generate() in sample_prefix_suffix_trajectory_cfg.py):
    for h in range(H):
        cache = blend_into_cache(prev_cache, states_h, blend)   # depends on chunk h-1
        for _ in range(chunk_size): AR-decode 1 token           # serial cache chain
Each chunk waits for the previous chunk -> fully serial over H*C tokens.

PARALLEL decoder (this file):
    all H chunks start from their OWN planned state only (blend=1 semantics:
    recurrent_state = planned, no mixing with a previous chunk's cache). The H
    chunks are stacked into a batch of size H and decoded token-by-token IN
    PARALLEL -- one rwkv forward of batch H per intra-chunk step, C steps total,
    instead of H*C serial steps. This is valid iff blend=1 decouples chunks
    (verified: cosine 0.9999, argmax ~97.7% vs serial).

Both share identical per-token sampling (temperature/top_k/top_p/rep-penalty),
so any output difference is purely the serial-vs-parallel state decoupling.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, cast

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.eval.relay_utils import load_relay_model
from scripts.eval.sample_prefix_suffix_trajectory_cfg import (
    apply_repetition_penalty,
    apply_top_p,
    encode_prefix,
    sample_trajectory_cfg,
)


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _sample_next(logits_1d, generated_ids, args):
    """Mirror the serial generate() per-token sampling exactly. logits_1d: [V]."""
    logits = apply_repetition_penalty(logits_1d.float(), generated_ids, args.repetition_penalty)
    if args.temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits / args.temperature
    probs = torch.softmax(logits, dim=-1)
    if args.top_k > 0:
        topk_vals, topk_idx = torch.topk(probs, args.top_k)
        probs = torch.zeros_like(probs).scatter(-1, topk_idx, topk_vals)
        probs = probs / probs.sum().clamp(min=1e-12)
    probs = apply_top_p(probs, args.top_p)
    return int(torch.multinomial(probs, 1).item())


@torch.no_grad()
def generate_serial(model, tokenizer, input_ids, prefix_cache, prefix_logits, z_traj, args, blend):
    """Serial reference (same as repo generate(), condensed)."""
    C = int(model.trajectory_chunk_size)
    generated = list(input_ids[0].tolist())
    past_kv = prefix_cache
    logits = prefix_logits
    layer_states = model.predict_trajectory_states(z_traj)
    H = z_traj.shape[1]
    for h in range(H):
        states_h = [ls[:, h] for ls in layer_states]
        past_kv = model.blend_into_cache(past_kv, states_h, blend)
        for _ in range(C):
            if len(generated) - input_ids.shape[1] >= args.max_new_tokens:
                break
            nid = _sample_next(logits, generated, args)
            generated.append(nid)
            out = model.rwkv_model(
                input_ids=torch.tensor([[nid]], device=input_ids.device),
                past_key_values=past_kv, use_cache=True, return_dict=True)
            past_kv = out.past_key_values
            logits = out.logits[0, -1]
    return tokenizer.decode(generated[input_ids.shape[1]:])


@torch.no_grad()
def generate_parallel(model, tokenizer, z_traj, args, first_tokens):
    """Parallel: all H chunks decoded simultaneously from their own planned state.

    Each chunk's FIRST token is sampled from ITS OWN planned-state logits (not a
    shared seed), so chunk h opens with what its own plan predicts. This removes
    the shared-seed degeneracy where every chunk started identically and produced
    repeated openings. first_tokens[h] is only the neutral primer used to build the
    fresh per-chunk cache before injecting chunk h's state.
    """
    rwkv = model.rwkv_model
    device = z_traj.device
    C = int(model.trajectory_chunk_size)
    H = z_traj.shape[1]
    layer_states = model.predict_trajectory_states(z_traj)

    seed = first_tokens.view(H, 1)
    out = rwkv(input_ids=seed, use_cache=True, return_dict=True)
    states_all = [ls[0] for ls in layer_states]
    cache = model.inject_into_cache(out.past_key_values, states_all)

    out = rwkv(input_ids=seed, past_key_values=cache, use_cache=True, return_dict=True)
    cache = out.past_key_values
    logits = out.logits[:, -1, :]

    per_chunk_tokens = [[] for _ in range(H)]
    first_ids = torch.empty(H, 1, dtype=torch.long, device=device)
    for h in range(H):
        nid = _sample_next(logits[h], per_chunk_tokens[h], args)
        per_chunk_tokens[h].append(nid)
        first_ids[h, 0] = nid
    out = rwkv(input_ids=first_ids, past_key_values=cache, use_cache=True, return_dict=True)
    cache = out.past_key_values
    logits = out.logits[:, -1, :]

    for _ in range(C - 1):
        next_ids = torch.empty(H, 1, dtype=torch.long, device=device)
        for h in range(H):
            nid = _sample_next(logits[h], per_chunk_tokens[h], args)
            per_chunk_tokens[h].append(nid)
            next_ids[h, 0] = nid
        out = rwkv(input_ids=next_ids, past_key_values=cache, use_cache=True, return_dict=True)
        cache = out.past_key_values
        logits = out.logits[:, -1, :]

    flat = [tok for chunk in per_chunk_tokens for tok in chunk]
    return tokenizer.decode(flat), per_chunk_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--prompt", default="The history of artificial intelligence")
    ap.add_argument("--output", default="outputs_eval/parallel_decode.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cfg_scale", type=float, default=3.0)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)  # greedy for clean speed compare
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.2)
    ap.add_argument("--blend", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compare_serial", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dtype = torch.bfloat16
    model, _rwkv, tokenizer, ckpt, cfg = load_relay_model(args.ckpt_dir, args.device)
    model_any = cast(Any, model)
    model_any._prefix_suffix_trajectory_s2 = True
    model_any._training_stage = 2
    model_any._cfg_drop_prob = float(cfg.training.get("cfg_drop_prob", 0.0))
    H = int(model.trajectory_horizon)
    C = int(model.trajectory_chunk_size)

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(args.device)
    attention_mask = torch.ones_like(input_ids)
    z_prefix, prefix_cache, prefix_logits = encode_prefix(model_any, input_ids, attention_mask)
    z_traj = sample_trajectory_cfg(model_any, z_prefix, args.steps, args.cfg_scale, args.device, dtype)

    # seed tokens for each chunk: chunk 0 from prefix next-token argmax; others reuse it
    first_tok0 = int(prefix_logits.float().argmax().item())
    first_tokens = torch.full((H,), first_tok0, dtype=torch.long, device=args.device)

    _sync(args.device)
    t0 = time.perf_counter()
    par_text, per_chunk = generate_parallel(model_any, tokenizer, z_traj, args, first_tokens)
    _sync(args.device)
    par_ms = (time.perf_counter() - t0) * 1000.0

    result = {
        "ckpt": args.ckpt_dir, "prompt": args.prompt, "blend": args.blend,
        "H": H, "C": C, "parallel_ms": par_ms, "parallel_text": par_text,
    }

    if args.compare_serial:
        _sync(args.device)
        t0 = time.perf_counter()
        ser_text = generate_serial(model_any, tokenizer, input_ids, prefix_cache,
                                   prefix_logits, z_traj, args, args.blend)
        _sync(args.device)
        ser_ms = (time.perf_counter() - t0) * 1000.0
        result["serial_ms"] = ser_ms
        result["serial_text"] = ser_text
        result["wallclock_speedup"] = ser_ms / par_ms if par_ms > 0 else 0.0
        print(f"serial={ser_ms:.0f}ms  parallel={par_ms:.0f}ms  speedup={result['wallclock_speedup']:.2f}x")
    else:
        print(f"parallel={par_ms:.0f}ms (H={H} chunks, C={C} tokens each)")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"saved: {args.output}")
    print(f"\n--- parallel text (first 400 chars) ---\n{par_text[:400]}")


if __name__ == "__main__":
    main()
