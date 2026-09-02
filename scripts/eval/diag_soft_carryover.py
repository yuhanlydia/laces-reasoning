#!/usr/bin/env python3
"""Soft-blending carryover diagnostic.

state_carryover=0% because inject_into_cache:
  1. REPLACES recurrent_state entirely
  2. ZEROS conv_state/ffn_state
  3. RESETS _seen_tokens=0

This script tests 3 fix strategies:
  A) blend_states: inject blended (alpha*mem1 + (1-alpha)*mem2) instead of hard switch
  B) soft_inject: modify recurrent_state in-place, PRESERVE conv/ffn/_seen_tokens
  C) soft_inject_blend: both A+B combined

Usage:
  CUDA_VISIBLE_DEVICES=1 python -u scripts/eval/diag_soft_carryover.py
"""
from __future__ import annotations
import json, sys, random, traceback
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from scripts.eval.sample_prefix_suffix_trajectory_cfg import encode_prefix
from scripts.eval.exp_multiagent_star_fusion import capture_final_state, decode_chunkwise
from scripts.eval.relay_utils import load_relay_model
from scripts.eval.run_cola_dlm_tasks_prefix_suffix_trajectory_cfg import (
    recompute_logits_from_injected_cache, sample_next_token,
)
from scripts.eval.fusion_ablation_pca import gen_tasks

CKPT = str(REPO / "outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000")
NUM_TASKS = 20
STEPS = 100
CFG = 3.0
SEED = 42
MAX_NEW = 64


def inject_into_cache_soft(model, cache, predicted_states, alpha=1.0):
    """Soft inject: blend recurrent_state, PRESERVE conv/ffn/_seen_tokens.

    alpha=1.0: pure replacement (same as original inject_into_cache)
    alpha=0.0: pure existing state (no change to recurrent)
    alpha=0.75: 75% new + 25% existing (soft blend)
    """
    for l, st in enumerate(predicted_states):
        layer = cache.layers[l]
        if layer.state is None:
            layer.state = {
                "recurrent_state": None, "attn_state": None,
                "conv_state": None, "ffn_state": None,
            }
        cur = layer.state.get("recurrent_state")
        new_st = st.to(torch.float32)
        if cur is not None and alpha < 1.0:
            # Blend: alpha*new + (1-alpha)*existing
            layer.state["recurrent_state"] = alpha * new_st + (1.0 - alpha) * cur
        else:
            layer.state["recurrent_state"] = new_st
        # PRESERVE conv_state/ffn_state (don't zero them)
        # PRESERVE _seen_tokens (don't reset)
    return cache


@torch.no_grad()
def decode_carryover_soft(model, tokenizer, q_ids, mem1, mem2, H, args, mode="blend", alpha=0.75):
    """Soft carryover: blend mem1/mem2 or preserve running state.

    mode="blend": for second half, inject alpha*mem1 + (1-alpha)*mem2
    mode="soft_inject": for second half, use inject_into_cache_soft with alpha
    mode="soft_inject_blend": both — blend states AND soft-inject
    """
    chunk_size = int(model.trajectory_chunk_size)
    device = q_ids.device
    eos_id = getattr(tokenizer, "eos_token_id", None)

    # Step 1: process query prefix
    out = model.rwkv_model(input_ids=q_ids, attention_mask=torch.ones_like(q_ids).bool(),
                           use_cache=True, return_dict=True)
    past_kv = out.past_key_values
    all_ids = list(q_ids[0].tolist())
    new_ids: list[int] = []
    stop = False
    half_H = H // 2

    for h in range(H):
        if stop or len(new_ids) >= args.max_new_tokens:
            break

        if h < half_H:
            # First half: inject mem1 (agent-1 knowledge)
            states_h = mem1
            past_kv = model.inject_into_cache(past_kv, states_h)
        else:
            # Second half: soft approach
            if mode == "blend":
                # Blend mem1 and mem2, then hard-inject (still zeros conv/ffn)
                blended = []
                for l in range(len(mem1)):
                    blended.append(alpha * mem1[l] + (1.0 - alpha) * mem2[l])
                past_kv = model.inject_into_cache(past_kv, blended)
            elif mode == "soft_inject":
                # Hard switch to mem2, but soft-inject (preserve conv/ffn/_seen)
                past_kv = inject_into_cache_soft(model, past_kv, mem2, alpha=alpha)
            elif mode == "soft_inject_blend":
                # Blend states AND soft-inject
                blended = []
                for l in range(len(mem1)):
                    blended.append(alpha * mem1[l] + (1.0 - alpha) * mem2[l])
                past_kv = inject_into_cache_soft(model, past_kv, blended, alpha=alpha)

        ctx = torch.tensor([all_ids], device=device, dtype=torch.long)
        past_kv, lb = recompute_logits_from_injected_cache(model, ctx, torch.ones_like(ctx), past_kv)
        logits = lb[0]
        for _ in range(chunk_size):
            if len(new_ids) >= args.max_new_tokens:
                stop = True
                break
            nid = sample_next_token(logits, all_ids, args)
            if eos_id is not None and nid == eos_id:
                stop = True
                break
            new_ids.append(nid)
            all_ids.append(nid)
            out = model.rwkv_model(input_ids=torch.tensor([[nid]], device=device),
                                   past_key_values=past_kv, use_cache=True, return_dict=True)
            past_kv = out.past_key_values
            logits = out.logits[0, -1]
    return new_ids


@torch.no_grad()
def main():
    device = torch.device("cuda:0")
    print(f"[init] Loading model from {CKPT}", flush=True)
    model, _rwkv, tokenizer, _ckpt, _cfg = load_relay_model(CKPT, str(device))
    model.eval()
    model._prefix_suffix_trajectory_s2 = True
    dtype = next(model.alpha_heads.parameters()).dtype
    H = int(model.trajectory_horizon)
    print(f"[init] H={H} dtype={dtype}", flush=True)

    def enc(t):
        return tokenizer(t, return_tensors="pt").input_ids.to(device)

    tasks = gen_tasks(NUM_TASKS, SEED)

    # Collect per-task latents + memory states
    print(f"[pass0] Collecting states for {len(tasks)} tasks...", flush=True)
    task_data = []
    for ii, it in enumerate(tasks):
        a_ids, b_ids = enc(it["a"]), enc(it["b"])
        mem1 = capture_final_state(model, a_ids)
        mem2 = capture_final_state(model, b_ids)
        task_data.append((it, mem1, mem2))
        if (ii + 1) % 10 == 0:
            print(f"[pass0] {ii+1}/{len(tasks)}", flush=True)

    # Conditions to test
    conditions = [
        # (name, decode_fn, kwargs)
        ("agent2_only", "agent_only", {"mem": "mem2"}),  # floor baseline
        ("agent1_only", "agent_only", {"mem": "mem1"}),  # floor baseline
        ("hard_carryover", "carryover", {"alpha": 0.0}),  # baseline: hard switch
        ("blend_0.75_0.25", "carryover_soft", {"mode": "blend", "alpha": 0.75}),
        ("blend_0.5_0.5", "carryover_soft", {"mode": "blend", "alpha": 0.5}),
        ("blend_0.25_0.75", "carryover_soft", {"mode": "blend", "alpha": 0.25}),
        ("soft_inject_0.75", "carryover_soft", {"mode": "soft_inject", "alpha": 0.75}),
        ("soft_inject_0.5", "carryover_soft", {"mode": "soft_inject", "alpha": 0.5}),
        ("soft_inject_blend_0.75", "carryover_soft", {"mode": "soft_inject_blend", "alpha": 0.75}),
    ]

    class Args:
        max_new_tokens = MAX_NEW
        temperature = 0.0  # greedy
        top_k = 1
        top_p = 1.0
        repetition_penalty = 1.0

    args = Args()
    results = {name: {"hits": 0, "total": 0} for name, _, _ in conditions}

    print(f"\n[eval] Testing {len(conditions)} conditions on {len(tasks)} tasks", flush=True)
    print("=" * 80, flush=True)

    for ti, (it, mem1, mem2) in enumerate(task_data):
        q_ids = enc(it["q"])

        for cond_name, decode_type, kwargs in conditions:
            try:
                torch.cuda.empty_cache()
                if decode_type == "agent_only":
                    # Simple baseline: inject one agent's memory for all chunks
                    mem = mem1 if kwargs.get("mem") == "mem1" else mem2
                    from scripts.eval.exp_multiagent_star_fusion import decode_chunkwise
                    ids = decode_chunkwise(model, tokenizer, q_ids, lambda h: mem, H, args)
                elif decode_type == "carryover":
                    # Original hard carryover (from fusion_ablation_pca.py)
                    from scripts.eval.fusion_ablation_pca import decode_sequential_carryover
                    ids = decode_sequential_carryover(model, tokenizer, q_ids, mem1, mem2, H, args)
                elif decode_type == "carryover_soft":
                    ids = decode_carryover_soft(model, tokenizer, q_ids, mem1, mem2, H, args, **kwargs)
                else:
                    ids = []

                text = tokenizer.decode(ids, skip_special_tokens=True).strip().lower()
                hit = it["gold"] in text
                results[cond_name]["hits"] += int(hit)
                results[cond_name]["total"] += 1

            except Exception as e:
                print(f"  ERROR [{cond_name}] task {ti}: {e}", flush=True)
                traceback.print_exc()
                results[cond_name]["total"] += 1

        if (ti + 1) % 5 == 0:
            print(f"[eval] {ti+1}/{len(tasks)} done", flush=True)
            for name, _, _ in conditions:
                r = results[name]
                acc = r["hits"] / max(r["total"], 1) * 100
                print(f"  {name:25s} = {acc:.1f}% ({r['hits']}/{r['total']})", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("FINAL RESULTS", flush=True)
    print("=" * 80, flush=True)
    for name, _, _ in conditions:
        r = results[name]
        acc = r["hits"] / max(r["total"], 1) * 100
        print(f"  {name:25s} = {acc:.1f}% ({r['hits']}/{r['total']})", flush=True)

    # Save
    out = {name: r for name, r in results.items()}
    out_path = REPO / "results/fusion_ablation/soft_carryover_diag.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
