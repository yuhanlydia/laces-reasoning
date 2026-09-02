#!/usr/bin/env python3
"""Hyperparameter sweep for dual + PCA fusion method.

Sweeps:
  1. Fusion weights: plan/mem1/mem2 ratios
  2. PCA K: 0,1,2,3,4,5,6,7,8
  3. lam: residual scaling factor

Usage: CUDA_VISIBLE_DEVICES=1 python -u scripts/eval/diag_fusion_sweep.py
"""
from __future__ import annotations
import json, sys, random, traceback
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from scripts.eval.sample_prefix_suffix_trajectory_cfg import encode_prefix, sample_trajectory_cfg
from scripts.eval.exp_multiagent_star_fusion import capture_final_state, fuse_states, decode_chunkwise
from scripts.eval.relay_utils import load_relay_model
from scripts.eval.run_cola_dlm_tasks_prefix_suffix_trajectory_cfg import recompute_logits_from_injected_cache, sample_next_token
from scripts.eval.fusion_ablation_pca import gen_tasks, pca_remove_shared

CKPT = str(REPO / "outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000")
NUM_TASKS = 10
STEPS = 100
CFG = 3.0
SEED = 42
MAX_NEW = 24


@torch.no_grad()
def run():
    torch.manual_seed(SEED)
    device = torch.device("cuda:0")
    print(f"[init] Loading model from {CKPT}", flush=True)
    model, _, tokenizer, _, _ = load_relay_model(CKPT, str(device))
    model.eval()
    model._prefix_suffix_trajectory_s2 = True
    dtype = next(model.alpha_heads.parameters()).dtype
    H = int(model.trajectory_horizon)
    print(f"[init] H={H} dtype={dtype}", flush=True)

    def enc(t):
        return tokenizer(t, return_tensors="pt").input_ids.to(device)

    tasks = gen_tasks(NUM_TASKS, SEED)

    # --- Pass 0: collect agent latents + memory states ---
    print(f"[pass0] Collecting states for {len(tasks)} tasks...", flush=True)
    all_z = []
    task_data = []
    for ii, it in enumerate(tasks):
        a_ids, b_ids = enc(it["a"]), enc(it["b"])
        z_a1 = encode_prefix(model, a_ids, torch.ones_like(a_ids))[0].to(dtype)
        z_a2 = encode_prefix(model, b_ids, torch.ones_like(b_ids))[0].to(dtype)
        all_z.append(z_a1[0].float().cpu())
        all_z.append(z_a2[0].float().cpu())
        mem1 = capture_final_state(model, a_ids)
        mem2 = capture_final_state(model, b_ids)
        task_data.append((it, z_a1, z_a2, mem1, mem2))
        if (ii + 1) % 10 == 0:
            print(f"[pass0] {ii+1}/{len(tasks)}", flush=True)

    z_all = torch.stack(all_z, 0).to(device)
    zbar = z_all.mean(dim=0, keepdim=True).to(dtype)
    z_mean_pca = zbar

    # PCA shared dirs for K=0..8
    shared_dirs_cache = {}
    for K in range(9):
        if K == 0:
            shared_dirs_cache[0] = None
        else:
            dirs, sv, mean, _ = pca_remove_shared(z_all, K=K)
            shared_dirs_cache[K] = (dirs, sv, mean)
            sv_ratio = sv[0].item() / sv.sum().item() if K >= 1 else 0
            print(f"[pca] K={K} sv1/total={sv_ratio:.4f}", flush=True)

    # --- Define conditions ---
    # Weight sweep: (w_plan, w_mem1, w_mem2)
    weight_sweep = [
        (0.5, 0.25, 0.25, "dual_0.5_0.25_0.25"),      # current
        (0.5, 0.50, 0.00, "plan_mem1_only"),
        (0.5, 0.00, 0.50, "plan_mem2_only"),
        (0.5, 0.50, 0.50, "plan_full_mems"),            # unnormalized
        (0.3, 0.35, 0.35, "less_plan"),
        (0.7, 0.15, 0.15, "more_plan"),
        (0.0, 0.50, 0.50, "no_plan_stateonly"),         # known 0%
        (1.0, 0.00, 0.00, "plan_only"),                 # known ~0%
    ]

    # PCA K sweep: K=0(dual),1,2,3,4,5,6,7,8
    pca_ks = list(range(9))

    # lam sweep: 0.5, 1.0, 1.5, 2.0
    lams = [0.5, 1.0, 1.5, 2.0]

    class Args:
        max_new_tokens = MAX_NEW
        temperature = 0.5
        top_k = 10
        top_p = 0.9
        repetition_penalty = 1.2

    args = Args()

    # --- Compute plans for each PCA K + lam combination ---
    # plan_states[K][lam] = list of per-layer states per chunk h
    print(f"\n[plan] Computing plans for {len(pca_ks)} K x {len(lams)} lam = {len(pca_ks)*len(lams)} combinations...", flush=True)
    plan_cache = {}  # (K, lam) -> plan_states[h]
    for K in pca_ks:
        for lam in lams:
            key = (K, lam)
            # Compute fused condition
            if K == 0:
                # Mean-subtraction residual (current dual method)
                cond = zbar + lam * (0.5 * (task_data[0][1] - zbar) + 0.5 * (task_data[0][2] - zbar))
                # This is per-task, can't cache globally. Need per-task plan.
                # Skip global caching, compute per-task later
                plan_cache[key] = "per_task"
            else:
                dirs, sv, mean = shared_dirs_cache[K]
                # PCA is global (same dirs for all tasks), but fusion is per-task
                plan_cache[key] = "per_task"

    # --- Evaluate ---
    # We'll test: weight sweep (with K=0, lam=1.0), PCA K sweep (with 0.5/0.25/0.25, lam=1.0), lam sweep (with K=0, 0.5/0.25/0.25)
    results = {}

    def dc(q_ids, fn):
        return decode_chunkwise(model, tokenizer, q_ids, fn, H, args)

    # Phase 1: Weight sweep (K=0, lam=1.0)
    print(f"\n[eval] Phase 1: Weight sweep ({len(weight_sweep)} conditions)", flush=True)
    print("=" * 80, flush=True)
    for wp, wm1, wm2, name in weight_sweep:
        hits = 0
        total = 0
        for ti, (it, z_a1, z_a2, mem1, mem2) in enumerate(task_data):
            try:
                torch.cuda.empty_cache()
                q_ids = enc(it["q"])
                # Compute plan per-task (K=0, lam=1.0)
                cond = zbar + 1.0 * (0.5 * (z_a1 - zbar) + 0.5 * (z_a2 - zbar))
                Z = sample_trajectory_cfg(model, cond, STEPS, CFG, device, dtype)
                plan = [model.predict_states(Z[:, h]) for h in range(H)]
                # Fuse
                fused = dc(q_ids, lambda h: fuse_states([(wp, plan[h]), (wm1, mem1), (wm2, mem2)]))
                text = tokenizer.decode(fused, skip_special_tokens=True).strip().lower()
                if it["gold"] in text:
                    hits += 1
                total += 1
            except Exception as e:
                print(f"  ERROR [{name}] task {ti}: {e}", flush=True)
                total += 1
        acc = hits / total * 100 if total > 0 else 0
        results[name] = {"hits": hits, "total": total, "acc": acc}
        print(f"  {name:30s} = {acc:.1f}% ({hits}/{total})", flush=True)

    # Phase 2: PCA K sweep (weights=0.5/0.25/0.25, lam=1.0)
    print(f"\n[eval] Phase 2: PCA K sweep ({len(pca_ks)} conditions)", flush=True)
    print("=" * 80, flush=True)
    for K in pca_ks:
        name = f"PCA_K{K}" if K > 0 else "dual_K0"
        hits = 0
        total = 0
        for ti, (it, z_a1, z_a2, mem1, mem2) in enumerate(task_data):
            try:
                torch.cuda.empty_cache()
                q_ids = enc(it["q"])
                if K == 0:
                    cond = zbar + 1.0 * (0.5 * (z_a1 - zbar) + 0.5 * (z_a2 - zbar))
                else:
                    dirs, sv, mean = shared_dirs_cache[K]
                    z1c = z_a1[0].float() - mean[0].float()
                    z2c = z_a2[0].float() - mean[0].float()
                    p1 = z1c @ dirs.T
                    p2 = z2c @ dirs.T
                    r1 = z1c - p1 @ dirs
                    r2 = z2c - p2 @ dirs
                    fused_resid = 1.0 * (0.5 * r1 + 0.5 * r2)
                    cond = (mean[0].float() + fused_resid).to(dtype).unsqueeze(0)
                Z = sample_trajectory_cfg(model, cond, STEPS, CFG, device, dtype)
                plan = [model.predict_states(Z[:, h]) for h in range(H)]
                plan_l = plan
                mem1_l, mem2_l = mem1, mem2
                fused = dc(q_ids, lambda h: fuse_states([(0.5, plan_l[h]), (0.25, mem1_l), (0.25, mem2_l)]))
                text = tokenizer.decode(fused, skip_special_tokens=True).strip().lower()
                if it["gold"] in text:
                    hits += 1
                total += 1
            except Exception as e:
                print(f"  ERROR [{name}] task {ti}: {e}", flush=True)
                traceback.print_exc()
                total += 1
        acc = hits / total * 100 if total > 0 else 0
        results[name] = {"hits": hits, "total": total, "acc": acc}
        print(f"  {name:30s} = {acc:.1f}% ({hits}/{total})", flush=True)

    # Phase 3: lam sweep (K=0, weights=0.5/0.25/0.25)
    print(f"\n[eval] Phase 3: lam sweep ({len(lams)} conditions)", flush=True)
    print("=" * 80, flush=True)
    for lam in lams:
        name = f"lam_{lam}"
        hits = 0
        total = 0
        for ti, (it, z_a1, z_a2, mem1, mem2) in enumerate(task_data):
            try:
                torch.cuda.empty_cache()
                q_ids = enc(it["q"])
                cond = zbar + lam * (0.5 * (z_a1 - zbar) + 0.5 * (z_a2 - zbar))
                Z = sample_trajectory_cfg(model, cond, STEPS, CFG, device, dtype)
                plan = [model.predict_states(Z[:, h]) for h in range(H)]
                plan_l = plan
                mem1_l, mem2_l = mem1, mem2
                fused = dc(q_ids, lambda h: fuse_states([(0.5, plan_l[h]), (0.25, mem1_l), (0.25, mem2_l)]))
                text = tokenizer.decode(fused, skip_special_tokens=True).strip().lower()
                if it["gold"] in text:
                    hits += 1
                total += 1
            except Exception as e:
                print(f"  ERROR [{name}] task {ti}: {e}", flush=True)
                traceback.print_exc()
                total += 1
        acc = hits / total * 100 if total > 0 else 0
        results[name] = {"hits": hits, "total": total, "acc": acc}
        print(f"  {name:30s} = {acc:.1f}% ({hits}/{total})", flush=True)

    # --- Summary ---
    print("\n" + "=" * 80, flush=True)
    print("FULL SWEEP RESULTS (sorted by accuracy)", flush=True)
    print("=" * 80, flush=True)
    sorted_results = sorted(results.items(), key=lambda x: x[1]["acc"], reverse=True)
    for name, r in sorted_results:
        print(f"  {name:30s} = {r['acc']:.1f}% ({r['hits']}/{r['total']})", flush=True)

    # Save
    out_path = REPO / "results/fusion_ablation/fusion_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    run()
