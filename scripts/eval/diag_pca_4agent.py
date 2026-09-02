#!/usr/bin/env python3
"""4-agent PCA ablation: residual vs PCA_K1 vs PCA_K1_no_center vs PCA_K3 vs perstep.
All use sequential carryover state. Tests: chain A→B→C→D (4 hops).
"""
from __future__ import annotations
import json, sys, random, traceback
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from scripts.eval.sample_prefix_suffix_trajectory_cfg import encode_prefix, sample_trajectory_cfg
from scripts.eval.exp_multiagent_star_fusion import (
    capture_final_state, fuse_states, decode_chunkwise, seq_carryover_state,
)
from scripts.eval.relay_utils import load_relay_model
from scripts.eval.run_cola_dlm_tasks_prefix_suffix_trajectory_cfg import (
    recompute_logits_from_injected_cache, sample_next_token,
)

CKPT = str(REPO / "outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000")
STEPS = 100; CFG = 3.0; SEED = 42; MAX_NEW = 24; LAM = 1.0; N_TASKS = 100

_LINK_ENTITIES = [
    "oak tree", "blue vial", "red door", "green jacket", "iron gate", "stone bridge",
    "silver key", "black notebook", "old lighthouse", "marble statue", "copper pipe",
    "glass tower", "wooden crate", "brass compass", "velvet curtain", "ceramic urn",
]
_PLACES = ["Wexford", "Ashby", "Thornfield", "Kestrel", "Draymoor", "Fenwick", "Halcyon",
           "Brimmel", "Corvane", "Ostend", "Marlow", "Renwick", "Selby", "Tanager", "Verdon", "Wychley"]
_SUBJECTS = ["treasure", "antidote", "manuscript", "artifact", "prototype", "ledger",
             "relic", "serum", "blueprint", "medallion", "specimen", "recording",
             "cipher", "sculpture", "vaccine", "heirloom", "diary", "sample"]


def gen_4agent(n, seed):
    rng = random.Random(seed)
    tasks = []
    for i in range(n):
        subj = _SUBJECTS[i % len(_SUBJECTS)]
        link1 = rng.choice(_LINK_ENTITIES); link2 = rng.choice(_LINK_ENTITIES); link3 = rng.choice(_LINK_ENTITIES)
        while link2 == link1: link2 = rng.choice(_LINK_ENTITIES)
        while link3 == link1 or link3 == link2: link3 = rng.choice(_LINK_ENTITIES)
        gold = rng.choice(_PLACES)
        a = f"Agent A knows: the {subj} is hidden by the {link1}."
        b = f"Agent B knows: the {link1} is located at the {link2}."
        c = f"Agent C knows: the {link2} is near the {link3}."
        d = f"Agent D knows: the {link3} is in the town of {gold}."
        q = f"Question: in which town is the {subj} hidden? Answer: the town of"
        tasks.append({"agents": [a, b, c, d], "q": q, "gold": gold.lower()})
    return tasks


def pca_no_center(z_all, K=1):
    """PCA WITHOUT centering: first PC captures mean direction."""
    U, S, Vh = torch.linalg.svd(z_all.float(), full_matrices=False)
    shared_dirs = Vh[:K]  # [K, D]
    proj_scores = z_all.float() @ shared_dirs.T
    shared_part = proj_scores @ shared_dirs
    z_residuals = z_all.float() - shared_part
    return shared_dirs, S, z_residuals


def pca_centered(z_all, K=1):
    """PCA WITH centering: remove top-K PCs from centered data."""
    z_mean = z_all.mean(dim=0, keepdim=True)
    z_centered = z_all.float() - z_mean.float()
    U, S, Vh = torch.linalg.svd(z_centered, full_matrices=False)
    shared_dirs = Vh[:K]
    proj_scores = z_centered @ shared_dirs.T
    shared_part = proj_scores @ shared_dirs
    z_residuals = z_all.float() - shared_part
    return shared_dirs, S, z_mean, z_residuals


def compute_plan(model, agent_zs, method, zbar, z_mean, dirs, device, dtype):
    """Compute plan state for a given fusion method."""
    H = int(model.trajectory_horizon)
    if method == "residual":
        resid_sum = sum(z - zbar for z in agent_zs) / len(agent_zs)
        cond = zbar.to(dtype) + LAM * resid_sum.to(dtype)
    elif method == "PCA_K1_no_center":
        # No centering: remove top-1 PC directly
        resid_sum = None
        for z in agent_zs:
            zc = z[0].float()
            proj = zc @ dirs.T
            r = zc - proj @ dirs
            resid_sum = r if resid_sum is None else resid_sum + r
        fused = LAM * (resid_sum / len(agent_zs))
        cond = fused.to(dtype).unsqueeze(0)
    else:
        # Centered PCA: remove top-K from centered data
        K = int(method.split("_K")[1])
        resid_sum = None
        for z in agent_zs:
            zc = z[0].float() - z_mean[0].float()
            proj = zc @ dirs.T
            r = zc - proj @ dirs
            resid_sum = r if resid_sum is None else resid_sum + r
        fused = LAM * (resid_sum / len(agent_zs))
        cond = (z_mean[0].float() + fused).to(dtype).unsqueeze(0)
    Z = sample_trajectory_cfg(model, cond, STEPS, CFG, device, dtype)
    plan = [model.predict_states(Z[:, h]) for h in range(H)]
    return plan


@torch.no_grad()
def run():
    torch.manual_seed(SEED)
    device = torch.device("cuda:0")
    print(f"[init] Loading model...", flush=True)
    model, _, tokenizer, _, _ = load_relay_model(CKPT, str(device))
    model.eval(); model._prefix_suffix_trajectory_s2 = True
    dtype = next(model.alpha_heads.parameters()).dtype
    H = int(model.trajectory_horizon)
    print(f"[init] H={H} dtype={dtype}", flush=True)

    def enc(t): return tokenizer(t, return_tensors="pt").input_ids.to(device)
    class Args: max_new_tokens = MAX_NEW; temperature = 0.5; top_k = 10; top_p = 0.9; repetition_penalty = 1.2
    args = Args()

    tasks = gen_4agent(N_TASKS, SEED)
    print(f"[4agent] {len(tasks)} tasks", flush=True)

    # Collect all latents
    all_z = []
    for it in tasks:
        for a_text in it["agents"]:
            z = encode_prefix(model, enc(a_text), torch.ones_like(enc(a_text)))[0].to(dtype)
            all_z.append(z[0].float().cpu())
    z_all = torch.stack(all_z, 0).to(device)
    zbar = z_all.mean(dim=0, keepdim=True).to(dtype)

    # PCA WITH centering
    dirs_K1, sv_K1, z_mean_K1, _ = pca_centered(z_all, K=1)
    dirs_K3, sv_K3, z_mean_K3, _ = pca_centered(z_all, K=3)

    # PCA WITHOUT centering
    dirs_nc_K1, sv_nc_K1, _ = pca_no_center(z_all, K=1)

    print(f"[PCA] zbar_norm={zbar.norm():.3f} sv1_centered={sv_K1[0]:.4f} sv1_no_center={sv_nc_K1[0]:.4f}", flush=True)

    methods = ["residual", "PCA_K1", "PCA_K3", "PCA_K1_no_center"]
    results = {m: [] for m in methods}
    results["text_concat"] = []
    results["best_agent"] = []

    for ii, it in enumerate(tasks):
        gold = it["gold"].lower(); q_ids = enc(it["q"])
        agent_zs = [encode_prefix(model, enc(a), torch.ones_like(enc(a)))[0].to(dtype) for a in it["agents"]]
        seq_state = seq_carryover_state(model, tokenizer, it["agents"], device)

        for method in methods:
            if method == "residual":
                dirs, z_mean = None, None
            elif method == "PCA_K1":
                dirs, z_mean = dirs_K1, z_mean_K1
            elif method == "PCA_K3":
                dirs, z_mean = dirs_K3, z_mean_K3
            elif method == "PCA_K1_no_center":
                dirs, z_mean = dirs_nc_K1, torch.zeros_like(zbar)  # no mean

            plan = compute_plan(model, agent_zs, method, zbar, z_mean, dirs, device, dtype)
            pred = decode_chunkwise(model, tokenizer, q_ids,
                lambda h: fuse_states([(0.5, plan[h]), (0.5, seq_state)]), H, args)
            text = tokenizer.decode(pred, skip_special_tokens=True).strip().lower()
            results[method].append(gold in text)

        # Text concat
        combined = " ".join(it["agents"]); c_ids = enc(combined)
        out = model.rwkv_model(input_ids=c_ids, attention_mask=torch.ones_like(c_ids).bool(), use_cache=True, return_dict=True)
        past_kv = out.past_key_values
        all_ids = list(c_ids[0].tolist()) + list(q_ids[0].tolist())
        ctx = torch.tensor([all_ids], device=device, dtype=torch.long)
        past_kv, lb = recompute_logits_from_injected_cache(model, ctx, torch.ones_like(ctx), past_kv)
        logits = lb[0]; new_ids = []
        eos_id = getattr(tokenizer, "eos_token_id", None)
        for _ in range(MAX_NEW):
            nid = sample_next_token(logits, all_ids, args)
            if eos_id is not None and nid == eos_id: break
            new_ids.append(nid); all_ids.append(nid)
            out = model.rwkv_model(input_ids=torch.tensor([[nid]], device=device), past_key_values=past_kv, use_cache=True, return_dict=True)
            past_kv = out.past_key_values; logits = out.logits[0, -1]
        results["text_concat"].append(gold in tokenizer.decode(new_ids, skip_special_tokens=True).strip().lower())

        # Best agent
        best = False
        for ai in range(len(it["agents"])):
            s = capture_final_state(model, enc(it["agents"][ai]))
            p = decode_chunkwise(model, tokenizer, q_ids, lambda h: s, H, args)
            if gold in tokenizer.decode(p, skip_special_tokens=True).strip().lower(): best = True
        results["best_agent"].append(best)

        if (ii+1) % 10 == 0:
            def a(k): return sum(results[k])/len(results[k])*100
            print(f"[{ii+1}/{N_TASKS}] residual={a('residual'):.0f}% "
                  f"PCA_K1={a('PCA_K1'):.0f}% PCA_K3={a('PCA_K3'):.0f}% "
                  f"K1_no_center={a('PCA_K1_no_center'):.0f}% "
                  f"concat={a('text_concat'):.0f}% best={a('best_agent'):.0f}%", flush=True)

    summary = {k: round(sum(v)/len(v)*100, 1) for k, v in results.items()}
    print(f"\n=== 4agent PCA ABLATION (n={N_TASKS}) ===", flush=True)
    for m in ["residual", "PCA_K1", "PCA_K3", "PCA_K1_no_center", "text_concat", "best_agent"]:
        print(f"  {m:20s}: {summary[m]:5.1f}%", flush=True)

    out_path = REPO / "results/hard_problems/pca_4agent.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"n": N_TASKS, "summary": summary}, open(out_path, "w"), indent=2)
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    run()