#!/usr/bin/env python3
"""Hard problems V2: correct fusion operators from paper.

Plan: residual fusion + diffusion resample (+ PCA)
State: sequential carryover (NOT parallel averaging)

Conditions per task:
  - dual (residual plan + seq carryover state)
  - PCA_K3 (PCA top-3 removed + seq carryover state)
  - text_concat (oracle)
  - best_agent_only (upper bound per agent)
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
from scripts.eval.fusion_ablation_pca import gen_tasks as gen_2agent_tasks, pca_remove_shared

CKPT = str(REPO / "outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000")
STEPS = 100; CFG = 3.0; SEED = 42; MAX_NEW = 24; LAM = 1.0

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


def gen_2agent(n, seed):
    rng = random.Random(seed)
    tasks = []
    for i in range(n):
        subj = _SUBJECTS[i % len(_SUBJECTS)]; link = rng.choice(_LINK_ENTITIES); gold = rng.choice(_PLACES)
        a = f"Agent A knows: the {subj} is hidden by the {link}."
        b = f"Agent B knows: the {link} is located in the town of {gold}."
        q = f"Question: in which town is the {subj} hidden? Answer: the town of"
        tasks.append({"agents": [a, b], "q": q, "gold": gold.lower()})
    return tasks


def gen_3agent(n, seed):
    rng = random.Random(seed)
    tasks = []
    for i in range(n):
        subj = _SUBJECTS[i % len(_SUBJECTS)]
        link1 = rng.choice(_LINK_ENTITIES); link2 = rng.choice(_LINK_ENTITIES)
        while link2 == link1: link2 = rng.choice(_LINK_ENTITIES)
        gold = rng.choice(_PLACES)
        a = f"Agent A knows: the {subj} is hidden by the {link1}."
        b = f"Agent B knows: the {link1} is located at the {link2}."
        c = f"Agent C knows: the {link2} is in the town of {gold}."
        q = f"Question: in which town is the {subj} hidden? Answer: the town of"
        tasks.append({"agents": [a, b, c], "q": q, "gold": gold.lower()})
    return tasks


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


def gen_conflict(n, seed):
    rng = random.Random(seed)
    tasks = []
    for i in range(n):
        subj = _SUBJECTS[i % len(_SUBJECTS)]; gold = rng.choice(_PLACES); wrong = rng.choice(_PLACES)
        while wrong == gold: wrong = rng.choice(_PLACES)
        a = f"Agent A knows: the {subj} is hidden in the town of {wrong}."
        b = f"Agent B knows: the {subj} is hidden in the town of {gold}."
        q = f"Question: in which town is the {subj} hidden? Answer: the town of"
        tasks.append({"agents": [a, b], "q": q, "gold": gold.lower(), "wrong": wrong.lower()})
    return tasks


def compute_residual_plan(model, agent_zs, zbar, device, dtype):
    """Residual plan: zbar + lam * (avg of (z_i - zbar))"""
    resid_sum = sum(z - zbar for z in agent_zs) / len(agent_zs)
    cond = zbar.to(dtype) + LAM * resid_sum.to(dtype)
    Z = sample_trajectory_cfg(model, cond, STEPS, CFG, device, dtype)
    plan = [model.predict_states(Z[:, h]) for h in range(int(model.trajectory_horizon))]
    return plan


def compute_pca_plan(model, agent_zs, z_mean, shared_dirs, K, device, dtype):
    """PCA plan: remove top-K shared directions from each z, fuse residuals"""
    resid_sum = None
    for z in agent_zs:
        zc = z[0].float() - z_mean[0].float()
        proj = zc @ shared_dirs.T
        r = zc - proj @ shared_dirs
        resid_sum = r if resid_sum is None else resid_sum + r
    fused_resid = LAM * (resid_sum / len(agent_zs))
    cond = (z_mean[0].float() + fused_resid).to(dtype).unsqueeze(0)
    Z = sample_trajectory_cfg(model, cond, STEPS, CFG, device, dtype)
    plan = [model.predict_states(Z[:, h]) for h in range(int(model.trajectory_horizon))]
    return plan


@torch.no_grad()
def run():
    torch.manual_seed(SEED)
    device = torch.device("cuda:0")
    print(f"[init] Loading model from {CKPT}", flush=True)
    model, _, tokenizer, _, _ = load_relay_model(CKPT, str(device))
    model.eval(); model._prefix_suffix_trajectory_s2 = True
    dtype = next(model.alpha_heads.parameters()).dtype
    H = int(model.trajectory_horizon)
    print(f"[init] H={H} dtype={dtype}", flush=True)

    def enc(t): return tokenizer(t, return_tensors="pt").input_ids.to(device)

    class Args:
        max_new_tokens = MAX_NEW; temperature = 0.5; top_k = 10; top_p = 0.9; repetition_penalty = 1.2
    args = Args()

    task_types = {
        "2agent": gen_2agent(20, SEED),
        "3agent": gen_3agent(20, SEED),
        "4agent": gen_4agent(20, SEED),
        "conflict": gen_conflict(20, SEED),
    }

    all_results = {}

    for ttype, tasks in task_types.items():
        print(f"\n{'='*60}", flush=True)
        print(f"[{ttype}] {len(tasks)} tasks", flush=True)

        # Collect all latents for PCA
        all_z = []
        for it in tasks:
            for a_text in it["agents"]:
                a_ids = enc(a_text)
                z = encode_prefix(model, a_ids, torch.ones_like(a_ids))[0].to(dtype)
                all_z.append(z[0].float().cpu())
        z_all = torch.stack(all_z, 0).to(device)
        zbar = z_all.mean(dim=0, keepdim=True).to(dtype)
        # PCA K=3
        dirs_K3, sv_K3, z_mean_pca, _ = pca_remove_shared(z_all, K=3)

        results = {"dual": [], "text_concat": [], "best_agent": [], "PCA_K3": []}

        for ii, it in enumerate(tasks):
            gold = it["gold"].lower()
            q_ids = enc(it["q"])
            agent_zs = []
            for a_text in it["agents"]:
                a_ids = enc(a_text)
                z = encode_prefix(model, a_ids, torch.ones_like(a_ids))[0].to(dtype)
                agent_zs.append(z)

            # --- DUAL: residual plan + sequential carryover state ---
            try:
                plan_resid = compute_residual_plan(model, agent_zs, zbar, device, dtype)
                seq_state = seq_carryover_state(model, tokenizer, it["agents"], device)
                dual_hit = decode_chunkwise(model, tokenizer, q_ids,
                    lambda h: fuse_states([(0.5, plan_resid[h]), (0.5, seq_state)]), H, args)
                dual_text = tokenizer.decode(dual_hit, skip_special_tokens=True).strip().lower()
                results["dual"].append({"hit": gold in dual_text, "text": dual_text[:60]})
            except Exception as e:
                traceback.print_exc(); results["dual"].append({"hit": False, "text": f"ERROR: {e}"})

            # --- PCA_K3: PCA plan + sequential carryover state ---
            try:
                plan_pca = compute_pca_plan(model, agent_zs, z_mean_pca, dirs_K3, 3, device, dtype)
                seq_state = seq_carryover_state(model, tokenizer, it["agents"], device)
                pca_hit = decode_chunkwise(model, tokenizer, q_ids,
                    lambda h: fuse_states([(0.5, plan_pca[h]), (0.5, seq_state)]), H, args)
                pca_text = tokenizer.decode(pca_hit, skip_special_tokens=True).strip().lower()
                results["PCA_K3"].append({"hit": gold in pca_text, "text": pca_text[:60]})
            except Exception as e:
                traceback.print_exc(); results["PCA_K3"].append({"hit": False, "text": f"ERROR: {e}"})

            # --- TEXT CONCAT: oracle ---
            try:
                combined = " ".join(it["agents"])
                c_ids = enc(combined)
                out = model.rwkv_model(input_ids=c_ids, attention_mask=torch.ones_like(c_ids).bool(),
                                       use_cache=True, return_dict=True)
                past_kv = out.past_key_values
                all_ids = list(c_ids[0].tolist()) + list(q_ids[0].tolist())
                ctx = torch.tensor([all_ids], device=device, dtype=torch.long)
                past_kv, lb = recompute_logits_from_injected_cache(model, ctx, torch.ones_like(ctx), past_kv)
                logits = lb[0]
                new_ids = []
                eos_id = getattr(tokenizer, "eos_token_id", None)
                for _ in range(MAX_NEW):
                    nid = sample_next_token(logits, all_ids, args)
                    if eos_id is not None and nid == eos_id: break
                    new_ids.append(nid); all_ids.append(nid)
                    out = model.rwkv_model(input_ids=torch.tensor([[nid]], device=device),
                                           past_key_values=past_kv, use_cache=True, return_dict=True)
                    past_kv = out.past_key_values; logits = out.logits[0, -1]
                tc_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip().lower()
                results["text_concat"].append({"hit": gold in tc_text, "text": tc_text[:60]})
            except Exception as e:
                traceback.print_exc(); results["text_concat"].append({"hit": False, "text": f"ERROR: {e}"})

            # --- BEST SINGLE AGENT ---
            best_hit = False
            for ai in range(len(it["agents"])):
                single_state = capture_final_state(model, enc(it["agents"][ai]))
                single_pred = decode_chunkwise(model, tokenizer, q_ids,
                    lambda h: single_state, H, args)
                single_text = tokenizer.decode(single_pred, skip_special_tokens=True).strip().lower()
                if gold in single_text: best_hit = True
            results["best_agent"].append({"hit": best_hit, "text": ""})

            if (ii + 1) % 5 == 0:
                def acc(key): return sum(r["hit"] for r in results[key]) / len(results[key]) * 100
                print(f"[{ttype} {ii+1}/{len(tasks)}] dual={acc('dual'):.0f}% "
                      f"PCA={acc('PCA_K3'):.0f}% concat={acc('text_concat'):.0f}% "
                      f"best={acc('best_agent'):.0f}%", flush=True)

        summary = {k: round(sum(r["hit"] for r in v) / len(v) * 100, 1) for k, v in results.items()}
        all_results[ttype] = summary
        print(f"[{ttype}] FINAL: dual={summary['dual']:.0f}% PCA={summary['PCA_K3']:.0f}% "
              f"concat={summary['text_concat']:.0f}% best={summary['best_agent']:.0f}%", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("=== SUMMARY ===", flush=True)
    for t, r in all_results.items():
        print(f"{t:12s}: dual={r['dual']:4.0f}% PCA={r['PCA_K3']:4.0f}% "
              f"concat={r['text_concat']:4.0f}% best={r['best_agent']:4.0f}%", flush=True)

    out_path = REPO / "results/hard_problems/diag_hard_v2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(all_results, open(out_path, "w"), indent=2)
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    run()