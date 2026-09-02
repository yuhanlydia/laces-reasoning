#!/usr/bin/env python3
"""Test optimal fusion config on harder problems.

Tests:
  1. 2-agent chain (baseline, verify optimal config)
  2. 3-agent chain (harder)
  3. 4-agent chain (even harder)
  4. Conflict resolution (different challenge)

Conditions per task:
  - dual (current: K=0, lam=1.0, 0.5/0.25/0.25)
  - PCA_K3 (K=3, lam=1.0, 0.5/0.25/0.25)
  - lam_0.5 (K=0, lam=0.5, 0.5/0.25/0.25)
  - PCA_K3_lam_0.5 (combined optimal)
  - text_concat (oracle)
  - best_agent_only

Usage: CUDA_VISIBLE_DEVICES=1 python -u scripts/eval/diag_hard_problems.py
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
STEPS = 100
CFG = 3.0
SEED = 42
MAX_NEW = 24
N_TASKS = 20

_LINK_ENTITIES = [
    "oak tree", "blue vial", "red door", "green jacket", "iron gate", "stone bridge",
    "silver key", "black notebook", "old lighthouse", "marble statue", "copper pipe",
    "glass tower", "wooden crate", "brass compass", "velvet curtain", "ceramic urn",
]
_PLACES = [
    "Wexford", "Ashby", "Thornfield", "Kestrel", "Draymoor", "Fenwick", "Halcyon",
    "Brimmel", "Corvane", "Ostend", "Marlow", "Renwick", "Selby", "Tanager",
    "Verdon", "Wychley",
]
_SUBJECTS = [
    "treasure", "antidote", "manuscript", "artifact", "prototype", "ledger",
    "relic", "serum", "blueprint", "medallion", "specimen", "recording",
    "cipher", "sculpture", "vaccine", "heirloom", "diary", "sample",
]


def gen_2agent(n, seed):
    rng = random.Random(seed)
    tasks = []
    for i in range(n):
        subj = _SUBJECTS[i % len(_SUBJECTS)]
        link = rng.choice(_LINK_ENTITIES)
        gold = rng.choice(_PLACES)
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
        link1 = rng.choice(_LINK_ENTITIES)
        link2 = rng.choice(_LINK_ENTITIES)
        while link2 == link1:
            link2 = rng.choice(_LINK_ENTITIES)
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
        link1 = rng.choice(_LINK_ENTITIES)
        link2 = rng.choice(_LINK_ENTITIES)
        link3 = rng.choice(_LINK_ENTITIES)
        while link2 == link1:
            link2 = rng.choice(_LINK_ENTITIES)
        while link3 == link1 or link3 == link2:
            link3 = rng.choice(_LINK_ENTITIES)
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
        subj = _SUBJECTS[i % len(_SUBJECTS)]
        gold = rng.choice(_PLACES)
        wrong = rng.choice(_PLACES)
        while wrong == gold:
            wrong = rng.choice(_PLACES)
        a = f"Agent A knows: the {subj} is hidden in the town of {wrong}."
        b = f"Agent B knows: the {subj} is hidden in the town of {gold}."
        q = f"Question: in which town is the {subj} hidden? Answer: the town of"
        tasks.append({"agents": [a, b], "q": q, "gold": gold.lower(), "wrong": wrong.lower()})
    return tasks


def compute_plan(model, agent_zs, zbar, shared_dirs, z_mean, K, lam, H, device, dtype):
    if K == 0:
        cond = zbar + lam * sum(0.5 * (z - zbar) for z in agent_zs) / len(agent_zs) * 2
        cond = zbar + lam * (sum(z - zbar for z in agent_zs) / len(agent_zs))
    else:
        dirs = shared_dirs
        resid_sum = None
        for z in agent_zs:
            zc = z[0].float() - z_mean[0].float()
            p = zc @ dirs.T
            r = zc - p @ dirs
            resid_sum = r if resid_sum is None else resid_sum + r
        fused_resid = lam * (resid_sum / len(agent_zs))
        cond = (z_mean[0].float() + fused_resid).to(dtype).unsqueeze(0)
    Z = sample_trajectory_cfg(model, cond, STEPS, CFG, device, dtype)
    plan = [model.predict_states(Z[:, h]) for h in range(H)]
    return plan


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

    class Args:
        max_new_tokens = MAX_NEW
        temperature = 0.5
        top_k = 10
        top_p = 0.9
        repetition_penalty = 1.2
    args = Args()

    def dc(q_ids, fn):
        return decode_chunkwise(model, tokenizer, q_ids, fn, H, args)

    task_types = {
        "2agent": gen_2agent(N_TASKS, SEED),
        "3agent": gen_3agent(N_TASKS, SEED),
        "4agent": gen_4agent(N_TASKS, SEED),
        "conflict": gen_conflict(N_TASKS, SEED),
    }

    all_results = {}

    for ttype, tasks in task_types.items():
        print(f"\n{'='*80}", flush=True)
        print(f"[{ttype}] Testing {len(tasks)} tasks", flush=True)
        print(f"{'='*80}", flush=True)

        # Collect per-task: agent latents + memory states
        print(f"[{ttype}] Collecting states...", flush=True)
        task_data = []
        all_z = []
        for ti, it in enumerate(tasks):
            agent_zs = []
            mems = []
            for atext in it["agents"]:
                a_ids = enc(atext)
                z = encode_prefix(model, a_ids, torch.ones_like(a_ids))[0].to(dtype)
                agent_zs.append(z)
                all_z.append(z[0].float().cpu())
                mem = capture_final_state(model, a_ids)
                mems.append(mem)
            task_data.append((it, agent_zs, mems))
            if (ti + 1) % 10 == 0:
                print(f"[{ttype}] {ti+1}/{len(tasks)}", flush=True)

        z_all = torch.stack(all_z, 0).to(device)
        zbar = z_all.mean(dim=0, keepdim=True).to(dtype)
        z_mean_pca = zbar

        # PCA K=3
        dirs_K3, sv_K3, _, _ = pca_remove_shared(z_all, K=3)

        n_agents = len(it["agents"])

        # Define conditions
        conditions = [
            ("dual_current", 0, 1.0),
            ("PCA_K3", 3, 1.0),
            ("lam_0.5", 0, 0.5),
            ("PCA_K3_lam_0.5", 3, 0.5),
        ]

        type_results = {}

        # Test each fusion condition
        for cond_name, K, lam in conditions:
            hits = 0
            total = 0
            for ti, (it, agent_zs, mems) in enumerate(task_data):
                try:
                    torch.cuda.empty_cache()
                    q_ids = enc(it["q"])
                    if K == 0:
                        plan = compute_plan(model, agent_zs, zbar, None, z_mean_pca, 0, lam, H, device, dtype)
                    else:
                        plan = compute_plan(model, agent_zs, zbar, dirs_K3, z_mean_pca, K, lam, H, device, dtype)
                    w_plan = 0.5
                    w_mem = (1.0 - w_plan) / n_agents
                    plan_l = plan
                    mems_l = mems
                    fused = dc(q_ids, lambda h: fuse_states(
                        [(w_plan, plan_l[h])] + [(w_mem, mems_l[i]) for i in range(n_agents)]
                    ))
                    text = tokenizer.decode(fused, skip_special_tokens=True).strip().lower()
                    if it["gold"] in text:
                        hits += 1
                    total += 1
                except Exception as e:
                    print(f"  ERROR [{cond_name}] task {ti}: {e}", flush=True)
                    traceback.print_exc()
                    total += 1
            acc = hits / total * 100 if total > 0 else 0
            type_results[cond_name] = {"hits": hits, "total": total, "acc": acc}
            print(f"  {cond_name:25s} = {acc:.1f}% ({hits}/{total})", flush=True)

        # Text concat (oracle)
        hits = 0
        total = 0
        for ti, (it, _, _) in enumerate(task_data):
            try:
                torch.cuda.empty_cache()
                combined = " ".join(it["agents"])
                c_ids = enc(combined)
                out = model.rwkv_model(input_ids=c_ids, attention_mask=torch.ones_like(c_ids).bool(),
                                       use_cache=True, return_dict=True)
                past_kv = out.past_key_values
                q_ids = enc(it["q"])
                all_ids = list(c_ids[0].tolist()) + list(q_ids[0].tolist())
                ctx = torch.tensor([all_ids], device=device, dtype=torch.long)
                past_kv, lb = recompute_logits_from_injected_cache(model, ctx, torch.ones_like(ctx), past_kv)
                logits = lb[0]
                new_ids = []
                eos_id = getattr(tokenizer, "eos_token_id", None)
                for _ in range(MAX_NEW):
                    nid = sample_next_token(logits, all_ids, args)
                    if eos_id is not None and nid == eos_id:
                        break
                    new_ids.append(nid)
                    all_ids.append(nid)
                    out = model.rwkv_model(input_ids=torch.tensor([[nid]], device=device),
                                           past_key_values=past_kv, use_cache=True, return_dict=True)
                    past_kv = out.past_key_values
                    logits = out.logits[0, -1]
                text = tokenizer.decode(new_ids, skip_special_tokens=True).strip().lower()
                if it["gold"] in text:
                    hits += 1
                total += 1
            except Exception as e:
                print(f"  ERROR [text_concat] task {ti}: {e}", flush=True)
                total += 1
        acc = hits / total * 100 if total > 0 else 0
        type_results["text_concat"] = {"hits": hits, "total": total, "acc": acc}
        print(f"  {'text_concat':25s} = {acc:.1f}% ({hits}/{total})", flush=True)

        # Best agent only
        best_acc = 0
        best_agent_idx = 0
        for ai in range(n_agents):
            hits = 0
            total = 0
            for ti, (it, _, mems) in enumerate(task_data):
                try:
                    torch.cuda.empty_cache()
                    q_ids = enc(it["q"])
                    mem = mems[ai]
                    fused = dc(q_ids, lambda h: mem)
                    text = tokenizer.decode(fused, skip_special_tokens=True).strip().lower()
                    if it["gold"] in text:
                        hits += 1
                    total += 1
                except Exception as e:
                    print(f"  ERROR [agent{ai}] task {ti}: {e}", flush=True)
                    total += 1
            acc = hits / total * 100 if total > 0 else 0
            if acc > best_acc:
                best_acc = acc
                best_agent_idx = ai
        type_results[f"best_agent_only"] = {"hits": int(best_acc * total / 100), "total": total, "acc": best_acc}
        print(f"  {'best_agent_only':25s} = {best_acc:.1f}% (agent {best_agent_idx})", flush=True)

        all_results[ttype] = type_results

    # Final summary
    print(f"\n{'='*80}", flush=True)
    print("FINAL SUMMARY: Optimal config on hard problems", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Condition':30s} | {'2-agent':>10s} | {'3-agent':>10s} | {'4-agent':>10s} | {'conflict':>10s}", flush=True)
    print("-" * 80, flush=True)
    for cond in ["best_agent_only", "dual_current", "PCA_K3", "lam_0.5", "PCA_K3_lam_0.5", "text_concat"]:
        row = f"{cond:30s} |"
        for ttype in ["2agent", "3agent", "4agent", "conflict"]:
            if ttype in all_results and cond in all_results[ttype]:
                acc = all_results[ttype][cond]["acc"]
                row += f" {acc:>8.1f}% |"
            else:
                row += f" {'N/A':>9s} |"
        print(row, flush=True)

    out_path = REPO / "results/fusion_ablation/hard_problems.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(all_results, open(out_path, "w"), indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    run()
