#!/usr/bin/env python3
"""Experiment 1: Multi-agent communication cost vs accuracy.

Three-way comparison (similar to LatentMAS framework):
  1. text_mas: sequential text passing between agents (LatentMAS text_mas equivalent)
  2. state_carryover: sequential RWKV state passing (LatentMAS latent_mas equivalent on RWKV)
  3. dual_fusion: our plan-based dual fusion method

Measurements:
  - Accuracy (EM)
  - Communication cost (total bytes transmitted between agents)
  - Task types: 2-hop, 3-hop, 4-hop chain reasoning

Usage: CUDA_VISIBLE_DEVICES=1 python -u scripts/eval/exp1_comm_cost.py
"""
from __future__ import annotations
import json, sys, random, traceback, time
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

_LINK_ENTITIES = [
    "oak tree", "blue vial", "red door", "green jacket", "iron gate", "stone bridge",
    "silver key", "black notebook", "old lighthouse", "marble statue",
]
_PLACES = [
    "Wexford", "Ashby", "Thornfield", "Kestrel", "Draymoor", "Fenwick",
    "Halcyon", "Brimmel", "Corvane", "Ostend",
]
_SUBJECTS = [
    "treasure", "antidote", "manuscript", "artifact", "prototype",
    "ledger", "relic", "serum", "blueprint", "medallion",
]


def gen_nagent_chain(n_agents, n_tasks, seed):
    rng = random.Random(seed)
    tasks = []
    for i in range(n_tasks):
        subj = _SUBJECTS[i % len(_SUBJECTS)]
        links = [rng.choice(_LINK_ENTITIES) for _ in range(n_agents - 1)]
        for j in range(1, len(links)):
            while links[j] == links[j-1]:
                links[j] = rng.choice(_LINK_ENTITIES)
        gold = rng.choice(_PLACES)

        agents = []
        agents.append(f"Agent A knows: the {subj} is hidden by the {links[0]}.")
        for j in range(1, n_agents - 1):
            agent_letter = chr(ord('A') + j)
            agents.append(f"Agent {agent_letter} knows: the {links[j-1]} is located at the {links[j]}.")
        last_letter = chr(ord('A') + n_agents - 1)
        agents.append(f"Agent {last_letter} knows: the {links[-1]} is in the town of {gold}.")

        q = f"Question: in which town is the {subj} hidden? Answer: the town of"
        tasks.append({"agents": agents, "q": q, "gold": gold.lower(), "n_agents": n_agents})
    return tasks


@torch.no_grad()
def method_text_mas(model, tokenizer, task, args, device, H):
    """Sequential text passing: each agent's output is passed to next agent."""
    all_text = ""
    total_bytes = 0

    for ai, agent_text in enumerate(task["agents"]):
        if ai == len(task["agents"]) - 1:
            # Last agent: answer the question
            prompt = all_text + " " + agent_text + " " + task["q"]
        else:
            # Intermediate agent: generate a summary
            prompt = all_text + " " + agent_text + "\nSummarize:"

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        total_bytes += len(prompt.encode('utf-8'))

        if ai < len(task["agents"]) - 1:
            # Generate summary (short)
            out = model.rwkv_model(input_ids=input_ids,
                                   attention_mask=torch.ones_like(input_ids).bool(),
                                   use_cache=True, return_dict=True)
            past_kv = out.past_key_values
            logits = out.logits[0, -1]
            all_ids = list(input_ids[0].tolist())
            new_ids = []
            eos_id = getattr(tokenizer, "eos_token_id", None)
            for _ in range(64):
                nid = sample_next_token(logits, all_ids, args)
                if eos_id is not None and nid == eos_id:
                    break
                new_ids.append(nid)
                all_ids.append(nid)
                out = model.rwkv_model(input_ids=torch.tensor([[nid]], device=device),
                                       past_key_values=past_kv, use_cache=True, return_dict=True)
                past_kv = out.past_key_values
                logits = out.logits[0, -1]
            summary = tokenizer.decode(new_ids, skip_special_tokens=True)
            all_text = all_text + " " + agent_text + " " + summary
            total_bytes += len(summary.encode('utf-8'))
        else:
            # Last agent: answer
            out = model.rwkv_model(input_ids=input_ids,
                                   attention_mask=torch.ones_like(input_ids).bool(),
                                   use_cache=True, return_dict=True)
            past_kv = out.past_key_values
            logits = out.logits[0, -1]
            all_ids = list(input_ids[0].tolist())
            new_ids = []
            eos_id = getattr(tokenizer, "eos_token_id", None)
            for _ in range(args.max_new_tokens):
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
            hit = task["gold"] in text
            return hit, total_bytes, text

    return False, total_bytes, ""


@torch.no_grad()
def method_text_concat(model, tokenizer, task, args, device, H):
    """Simple text concatenation: all agents' text in one context."""
    combined = " ".join(task["agents"]) + " " + task["q"]
    total_bytes = len(combined.encode('utf-8'))

    input_ids = tokenizer(combined, return_tensors="pt").input_ids.to(device)
    out = model.rwkv_model(input_ids=input_ids,
                          attention_mask=torch.ones_like(input_ids).bool(),
                          use_cache=True, return_dict=True)
    past_kv = out.past_key_values
    logits = out.logits[0, -1]
    all_ids = list(input_ids[0].tolist())
    new_ids = []
    eos_id = getattr(tokenizer, "eos_token_id", None)
    for _ in range(args.max_new_tokens):
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
    hit = task["gold"] in text
    return hit, total_bytes, text


@torch.no_grad()
def method_state_carryover(model, tokenizer, task, args, device, H):
    """Sequential state passing (LatentMAS latent_mas equivalent on RWKV).
    Each agent reads their text, state is passed to next agent.
    """
    state_size_bytes = 0
    past_kv = None

    for ai, agent_text in enumerate(task["agents"]):
        input_ids = tokenizer(agent_text, return_tensors="pt").input_ids.to(device)
        out = model.rwkv_model(input_ids=input_ids,
                              attention_mask=torch.ones_like(input_ids).bool(),
                              past_key_values=past_kv,
                              use_cache=True, return_dict=True)
        past_kv = out.past_key_values
        # Measure state size (once, it's fixed)
        if ai == 0:
            for layer in past_kv.layers:
                st = layer.state
                if st and st.get("recurrent_state") is not None:
                    state_size_bytes += st["recurrent_state"].numel() * st["recurrent_state"].element_size()
                if st and st.get("conv_state") is not None:
                    state_size_bytes += st["conv_state"].numel() * st["conv_state"].element_size()

    # Answer question from accumulated state
    total_bytes = len(task["agents"]) * state_size_bytes  # N agents × fixed state size

    q_ids = tokenizer(task["q"], return_tensors="pt").input_ids.to(device)
    past_kv, lb = recompute_logits_from_injected_cache(model, q_ids, torch.ones_like(q_ids), past_kv)
    logits = lb[0]
    all_ids = list(q_ids[0].tolist())
    new_ids = []
    eos_id = getattr(tokenizer, "eos_token_id", None)
    for _ in range(args.max_new_tokens):
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
    hit = task["gold"] in text
    return hit, total_bytes, text


@torch.no_grad()
def method_dual_fusion(model, tokenizer, task, args, device, H, dtype, zbar, shared_dirs_K3, z_mean_pca):
    """Our method: plan-based dual fusion (PCA_K3 + lam=1.0 + 0.5/0.25/0.25)."""
    n_agents = len(task["agents"])

    # Encode each agent
    agent_zs = []
    mems = []
    for agent_text in task["agents"]:
        a_ids = tokenizer(agent_text, return_tensors="pt").input_ids.to(device)
        z = encode_prefix(model, a_ids, torch.ones_like(a_ids))[0].to(dtype)
        agent_zs.append(z)
        mem = capture_final_state(model, a_ids)
        mems.append(mem)

    # Compute plan with PCA_K3
    dirs = shared_dirs_K3
    resid_sum = None
    for z in agent_zs:
        zc = z[0].float() - z_mean_pca[0].float()
        p = zc @ dirs.T
        r = zc - p @ dirs
        resid_sum = r if resid_sum is None else resid_sum + r
    fused_resid = 1.0 * (resid_sum / n_agents)
    cond = (z_mean_pca[0].float() + fused_resid).to(dtype).unsqueeze(0)
    Z = sample_trajectory_cfg(model, cond, STEPS, CFG, device, dtype)
    plan = [model.predict_states(Z[:, h]) for h in range(H)]

    # Measure communication cost: N × state_size
    state_size = 0
    for layer_mem in mems[0]:
        if layer_mem is not None:
            state_size += layer_mem.numel() * 4  # fp32
    total_bytes = n_agents * state_size + 32 * 4  # N states + latent

    # Fuse and decode
    w_plan = 0.5
    w_mem = (1.0 - w_plan) / n_agents
    plan_l = plan
    mems_l = mems
    q_ids = tokenizer(task["q"], return_tensors="pt").input_ids.to(device)
    fused = decode_chunkwise(model, tokenizer, q_ids,
                             lambda h: fuse_states(
                                 [(w_plan, plan_l[h])] + [(w_mem, mems_l[i]) for i in range(n_agents)]
                             ), H, args)
    text = tokenizer.decode(fused, skip_special_tokens=True).strip().lower()
    hit = task["gold"] in text
    return hit, total_bytes, text


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

    N_TASKS = 20
    task_configs = [
        ("2agent", 2),
        ("3agent", 3),
        ("4agent", 4),
    ]

    all_results = {}

    for ttype, n_agents in task_configs:
        print(f"\n{'='*80}", flush=True)
        print(f"[{ttype}] {n_agents} agents, {N_TASKS} tasks", flush=True)
        print(f"{'='*80}", flush=True)

        tasks = gen_nagent_chain(n_agents, N_TASKS, SEED)

        # Collect agent latents for PCA (global)
        all_z = []
        for it in tasks:
            for agent_text in it["agents"]:
                a_ids = enc(agent_text)
                z = encode_prefix(model, a_ids, torch.ones_like(a_ids))[0].to(dtype)
                all_z.append(z[0].float().cpu())
        z_all = torch.stack(all_z, 0).to(device)
        zbar = z_all.mean(dim=0, keepdim=True).to(dtype)
        z_mean_pca = zbar
        dirs_K3, _, _, _ = pca_remove_shared(z_all, K=3)

        methods = [
            ("text_concat", method_text_concat),
            ("text_mas", method_text_mas),
            ("state_carryover", method_state_carryover),
            ("dual_fusion", None),  # special handling
        ]

        type_results = {}

        for method_name, method_fn in methods:
            hits = 0
            total = 0
            total_bytes_all = 0
            t0 = time.time()

            for ti, it in enumerate(tasks):
                try:
                    torch.cuda.empty_cache()
                    if method_name == "dual_fusion":
                        hit, nbytes, text = method_dual_fusion(
                            model, tokenizer, it, args, device, H, dtype, zbar, dirs_K3, z_mean_pca)
                    else:
                        hit, nbytes, text = method_fn(model, tokenizer, it, args, device, H)
                    if hit:
                        hits += 1
                    total += 1
                    total_bytes_all += nbytes
                except Exception as e:
                    print(f"  ERROR [{method_name}] task {ti}: {e}", flush=True)
                    traceback.print_exc()
                    total += 1

            acc = hits / total * 100 if total > 0 else 0
            avg_bytes = total_bytes_all / total if total > 0 else 0
            elapsed = time.time() - t0
            type_results[method_name] = {
                "acc": acc, "hits": hits, "total": total,
                "avg_comm_bytes": avg_bytes, "time_sec": elapsed
            }
            print(f"  {method_name:20s} = {acc:5.1f}% ({hits}/{total}) | "
                  f"comm={avg_bytes:>10.0f} bytes | time={elapsed:.1f}s", flush=True)

        all_results[ttype] = type_results

    # Summary
    print(f"\n{'='*80}", flush=True)
    print("SUMMARY: Communication cost vs accuracy", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Method':20s} | {'2-agent acc':>12s} {'bytes':>10s} | "
          f"{'3-agent acc':>12s} {'bytes':>10s} | {'4-agent acc':>12s} {'bytes':>10s}", flush=True)
    print("-" * 90, flush=True)
    for method in ["text_concat", "text_mas", "state_carryover", "dual_fusion"]:
        row = f"{method:20s} |"
        for ttype in ["2agent", "3agent", "4agent"]:
            if ttype in all_results and method in all_results[ttype]:
                r = all_results[ttype][method]
                row += f" {r['acc']:>10.1f}% {r['avg_comm_bytes']:>10.0f} |"
            else:
                row += f" {'N/A':>11s} {'N/A':>10s} |"
        print(row, flush=True)

    out_path = REPO / "results/fusion_ablation/exp1_comm_cost.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(all_results, open(out_path, "w"), indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    run()
