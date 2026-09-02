#!/usr/bin/env python3
"""4-agent PCA ablation (EXPERIMENT_PLAN experiment 4).

Tests whether PCA-topK shared-direction removal beats plain residual fusion
on 4-agent chains, and whether K1-without-centering degenerates to residual.

Conditions: residual, PCA_K1, PCA_K3, K1_no_center, text_concat, best_agent.
One (seed, device) per run; launch 4 seeds across 4 GPUs and aggregate.
"""
from __future__ import annotations
import argparse, json, random, sys
import torch

sys.path.insert(0, ".")
from scripts.eval.diag_hard_problems_v2 import (
    REPO, gen_4agent, compute_residual_plan, compute_pca_plan,
    _LINK_ENTITIES, _PLACES, _SUBJECTS,
)
from scripts.eval.sample_prefix_suffix_trajectory_cfg import encode_prefix
from scripts.eval.exp_multiagent_star_fusion import (
    capture_final_state, fuse_states, decode_chunkwise, seq_carryover_state,
)
from scripts.eval.relay_utils import load_relay_model
from scripts.eval.run_cola_dlm_tasks_prefix_suffix_trajectory_cfg import (
    recompute_logits_from_injected_cache, sample_next_token,
)

CKPT = "outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000"


def gen_chain(nagents, n, seed):
    """Generic n-agent chain: subj -> link1 -> ... -> link{n-1} -> town."""
    rng = random.Random(seed)
    tasks = []
    nlinks = nagents - 1
    assert nlinks <= len(_LINK_ENTITIES)
    for i in range(n):
        subj = _SUBJECTS[i % len(_SUBJECTS)]
        links = rng.sample(_LINK_ENTITIES, nlinks)
        gold = rng.choice(_PLACES)
        agents = [f"Agent A knows: the {subj} is hidden by the {links[0]}."]
        for j in range(1, nlinks):
            rel = "is located at" if j == 1 else "is near"
            agents.append(f"Agent {chr(65 + j)} knows: the {links[j-1]} {rel} the {links[j]}.")
        agents.append(f"Agent {chr(65 + nagents - 1)} knows: the {links[-1]} is in the town of {gold}.")
        q = f"Question: in which town is the {subj} hidden? Answer: the town of"
        tasks.append({"agents": agents, "q": q, "gold": gold.lower()})
    return tasks


class Args:
    max_new_tokens = 24
    temperature = 0.5
    top_k = 10
    top_p = 0.9
    repetition_penalty = 1.2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--mode", type=str, default="full",
                    choices=["full", "plan_only", "state_only"],
                    help="full = 0.5*plan+0.5*state; plan_only isolates the plan channel; "
                         "state_only isolates the carryover channel")
    ap.add_argument("--agents", type=int, default=4)
    cli = ap.parse_args()

    device = torch.device(cli.device)
    model, _, tokenizer, _, _ = load_relay_model(str(REPO / CKPT), str(device))
    model.eval()
    model._prefix_suffix_trajectory_s2 = True
    dtype = next(model.alpha_heads.parameters()).dtype
    H = int(model.trajectory_horizon)

    def enc(t):
        return tokenizer(t, return_tensors="pt").input_ids.to(device)

    args = Args()
    N = cli.n
    tasks = gen_4agent(N, cli.seed) if cli.agents == 4 else gen_chain(cli.agents, N, cli.seed)
    print(f"{cli.agents}agent N={N} H={H} seed={cli.seed} device={cli.device} mode={cli.mode}", flush=True)

    all_z = []
    for it in tasks:
        for a in it["agents"]:
            all_z.append(encode_prefix(model, enc(a), torch.ones_like(enc(a)))[0].to(dtype)[0].float().cpu())
    z_all = torch.stack(all_z, 0).to(device)
    zbar = z_all.mean(dim=0, keepdim=True).to(dtype)
    zc = z_all.float() - z_all.float().mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(zc, full_matrices=False)
    dirs_K1 = Vh[:1]
    dirs_K3 = Vh[:3]
    z_mean = z_all.float().mean(dim=0, keepdim=True)
    U2, S2, Vh2 = torch.linalg.svd(z_all.float(), full_matrices=False)
    dirs_nc = Vh2[:1]
    print(f"PCA sv1_c={S[0]:.4f} sv1_nc={S2[0]:.4f}", flush=True)

    res = {"residual": [], "PCA_K1": [], "PCA_K3": [], "K1_nc": [], "concat": [], "best": []}

    if cli.mode == "full":
        def fuse(ps, ss):
            return lambda h: fuse_states([(0.5, ps[h]), (0.5, ss)])
    elif cli.mode == "plan_only":
        def fuse(ps, ss):
            return lambda h: ps[h]
    else:
        def fuse(ps, ss):
            return lambda h: ss

    for ii, it in enumerate(tasks):
        gold = it["gold"].lower()
        q_ids = enc(it["q"])
        azs = [encode_prefix(model, enc(a), torch.ones_like(enc(a)))[0].to(dtype) for a in it["agents"]]
        ss = seq_carryover_state(model, tokenizer, it["agents"], device)

        p = compute_residual_plan(model, azs, zbar, device, dtype)
        d = decode_chunkwise(model, tokenizer, q_ids, fuse(p, ss), H, args)
        res["residual"].append(gold in tokenizer.decode(d, skip_special_tokens=True).strip().lower())

        for mk, k, dirs, mean in [
            ("PCA_K1", 1, dirs_K1, z_mean),
            ("PCA_K3", 3, dirs_K3, z_mean),
            ("K1_nc", 1, dirs_nc, torch.zeros(1, 32, device=device)),
        ]:
            p = compute_pca_plan(model, azs, mean, dirs, k, device, dtype)
            d = decode_chunkwise(model, tokenizer, q_ids, fuse(p, ss), H, args)
            res[mk].append(gold in tokenizer.decode(d, skip_special_tokens=True).strip().lower())

        combined = " ".join(it["agents"])
        c_ids = enc(combined)
        out = model.rwkv_model(input_ids=c_ids, attention_mask=torch.ones_like(c_ids).bool(),
                               use_cache=True, return_dict=True)
        pk = out.past_key_values
        aids = list(c_ids[0].tolist()) + list(q_ids[0].tolist())
        ctx = torch.tensor([aids], device=device, dtype=torch.long)
        pk, lb = recompute_logits_from_injected_cache(model, ctx, torch.ones_like(ctx), pk)
        logits = lb[0]
        nids = []
        for _ in range(24):
            nid = sample_next_token(logits, aids, args)
            if nid == getattr(tokenizer, "eos_token_id", None):
                break
            nids.append(nid)
            aids.append(nid)
            out = model.rwkv_model(input_ids=torch.tensor([[nid]], device=device),
                                   past_key_values=pk, use_cache=True, return_dict=True)
            pk = out.past_key_values
            logits = out.logits[0, -1]
        res["concat"].append(gold in tokenizer.decode(nids, skip_special_tokens=True).strip().lower())

        b = False
        for ai in range(len(it["agents"])):
            s = capture_final_state(model, enc(it["agents"][ai]))
            d = decode_chunkwise(model, tokenizer, q_ids, lambda h: s, H, args)
            if gold in tokenizer.decode(d, skip_special_tokens=True).strip().lower():
                b = True
        res["best"].append(b)

        if (ii + 1) % 5 == 0:
            def a(k):
                return sum(res[k]) / len(res[k]) * 100
            print(f"[{ii+1}/{N}] residual={a('residual'):.0f}% K1={a('PCA_K1'):.0f}% "
                  f"K3={a('PCA_K3'):.0f}% nc={a('K1_nc'):.0f}% concat={a('concat'):.0f}%", flush=True)

    s = {k: round(sum(v) / len(v) * 100, 1) for k, v in res.items()}
    for k in ["residual", "PCA_K1", "PCA_K3", "K1_nc", "concat", "best"]:
        print(f"{k:12s}: {s[k]:5.1f}% ({sum(res[k])}/{len(res[k])})", flush=True)
    payload = {"seed": cli.seed, "n": N, "acc": s,
               "raw": {k: [int(x) for x in v] for k, v in res.items()},
               "sv1_centered": float(S[0]), "sv1_nocenter": float(S2[0])}
    with open(cli.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {cli.out}", flush=True)


if __name__ == "__main__":
    main()
