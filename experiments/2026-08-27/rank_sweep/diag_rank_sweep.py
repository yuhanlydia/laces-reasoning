#!/usr/bin/env python3
"""E2: Rank sweep for the dynamic low-rank state writer (zero-training).

Question: how much rank r does a low-rank residual state writer
    dS_l = U_l C_l V_l^T   (U [H*D, r], C [r,r], V [r, D])
need to (a) capture the gold correction dS* = S_gold - S_context, and (b) flip the answer?

The rank-4 audit gave 25% energy / 0% accuracy. Here we sweep r and report:
  - SVD truncation error e_r = ||dS* - dS*_r||^2 / ||dS*||^2  (theoretical energy floor, exact)
  - optimization MSE + generate-then-match accuracy (behavioral)

r=64 is full rank (the flattened per-layer matrix is [H*D, D] = [2560, 64], rank <= 64),
so r=64 ~= inject_gold ceiling.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from scripts.eval.exp_multiagent_star_fusion import capture_final_state  # noqa: E402
from scripts.eval.relay_utils import load_relay_model  # noqa: E402
from scripts.eval.run_cola_dlm_tasks_prefix_suffix_trajectory_cfg import (  # noqa: E402
    recompute_logits_from_injected_cache, sample_next_token,
)
from scripts.eval.diag_hard_problems_v2 import gen_2agent, gen_3agent  # noqa: E402

CKPT = str(REPO / "outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000")
RANKS = [4, 8, 16, 32, 64]


@torch.no_grad()
def generate_residual(model, tokenizer, q_ids, delta_states, args):
    out = model.rwkv_model(input_ids=q_ids, attention_mask=torch.ones_like(q_ids).bool(),
                           use_cache=True, return_dict=True)
    past_kv = out.past_key_values
    for l, st in enumerate(delta_states):
        layer = past_kv.layers[l]
        cur = layer.state.get("recurrent_state") if layer.state is not None else None
        layer.state["recurrent_state"] = (cur.float() + st.float()) if isinstance(cur, torch.Tensor) else st.float()
    all_ids = list(q_ids[0].tolist())
    ctx = torch.tensor([all_ids], device=q_ids.device, dtype=torch.long)
    past_kv, lb = recompute_logits_from_injected_cache(model, ctx, torch.ones_like(ctx), past_kv)
    logits = lb[0]
    new_ids = []
    eos = getattr(tokenizer, "eos_token_id", None)
    for _ in range(args.max_new_tokens):
        nid = sample_next_token(logits, all_ids, args)
        if eos is not None and nid == eos:
            break
        new_ids.append(nid)
        all_ids.append(nid)
        out = model.rwkv_model(input_ids=torch.tensor([[nid]], device=q_ids.device),
                               past_key_values=past_kv, use_cache=True, return_dict=True)
        past_kv = out.past_key_values
        logits = out.logits[0, -1]
    return new_ids


def rel_mse(states, target):
    tot = 0.0
    for s, t in zip(states, target):
        s = s.float(); t = t.float()
        tot = tot + (s - t).pow(2).sum() / t.pow(2).sum().clamp(min=1e-8)
    return tot


def optimize_lowrank(target, L, H, D, r, steps, lr, device):
    Us = [torch.randn(H * D, r, device=device, dtype=torch.float32) * 0.01 for _ in range(L)]
    Cs = [torch.randn(r, r, device=device, dtype=torch.float32) * 0.01 for _ in range(L)]
    Vs = [torch.randn(D, r, device=device, dtype=torch.float32) * 0.01 for _ in range(L)]
    for u in Us + Cs + Vs:
        u.requires_grad_(True)

    def fn():
        out = []
        for l in range(L):
            mat = Us[l] @ Cs[l] @ Vs[l].t()
            out.append(mat.reshape(H, D, D).unsqueeze(0))
        return out

    opt = torch.optim.Adam(Us + Cs + Vs, lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = rel_mse(fn(), target)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return [s.detach() for s in fn()]


def svd_trunc_err(target, L, H, D, r):
    """Exact rank-r SVD truncation error of dS* (per-layer [H,D,D] flattened to [H*D, D])."""
    e = 0.0
    for l in range(L):
        M = target[l].float().reshape(H * D, D)
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        if r < S.shape[0]:
            Mr = U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :]
        else:
            Mr = M
        e += ((M - Mr).pow(2).sum() / M.pow(2).sum().clamp(min=1e-8)).item()
    return e / L


def run(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"[init] loading {args.ckpt_dir}", flush=True)
    model, _rwkv, tokenizer, _ckpt, _cfg = load_relay_model(args.ckpt_dir, device)
    model.eval()
    model._prefix_suffix_trajectory_s2 = True
    L = int(model.num_layers)
    H = int(model.num_heads)
    D = int(model.head_dim)
    print(f"[init] L={L} heads={H} head_dim={D}", flush=True)

    def enc(t):
        return tokenizer(t, return_tensors="pt").input_ids.to(device)

    class Args:
        max_new_tokens = args.max_new_tokens
        temperature = args.temperature
        top_k = args.top_k
        top_p = args.top_p
        repetition_penalty = args.repetition_penalty
    samp = Args()
    samp.device = device

    gen = {"2agent": gen_2agent, "3agent": gen_3agent}
    tasks = []
    for tname in args.tasks:
        tasks.extend(gen[tname](args.n, args.seed))

    # which ranks to actually optimize+decode (r=64 == full rank == inject_gold)
    opt_ranks = [r for r in args.opt_ranks if r < 64]
    sweep = {r: {"svd_err": [], "mse": [], "acc": 0} for r in RANKS}
    n = 0

    for ii, it in enumerate(tasks):
        gold = it["gold"].lower()
        facts = it["agents"]
        q_ids = enc(it["q"])
        facts_text = " ".join(facts)

        S_context = capture_final_state(model, q_ids)
        S_gold = capture_final_state(model, enc(facts_text + " " + it["q"]))
        dS_star = [g - c for g, c in zip(S_gold, S_context)]

        for r in RANKS:
            sweep[r]["svd_err"].append(svd_trunc_err(dS_star, L, H, D, r))

        # full rank r=64 == inject S_gold residual == inject dS* exactly
        acc64_ids = generate_residual(model, tokenizer, q_ids, dS_star, samp)
        acc64 = gold in tokenizer.decode(acc64_ids, skip_special_tokens=True).strip().lower()
        sweep[64]["acc"] += int(acc64)
        sweep[64]["mse"].append(0.0)

        for r in opt_ranks:
            dS_r = optimize_lowrank(dS_star, L, H, D, r, args.steps, args.lr, device)
            sweep[r]["mse"].append(rel_mse(dS_r, dS_star).item())
            ids = generate_residual(model, tokenizer, q_ids, dS_r, samp)
            hit = gold in tokenizer.decode(ids, skip_special_tokens=True).strip().lower()
            sweep[r]["acc"] += int(hit)

        n += 1
        line = " ".join(f"r{r}_svd={sum(sweep[r]['svd_err'])/n:.3f}" for r in [4, 8, 16, 32])
        line += " " + " ".join(f"r{r}_acc={sweep[r]['acc']}/{n}" for r in opt_ranks + [64])
        print(f"[{n}/{len(tasks)}] {line}", flush=True)

    summary = {
        "ckpt_dir": args.ckpt_dir,
        "n": n,
        "tasks": args.tasks,
        "ranks": RANKS,
        "svd_err": {str(r): round(sum(sweep[r]["svd_err"]) / n, 4) for r in RANKS},
        "mse": {str(r): round(sum(sweep[r]["mse"]) / len(sweep[r]["mse"]), 4) if sweep[r]["mse"] else None for r in RANKS},
        "accuracy": {str(r): round(sweep[r]["acc"] / n, 4) for r in RANKS},
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print("\n================ RANK SWEEP ================", flush=True)
    print(f"  {'rank':>6} {'svd_err':>10} {'mse':>10} {'acc':>8}", flush=True)
    for r in RANKS:
        m = summary["mse"][str(r)]
        m = f"{m:.3f}" if m is not None else "   full"
        print(f"  {r:>6} {summary['svd_err'][str(r)]:>10.4f} {m:>10} {summary['accuracy'][str(r)]*100:>7.1f}%", flush=True)
    print(f"written: {out}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default=CKPT)
    p.add_argument("--tasks", nargs="+", default=["2agent", "3agent"])
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--opt_ranks", nargs="+", type=int, default=[8, 16, 32])
    p.add_argument("--max_new_tokens", type=int, default=24)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="results/capacity_audit/rank_sweep.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
