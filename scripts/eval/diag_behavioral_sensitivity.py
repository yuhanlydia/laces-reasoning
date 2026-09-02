#!/usr/bin/env python3
"""E3: Behavioral sensitivity of basis directions (zero-order, forward-only).

Question: is the 16-basis span "behaviorally inert" for reasoning (r_B ~ 0)?

For each task we measure the gold-answer log-probability under a residual injection
    S_context + mag * dhat
where mag = ||dS*||_F is the magnitude of the FULL gold correction, and dhat is a
unit-norm direction. Directions compared (all unit-norm, all same magnitude mag):

  base   : no injection (S_context)                       -> lp_base (floor)
  gold   : dhat = dS* / ||dS*||   (the true correction)   -> lp_gold (ceiling, must be high)
  basis_k: dhat = B_k / ||B_k||   (learned basis, k=0..15)-> lp_k   (does steering help?)
  random : dhat = random                                   -> lp_rand (negative control)

If lp_k ~ lp_base << lp_gold for all k, the basis is a "language-steering subspace" with
~0 behavioral rank for reasoning (||B_k||_F large but the answer never changes).
NO backprop (FLA does not backprop the recurrent state).
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
    recompute_logits_from_injected_cache,
)
from scripts.eval.diag_hard_problems_v2 import gen_2agent  # noqa: E402

CKPT = str(REPO / "outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000")


def fro_norm(states):
    return sum(s.float().pow(2).sum() for s in states).sqrt().item()


def scale_dir(states, mag):
    """Scale a direction list to have Frobenius norm `mag`."""
    n = fro_norm(states)
    n = n if n > 1e-12 else 1.0
    return [s.float() * (mag / n) for s in states]


@torch.no_grad()
def gold_logprob(model, tokenizer, q_ids, gold_ids, delta_states):
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
    total = 0.0
    for gid in gold_ids:
        total += torch.log_softmax(logits.float(), dim=-1)[int(gid)].item()
        out = model.rwkv_model(input_ids=torch.tensor([[gid]], device=q_ids.device),
                               past_key_values=past_kv, use_cache=True, return_dict=True)
        past_kv = out.past_key_values
        logits = out.logits[0, -1]
    return total


def run(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"[init] loading {args.ckpt_dir}", flush=True)
    model, _rwkv, tokenizer, _ckpt, _cfg = load_relay_model(args.ckpt_dir, device)
    model.eval()
    model._prefix_suffix_trajectory_s2 = True
    L = int(model.num_layers)
    K = int(model.n_basis)
    basis = model.state_basis.detach().float()  # [L, K, H, D, D]
    print(f"[init] L={L} K={K}", flush=True)

    def enc(t):
        return tokenizer(t, return_tensors="pt").input_ids.to(device)

    tasks = gen_2agent(args.n, args.seed)

    lp_base = []
    lp_gold = []
    lp_rand = []
    lp_k = [0.0] * K
    norm_k = [0.0] * K

    for ii, it in enumerate(tasks):
        gold = it["gold"].lower()
        facts = it["agents"]
        q_ids = enc(it["q"])
        facts_text = " ".join(facts)
        gold_ids = tokenizer(" " + gold, return_tensors="pt").input_ids[0].tolist()

        S_context = capture_final_state(model, q_ids)
        S_gold = capture_final_state(model, enc(facts_text + " " + it["q"]))
        dS_star = [g - c for g, c in zip(S_gold, S_context)]
        mag = fro_norm(dS_star)

        zero = [torch.zeros_like(s) for s in S_context]
        lp_base.append(gold_logprob(model, tokenizer, q_ids, gold_ids, zero))
        lp_gold.append(gold_logprob(model, tokenizer, q_ids, gold_ids, dS_star))  # full correction

        drand = [torch.randn_like(s) for s in S_context]
        lp_rand.append(gold_logprob(model, tokenizer, q_ids, gold_ids, scale_dir(drand, mag)))

        for k in range(K):
            dk = [basis[l, k].unsqueeze(0) for l in range(L)]
            norm_k[k] += fro_norm(dk)
            lp_k[k] += gold_logprob(model, tokenizer, q_ids, gold_ids, scale_dir(dk, mag))

        print(f"[{ii+1}/{len(tasks)}] gold={gold} base={lp_base[-1]:.2f} "
              f"gold_full={lp_gold[-1]:.2f} rand={lp_rand[-1]:.2f} "
              f"best_basis={max(lp_k[k]/(ii+1) for k in range(K)):.2f}", flush=True)

    n = len(tasks)
    summary = {
        "ckpt_dir": args.ckpt_dir,
        "n": n,
        "mean_lp_base": round(sum(lp_base) / n, 4),
        "mean_lp_gold_full": round(sum(lp_gold) / n, 4),
        "mean_lp_rand": round(sum(lp_rand) / n, 4),
        "basis": [{"k": k, "norm_fro": round(norm_k[k] / n, 3),
                   "lp": round(lp_k[k] / n, 4)} for k in range(K)],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print("\n================ BEHAVIORAL SENSITIVITY ================", flush=True)
    print(f"  mean lp_base (floor)     = {summary['mean_lp_base']:.3f}", flush=True)
    print(f"  mean lp_gold_full (ceil) = {summary['mean_lp_gold_full']:.3f}", flush=True)
    print(f"  mean lp_rand (control)   = {summary['mean_lp_rand']:.3f}", flush=True)
    print(f"  {'k':>3} {'||B_k||_F':>12} {'lp_k':>12}", flush=True)
    for b in summary["basis"]:
        print(f"  {b['k']:>3} {b['norm_fro']:>12.3f} {b['lp']:>12.3f}", flush=True)
    print(f"written: {out}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default=CKPT)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="results/capacity_audit/behavioral_sensitivity.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
