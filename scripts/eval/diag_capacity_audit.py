#!/usr/bin/env python3
"""Capacity audit for the LACES writable-state manifold (zero-training, no RWKV backprop).

Answers three questions about the frozen-RWKV latent/state interface:
  1. Can the interface carry "reasoning" (here: multi-hop fact fusion)?
  2. Is the 32-D plan too small?
  3. Are the K=16 basis states the real bottleneck?

Setup (residual write, matching the "reasoning correction" framing):
  S_context  = frozen RWKV recurrent state after reading ONLY the question
  S_gold     = frozen RWKV recurrent state after reading (facts + question)
  dS*        = S_gold - S_context   = the *correction* needed to answer

We ask: can the writable manifold represent the correction dS*? Three levels,
each minimizing per-layer relative MSE against dS* (all module weights frozen,
no backprop through the 2.9B RWKV -- the FLA kernel does not backprop the
recurrent state):

  Level 1 (optimize Z)     : dS = basis @ alpha_heads(z)   (latent -> head -> basis)
  Level 2 (optimize alpha) : dS = basis @ alpha             (bypass 32-D latent + head)
  Level 3 (optimize dS)    : dS_l = U_l C_l V_l^T (low-rank, bypass basis)

The optimized dS is injected RESIDUALLY (state = S_context + dS) and the answer
is generated. Also reported:
  e16_full  = ||S_gold  - proj_basis(S_gold)||^2 / ||S_gold||^2   (basis can't hold full memory)
  e16_delta = ||dS*     - proj_basis(dS*)||^2    / ||dS*||^2      (basis can't hold the correction)

Conditions (generate-then-match, gold token in generated text):
  text_concat : oracle (read facts as text)
  inject_gold : full-replace inject the true S_gold (state-injection ceiling)
  R0_dual     : existing method (residual plan + seq-carryover fusion), no optimization
  L1_z / L2_alpha / L3_delta : the three optimized residual corrections

Interpretation:
  e16_delta ~ 0  -> 16-basis is enough for the correction; bottleneck is elsewhere
  e16_delta ~ 1  -> 16-basis span is the hard bottleneck (motivates residual low-rank write)
  L3 >> L2 >> L1 -> basis/rank and 32-D latent are both limiting
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from scripts.eval.sample_prefix_suffix_trajectory_cfg import (  # noqa: E402
    encode_prefix,
)
from scripts.eval.exp_multiagent_star_fusion import (  # noqa: E402
    capture_final_state, fuse_states, decode_chunkwise, seq_carryover_state, raw_text,
)
from scripts.eval.relay_utils import load_relay_model  # noqa: E402
from scripts.eval.run_cola_dlm_tasks_prefix_suffix_trajectory_cfg import (  # noqa: E402
    recompute_logits_from_injected_cache, sample_next_token,
)
from scripts.eval.diag_hard_problems_v2 import (  # noqa: E402
    gen_2agent, gen_3agent, gen_4agent, gen_conflict, compute_residual_plan,
)

CKPT = str(REPO / "outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000")


# --------------------------------------------------------------------------- #
#  Generation helpers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate_with_state(model, tokenizer, q_ids, state, args, residual=False):
    """Read question, inject a single per-layer state (full-replace or residual add), generate."""
    out = model.rwkv_model(input_ids=q_ids, attention_mask=torch.ones_like(q_ids).bool(),
                           use_cache=True, return_dict=True)
    past_kv = out.past_key_values
    if residual:
        for l, st in enumerate(state):
            layer = past_kv.layers[l]
            cur = layer.state.get("recurrent_state") if layer.state is not None else None
            if cur is not None and isinstance(cur, torch.Tensor):
                layer.state["recurrent_state"] = cur.float() + st.float()
            else:
                layer.state["recurrent_state"] = st.float()
    else:
        past_kv = model.inject_into_cache(past_kv, state)
    all_ids = list(q_ids[0].tolist())
    ctx = torch.tensor([all_ids], device=q_ids.device, dtype=torch.long)
    past_kv, lb = recompute_logits_from_injected_cache(model, ctx, torch.ones_like(ctx), past_kv)
    logits = lb[0]
    new_ids: list[int] = []
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


def sub_states(a, b):
    """Element-wise per-layer subtraction of two state lists."""
    return [x - y for x, y in zip(a, b)]


def rel_mse(states, target):
    """Sum of per-layer relative MSE between two lists of [1,H,D,D] states."""
    tot = 0.0
    for s, t in zip(states, target):
        s = s.float()
        t = t.float()
        tot = tot + (s - t).pow(2).sum() / t.pow(2).sum().clamp(min=1e-8)
    return tot


def optimize(params, state_fn, target, steps, lr):
    """Adam-minimize per-layer relative MSE between state_fn() and target. Returns detached states."""
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        states = state_fn()
        loss = rel_mse(states, target)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return [s.detach() for s in state_fn()]


def basis_projection_error(basis, target, L, K):
    """Mean relative projection error of `target` (list of [1,H,D,D]) onto span(basis)."""
    e = 0.0
    for l in range(L):
        bf = basis[l].float().reshape(K, -1)          # [K, H*D*D]
        s = target[l].float().reshape(-1)             # [H*D*D]
        coeffs = torch.linalg.lstsq(bf.t(), s).solution
        resid = s - bf.t() @ coeffs
        e += (resid.pow(2).sum() / s.pow(2).sum().clamp(min=1e-8)).item()
    return e / L


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def run(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"[init] loading {args.ckpt_dir}", flush=True)
    model, _rwkv, tokenizer, _ckpt, _cfg = load_relay_model(args.ckpt_dir, device)
    model.eval()
    model._prefix_suffix_trajectory_s2 = True
    dtype = next(model.alpha_heads.parameters()).dtype
    L = int(model.num_layers)
    K = int(model.n_basis)
    H = int(model.num_heads)
    D = int(model.head_dim)
    Hh = int(model.trajectory_horizon)
    basis = model.state_basis.detach()  # [L, K, H, D, D]
    scale = float(model.state_scale.detach())
    print(f"[init] L={L} K={K} heads={H} head_dim={D} horizon={Hh} "
          f"dtype={dtype} state_scale={scale:.4f}", flush=True)

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

    gen = {"2agent": gen_2agent, "3agent": gen_3agent, "4agent": gen_4agent, "conflict": gen_conflict}
    tasks = []
    for tname in args.tasks:
        tasks.extend(gen[tname](args.n, args.seed))
    print(f"[tasks] {len(tasks)} total", flush=True)

    # shared mean over all agent latents (for R0_dual residual plan)
    all_z = []
    for it in tasks:
        for a_text in it["agents"]:
            a_ids = enc(a_text)
            all_z.append(encode_prefix(model, a_ids, torch.ones_like(a_ids))[0].to(dtype)[0])
    zbar = torch.stack(all_z, 0).mean(dim=0, keepdim=True).to(dtype)

    conds = ["text_concat", "inject_gold", "R0_dual", "L1_z", "L2_alpha", "L3_delta"]
    correct = {c: 0 for c in conds}
    mse = {"L1_z": [], "L2_alpha": [], "L3_delta": []}
    e16_full = []
    e16_delta = []
    items = []

    for ii, it in enumerate(tasks):
        gold = it["gold"].lower()
        facts = it["agents"]
        q_ids = enc(it["q"])
        facts_text = " ".join(facts)

        S_context = capture_final_state(model, q_ids)                 # question only
        S_gold = capture_final_state(model, enc(facts_text + " " + it["q"]))  # facts + question
        dS_star = sub_states(S_gold, S_context)                       # the correction

        rec = {"gold": gold, "gen": {}, "mse": {}}

        # ---- text_concat oracle ----
        tc_ids = raw_text(model, tokenizer, facts_text, it["q"], samp)
        tc_txt = tokenizer.decode(tc_ids, skip_special_tokens=True).strip().lower()
        rec["gen"]["text_concat"] = {"text": tc_txt[:60], "hit": gold in tc_txt}

        # ---- inject_gold: full-replace inject true S_gold ----
        ig_ids = generate_with_state(model, tokenizer, q_ids, S_gold, samp, residual=False)
        ig_txt = tokenizer.decode(ig_ids, skip_special_tokens=True).strip().lower()
        rec["gen"]["inject_gold"] = {"text": ig_txt[:60], "hit": gold in ig_txt}

        # ---- Level 1: optimize Z (latent -> head -> basis), no scale ----
        z = torch.zeros(1, 32, device=device, dtype=torch.float32, requires_grad=True)
        def l1_fn(z=z, model=model, L=L, dtype=dtype):
            zd = z.to(dtype)
            out = []
            for l in range(L):
                alpha = model.alpha_heads[l](zd)                     # [1,16]
                s = torch.einsum('bk,khde->bhde', alpha, model.state_basis[l].to(dtype))
                out.append(s.float())                                # [1,H,D,D]
            return out
        l1 = optimize([z], l1_fn, dS_star, args.steps, args.lr_z)
        mse["L1_z"].append(rel_mse(l1, dS_star).item())
        l1_ids = generate_with_state(model, tokenizer, q_ids, l1, samp, residual=True)
        l1_txt = tokenizer.decode(l1_ids, skip_special_tokens=True).strip().lower()
        rec["gen"]["L1_z"] = {"text": l1_txt[:60], "hit": gold in l1_txt}

        # ---- Level 2: optimize alpha (bypass latent + head) ----
        alpha = torch.zeros(L, K, device=device, dtype=torch.float32, requires_grad=True)
        def l2_fn(alpha=alpha, basis=basis, L=L, dtype=dtype):
            ad = alpha.to(dtype)
            out = []
            for l in range(L):
                s = torch.einsum('k,khde->hde', ad[l], basis[l].to(dtype))
                out.append(s.unsqueeze(0).float())
            return out
        l2 = optimize([alpha], l2_fn, dS_star, args.steps, args.lr_alpha)
        mse["L2_alpha"].append(rel_mse(l2, dS_star).item())
        l2_ids = generate_with_state(model, tokenizer, q_ids, l2, samp, residual=True)
        l2_txt = tokenizer.decode(l2_ids, skip_special_tokens=True).strip().lower()
        rec["gen"]["L2_alpha"] = {"text": l2_txt[:60], "hit": gold in l2_txt}

        # ---- Level 3: optimize low-rank delta dS_l = U_l C_l V_l^T ----
        r = args.rank
        Us = [torch.randn(H * D, r, device=device, dtype=torch.float32) * 0.01 for _ in range(L)]
        Cs = [torch.randn(r, r, device=device, dtype=torch.float32) * 0.01 for _ in range(L)]
        Vs = [torch.randn(D, r, device=device, dtype=torch.float32) * 0.01 for _ in range(L)]
        for u in Us + Cs + Vs:
            u.requires_grad_(True)
        def l3_fn(Us=Us, Cs=Cs, Vs=Vs, L=L, H=H, D=D):
            out = []
            for l in range(L):
                mat = Us[l] @ Cs[l] @ Vs[l].t()                     # [H*D, D]
                out.append(mat.reshape(H, D, D).unsqueeze(0))
            return out
        l3 = optimize(Us + Cs + Vs, l3_fn, dS_star, args.steps, args.lr_delta)
        mse["L3_delta"].append(rel_mse(l3, dS_star).item())
        l3_ids = generate_with_state(model, tokenizer, q_ids, l3, samp, residual=True)
        l3_txt = tokenizer.decode(l3_ids, skip_special_tokens=True).strip().lower()
        rec["gen"]["L3_delta"] = {"text": l3_txt[:60], "hit": gold in l3_txt}

        # ---- R0_dual: existing method (residual plan + seq carryover), no optimization ----
        try:
            agent_zs = [encode_prefix(model, enc(a), torch.ones_like(enc(a)))[0].to(dtype)
                        for a in facts]
            plan = compute_residual_plan(model, agent_zs, zbar, device, dtype)
            seq_state = seq_carryover_state(model, tokenizer, facts, device)
            dual_ids = decode_chunkwise(model, tokenizer, q_ids,
                lambda h: fuse_states([(0.5, plan[h]), (0.5, seq_state)]), Hh, samp)
            dual_txt = tokenizer.decode(dual_ids, skip_special_tokens=True).strip().lower()
            rec["gen"]["R0_dual"] = {"text": dual_txt[:60], "hit": gold in dual_txt}
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            rec["gen"]["R0_dual"] = {"text": f"ERROR {e}", "hit": False}

        # ---- projection errors ----
        e16_full.append(basis_projection_error(basis, S_gold, L, K))
        e16_delta.append(basis_projection_error(basis, dS_star, L, K))

        rec["mse"] = {k: mse[k][-1] for k in mse}
        rec["e16_full"] = e16_full[-1]
        rec["e16_delta"] = e16_delta[-1]
        items.append(rec)

        for c in conds:
            correct[c] += int(rec["gen"][c]["hit"])
        n = ii + 1
        line = " ".join(f"{c}={correct[c]}/{n}" for c in conds)
        print(f"[{n}/{len(tasks)}] {line} | e16_full={e16_full[-1]:.3f} e16_delta={e16_delta[-1]:.3f}", flush=True)

    # ---- summary ----
    summary = {
        "ckpt_dir": args.ckpt_dir,
        "n": len(tasks),
        "tasks": args.tasks,
        "steps": args.steps,
        "rank": args.rank,
        "state_scale": scale,
        "accuracy": {c: round(correct[c] / len(tasks), 4) for c in conds},
        "mean_mse": {k: round(sum(v) / len(v), 4) for k, v in mse.items()},
        "mean_e16_full": round(sum(e16_full) / len(e16_full), 4),
        "mean_e16_delta": round(sum(e16_delta) / len(e16_delta), 4),
        "items": items,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print("\n================ CAPACITY AUDIT ================", flush=True)
    for c in conds:
        print(f"  {c:12s} acc = {summary['accuracy'][c]*100:5.1f}%", flush=True)
    print("  ---- matching MSE vs dS* (lower=better) ----", flush=True)
    for k, v in summary["mean_mse"].items():
        print(f"  {k:12s} mse = {v}", flush=True)
    print(f"  basis projection error e16 (full S_gold)  = {summary['mean_e16_full']:.4f}", flush=True)
    print(f"  basis projection error e16 (delta dS*)    = {summary['mean_e16_delta']:.4f}", flush=True)
    print(f"written: {out}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default=CKPT)
    p.add_argument("--tasks", nargs="+", default=["2agent", "3agent"])
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--lr_z", type=float, default=0.05)
    p.add_argument("--lr_alpha", type=float, default=0.05)
    p.add_argument("--lr_delta", type=float, default=0.05)
    p.add_argument("--max_new_tokens", type=int, default=24)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="results/capacity_audit/audit.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
