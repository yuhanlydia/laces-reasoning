#!/usr/bin/env python3
"""E6: BDH dynamic reasoning writer v2 (the user's section-15 "first version").

Fixes the two E5 failures:
  1. INPUT: token-level hidden sequence H_{1:T} via cross-attention (NOT mean-pool).
  2. WRITER: truly dynamic per-layer directions U_l(W_R), V_l(W_R) (NOT fixed directions).

Architecture:
    H_{1:T} --cross-attn--> W_0 in R^512        (learned query attends to token sequence)
    W_{r+1} = GRU(W_r, CrossAttn(W_r, H))        R=4 recurrent "thinking" steps
    U_l(W_R) = U_core(W_R) @ M_l,   V_l(W_R) = V_core(W_R) @ N_l     (dynamic + per-layer)
    dS_l = U_l V_l^T                              rank r=32 per layer

Train ONLY L_state = per-layer relative MSE against dS* = S_gold - S_context (state-matching;
the FLA RWKV kernel does not backprop the recurrent state).

Gate (section 15): on held-out 2-hop/3-hop tasks, rank-32 dynamic writer reaches
MSE < 7 AND accuracy > 50%.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from scripts.eval.exp_multiagent_star_fusion import capture_final_state, raw_text  # noqa: E402
from scripts.eval.relay_utils import load_relay_model  # noqa: E402
from scripts.eval.run_cola_dlm_tasks_prefix_suffix_trajectory_cfg import (  # noqa: E402
    recompute_logits_from_injected_cache, sample_next_token,
)
from scripts.eval.diag_hard_problems_v2 import gen_2agent, gen_3agent  # noqa: E402

CKPT = str(REPO / "outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000")


class DynamicReasonWriter(nn.Module):
    """token-level cross-attn -> recurrent workspace -> dynamic per-head low-rank state writer.

    The RWKV-7 recurrent state is PER-HEAD [H, D, D]. We write per-head rank-r_h corrections
        dS_{l,h} = sum_m u_{l,h,m} v_{l,h,m}^T
    where u, v are produced by heads from the workspace (truly dynamic: per-layer, per-head,
    input-dependent directions). Per-head rank-1 gives a block-diagonal rank ~H per layer
    (>= flattened rank-32), matching E2's rank sweep.
    """

    def __init__(self, hidden_dim, L, H, D, r_h, ws_dim=512, n_steps=4):
        super().__init__()
        self.L, self.H, self.D, self.r_h = L, H, D, r_h
        self.n_steps = n_steps
        self.h_proj = nn.Linear(hidden_dim, ws_dim)                 # H [T,2560] -> [T,512]
        self.cross_attn = nn.MultiheadAttention(ws_dim, num_heads=8, batch_first=True)
        self.q0 = nn.Parameter(torch.randn(1, ws_dim) * 0.02)       # initial query
        self.gru = nn.GRUCell(ws_dim, ws_dim)                       # recurrent thinking step
        # dynamic per-head directions: u [L,H,D,r_h], v [L,H,D,r_h]
        self.u_head = nn.Linear(ws_dim, L * H * D * r_h)
        self.v_head = nn.Linear(ws_dim, L * H * D * r_h)

    def forward(self, H_tokens):
        # H_tokens: [T, hidden_dim] single sequence
        Hp = self.h_proj(H_tokens).unsqueeze(0)                     # [1, T, 512]
        ctx, _ = self.cross_attn(self.q0.unsqueeze(1), Hp, Hp)      # [1, 1, 512]
        W = ctx.squeeze(1)                                          # [1, 512]
        for _ in range(self.n_steps - 1):
            ctx, _ = self.cross_attn(W.unsqueeze(1), Hp, Hp)
            W = self.gru(W, ctx.squeeze(1))                         # [1, 512]
        w = W.squeeze(0)                                            # [512]
        u = self.u_head(w).reshape(self.L, self.H, self.D, self.r_h)  # [L,H,D,r_h]
        v = self.v_head(w).reshape(self.L, self.H, self.D, self.r_h)
        states = []
        for l in range(self.L):
            dS_l = torch.einsum('hdr,her->hde', u[l], v[l])         # [H,D,D] (rank r_h per head)
            states.append(dS_l.unsqueeze(0))
        return states


@torch.no_grad()
def token_hidden(model, ids):
    mask = torch.ones_like(ids).bool()
    out = model.rwkv_model(input_ids=ids, attention_mask=mask, output_hidden_states=True,
                           use_cache=True, return_dict=True)
    return out.hidden_states[-1][0].float()                         # [T, hidden_dim]


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


def rel_mse(delta, target):
    tot = 0.0
    for d, t in zip(delta, target):
        d = d.float(); t = t.float()
        tot = tot + (d - t).pow(2).sum() / t.pow(2).sum().clamp(min=1e-8)
    return tot


def direction_cos(delta, target):
    """Mean per-layer cosine similarity (scale-invariant)."""
    tot = 0.0
    n = 0
    for d, t in zip(delta, target):
        d = d.float().reshape(-1)
        t = t.float().reshape(-1)
        dn, tn = d.norm(), t.norm()
        if dn > 1e-8 and tn > 1e-8:
            tot = tot + (d * t).sum() / (dn * tn)
            n += 1
    return tot / max(n, 1)


def combined_loss(delta, target, L, lambda_mse=1.0):
    """(1 - mean cos) + lambda_mse * mean per-layer relative MSE."""
    return (1.0 - direction_cos(delta, target)) + lambda_mse * (rel_mse(delta, target) / L)


def make_dataset(model, tokenizer, tasks, device):
    data = []
    for it in tasks:
        facts = it["agents"]
        q_ids = tokenizer(it["q"], return_tensors="pt").input_ids.to(device)
        # input: token-level hidden of (facts + question)
        fq = tokenizer(" ".join(facts) + " " + it["q"], return_tensors="pt").input_ids.to(device)
        H = token_hidden(model, fq).cpu().half()                    # [T, hidden] fp16
        S_context = capture_final_state(model, q_ids)
        S_gold = capture_final_state(model, fq)
        dS = [g - c for g, c in zip(S_gold, S_context)]
        data.append((H, [t.cpu() for t in dS]))                     # dS kept fp32
    return data


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
    hidden_dim = int(model.hidden_size)
    print(f"[init] L={L} H={H} D={D} hidden={hidden_dim} r_h={args.r_h}", flush=True)

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
    train_tasks, test_tasks = [], []
    for tname in args.tasks:
        train_tasks.extend(gen[tname](args.n_train, args.seed))
        test_tasks.extend(gen[tname](args.n_test, args.seed + 999))

    print(f"[data] train set ({len(train_tasks)}) ...", flush=True)
    train_data = make_dataset(model, tokenizer, train_tasks, device)
    print(f"[data] test set ({len(test_tasks)}) ...", flush=True)
    test_data = make_dataset(model, tokenizer, test_tasks, device)

    writer = DynamicReasonWriter(hidden_dim, L, H, D, args.r_h, ws_dim=args.ws_dim,
                                 n_steps=args.n_steps).to(device)
    opt = torch.optim.Adam(writer.parameters(), lr=args.lr)
    print(f"[train] writer params = {sum(p.numel() for p in writer.parameters())/1e6:.2f}M, "
          f"epochs={args.epochs}", flush=True)

    for ep in range(args.epochs):
        writer.train()
        opt.zero_grad()
        tot = 0.0
        for Hh, dS in train_data:
            delta = writer(Hh.to(device).float())
            tot = tot + combined_loss(delta, [t.to(device).float() for t in dS], L, args.lambda_mse)
        loss = tot / len(train_data)
        loss.backward()
        opt.step()
        if (ep + 1) % max(1, args.epochs // 20) == 0:
            print(f"[train] epoch {ep+1}/{args.epochs} loss={float(loss):.4f}", flush=True)

    # ---- evaluate ----
    writer.eval()
    correct = {"text_concat": 0, "inject_gold": 0, "dynamic_writer": 0}
    mse = 0.0
    for ii, (it, (Hh, dS)) in enumerate(zip(test_tasks, test_data)):
        gold = it["gold"].lower()
        facts = it["agents"]
        q_ids = enc(it["q"])
        facts_text = " ".join(facts)

        tc = tokenizer.decode(raw_text(model, tokenizer, facts_text, it["q"], samp),
                              skip_special_tokens=True).strip().lower()
        correct["text_concat"] += int(gold in tc)

        S_context = capture_final_state(model, q_ids)
        S_gold = capture_final_state(model, enc(facts_text + " " + it["q"]))
        dS_star = [g - c for g, c in zip(S_gold, S_context)]
        ig = generate_residual(model, tokenizer, q_ids, dS_star, samp)
        correct["inject_gold"] += int(gold in tokenizer.decode(ig, skip_special_tokens=True).strip().lower())

        with torch.no_grad():
            delta = writer(Hh.to(device).float())
        delta = [d.detach().float() for d in delta]
        mse += rel_mse(delta, [t.to(device).float() for t in dS]).item()
        dw = generate_residual(model, tokenizer, q_ids, delta, samp)
        correct["dynamic_writer"] += int(gold in tokenizer.decode(dw, skip_special_tokens=True).strip().lower())

        print(f"[eval {ii+1}/{len(test_tasks)}] " +
              " ".join(f"{c}={correct[c]}/{ii+1}" for c in correct), flush=True)

    n = len(test_tasks)
    summary = {
        "ckpt_dir": args.ckpt_dir, "n_train": len(train_tasks), "n_test": n,
        "r_h": args.r_h, "ws_dim": args.ws_dim, "n_steps": args.n_steps,
        "epochs": args.epochs, "lr": args.lr,
        "accuracy": {c: round(correct[c] / n, 4) for c in correct},
        "test_mse": round(mse / n, 4),
        "gate_mse_passed": bool((mse / n) < 7.0),
        "gate_acc_passed": bool(correct["dynamic_writer"] / n > 0.5),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    torch.save({"writer": writer.state_dict(), "args": vars(args)},
               str(Path(args.output).with_suffix(".pt")))

    print("\n================ E6 DYNAMIC WRITER v2 ================", flush=True)
    for c in correct:
        print(f"  {c:16s} acc = {correct[c]/n*100:5.1f}%", flush=True)
    print(f"  test MSE = {summary['test_mse']:.4f}  (gate: MSE<7 = {summary['gate_mse_passed']}, "
          f"acc>50% = {summary['gate_acc_passed']})", flush=True)
    print(f"written: {out}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default=CKPT)
    p.add_argument("--tasks", nargs="+", default=["2agent", "3agent"])
    p.add_argument("--n_train", type=int, default=120)
    p.add_argument("--n_test", type=int, default=40)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--r_h", type=int, default=1)
    p.add_argument("--ws_dim", type=int, default=512)
    p.add_argument("--n_steps", type=int, default=4)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda_mse", type=float, default=1.0)
    p.add_argument("--max_new_tokens", type=int, default=24)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="results/capacity_audit/e6_dynamic_writer_v2.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
