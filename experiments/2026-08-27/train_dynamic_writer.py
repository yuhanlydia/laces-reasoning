#!/usr/bin/env python3
"""E5: Train a BDH-style dynamic low-rank state writer (the "long goal").

Motivation (from experiments/2026-08-27/NOTES.md):
  - E1: the K=16 FIXED basis span captures ~0% of the reasoning correction dS* (e16_delta~1).
  - E2: a low-rank residual writer needs rank ~32+ to flip the answer (r32=60%, r64=100%).
  - E3: the fixed basis is behaviorally inert (language-steering, not capability).

This trains a DYNAMIC writer that replaces the fixed basis:
    W_0 = E(x)                          # encode input to a high-dim workspace
    W_{r+1} = W_r + F_rho(W_r, x)        # BDH-style recurrent update (n_steps)
    dS_l = (U @ diag(c_l)) @ V^T         # low-rank state correction, U,V from W_R (rank r)
  where U [H*D, r] and V [D, r] are produced by heads from W_R (directions depend on input,
  unlike fixed basis), and c_l [r] is a per-layer scale.

Supervision is STATE-MATCHING: minimize per-layer relative MSE between dS and the gold
correction dS* = S_gold - S_context (the FLA RWKV kernel does NOT backprop the recurrent
state, so answer-loss is impossible; state-matching is the only gradient path).

Evaluate: inject S_context + dS (residual), generate-then-match vs text_concat / inject_gold.
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

from scripts.eval.exp_multiagent_star_fusion import capture_final_state  # noqa: E402
from scripts.eval.relay_utils import load_relay_model  # noqa: E402
from scripts.eval.run_cola_dlm_tasks_prefix_suffix_trajectory_cfg import (  # noqa: E402
    recompute_logits_from_injected_cache, sample_next_token,
)
from scripts.eval.diag_hard_problems_v2 import gen_2agent, gen_3agent  # noqa: E402

CKPT = str(REPO / "outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000")


class BDHWriter(nn.Module):
    """Recurrent workspace -> per-layer low-rank state correction.

    Each layer l has r learned rank-1 directions (U_l [r, H*D], V_l [r, D]); the workspace
    produces input-dependent coefficients c [B, L, r], and
        dS_l = sum_m c[l,m] * (u_m v_m^T)   = einsum('bm,mh,md->bhd', c_l, U_l, V_l)
    This is a "wide low-rank basis" (r directions/layer) with DYNAMIC coefficients, in
    contrast to the K=16 fixed full-rank basis of the champion.
    """

    def __init__(self, in_dim, L, H, D, r, ws_dim=512, n_steps=4):
        super().__init__()
        self.L, self.H, self.D, self.r = L, H, D, r
        self.U = nn.Parameter(torch.randn(L, r, H * D) * 0.02)   # [L, r, H*D]
        self.V = nn.Parameter(torch.randn(L, r, D) * 0.02)       # [L, r, D]
        self.enc = nn.Linear(in_dim, ws_dim)                       # W_0 = E(x)
        self.f = nn.Sequential(                                     # F_rho: W += F(W, x)
            nn.Linear(ws_dim + in_dim, ws_dim), nn.GELU(),
            nn.Linear(ws_dim, ws_dim), nn.GELU(),
        )
        self.coef_head = nn.Linear(ws_dim, L * r)                 # coefficients [B, L, r]
        self.n_steps = n_steps

    def forward(self, x):
        # x: [B, in_dim]
        B = x.shape[0]
        W = self.enc(x)
        for _ in range(self.n_steps):
            W = W + self.f(torch.cat([W, x], dim=-1))
        c = self.coef_head(W).reshape(B, self.L, self.r)          # [B, L, r]
        states = []
        for l in range(self.L):
            dl = torch.einsum('bm,mh,md->bhd', c[:, l], self.U[l], self.V[l])  # [B, H*D, D]
            states.append(dl.reshape(B, self.H, self.D, self.D))
        return states


@torch.no_grad()
def pooled_hidden(model, ids, mask):
    out = model.rwkv_model(input_ids=ids, attention_mask=mask.bool(),
                           output_hidden_states=True, use_cache=True, return_dict=True)
    return model._pool_hidden(out.hidden_states[-1], mask).float()  # [B, in_dim]


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


def state_rel_loss(delta, target):
    tot = 0.0
    for d, t in zip(delta, target):
        d = d.float(); t = t.float()
        tot = tot + (d - t).pow(2).sum() / t.pow(2).sum().clamp(min=1e-8)
    return tot


def make_dataset(model, tokenizer, tasks, device):
    """Returns list of (input [in_dim] cpu, target list of [1,H,D,D] cpu)."""
    data = []
    for it in tasks:
        facts = it["agents"]
        q_ids = tokenizer(it["q"], return_tensors="pt").input_ids.to(device)
        f_ids = tokenizer(" ".join(facts), return_tensors="pt").input_ids.to(device)
        fmask = torch.ones_like(f_ids)
        inp = pooled_hidden(model, f_ids, fmask)[0].cpu()          # [in_dim]
        S_context = capture_final_state(model, q_ids)
        S_gold = capture_final_state(model, tokenizer(" ".join(facts) + " " + it["q"],
                                                      return_tensors="pt").input_ids.to(device))
        dS = [g - c for g, c in zip(S_gold, S_context)]
        data.append((inp, [t.cpu() for t in dS]))
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
    in_dim = int(model.hidden_size)
    print(f"[init] L={L} H={H} D={D} in_dim={in_dim} rank={args.rank}", flush=True)

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

    # datasets
    train_tasks = []
    test_tasks = []
    for tname in args.tasks:
        train_tasks.extend({"2agent": gen_2agent, "3agent": gen_3agent}[tname](args.n_train, args.seed))
        test_tasks.extend({"2agent": gen_2agent, "3agent": gen_3agent}[tname](args.n_test, args.seed + 999))

    print(f"[data] building train set ({len(train_tasks)}) ...", flush=True)
    train_data = make_dataset(model, tokenizer, train_tasks, device)
    print(f"[data] building test set ({len(test_tasks)}) ...", flush=True)
    test_data = make_dataset(model, tokenizer, test_tasks, device)

    writer = BDHWriter(in_dim, L, H, D, args.rank, ws_dim=args.ws_dim, n_steps=args.n_steps).to(device)
    opt = torch.optim.Adam(writer.parameters(), lr=args.lr)
    print(f"[train] writer params = {sum(p.numel() for p in writer.parameters())/1e6:.2f}M, epochs={args.epochs}", flush=True)

    for ep in range(args.epochs):
        writer.train()
        opt.zero_grad()
        tot = 0.0
        for inp, dS in train_data:
            delta = writer(inp.unsqueeze(0).to(device))
            tot = tot + state_rel_loss(delta, [t.to(device) for t in dS])
        loss = tot / len(train_data)
        loss.backward()
        opt.step()
        if (ep + 1) % max(1, args.epochs // 10) == 0:
            print(f"[train] epoch {ep+1}/{args.epochs} loss={float(loss):.4f}", flush=True)

    # ---- evaluate ----
    writer.eval()
    conds = ["text_concat", "inject_gold", "dynamic_writer"]
    correct = {c: 0 for c in conds}
    mse = 0.0
    for ii, (it, (inp, dS)) in enumerate(zip(test_tasks, test_data)):
        gold = it["gold"].lower()
        facts = it["agents"]
        q_ids = enc(it["q"])
        facts_text = " ".join(facts)

        # text_concat
        from scripts.eval.exp_multiagent_star_fusion import raw_text
        tc = tokenizer.decode(raw_text(model, tokenizer, facts_text, it["q"], samp),
                              skip_special_tokens=True).strip().lower()
        correct["text_concat"] += int(gold in tc)

        # inject_gold (residual dS* = S_gold - S_context)
        S_context = capture_final_state(model, q_ids)
        S_gold = capture_final_state(model, enc(facts_text + " " + it["q"]))
        dS_star = [g - c for g, c in zip(S_gold, S_context)]
        ig = generate_residual(model, tokenizer, q_ids, dS_star, samp)
        correct["inject_gold"] += int(gold in tokenizer.decode(ig, skip_special_tokens=True).strip().lower())

        # dynamic writer
        with torch.no_grad():
            delta = writer(inp.unsqueeze(0).to(device))
        delta = [d.detach().float() for d in delta]
        mse += state_rel_loss(delta, [t.to(device) for t in dS]).item()
        dw = generate_residual(model, tokenizer, q_ids, delta, samp)
        correct["dynamic_writer"] += int(gold in tokenizer.decode(dw, skip_special_tokens=True).strip().lower())

        print(f"[eval {ii+1}/{len(test_tasks)}] " +
              " ".join(f"{c}={correct[c]}/{ii+1}" for c in conds), flush=True)

    n = len(test_tasks)
    summary = {
        "ckpt_dir": args.ckpt_dir,
        "n_train": len(train_tasks), "n_test": n,
        "rank": args.rank, "ws_dim": args.ws_dim, "n_steps": args.n_steps,
        "epochs": args.epochs, "lr": args.lr,
        "accuracy": {c: round(correct[c] / n, 4) for c in conds},
        "test_mse": round(mse / n, 4),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    torch.save({"writer": writer.state_dict(), "args": vars(args)},
               str(Path(args.output).with_suffix(".pt")))

    print("\n================ E5 DYNAMIC WRITER ================", flush=True)
    for c in conds:
        print(f"  {c:16s} acc = {summary['accuracy'][c]*100:5.1f}%", flush=True)
    print(f"  test state-matching MSE = {summary['test_mse']:.4f}", flush=True)
    print(f"written: {out}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default=CKPT)
    p.add_argument("--tasks", nargs="+", default=["2agent", "3agent"])
    p.add_argument("--n_train", type=int, default=40)
    p.add_argument("--n_test", type=int, default=20)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--ws_dim", type=int, default=512)
    p.add_argument("--n_steps", type=int, default=4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_new_tokens", type=int, default=24)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="results/capacity_audit/e5_dynamic_writer.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
