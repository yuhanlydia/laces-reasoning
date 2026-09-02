#!/usr/bin/env python3
"""E10: Recurrent latent reasoning (trained F_rho + dynamic writer, step-wise distillation).

The user's unified "Recurrent Latent-State Planner": keep z in R^32, but iterate it through a
trained recurrent reasoner and write state through a DYNAMIC (input-dependent) writer.

    c      = Compress(H_{1:T}, q)                 (cross-attn with learned queries -> R^128)
    z^(0)  = Linear(c)
    z^(r+1)= z^(r) + F_rho([z^(r), c])             (R recurrent steps, all in R^32)
    dS^(r) = U(z^(r)) V(z^(r))^T                    (dynamic low-rank writer, rank r_s)

Training (step-wise state trajectory distillation): for an R-hop chain, oracle intermediate
states S_r* = f_theta(f_1+...+f_r+q), corrections dS_r* = S_r* - S_{r-1}*, supervise
    dS^(r) ~ dS_r*  for r = 1..R
via relative MSE + direction (cosine) loss. NO backprop through the RWKV (FLA does not
backprop the recurrent state); the writer is a normal nn.Module so it backprops fine.

Inference: forward-only (no weight update). Final injection = S_context + sum_r dS^(r).

Experiment matrix (--R and --mode):
  A1 R=1 final | A2 R=2 final | A3 R=4 final | A4 R=4 stepwise | A5 R=8 stepwise+final
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
from scripts.eval.diag_hard_problems_v2 import gen_2agent, gen_3agent, gen_4agent  # noqa: E402

CKPT = str(REPO / "outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000")


class RecurrentReasoner(nn.Module):
    def __init__(self, hidden_dim, L, H, D, z_dim=32, c_dim=128, r=32, n_queries=4):
        super().__init__()
        self.L, self.H, self.D, self.z_dim, self.r = L, H, D, z_dim, r
        self.h_proj = nn.Linear(hidden_dim, c_dim)
        self.queries = nn.Parameter(torch.randn(n_queries, c_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(c_dim, num_heads=4, batch_first=True)
        self.c_pool = nn.Linear(n_queries * c_dim, c_dim)
        self.z0_head = nn.Linear(c_dim, z_dim)
        self.f = nn.Sequential(nn.Linear(z_dim + c_dim, 128), nn.GELU(), nn.Linear(128, z_dim))
        self.u_head = nn.Linear(z_dim, L * H * D * r)
        self.v_head = nn.Linear(z_dim, L * D * r)

    def forward(self, H_tokens, R):
        Hp = self.h_proj(H_tokens).unsqueeze(0)                    # [1,T,c_dim]
        q = self.queries.unsqueeze(0)                               # [1,n_q,c_dim]
        ctx, _ = self.cross_attn(q, Hp, Hp)                         # [1,n_q,c_dim]
        c = self.c_pool(ctx.reshape(1, -1))                         # [1,c_dim]
        z = self.z0_head(c)                                         # [1,z_dim] = z^(0)
        zs = [z]
        for _ in range(R):
            z = z + self.f(torch.cat([z, c], dim=-1))               # z^(r+1)
            zs.append(z)
        traj = []
        for z in zs:
            U = self.u_head(z).reshape(self.L, self.H * self.D, self.r)
            V = self.v_head(z).reshape(self.L, self.D, self.r)
            dS = [(U[l] @ V[l].transpose(-1, -2)).reshape(self.H, self.D, self.D).unsqueeze(0)
                  for l in range(self.L)]
            traj.append(dS)
        return traj


@torch.no_grad()
def token_hidden(model, ids):
    mask = torch.ones_like(ids).bool()
    out = model.rwkv_model(input_ids=ids, attention_mask=mask, output_hidden_states=True,
                           use_cache=True, return_dict=True)
    return out.hidden_states[-1][0].float()


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
    return sum((d.float() - t.float()).pow(2).sum() / t.float().pow(2).sum().clamp(min=1e-8)
               for d, t in zip(delta, target))


def direction_cos(delta, target):
    tot, n = 0.0, 0
    for d, t in zip(delta, target):
        d = d.float().reshape(-1)
        t = t.float().reshape(-1)
        dn, tn = d.norm(), t.norm()
        if dn > 1e-8 and tn > 1e-8:
            tot += (d * t).sum().item() / (dn.item() * tn.item())
            n += 1
    return tot / max(n, 1)


def sum_states(states_list):
    return [sum(s[l] for s in states_list) for l in range(len(states_list[0]))]


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

    gen = {"2agent": gen_2agent, "3agent": gen_3agent, "4agent": gen_4agent}
    train_tasks, test_tasks = [], []
    for tname in args.tasks:
        train_tasks.extend(gen[tname](args.n_train, args.seed))
        test_tasks.extend(gen[tname](args.n_test, args.seed + 999))

    # build oracle trajectory: for facts f_1..f_R, dS_r* = S_r* - S_{r-1}*
    def build(tasks):
        data = []
        for it in tasks:
            facts = it["agents"]
            R = len(facts)
            q_ids = enc(it["q"])
            Hh = token_hidden(model, enc(" ".join(facts))).cpu().half()
            S_prev = capture_final_state(model, q_ids)
            dS_star = []
            prefix = ""
            for r in range(R):
                prefix = (prefix + " " + facts[r]).strip()
                S_r = capture_final_state(model, enc(prefix + " " + it["q"]))
                dS_star.append([(g - c).cpu().half() for g, c in zip(S_r, S_prev)])
                S_prev = S_r
            data.append({"H": Hh, "dS_star": dS_star, "R": R,
                         "q_ids": q_ids, "gold": it["gold"].lower()})
        return data

    print(f"[data] train {len(train_tasks)} ...", flush=True)
    train_data = build(train_tasks)
    print(f"[data] test {len(test_tasks)} ...", flush=True)
    test_data = build(test_tasks)

    reasoner = RecurrentReasoner(hidden_dim, L, H, D, z_dim=32, c_dim=128,
                                 r=args.r_s, n_queries=args.n_queries).to(device)
    opt = torch.optim.Adam(reasoner.parameters(), lr=args.lr)
    nparam = sum(p.numel() for p in reasoner.parameters()) / 1e6
    print(f"[train] reasoner params = {nparam:.1f}M, epochs={args.epochs}, "
          f"R_mode={args.R_mode} mode={args.mode}", flush=True)

    for ep in range(args.epochs):
        reasoner.train()
        opt.zero_grad()
        tot = 0.0
        cnt = 0
        for rec in train_data:
            R = rec["R"] if args.R_mode == "auto" else args.R
            traj = reasoner(rec["H"].to(device).float(), R)   # [R+1] corrections
            if args.mode == "stepwise":
                # supervise dS^(r) ~ dS_r* for r=1..R
                for r in range(1, R + 1):
                    tgt = [t.to(device).float() for t in rec["dS_star"][r - 1]]
                    tot = tot + rel_mse(traj[r], tgt) / L + args.lambda_cos * (1.0 - direction_cos(traj[r], tgt))
                    cnt += 1
            else:  # final
                dS_final = sum_states(traj[1:R + 1])          # sum of corrections
                full = sum_states([[s.to(device).float() for s in step] for step in rec["dS_star"]])
                tot = tot + rel_mse(dS_final, full) / L + args.lambda_cos * (1.0 - direction_cos(dS_final, full))
                cnt += 1
        loss = tot / max(cnt, 1)
        loss.backward()
        opt.step()
        if (ep + 1) % max(1, args.epochs // 10) == 0:
            print(f"[train] ep {ep+1}/{args.epochs} loss={float(loss):.4f}", flush=True)

    # ---- eval ----
    reasoner.eval()
    correct = {"text_concat": 0, "inject_gold": 0, "reasoner": 0}
    lp_traj = []  # gold-lp at each cumulative step
    n = len(test_data)
    for ii, (it, rec) in enumerate(zip(test_tasks, test_data)):
        gold = rec["gold"]
        facts = it["agents"]
        q_ids = rec["q_ids"]

        tc = tokenizer.decode(raw_text(model, tokenizer, " ".join(facts), it["q"], samp),
                              skip_special_tokens=True).strip().lower()
        correct["text_concat"] += int(gold in tc)

        full = sum_states([[s.to(device).float() for s in step] for step in rec["dS_star"]])
        ig = generate_residual(model, tokenizer, q_ids, full, samp)
        correct["inject_gold"] += int(gold in tokenizer.decode(ig, skip_special_tokens=True).strip().lower())

        R = rec["R"] if args.R_mode == "auto" else args.R
        with torch.no_grad():
            traj = reasoner(rec["H"].to(device).float(), R)
        dS_cum = [torch.zeros_like(t).to(device) for t in full]
        for r in range(1, R + 1):
            dS_cum = [a + b.float() for a, b in zip(dS_cum, traj[r])]
        dw = generate_residual(model, tokenizer, q_ids, dS_cum, samp)
        correct["reasoner"] += int(gold in tokenizer.decode(dw, skip_special_tokens=True).strip().lower())

        print(f"[eval {ii+1}/{n}] " + " ".join(f"{c}={correct[c]}/{ii+1}" for c in correct), flush=True)

    summary = {
        "ckpt_dir": args.ckpt_dir, "n_train": len(train_tasks), "n_test": n,
        "tasks": args.tasks, "R_mode": args.R_mode, "mode": args.mode,
        "r_s": args.r_s, "epochs": args.epochs, "lr": args.lr,
        "accuracy": {c: round(correct[c] / n, 4) for c in correct},
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    torch.save({"reasoner": reasoner.state_dict()}, str(Path(args.output).with_suffix(".pt")))

    print("\n================ E10 RECURRENT REASONING ================", flush=True)
    for c in correct:
        print(f"  {c:14s} acc = {correct[c]/n*100:5.1f}%", flush=True)
    print(f"written: {out}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default=CKPT)
    p.add_argument("--tasks", nargs="+", default=["2agent", "3agent", "4agent"])
    p.add_argument("--n_train", type=int, default=60)
    p.add_argument("--n_test", type=int, default=25)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--R_mode", default="auto", choices=["auto", "fixed"])
    p.add_argument("--R", type=int, default=4)
    p.add_argument("--mode", default="stepwise", choices=["stepwise", "final"])
    p.add_argument("--r_s", type=int, default=32)
    p.add_argument("--n_queries", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda_cos", type=float, default=0.5)
    p.add_argument("--max_new_tokens", type=int, default=24)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="results/capacity_audit/e10_recurrent.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
