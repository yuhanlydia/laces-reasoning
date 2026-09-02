#!/usr/bin/env python3
"""E9: DECISIVE experiment — is the FIXED basis (not the 32-D latent) the bottleneck?

User's claim: "32-D latent was never the bottleneck; the fixed linear writable interface was."

Fix the INPUT to the trained 32-D z = S0(facts), and ONLY change the writer:
  FixedBasis-16 : dS_l = sum_k alpha_k(z) B_k          (champion's structure, 16 dirs)
  FixedBasis-32 : dS_l = sum_k alpha_k(z) B_k          (32 dirs)
  DynamicUV-r32 : dS_l = U_l(z) V_l(z)^T               (dynamic directions, flattened rank-32)

All trained on state-matching (dS* = S_gold - S_context). If DynamicUV >> FixedBasis,
the fixed linear interface (not the 32-D z) is the bottleneck.
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
from scripts.eval.sample_prefix_suffix_trajectory_cfg import encode_prefix  # noqa: E402
from scripts.eval.diag_hard_problems_v2 import gen_2agent, gen_3agent  # noqa: E402

CKPT = str(REPO / "outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000")


class FixedBasisWriter(nn.Module):
    def __init__(self, z_dim, L, H, D, K):
        super().__init__()
        self.L, self.H, self.D, self.K = L, H, D, K
        self.basis = nn.Parameter(torch.randn(L, K, H, D, D) * 0.02)
        self.alpha_heads = nn.ModuleList([nn.Linear(z_dim, K) for _ in range(L)])

    def forward(self, z):
        states = []
        for l in range(self.L):
            alpha = self.alpha_heads[l](z)                      # [1, K]
            dl = torch.einsum('bk,khde->bhde', alpha, self.basis[l])  # [1,H,D,D]
            states.append(dl)
        return states


class DynamicUVWriter(nn.Module):
    def __init__(self, z_dim, L, H, D, r, ws=128):
        super().__init__()
        self.L, self.H, self.D, self.r = L, H, D, r
        self.trunk = nn.Sequential(nn.Linear(z_dim, ws), nn.GELU(), nn.Linear(ws, ws), nn.GELU())
        self.u_head = nn.Linear(ws, L * H * D * r)
        self.v_head = nn.Linear(ws, L * D * r)

    def forward(self, z):
        h = self.trunk(z)                                        # [1, ws]
        U = self.u_head(h).reshape(self.L, self.H * self.D, self.r)  # [L, H*D, r]
        V = self.v_head(h).reshape(self.L, self.D, self.r)           # [L, D, r]
        states = []
        for l in range(self.L):
            dl = U[l] @ V[l].transpose(-1, -2)                    # [H*D, D]
            states.append(dl.reshape(self.H, self.D, self.D).unsqueeze(0))
        return states


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
            tot += (d * t).sum() / (dn * tn)
            n += 1
    return tot / max(n, 1)


def state_loss(delta, target, L):
    return (1.0 - direction_cos(delta, target)) + rel_mse(delta, target) / L


def train_writer(writer, train_data, device, epochs, lr):
    opt = torch.optim.Adam(writer.parameters(), lr=lr)
    for ep in range(epochs):
        writer.train()
        opt.zero_grad()
        tot = 0.0
        for z, dS in train_data:
            delta = writer(z.to(device))
            tot = tot + state_loss(delta, [t.to(device) for t in dS], writer.L)
        loss = tot / len(train_data)
        loss.backward()
        opt.step()
    return loss.item()


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
    dtype = next(model.alpha_heads.parameters()).dtype

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

    def build(tasks):
        data = []
        for it in tasks:
            facts = it["agents"]
            q_ids = enc(it["q"])
            fq_ids = enc(" ".join(facts) + " " + it["q"])
            z = encode_prefix(model, enc(" ".join(facts)), torch.ones_like(enc(" ".join(facts))))[0].to(dtype)
            Sc = capture_final_state(model, q_ids)
            Sg = capture_final_state(model, fq_ids)
            dS = [g - c for g, c in zip(Sg, Sc)]
            data.append((z.float().cpu(), [t.cpu() for t in dS]))
        return data

    print(f"[data] train {len(train_tasks)} test {len(test_tasks)} ...", flush=True)
    train_data = build(train_tasks)
    test_data = build(test_tasks)
    z_dim = train_data[0][0].shape[1]
    print(f"[init] L={L} H={H} D={D} z_dim={z_dim}", flush=True)

    writers = {}
    if "fixed16" in args.writers:
        writers["fixed16"] = FixedBasisWriter(z_dim, L, H, D, 16).to(device)
    if "fixed32" in args.writers:
        writers["fixed32"] = FixedBasisWriter(z_dim, L, H, D, 32).to(device)
    if "dynuv32" in args.writers:
        writers["dynuv32"] = DynamicUVWriter(z_dim, L, H, D, 32).to(device)

    results = {}
    for name, writer in writers.items():
        nparam = sum(p.numel() for p in writer.parameters()) / 1e6
        print(f"[train] {name} params={nparam:.1f}M ...", flush=True)
        final_loss = train_writer(writer, train_data, device, args.epochs, args.lr)

        writer.eval()
        mse = 0.0
        cos = 0.0
        acc = 0
        for it, (z, dS) in zip(test_tasks, test_data):
            gold = it["gold"].lower()
            q_ids = enc(it["q"])
            with torch.no_grad():
                delta = writer(z.to(device))
            delta = [d.detach().float() for d in delta]
            tgt = [t.to(device) for t in dS]
            mse += rel_mse(delta, tgt).item()
            cos += direction_cos(delta, tgt).item()
            dw = generate_residual(model, tokenizer, q_ids, delta, samp)
            acc += int(gold in tokenizer.decode(dw, skip_special_tokens=True).strip().lower())
        n = len(test_data)
        results[name] = {"params_M": round(nparam, 1), "final_loss": round(final_loss, 4),
                         "mse": round(mse / n, 4), "cos": round(cos / n, 4),
                         "acc": round(acc / n, 4)}

    summary = {"ckpt_dir": args.ckpt_dir, "n_train": len(train_tasks), "n_test": n,
               "tasks": args.tasks, "epochs": args.epochs, "lr": args.lr,
               "writers": results}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print("\n================ E9 DECISIVE (32-D z, writer only) ================", flush=True)
    print(f"  {'writer':10s} {'params':>8s} {'MSE':>8s} {'cos':>7s} {'acc':>6s}", flush=True)
    for name, r in results.items():
        print(f"  {name:10s} {r['params_M']:>7.1f}M {r['mse']:>8.3f} {r['cos']:>7.3f} {r['acc']*100:>5.1f}%", flush=True)
    print(f"written: {out}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default=CKPT)
    p.add_argument("--tasks", nargs="+", default=["2agent", "3agent"])
    p.add_argument("--n_train", type=int, default=60)
    p.add_argument("--n_test", type=int, default=30)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--writers", nargs="+", default=["fixed16", "fixed32", "dynuv32"])
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_new_tokens", type=int, default=24)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="results/capacity_audit/e9_decisive.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
