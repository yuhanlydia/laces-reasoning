#!/usr/bin/env python3
"""E7: Zero-order behavioral loss (user's section 10) on top of the dynamic writer.

E6 (state-matching) overfit (8.17) but did NOT generalize (held-out 12.27, 0% acc), because
matching the full high-rank correction dS* is misaligned with the actual goal (change the
answer). Section 10 fixes this: optimize the writer so its correction CHANGES the gold-answer
log-probability, using a zero-order (NES) behavioral gradient (the FLA RWKV kernel does not
backprop the recurrent state).

  r(DeltaS) = log p(gold | S_context + DeltaS)     (forward-only reward)
  grad_DS r ~ (1/2eps K) * sum_k (r(+eps*d_k) - r(-eps*d_k)) * d_k      (random directions)
  behavioral_loss = - sum_l <grad_DS_l, DeltaS_l>   (backprop through the writer, which IS
                                                       a normal differentiable nn.Module)

Combined loss = lambda_state * (MSE + cosine) + lambda_behavioral * behavioral_loss.
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

from scripts.eval.exp_multiagent_star_fusion import capture_final_state, raw_text  # noqa: E402
from scripts.eval.relay_utils import load_relay_model  # noqa: E402
from scripts.eval.run_cola_dlm_tasks_prefix_suffix_trajectory_cfg import (  # noqa: E402
    recompute_logits_from_injected_cache, sample_next_token,
)
from scripts.eval.diag_hard_problems_v2 import gen_2agent, gen_3agent  # noqa: E402
from scripts.eval.train_dynamic_writer_v2 import (  # noqa: E402
    DynamicReasonWriter, token_hidden, rel_mse, direction_cos, combined_loss, generate_residual,
)

CKPT = str(REPO / "outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000")


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


def nse_grad(dS, r_fn, K, eps, device):
    """Zero-order gradient of r w.r.t. dS (list of [1,H,D,D]) via K random directions."""
    grad = [torch.zeros_like(d) for d in dS]
    for _ in range(K):
        dk = [torch.randn_like(d) for d in dS]
        norm = sum(d.norm() ** 2 for d in dk).sqrt().clamp(min=1e-8)
        dk = [d / norm for d in dk]
        rp = r_fn([d + eps * k for d, k in zip(dS, dk)])
        rm = r_fn([d - eps * k for d, k in zip(dS, dk)])
        g = (rp - rm) / (2.0 * eps)
        grad = [gd + g * k for gd, k in zip(grad, dk)]
    return [gd / K for gd in grad]


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

    gen = {"2agent": gen_2agent, "3agent": gen_3agent}
    train_tasks, test_tasks = [], []
    for tname in args.tasks:
        train_tasks.extend(gen[tname](args.n_train, args.seed))
        test_tasks.extend(gen[tname](args.n_test, args.seed + 999))

    def build(tasks):
        data = []
        for it in tasks:
            facts = it["agents"]
            q = it["q"]
            q_ids = enc(q)
            fq_ids = enc(" ".join(facts) + " " + q)
            Hh = token_hidden(model, fq_ids).cpu()
            Sc = capture_final_state(model, q_ids)
            Sg = capture_final_state(model, fq_ids)
            dS = [g - c for g, c in zip(Sg, Sc)]
            gold_ids = tokenizer(" " + it["gold"].lower(), return_tensors="pt").input_ids[0].tolist()
            data.append({"H": Hh, "dS": [t.cpu() for t in dS], "q_ids": q_ids,
                         "gold_ids": gold_ids, "S_context": [t.float().clone() for t in Sc]})
        return data

    print(f"[data] train {len(train_tasks)} ...", flush=True)
    train_data = build(train_tasks)
    print(f"[data] test {len(test_tasks)} ...", flush=True)
    test_data = build(test_tasks)

    writer = DynamicReasonWriter(hidden_dim, L, H, D, args.r_h, ws_dim=args.ws_dim,
                                 n_steps=args.n_steps).to(device)
    opt = torch.optim.Adam(writer.parameters(), lr=args.lr)
    print(f"[train] writer params = {sum(p.numel() for p in writer.parameters())/1e6:.2f}M, "
          f"epochs={args.epochs} K={args.K} lambda_b={args.lambda_b}", flush=True)

    for ep in range(args.epochs):
        writer.train()
        opt.zero_grad()
        state_loss = 0.0
        beh_loss = 0.0
        for rec in train_data:
            Hh = rec["H"].to(device).float()
            tgt = [t.to(device).float() for t in rec["dS"]]
            dS = writer(Hh)
            state_loss = state_loss + combined_loss(dS, tgt, L, args.lambda_mse)

            if args.lambda_b > 0 and (ep % args.beh_every == 0):
                q_ids = rec["q_ids"]
                gold_ids = rec["gold_ids"]
                Sc = [t.to(device) for t in rec["S_context"]]
                r_fn = lambda delta, q_ids=q_ids, gold_ids=gold_ids, Sc=Sc: gold_logprob(
                    model, tokenizer, q_ids, gold_ids,
                    [c + d for c, d in zip(Sc, delta)])
                with torch.no_grad():
                    grad = nse_grad([d.detach().float() for d in dS], r_fn, args.K, args.eps, device)
                beh_loss = beh_loss - sum((g * d.float()).sum() for g, d in zip(grad, dS))

        loss = state_loss / len(train_data)
        if args.lambda_b > 0 and (ep % args.beh_every == 0):
            loss = loss + args.lambda_b * (beh_loss / len(train_data))
        loss.backward()
        opt.step()
        if (ep + 1) % max(1, args.epochs // 10) == 0:
            print(f"[train] ep {ep+1}/{args.epochs} loss={float(loss):.4f} "
                  f"(state={state_loss.item()/len(train_data):.3f})", flush=True)

    # ---- evaluate ----
    writer.eval()
    correct = {"text_concat": 0, "inject_gold": 0, "dynamic_writer": 0}
    mse = 0.0
    for ii, (it, rec) in enumerate(zip(test_tasks, test_data)):
        gold = it["gold"].lower()
        facts = it["agents"]
        q_ids = rec["q_ids"]

        tc = tokenizer.decode(raw_text(model, tokenizer, " ".join(facts), it["q"], samp),
                              skip_special_tokens=True).strip().lower()
        correct["text_concat"] += int(gold in tc)

        dS_star = [t.to(device).float() for t in rec["dS"]]
        ig = generate_residual(model, tokenizer, q_ids, dS_star, samp)
        correct["inject_gold"] += int(gold in tokenizer.decode(ig, skip_special_tokens=True).strip().lower())

        with torch.no_grad():
            dS = writer(rec["H"].to(device).float())
        dS = [d.detach().float() for d in dS]
        mse += rel_mse(dS, dS_star).item()
        dw = generate_residual(model, tokenizer, q_ids, dS, samp)
        correct["dynamic_writer"] += int(gold in tokenizer.decode(dw, skip_special_tokens=True).strip().lower())

        print(f"[eval {ii+1}/{len(test_tasks)}] " +
              " ".join(f"{c}={correct[c]}/{ii+1}" for c in correct), flush=True)

    n = len(test_tasks)
    summary = {
        "ckpt_dir": args.ckpt_dir, "n_train": len(train_tasks), "n_test": n,
        "r_h": args.r_h, "ws_dim": args.ws_dim, "n_steps": args.n_steps,
        "epochs": args.epochs, "lr": args.lr, "K": args.K, "eps": args.eps,
        "lambda_b": args.lambda_b, "lambda_mse": args.lambda_mse,
        "accuracy": {c: round(correct[c] / n, 4) for c in correct},
        "test_mse": round(mse / n, 4),
        "gate_mse_passed": bool((mse / n) < 7.0),
        "gate_acc_passed": bool(correct["dynamic_writer"] / n > 0.5),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    torch.save({"writer": writer.state_dict()}, str(Path(args.output).with_suffix(".pt")))

    print("\n================ E7 BEHAVIORAL WRITER ================", flush=True)
    for c in correct:
        print(f"  {c:16s} acc = {correct[c]/n*100:5.1f}%", flush=True)
    print(f"  test MSE = {summary['test_mse']:.4f}  (gate MSE<7={summary['gate_mse_passed']}, "
          f"acc>50%={summary['gate_acc_passed']})", flush=True)
    print(f"written: {out}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default=CKPT)
    p.add_argument("--tasks", nargs="+", default=["2agent", "3agent"])
    p.add_argument("--n_train", type=int, default=40)
    p.add_argument("--n_test", type=int, default=20)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--r_h", type=int, default=4)
    p.add_argument("--ws_dim", type=int, default=256)
    p.add_argument("--n_steps", type=int, default=4)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--eps", type=float, default=0.5)
    p.add_argument("--lambda_b", type=float, default=1.0)
    p.add_argument("--lambda_mse", type=float, default=1.0)
    p.add_argument("--beh_every", type=int, default=2)
    p.add_argument("--max_new_tokens", type=int, default=24)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="results/capacity_audit/e7_behavioral_writer.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
