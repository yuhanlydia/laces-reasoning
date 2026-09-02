#!/usr/bin/env python3
"""Multi-agent COOPERATION tasks beyond single-attribute retrieval, to test whether
sequential carryover preserves ALL agents' information (not just routes to one).
Three task modes on M agents sharing the frozen renderer, zero training, champion ckpt:
  aggregate : query needs TWO agents' facts jointly (both must survive fusion)
  count     : K of M agents share a property; answer is the count (needs all M scanned)
  conflict  : agents give conflicting values; answer is the majority value
Each mode compares text_concat (ceiling, growing cost) vs seq_carryover (ours, fixed)
vs parallel averaging (expected to fail) vs shuffled control.
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

from scripts.eval.sample_prefix_suffix_trajectory_cfg import encode_prefix, sample_trajectory_cfg  # noqa: E402
from scripts.eval.relay_utils import load_relay_model  # noqa: E402
from scripts.eval.exp_multiagent_star_fusion import (  # noqa: E402
    ITEMS, capture_final_state, fuse_states, decode_chunkwise, raw_text, seq_carryover_state,
)


def build_aggregate(item, m, qi):
    subj = item["subject"]
    attrs = item["attrs"][:m]
    facts = [f"Agent {i + 1} knows: the {subj}'s {n} is {v}." for i, (n, v) in enumerate(attrs)]
    i1, i2 = 1, min(2, m - 1)
    (n1, v1), (n2, v2) = attrs[i1], attrs[i2]
    q = f"Question: list the {subj}'s {n1} and {n2}. Answer: {n1} {v1}, {n2}"
    return facts, q, v2.lower()


def build_count(item, m, qi):
    subj = item["subject"]
    k = 2 + (qi % max(1, m - 2))
    facts = []
    for i in range(m):
        status = "active" if i < k else "inactive"
        facts.append(f"Agent {i + 1} reports: unit {i + 1} is {status}.")
    numwords = ["zero", "one", "two", "three", "four", "five", "six", "seven",
                "eight", "nine", "ten", "eleven", "twelve"]
    q = "Question: how many units are active? Answer: the number of active units is"
    return facts, q, numwords[k] if k < len(numwords) else str(k)


def build_conflict(item, m, qi):
    subj = item["subject"]
    name, true_val = item["attrs"][0]
    wrong_vals = [item["attrs"][j % len(item["attrs"])][1] for j in range(1, 4)]
    maj = (m // 2) + 1
    facts = []
    for i in range(m):
        v = true_val if i < maj else wrong_vals[i % len(wrong_vals)]
        facts.append(f"Agent {i + 1} claims: the {subj}'s {name} is {v}.")
    q = f"Question: by majority, what is the {subj}'s {name}? Answer: the {subj}'s {name} is"
    return facts, q, true_val.lower()


BUILDERS = {"aggregate": build_aggregate, "count": build_count, "conflict": build_conflict}


@torch.no_grad()
def run(args):
    torch.manual_seed(args.seed)
    device = args.device
    model, _r, tokenizer, _c, _cfg = load_relay_model(args.ckpt_dir, device)
    model.eval(); model._prefix_suffix_trajectory_s2 = True
    dtype = next(model.alpha_heads.parameters()).dtype
    H = int(model.trajectory_horizon)
    M = args.num_agents
    a = args.plan_weight
    builder = BUILDERS[args.task]

    def enc(t):
        return tokenizer(t, return_tensors="pt").input_ids.to(device)

    conds = ["text_concat", "seq_carryover", "parallel_avg", "shuffled"]
    correct = {c: 0 for c in conds}
    ctxchars = {c: 0 for c in conds}
    prepared = [builder(it, M, qi) for qi, it in enumerate(ITEMS)]

    for ii, (facts, q, gold) in enumerate(prepared):
        q_ids = enc(q)
        z_q = encode_prefix(model, q_ids, torch.ones_like(q_ids))[0].to(dtype)
        Zq = sample_trajectory_cfg(model, z_q, args.steps, args.cfg_scale, device, dtype)
        plan = [model.predict_states(Zq[:, h]) for h in range(H)]
        w = 1.0 / M

        mems = [capture_final_state(model, enc(f)) for f in facts]
        seq_state = seq_carryover_state(model, tokenizer, facts, device)
        gold_slots = [i for i, f in enumerate(facts) if gold in f.lower()] or [len(facts) - 1]
        wrong_facts = list(facts)
        donor = prepared[(ii + 1) % len(prepared)][0]
        for gs in gold_slots:
            wrong_facts[gs] = donor[gs % len(donor)]
        seq_shuf = seq_carryover_state(model, tokenizer, wrong_facts, device)

        preds = {}
        preds["text_concat"] = raw_text(model, tokenizer, " ".join(facts), q, args)
        preds["seq_carryover"] = decode_chunkwise(model, tokenizer, q_ids,
            lambda h: fuse_states([(a, plan[h]), ((1 - a), seq_state)]), H, args)
        preds["parallel_avg"] = decode_chunkwise(model, tokenizer, q_ids,
            lambda h: fuse_states([(a, plan[h])] + [((1 - a) * w, m) for m in mems]), H, args)
        preds["shuffled"] = decode_chunkwise(model, tokenizer, q_ids,
            lambda h: fuse_states([(a, plan[h]), ((1 - a), seq_shuf)]), H, args)

        ctxchars["text_concat"] += len(" ".join(facts))
        for c in ["seq_carryover", "parallel_avg", "shuffled"]:
            ctxchars[c] += len(q)

        for c in conds:
            txt = (tokenizer.decode(preds[c], skip_special_tokens=True).strip().lower()
                   if isinstance(preds[c], list) else str(preds[c]).lower())
            correct[c] += int(gold in txt)
        print(f"[{args.task} M={M} {ii+1}/{len(prepared)}] gold={gold!r} " +
              " ".join(f"{c[:4]}={'1' if gold in (tokenizer.decode(preds[c],skip_special_tokens=True).lower() if isinstance(preds[c],list) else str(preds[c]).lower()) else '.'}" for c in conds),
              flush=True)

    n = len(prepared)
    res = {"task": args.task, "num_agents": M, "n": n, "plan_weight": a,
           "accuracy": {c: round(correct[c] / n, 3) for c in conds},
           "mean_ctx_chars": {c: round(ctxchars[c] / n) for c in conds}}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(res, indent=2))
    print(f"\n=== COOP {args.task} (M={M}) ===")
    for c in conds:
        print(f"  {res['accuracy'][c]*100:5.0f}%  ctx={res['mean_ctx_chars'][c]:5d}  {c}")
    print(f"written: {args.output}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000")
    p.add_argument("--task", choices=list(BUILDERS), required=True)
    p.add_argument("--num_agents", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--cfg_scale", type=float, default=3.0)
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.2)
    p.add_argument("--plan_weight", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
