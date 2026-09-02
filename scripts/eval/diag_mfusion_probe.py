#!/usr/bin/env python3
"""Isolate the M-agent state-fusion failure. On the SAME Hidden-Profile items, compare
four ways to combine M agents' memory states for answering ONE agent's attribute:
  avg      : equal-weight mean of all M memories (the failing operator)
  oracle1  : inject ONLY the queried agent's memory (upper bound; tests if averaging is the killer)
  seq      : sequential carryover (run agent 1 -> keep state -> run agent 2 ... -> decode)
  concat   : all M facts concatenated as text into one state, then decode (single capture)
plus text_concat (decoder reads full text). Metric: gold token in generation.
Zero training, champion checkpoint. Purpose: root-cause the M-scaling collapse.
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
from scripts.eval.run_cola_dlm_tasks_prefix_suffix_trajectory_cfg import (  # noqa: E402
    recompute_logits_from_injected_cache, sample_next_token,
)
from scripts.eval.relay_utils import load_relay_model  # noqa: E402
from scripts.eval.exp_multiagent_star_fusion import ITEMS, build_chain  # noqa: E402


@torch.no_grad()
def capture_final_state(model, ids):
    out = model.rwkv_model(input_ids=ids, attention_mask=torch.ones_like(ids).bool(),
                           use_cache=True, return_dict=True)
    cache = out.past_key_values
    return [cache.layers[l].state.get("recurrent_state").float().clone()
            if cache.layers[l].state is not None and cache.layers[l].state.get("recurrent_state") is not None
            else None for l in range(model.num_layers)]


def fuse(weighted):
    L = next(len(sl) for _, sl in weighted if sl is not None)
    out = []
    for l in range(L):
        acc = None
        for wt, sl in weighted:
            if sl is None or sl[l] is None:
                continue
            t = wt * sl[l].float()
            acc = t if acc is None else acc + t
        out.append(acc)
    return out


def state_norm(states):
    return float(torch.stack([s.float().norm() for s in states if s is not None]).mean())


@torch.no_grad()
def decode_from_state(model, tokenizer, q_ids, states, H, args):
    out = model.rwkv_model(input_ids=q_ids, attention_mask=torch.ones_like(q_ids).bool(),
                           use_cache=True, return_dict=True)
    past = out.past_key_values
    all_ids = list(q_ids[0].tolist())
    new_ids: list[int] = []
    eos = getattr(tokenizer, "eos_token_id", None)
    past = model.inject_into_cache(past, states)
    ctx = torch.tensor([all_ids], device=q_ids.device)
    past, lb = recompute_logits_from_injected_cache(model, ctx, torch.ones_like(ctx), past)
    logits = lb[0]
    for _ in range(args.max_new_tokens):
        nid = sample_next_token(logits, all_ids, args)
        if eos is not None and nid == eos:
            break
        new_ids.append(nid); all_ids.append(nid)
        o = model.rwkv_model(input_ids=torch.tensor([[nid]], device=q_ids.device),
                             past_key_values=past, use_cache=True, return_dict=True)
        past = o.past_key_values; logits = o.logits[0, -1]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip().lower()


@torch.no_grad()
def seq_carryover_state(model, facts):
    """Thread state through agents sequentially: run fact_1, keep cache, continue fact_2, ..."""
    past = None
    for f in facts:
        ids = model_tok(model, f)
        if past is None:
            out = model.rwkv_model(input_ids=ids, attention_mask=torch.ones_like(ids).bool(),
                                   use_cache=True, return_dict=True)
        else:
            out = model.rwkv_model(input_ids=ids, past_key_values=past,
                                   attention_mask=torch.ones_like(ids).bool(),
                                   use_cache=True, return_dict=True)
        past = out.past_key_values
    return [past.layers[l].state.get("recurrent_state").float().clone()
            if past.layers[l].state is not None and past.layers[l].state.get("recurrent_state") is not None
            else None for l in range(model.num_layers)]


_TOK = {}


def model_tok(model, text):
    tk = _TOK["tk"]
    return tk(text, return_tensors="pt").input_ids.to(_TOK["dev"])


@torch.no_grad()
def run(args):
    torch.manual_seed(args.seed)
    dev = args.device
    model, _r, tok, _c, _cfg = load_relay_model(args.ckpt_dir, dev)
    model.eval(); model._prefix_suffix_trajectory_s2 = True
    _TOK["tk"] = tok; _TOK["dev"] = dev
    dtype = next(model.alpha_heads.parameters()).dtype
    H = int(model.trajectory_horizon)
    M = args.num_agents
    a = args.plan_weight

    def enc(t):
        return tok(t, return_tensors="pt").input_ids.to(dev)

    conds = ["text_concat", "avg", "oracle1", "seq", "concat"]
    correct = {c: 0 for c in conds}
    norms = {"avg": [], "oracle1": [], "seq": [], "single": []}

    for qi, it in enumerate(ITEMS):
        facts, q, gold = build_chain(it, M, qi)
        q_ids = enc(q)
        qslot = (M - 1) if M <= 2 else 1 + (qi % (M - 1))

        mems = [capture_final_state(model, enc(f)) for f in facts]
        z_q = encode_prefix(model, q_ids, torch.ones_like(q_ids))[0].to(dtype)
        Zq = sample_trajectory_cfg(model, z_q, args.steps, args.cfg_scale, dev, dtype)
        plan = [model.predict_states(Zq[:, h]) for h in range(H)]

        w = 1.0 / M
        # text_concat
        ct = " ".join(facts)
        cids = enc(ct + "\n" + q)
        o = model.rwkv_model(input_ids=cids, attention_mask=torch.ones_like(cids).bool(),
                             use_cache=True, return_dict=True)
        past = o.past_key_values; logits = o.logits[0, -1]
        ids_all = list(cids[0].tolist()); gen = []
        for _ in range(args.max_new_tokens):
            nid = sample_next_token(logits, ids_all, args)
            if nid == getattr(tok, "eos_token_id", -1):
                break
            gen.append(nid); ids_all.append(nid)
            oo = model.rwkv_model(input_ids=torch.tensor([[nid]], device=dev),
                                  past_key_values=past, use_cache=True, return_dict=True)
            past = oo.past_key_values; logits = oo.logits[0, -1]
        correct["text_concat"] += int(gold in tok.decode(gen, skip_special_tokens=True).lower())

        # avg: plan + equal-weight mean of M memories
        avg_state = fuse([(a, plan[0])] + [((1 - a) * w, m) for m in mems])
        norms["avg"].append(state_norm(avg_state))
        # oracle1: plan + only queried agent's memory
        or1_state = fuse([(a, plan[0]), ((1 - a), mems[qslot])])
        norms["oracle1"].append(state_norm(or1_state))
        norms["single"].append(state_norm(mems[qslot]))
        # seq: sequential carryover state (no plan) then decode
        seq_state = seq_carryover_state(model, facts)
        norms["seq"].append(state_norm(seq_state))
        # concat: single capture of concatenated facts
        concat_state = capture_final_state(model, enc(ct))

        for c, st in [("avg", avg_state), ("oracle1", or1_state),
                      ("seq", fuse([(a, plan[0]), ((1 - a), seq_state)])),
                      ("concat", fuse([(a, plan[0]), ((1 - a), concat_state)]))]:
            txt = decode_from_state(model, tok, q_ids, st, H, args)
            correct[c] += int(gold in txt)

        print(f"[M={M} {qi+1}/{len(ITEMS)}] " +
              " ".join(f"{c[:4]}={'1' if False else ''}" for c in []) +
              f"avg_norm={norms['avg'][-1]:.1f} or1_norm={norms['oracle1'][-1]:.1f}", flush=True)

    n = len(ITEMS)
    res = {"num_agents": M, "n": n, "plan_weight": a,
           "accuracy": {c: round(correct[c] / n, 3) for c in conds},
           "mean_state_norm": {k: round(sum(v) / len(v), 2) for k, v in norms.items() if v}}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(res, indent=2))
    print(f"\n=== M-FUSION PROBE (M={M}) ===")
    for c in conds:
        print(f"  {res['accuracy'][c]*100:5.0f}%  {c}")
    print(f"  state norms: {res['mean_state_norm']}")
    print(f"written: {args.output}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000")
    p.add_argument("--num_agents", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--cfg_scale", type=float, default=3.0)
    p.add_argument("--max_new_tokens", type=int, default=12)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.2)
    p.add_argument("--plan_weight", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="outputs_eval/latentcot_gen/star/probe_M4.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
