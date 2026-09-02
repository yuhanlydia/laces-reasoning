#!/usr/bin/env python3
"""Topology C (hierarchical / division-of-labor): M agents split into G sub-groups.
Within each group, states are combined by sequential carryover (the operator that works);
across groups, the group-summary states are combined again by carryover at a parent level.
This tests whether the winning sequential operator composes across a 2-level tree, versus
the parallel-average failure. Baselines: flat seq carryover (all M in one chain), text
concat (ceiling, growing cost), and a shuffled control. Zero training, champion checkpoint.
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
    ITEMS, build_chain, capture_final_state, fuse_states, decode_chunkwise, raw_text,
    seq_carryover_state,
)


@torch.no_grad()
def seq_from_states_then_facts(model, tokenizer, init_states, facts, device):
    """Inject init_states, then run the group's facts, capture resulting state.
    Used to thread a parent chain across sub-group summary states."""
    ids0 = tokenizer(facts[0], return_tensors="pt").input_ids.to(device)
    out = model.rwkv_model(input_ids=ids0, attention_mask=torch.ones_like(ids0).bool(),
                           use_cache=True, return_dict=True)
    past = out.past_key_values
    if init_states is not None:
        past = model.inject_into_cache(past, init_states)
        ctx = ids0
        out = model.rwkv_model(input_ids=ctx, attention_mask=torch.ones_like(ctx).bool(),
                               past_key_values=past, use_cache=True, return_dict=True)
        past = out.past_key_values
    for f in facts[1:]:
        ids = tokenizer(f, return_tensors="pt").input_ids.to(device)
        out = model.rwkv_model(input_ids=ids, past_key_values=past,
                               attention_mask=torch.ones_like(ids).bool(),
                               use_cache=True, return_dict=True)
        past = out.past_key_values
    return [past.layers[l].state.get("recurrent_state").float().clone()
            if past.layers[l].state is not None and past.layers[l].state.get("recurrent_state") is not None
            else None for l in range(model.num_layers)]


@torch.no_grad()
def run(args):
    torch.manual_seed(args.seed)
    device = args.device
    model, _r, tokenizer, _c, _cfg = load_relay_model(args.ckpt_dir, device)
    model.eval(); model._prefix_suffix_trajectory_s2 = True
    dtype = next(model.alpha_heads.parameters()).dtype
    H = int(model.trajectory_horizon)
    M, G = args.num_agents, args.num_groups
    a = args.plan_weight

    def enc(t):
        return tokenizer(t, return_tensors="pt").input_ids.to(device)

    conds = ["text_concat", "flat_seq", "hier_seq", "hier_avg", "hier_shuffled"]
    correct = {c: 0 for c in conds}
    ctxchars = {c: 0 for c in conds}
    prepared = [build_chain(it, M, qi) for qi, it in enumerate(ITEMS)]

    for ii, (facts, q, gold) in enumerate(prepared):
        q_ids = enc(q)
        z_q = encode_prefix(model, q_ids, torch.ones_like(q_ids))[0].to(dtype)
        Zq = sample_trajectory_cfg(model, z_q, args.steps, args.cfg_scale, device, dtype)
        plan_states = [model.predict_states(Zq[:, h]) for h in range(H)]

        # split M facts into G contiguous groups
        gsize = (M + G - 1) // G
        groups = [facts[g * gsize:(g + 1) * gsize] for g in range(G) if facts[g * gsize:(g + 1) * gsize]]

        flat_state = seq_carryover_state(model, tokenizer, facts, device)
        group_states = [seq_carryover_state(model, tokenizer, gr, device) for gr in groups]
        wg = 1.0 / len(group_states)
        hier_avg_state = fuse_states([(wg, gs) for gs in group_states])
        ordered = [f for gr in groups for f in gr]
        hier_state = seq_carryover_state(model, tokenizer, ordered, device)

        query_pool = min(8, M)
        qslot = (M - 1) if M <= 2 else 1 + (ii % max(1, query_pool - 1))
        shuffled_facts = list(facts)
        shuffled_facts[qslot] = prepared[(ii + 1) % len(prepared)][0][qslot]
        shuf_groups = [shuffled_facts[g * gsize:(g + 1) * gsize] for g in range(G) if shuffled_facts[g * gsize:(g + 1) * gsize]]
        shuf_group_states = [seq_carryover_state(model, tokenizer, gr, device) for gr in shuf_groups]
        hier_shuf = fuse_states([(wg, gs) for gs in shuf_group_states])

        preds = {}
        preds["text_concat"] = raw_text(model, tokenizer, " ".join(facts), q, args)
        preds["flat_seq"] = decode_chunkwise(model, tokenizer, q_ids,
            lambda h: fuse_states([(a, plan_states[h]), ((1 - a), flat_state)]), H, args)
        preds["hier_seq"] = decode_chunkwise(model, tokenizer, q_ids,
            lambda h: fuse_states([(a, plan_states[h]), ((1 - a), hier_state)]), H, args)
        preds["hier_avg"] = decode_chunkwise(model, tokenizer, q_ids,
            lambda h: fuse_states([(a, plan_states[h]), ((1 - a), hier_avg_state)]), H, args)
        preds["hier_shuffled"] = decode_chunkwise(model, tokenizer, q_ids,
            lambda h: fuse_states([(a, plan_states[h]), ((1 - a), hier_shuf)]), H, args)

        ctxchars["text_concat"] += len(" ".join(facts))
        for c in ["flat_seq", "hier_seq", "hier_avg", "hier_shuffled"]:
            ctxchars[c] += len(q)

        for c in conds:
            txt = (tokenizer.decode(preds[c], skip_special_tokens=True).strip().lower()
                   if isinstance(preds[c], list) else str(preds[c]).lower())
            correct[c] += int(gold in txt)
        print(f"[M={M} G={G} {ii+1}/{len(prepared)}] " +
              " ".join(f"{c[:4]}={'1' if gold in (tokenizer.decode(preds[c],skip_special_tokens=True).lower() if isinstance(preds[c],list) else str(preds[c]).lower()) else '.'}" for c in conds),
              flush=True)

    n = len(prepared)
    res = {"num_agents": M, "num_groups": G, "n": n, "plan_weight": a,
           "accuracy": {c: round(correct[c] / n, 3) for c in conds},
           "mean_ctx_chars": {c: round(ctxchars[c] / n) for c in conds}}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(res, indent=2))
    print(f"\n=== HIERARCHICAL (M={M}, G={G}) ===")
    for c in conds:
        print(f"  {res['accuracy'][c]*100:5.0f}%  ctx={res['mean_ctx_chars'][c]:5d}  {c}")
    print(f"written: {args.output}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000")
    p.add_argument("--num_agents", type=int, default=8)
    p.add_argument("--num_groups", type=int, default=2)
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
    p.add_argument("--output", default="outputs_eval/latentcot_gen/star/hier_M8G2.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
