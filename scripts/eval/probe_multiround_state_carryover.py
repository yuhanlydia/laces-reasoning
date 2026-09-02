#!/usr/bin/env python3
"""DECISIVE ICLR probe: does fixed-size RWKV state carryover beat growing text
under a fixed decoder-visible context budget as rounds N grow?

This is the go/no-go test for the whole multi-round positive route (advisory Exp 3).
Setup: N rounds, each reveals one short fact. A final question needs one EARLY fact.
Text history grows with N; RWKV state stays fixed-size.

Conditions:
  full_text_history   : all N facts visible to decoder (oracle when it fits)
  truncated_text      : only the most recent B tokens of history visible
  state_carryover     : each round's RWKV state carried into the next; final decoder
                        sees only the question (no history text), answers from state
  shuffled_state      : state_carryover but with a WRONG carried state (control)

Winning condition for the route: as N grows, state_carryover degrades more slowly
than truncated_text at the same visible-text budget, and beats shuffled_state.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from scripts.eval.run_cola_dlm_tasks_prefix_suffix_trajectory_cfg import (  # noqa: E402
    generate_raw_answer, sample_next_token,
)
from scripts.eval.relay_utils import load_relay_model  # noqa: E402
from scripts.eval.probe_hotpot_backbone import normalize, extract_answer  # noqa: E402


ENTITIES = [
    ("the red key", "opens the north gate"), ("the blue vial", "holds the antidote"),
    ("the oak chest", "contains the gold"), ("the stone bridge", "leads to the mill"),
    ("the silver ring", "belongs to the queen"), ("the black horse", "won the race"),
    ("the old map", "shows the harbor"), ("the green lamp", "marks the safe house"),
    ("the iron door", "hides the archive"), ("the white tower", "guards the coast"),
    ("the copper coin", "dates to the war"), ("the tall pine", "stands by the lake"),
    ("the glass jar", "stores the seeds"), ("the wooden flute", "plays the anthem"),
    ("the brass compass", "points to camp"), ("the leather bag", "carries the letters"),
    ("the round mirror", "reflects the moon"), ("the sharp blade", "cuts the rope"),
    ("the small boat", "crosses the river"), ("the golden crown", "sits in the vault"),
    ("the paper scroll", "names the heir"), ("the clay pot", "keeps the honey"),
    ("the steel chain", "locks the cellar"), ("the velvet cloak", "hides the scar"),
    ("the marble step", "counts to seven"), ("the amber bead", "warms in sunlight"),
    ("the linen sheet", "covers the loom"), ("the bronze bell", "rings at dawn"),
    ("the cedar box", "smells of pine"), ("the ivory key", "winds the clock"),
    ("the crimson flag", "flies at noon"), ("the frozen pond", "hides the ring"),
]


def em(pred, gold):
    p, g = normalize(extract_answer(pred)), normalize(gold)
    return bool(g) and (g in p or p in g)


class GenArgs:
    max_new_tokens = 16
    temperature = 0.0
    top_k = 50
    top_p = 0.9
    repetition_penalty = 1.1


@torch.no_grad()
def capture_state(model, ids, init_cache=None):
    kw = dict(input_ids=ids, attention_mask=torch.ones_like(ids).bool(),
              use_cache=True, return_dict=True)
    if init_cache is not None:
        kw["past_key_values"] = init_cache
    out = model.rwkv_model(**kw)
    c = out.past_key_values
    states = [c.layers[l].state.get("recurrent_state").float().clone()
              if c.layers[l].state and isinstance(c.layers[l].state.get("recurrent_state"), torch.Tensor)
              else None for l in range(model.num_layers)]
    return states


@torch.no_grad()
def decode_with_state(model, tokenizer, q_ids, states, gen):
    # First pass: process q_ids and get cache
    out = model.rwkv_model(input_ids=q_ids, attention_mask=torch.ones_like(q_ids).bool(),
                           use_cache=True, return_dict=True)
    # Inject states into cache
    past = model.inject_into_cache(out.past_key_values, states)
    # Get logits from the last position
    logits = out.logits[0, -1]
    all_ids = list(q_ids[0].tolist())
    new_ids = []
    eos = getattr(tokenizer, "eos_token_id", None)
    for _ in range(gen.max_new_tokens):
        nid = sample_next_token(logits, all_ids, gen)
        if eos is not None and nid == eos:
            break
        new_ids.append(nid); all_ids.append(nid)
        out = model.rwkv_model(input_ids=torch.tensor([[nid]], device=q_ids.device),
                               past_key_values=past, use_cache=True, return_dict=True)
        past = out.past_key_values; logits = out.logits[0, -1]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


@torch.no_grad()
def run(a):
    random.seed(a.seed); torch.manual_seed(a.seed)
    dev = a.device
    model, _rwkv, tokenizer, _ckpt, _cfg = load_relay_model(a.ckpt_dir, dev)
    model.eval()
    gen = GenArgs()

    def enc(text):
        return tokenizer(text, return_tensors="pt").input_ids.to(dev)

    Ns = [int(x) for x in a.rounds.split(",")]
    conds = ["full_text", "truncated_text", "state_carryover", "shuffled_state"]
    results = {"ckpt_dir": a.ckpt_dir, "budget_tokens": a.budget_tokens,
               "n_examples": a.n, "rounds": Ns, "by_round": {}}

    for N in Ns:
        correct = {c: 0 for c in conds}
        for ex in range(a.n):
            pool = random.sample(ENTITIES, N)
            target = random.randrange(N)
            tgt_ent, tgt_attr = pool[target]
            facts = [f"Fact {i+1}: {e} {attr}." for i, (e, attr) in enumerate(pool)]
            q = f"\nQuestion: What does {tgt_ent} do? Answer: {tgt_ent}"
            gold = tgt_attr

            # full text history
            full_prompt = "\n".join(facts) + q
            pf, _, _ = generate_raw_answer(model, tokenizer, enc(full_prompt), gen)

            # truncated text: only last B tokens of the history + question
            hist_ids = enc("\n".join(facts))
            trunc = hist_ids[:, -a.budget_tokens:]
            trunc_prompt = tokenizer.decode(trunc[0], skip_special_tokens=True) + q
            pt, _, _ = generate_raw_answer(model, tokenizer, enc(trunc_prompt), gen)

            st = None
            for f in facts:
                fids = enc(f + "\n")
                out = model.rwkv_model(input_ids=fids, attention_mask=torch.ones_like(fids).bool(),
                                       use_cache=True, return_dict=True)
                cache = out.past_key_values
                if st is not None:
                    cache = model.inject_into_cache(cache, st)
                    out = model.rwkv_model(input_ids=fids, past_key_values=cache,
                                           use_cache=True, return_dict=True)
                    cache = out.past_key_values
                st = [cache.layers[l].state.get("recurrent_state").float().clone()
                      if cache.layers[l].state and isinstance(cache.layers[l].state.get("recurrent_state"), torch.Tensor)
                      else None for l in range(model.num_layers)]
            q_only = enc(f"Question: What does {tgt_ent} do? Answer: {tgt_ent}")
            ps = decode_with_state(model, tokenizer, q_only, st, gen)

            # shuffled control: use a random other example's state
            pool2 = random.sample(ENTITIES, N)
            st2 = None
            for e2, at2 in pool2:
                fids = enc(f"Fact: {e2} {at2}.\n")
                out = model.rwkv_model(input_ids=fids, attention_mask=torch.ones_like(fids).bool(),
                                       use_cache=True, return_dict=True)
                st2 = [out.past_key_values.layers[l].state.get("recurrent_state").float().clone()
                       if out.past_key_values.layers[l].state and isinstance(out.past_key_values.layers[l].state.get("recurrent_state"), torch.Tensor)
                       else None for l in range(model.num_layers)]
            psh = decode_with_state(model, tokenizer, q_only, st2, gen)

            correct["full_text"] += em(pf, gold)
            correct["truncated_text"] += em(pt, gold)
            correct["state_carryover"] += em(ps, gold)
            correct["shuffled_state"] += em(psh, gold)

        results["by_round"][str(N)] = {c: round(correct[c] / a.n, 3) for c in conds}
        print(f"[N={N}] " + " | ".join(f"{c}={correct[c]/a.n*100:.0f}%" for c in conds), flush=True)

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(results, indent=2))
    print("\n=== Multi-round state carryover (budget={} tok) ===".format(a.budget_tokens))
    print(f"  {'N':>4} " + " ".join(f"{c[:8]:>10}" for c in conds))
    for N in Ns:
        r = results["by_round"][str(N)]
        print(f"  {N:>4} " + " ".join(f"{r[c]*100:>9.0f}%" for c in conds))
    print(f"\nwritten: {a.output}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--rounds", default="2,4,8,16")
    p.add_argument("--budget_tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="outputs_eval/probe_multiround_state_carryover.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
