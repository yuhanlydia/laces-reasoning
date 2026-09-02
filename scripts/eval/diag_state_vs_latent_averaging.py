#!/usr/bin/env python3
"""Latent-communication diagnostic: State Averaging vs Latent Averaging.

Two agents share the SAME frozen RWKV renderer + single-z S1/S2. Agent-1 answers q1;
we want its "memory" to condition Agent-2's answer to q2, alongside q2's own plan.
Zero training; single-z prefix/suffix checkpoint.

Method 1 (State Averaging): mix in RWKV recurrent-STATE space.
  state_mem   = final recurrent state after running q1 through the renderer
  Z_plan      ~ S2(cond=encode(q2))              # current-task latent plan
  state_plan  = S1(Z_plan)                        # per-layer states
  state_init  = (1-a)*state_mem + a*state_plan    # per layer
  decode q2 from state_init

Method 2 (Latent Averaging): mix in R^32 LATENT space, then map once through S1.
  z_mem       = encode(q1)                        # q1 stored as a latent packet
  Z_plan      ~ S2(cond=encode(q2))
  Z_hist      = (1-a)*z_mem + a*Z_plan            # R^32 interpolation
  state_init  = S1(Z_hist)
  decode q2 from state_init

Sweep a in {0.0,0.25,0.5,0.75,1.0}. We report, per method x alpha:
  - degeneration: repeat-4, distinct-2 of the q2 generation
  - state geometry: relative L2 of state_init vs the pure-plan state (a=1 anchor)
  - latent geometry (M2 only): ||Z_hist||, cos(Z_hist, Z_plan)
and dump the decoded text so quality can be judged directly.

a=0.0 -> pure history/memory (ignores q2 plan)
a=1.0 -> pure current plan  (ignores history) == the standard single-z baseline
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

from scripts.eval.sample_prefix_suffix_cfg import (  # noqa: E402
    apply_repetition_penalty, apply_top_p, encode_prefix, sample_ddim_cfg,
)
from scripts.eval.relay_utils import load_relay_model  # noqa: E402


# ---- q1/q2 prompt pairs: q1 = prior "agent-1" context, q2 = current task ----
PAIRS = [
    ("The team studied the migration of arctic terns across the Atlantic.",
     "Question: Which ocean did the birds cross? Answer:"),
    ("Marie brewed a pot of coffee and left it on the kitchen counter.",
     "Question: Where is the coffee? Answer:"),
    ("The spacecraft entered orbit around Mars after a seven-month journey.",
     "Question: Which planet did the spacecraft reach? Answer:"),
    ("A carpenter built a wooden chair using oak planks and iron nails.",
     "Question: What material were the planks? Answer:"),
    ("The novelist set her latest story in a small fishing village in Norway.",
     "Question: In which country is the village? Answer:"),
    ("Dr. Chen discovered a new enzyme that breaks down plastic waste.",
     "Question: What does the enzyme break down? Answer:"),
    ("The orchestra rehearsed a symphony composed by Beethoven for the gala.",
     "Question: Who composed the symphony? Answer:"),
    ("The hikers reached the summit of the mountain just before sunrise.",
     "Question: When did they reach the summit? Answer:"),
]

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def _repeat_n(ids, n=4):
    if len(ids) < n:
        return 0.0
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return 1.0 - len(set(grams)) / max(1, len(grams))


def _distinct_n(ids, n=2):
    if len(ids) < n:
        return 0.0
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return len(set(grams)) / max(1, len(grams))


def _rel_l2(states, ref):
    num = sum((s.float() - r.float()).pow(2).sum() for s, r in zip(states, ref))
    den = sum(r.float().pow(2).sum() for r in ref).clamp(min=1e-8)
    return float((num / den).sqrt())


@torch.no_grad()
def capture_final_state(model, ids, am):
    """Run q1 through renderer, return per-layer final recurrent_state (list of tensors)."""
    out = model.rwkv_model(input_ids=ids, attention_mask=am.bool(), use_cache=True, return_dict=True)
    cache = out.past_key_values
    states = []
    for l in range(model.num_layers):
        st = cache.layers[l].state.get("recurrent_state") if cache.layers[l].state is not None else None
        states.append(st.float().clone() if isinstance(st, torch.Tensor) else None)
    return states


@torch.no_grad()
def decode_from_states(model, q2_ids, q2_am, state_init, args):
    """Inject state_init into a fresh cache primed on q2, decode continuation."""
    out = model.rwkv_model(input_ids=q2_ids, attention_mask=q2_am.bool(), use_cache=True, return_dict=True)
    past = model.inject_into_cache(out.past_key_values, state_init)
    out = model.rwkv_model(input_ids=q2_ids, past_key_values=past, use_cache=True, return_dict=True)
    past = out.past_key_values
    generated = []
    logits = out.logits[0, -1]
    ctx = list(q2_ids[0].tolist())
    for _ in range(args.max_new_tokens):
        logits = apply_repetition_penalty(logits.float(), ctx + generated, args.repetition_penalty)
        logits = logits / max(args.temperature, 1e-6)
        probs = torch.softmax(logits, dim=-1)
        if args.top_k > 0:
            tv, ti = torch.topk(probs, args.top_k)
            probs = torch.zeros_like(probs).scatter(-1, ti, tv)
            probs = probs / probs.sum().clamp(min=1e-12)
        probs = apply_top_p(probs, args.top_p)
        nid = torch.multinomial(probs, 1).item()
        generated.append(nid)
        out = model.rwkv_model(input_ids=torch.tensor([[nid]], device=q2_ids.device),
                               past_key_values=past, use_cache=True, return_dict=True)
        past = out.past_key_values
        logits = out.logits[0, -1]
    return generated


@torch.no_grad()
def run(args):
    torch.manual_seed(args.seed)
    device = args.device
    model, _rwkv, tokenizer, _ckpt, _cfg = load_relay_model(args.ckpt_dir, device)
    model.eval()
    model._prefix_suffix_s2 = True
    dtype = next(model.alpha_heads.parameters()).dtype

    def enc(text):
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        am = torch.ones_like(ids)
        return ids, am

    results = {"ckpt_dir": args.ckpt_dir, "alphas": ALPHAS, "steps": args.steps,
               "cfg_scale": args.cfg_scale, "pairs": []}

    for pi, (q1, q2) in enumerate(PAIRS):
        q1_ids, q1_am = enc(q1)
        q2_ids, q2_am = enc(q2)

        # shared ingredients
        state_mem = capture_final_state(model, q1_ids, q1_am)          # M1 memory
        z_mem = encode_prefix(model, q1_ids, q1_am).to(dtype)          # M2 memory latent
        cond_q2 = encode_prefix(model, q2_ids, q2_am).to(dtype)
        Z_plan = sample_ddim_cfg(model, cond_q2, args.steps, args.cfg_scale, device, dtype)  # S2 plan
        state_plan = model.predict_states(Z_plan)                      # pure-plan state (a=1 anchor)

        pair_rec = {"q1": q1, "q2": q2, "m1": [], "m2": []}
        for a in ALPHAS:
            # Method 1: mix in STATE space
            si = []
            for sm, sp in zip(state_mem, state_plan):
                if sm is None:
                    si.append(sp.float())
                else:
                    si.append((1.0 - a) * sm.float() + a * sp.float())
            gen = decode_from_states(model, q2_ids, q2_am, si, args)
            txt = tokenizer.decode(gen)
            pair_rec["m1"].append({
                "alpha": a, "text": txt,
                "repeat4": _repeat_n(gen, 4), "distinct2": _distinct_n(gen, 2),
                "state_rel_l2_vs_plan": _rel_l2(si, state_plan),
            })

            # Method 2: mix in LATENT space, then one S1 pass
            Z_hist = (1.0 - a) * z_mem + a * Z_plan
            si2 = model.predict_states(Z_hist)
            gen2 = decode_from_states(model, q2_ids, q2_am, si2, args)
            txt2 = tokenizer.decode(gen2)
            pair_rec["m2"].append({
                "alpha": a, "text": txt2,
                "repeat4": _repeat_n(gen2, 4), "distinct2": _distinct_n(gen2, 2),
                "state_rel_l2_vs_plan": _rel_l2(si2, state_plan),
                "z_hist_norm": float(Z_hist.float().norm()),
                "cos_zhist_zplan": float(torch.nn.functional.cosine_similarity(
                    Z_hist.float().flatten(), Z_plan.float().flatten(), dim=0)),
            })
        results["pairs"].append(pair_rec)
        print(f"[pair {pi+1}/{len(PAIRS)}] done", flush=True)

    # aggregate degeneration per method x alpha
    agg = {"m1": {}, "m2": {}}
    for m in ("m1", "m2"):
        for ai, a in enumerate(ALPHAS):
            r4 = sum(p[m][ai]["repeat4"] for p in results["pairs"]) / len(results["pairs"])
            d2 = sum(p[m][ai]["distinct2"] for p in results["pairs"]) / len(results["pairs"])
            rl = sum(p[m][ai]["state_rel_l2_vs_plan"] for p in results["pairs"]) / len(results["pairs"])
            agg[m][str(a)] = {"repeat4": round(r4, 4), "distinct2": round(d2, 4),
                              "state_rel_l2_vs_plan": round(rl, 4)}
    results["aggregate"] = agg

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print("\n=== aggregate (mean over pairs) ===")
    for m, label in (("m1", "State Averaging"), ("m2", "Latent Averaging")):
        print(f"\n{label}:")
        print(f"  {'alpha':>6} {'repeat4':>9} {'distinct2':>10} {'rel_l2_vs_plan':>15}")
        for a in ALPHAS:
            r = agg[m][str(a)]
            print(f"  {a:>6} {r['repeat4']:>9} {r['distinct2']:>10} {r['state_rel_l2_vs_plan']:>15}")
    print(f"\nwritten: {args.output}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="State vs Latent averaging for latent communication.")
    p.add_argument("--ckpt_dir", default="outputs_relay/test-v6-2.9B-s2-prefix-suffix-cfg/step_00150000")
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--cfg_scale", type=float, default=3.0)
    p.add_argument("--max_new_tokens", type=int, default=48)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="outputs_eval/diag_state_vs_latent_avg.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
