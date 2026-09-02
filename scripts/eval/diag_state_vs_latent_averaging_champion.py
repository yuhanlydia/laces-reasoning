#!/usr/bin/env python3
"""Latent-communication diagnostic on the CHAMPION (scratch joint co-adapt trajectory).

Champion = traj32x16 single-z-bridge birwkv joint-scratch co-adapt + condboundary.
It is a TRAJECTORY model: a plan is Z ~ S2(q2) with shape [1, H=16, 32], decoded
chunk-by-chunk (each chunk latent z_h -> predict_states -> inject -> generate 32 tok).

Two agents share this frozen renderer. Agent-1 answered q1; we want its memory to
condition Agent-2's answer to q2 alongside q2's own trajectory plan. Zero training.

Method 1 (State Averaging): mix in RWKV recurrent-STATE space, per chunk.
  state_mem       = final recurrent state after running q1 (per layer)
  Z_plan          ~ S2(cond=encode(q2))                # [1,H,32]
  state_plan_h    = predict_states(Z_plan[:,h])        # per chunk h
  state_init_h    = (1-a)*state_mem + a*state_plan_h   # inject before chunk h
  decode q2 chunk-wise from these.

Method 2 (Latent Averaging): mix in R^32 LATENT space per chunk, then S1.
  z_mem           = encode(q1)  (single latent packet, broadcast to every chunk)
  Z_plan          ~ S2(cond=encode(q2))                # [1,H,32]
  Z_hist[:,h]     = (1-a)*z_mem + a*Z_plan[:,h]
  state_init_h    = predict_states(Z_hist[:,h])
  decode q2 chunk-wise from these.

Sweep a in {0,0.25,0.5,0.75,1.0}. a=1.0 == the standard champion baseline (pure plan).
Reports repeat4/distinct2, per-chunk state rel-L2 vs pure-plan, and latent geometry (M2),
plus decoded text for direct quality judgment.
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

from scripts.eval.sample_prefix_suffix_trajectory_cfg import (  # noqa: E402
    encode_prefix, sample_trajectory_cfg,
)
from scripts.eval.run_cola_dlm_tasks_prefix_suffix_trajectory_cfg import (  # noqa: E402
    recompute_logits_from_injected_cache, sample_next_token,
)
from scripts.eval.relay_utils import load_relay_model  # noqa: E402


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
    ("The chef prepared a spicy curry using fresh ginger and coconut milk.",
     "Question: What milk did the chef use? Answer:"),
    ("The archaeologists unearthed an ancient bronze statue near the river.",
     "Question: What metal was the statue? Answer:"),
    ("The pilot landed the plane safely in Tokyo during a heavy storm.",
     "Question: In which city did the plane land? Answer:"),
    ("The gardener planted rows of yellow tulips along the stone path.",
     "Question: What color were the tulips? Answer:"),
    ("The scientist measured the temperature of the volcano before the eruption.",
     "Question: What did the scientist measure? Answer:"),
    ("The children built a large sandcastle on the beach at low tide.",
     "Question: Where did the children build the castle? Answer:"),
    ("The professor lectured about quantum physics to the graduate students.",
     "Question: What subject did the professor teach? Answer:"),
    ("The blacksmith forged a sharp sword from a bar of heated steel.",
     "Question: What did the blacksmith forge? Answer:"),
    ("The astronaut repaired the solar panel during a spacewalk outside the station.",
     "Question: What did the astronaut repair? Answer:"),
    ("The farmer harvested a field of golden wheat before the autumn rains.",
     "Question: What crop did the farmer harvest? Answer:"),
    ("The violinist performed a concerto in the grand hall of Vienna.",
     "Question: In which city did she perform? Answer:"),
    ("The diver photographed a coral reef teeming with colorful fish.",
     "Question: What did the diver photograph? Answer:"),
    ("The engineer designed a suspension bridge spanning the wide canyon.",
     "Question: What did the engineer design? Answer:"),
    ("The baker decorated a chocolate cake for the birthday celebration.",
     "Question: What flavor was the cake? Answer:"),
    ("The explorer discovered a hidden cave behind the frozen waterfall.",
     "Question: What did the explorer discover? Answer:"),
    ("The librarian catalogued a rare manuscript written in old Latin.",
     "Question: What language was the manuscript? Answer:"),
    ("The surgeon performed a delicate operation on the patient's heart.",
     "Question: What organ did the surgeon operate on? Answer:"),
    ("The photographer captured a lightning bolt striking the tall tower.",
     "Question: What struck the tower? Answer:"),
    ("The merchant sold silk and spices along the ancient trade route.",
     "Question: What did the merchant sell? Answer:"),
    ("The climber scaled the icy cliff using ropes and metal picks.",
     "Question: What did the climber scale? Answer:"),
    ("The teacher explained photosynthesis to the curious biology class.",
     "Question: What process did the teacher explain? Answer:"),
    ("The captain steered the ship through the narrow strait at midnight.",
     "Question: When did the captain steer the ship? Answer:"),
    ("The painter mixed blue and yellow to create a vivid green mural.",
     "Question: What color was the mural? Answer:"),
]
GOLD = ["atlantic", "kitchen", "mars", "oak", "norway", "plastic", "beethoven", "sunrise",
        "coconut", "bronze", "tokyo", "yellow", "temperature", "beach", "physics", "sword",
        "solar", "wheat", "vienna", "coral", "bridge", "chocolate", "cave", "latin",
        "heart", "lightning", "silk", "cliff", "photosynthesis", "midnight", "green"]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def _repeat_n(ids, n=4):
    if len(ids) < n:
        return 0.0
    g = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return 1.0 - len(set(g)) / max(1, len(g))


def _distinct_n(ids, n=2):
    if len(ids) < n:
        return 0.0
    g = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return len(set(g)) / max(1, len(g))


def _rel_l2(states, ref):
    num = sum((s.float() - r.float()).pow(2).sum() for s, r in zip(states, ref))
    den = sum(r.float().pow(2).sum() for r in ref).clamp(min=1e-8)
    return float((num / den).sqrt())


@torch.no_grad()
def capture_final_state(model, ids, am):
    out = model.rwkv_model(input_ids=ids, attention_mask=am.bool(), use_cache=True, return_dict=True)
    cache = out.past_key_values
    states = []
    for l in range(model.num_layers):
        st = cache.layers[l].state.get("recurrent_state") if cache.layers[l].state is not None else None
        states.append(st.float().clone() if isinstance(st, torch.Tensor) else None)
    return states


@torch.no_grad()
def decode_chunkwise(model, tokenizer, q2_ids, prefix_cache, prefix_logits,
                     per_chunk_state_fn, H, args):
    """Chunk-wise trajectory decode; per_chunk_state_fn(h) -> list[per-layer state] to inject."""
    chunk_size = int(model.trajectory_chunk_size)
    all_ids = list(q2_ids[0].tolist())
    new_ids: list[int] = []
    past_kv = prefix_cache
    logits = prefix_logits
    eos_id = getattr(tokenizer, "eos_token_id", None)
    stop = False
    for h in range(H):
        if stop or len(new_ids) >= args.max_new_tokens:
            break
        states_h = per_chunk_state_fn(h)
        past_kv = model.inject_into_cache(past_kv, states_h)
        context_ids = torch.tensor([all_ids], device=q2_ids.device, dtype=torch.long)
        context_mask = torch.ones_like(context_ids)
        past_kv, logits_batch = recompute_logits_from_injected_cache(model, context_ids, context_mask, past_kv)
        logits = logits_batch[0]
        for _ in range(chunk_size):
            if len(new_ids) >= args.max_new_tokens:
                stop = True
                break
            nid = sample_next_token(logits, all_ids, args)
            if eos_id is not None and nid == eos_id:
                stop = True
                break
            new_ids.append(nid)
            all_ids.append(nid)
            out = model.rwkv_model(input_ids=torch.tensor([[nid]], device=q2_ids.device),
                                   past_key_values=past_kv, use_cache=True, return_dict=True)
            past_kv = out.past_key_values
            logits = out.logits[0, -1]
    return new_ids


@torch.no_grad()
def run(args):
    torch.manual_seed(args.seed)
    device = args.device
    model, _rwkv, tokenizer, _ckpt, _cfg = load_relay_model(args.ckpt_dir, device)
    model.eval()
    model._prefix_suffix_trajectory_s2 = True
    dtype = next(model.alpha_heads.parameters()).dtype
    H = int(model.trajectory_horizon)

    def enc_ids(text):
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        return ids, torch.ones_like(ids)

    results = {"ckpt_dir": args.ckpt_dir, "alphas": ALPHAS, "H": H,
               "steps": args.steps, "cfg_scale": args.cfg_scale, "pairs": []}

    for pi, (q1, q2) in enumerate(PAIRS):
        q1_ids, q1_am = enc_ids(q1)
        q2_ids, q2_am = enc_ids(q2)

        state_mem = capture_final_state(model, q1_ids, q1_am)          # M1 memory
        z_mem = encode_prefix(model, q1_ids, q1_am)[0].to(dtype)        # M2 memory latent (z_prefix of q1)
        z_prefix_q2, prefix_cache, prefix_logits = encode_prefix(model, q2_ids, q2_am)
        Z_plan = sample_trajectory_cfg(model, z_prefix_q2.to(dtype), args.steps,
                                       args.cfg_scale, device, dtype)   # [1,H,32]

        # pure-plan per-chunk states (a=1 anchor for rel-L2)
        plan_states = [model.predict_states(Z_plan[:, h]) for h in range(H)]

        pair_rec = {"q1": q1, "q2": q2, "m1": [], "m2": []}
        for a in ALPHAS:
            # need a fresh prefix cache each decode (inject mutates it) -> re-encode
            def fresh_prefix():
                zp, pc, pl = encode_prefix(model, q2_ids, q2_am)
                return pc, pl

            # Method 1: state-space mix per chunk
            def m1_fn(h, a=a):
                sp = plan_states[h]
                out = []
                for sm, s in zip(state_mem, sp):
                    out.append(s.float() if sm is None else (1.0 - a) * sm.float() + a * s.float())
                return out
            pc, pl = fresh_prefix()
            gen = decode_chunkwise(model, tokenizer, q2_ids, pc, pl, m1_fn, H, args)
            # per-chunk state rel-l2 vs plan (measure at chunk 0)
            rl1 = _rel_l2(m1_fn(0), plan_states[0])
            pair_rec["m1"].append({
                "alpha": a, "text": tokenizer.decode(gen, skip_special_tokens=True).strip(),
                "repeat4": _repeat_n(gen, 4), "distinct2": _distinct_n(gen, 2),
                "state_rel_l2_vs_plan": rl1,
            })

            # Method 2: latent-space mix per chunk, then S1
            Z_hist = torch.stack([(1.0 - a) * z_mem + a * Z_plan[:, h] for h in range(H)], dim=1)  # [1,H,32]
            hist_states = [model.predict_states(Z_hist[:, h]) for h in range(H)]

            def m2_fn(h):
                return hist_states[h]
            pc, pl = fresh_prefix()
            gen2 = decode_chunkwise(model, tokenizer, q2_ids, pc, pl, m2_fn, H, args)
            rl2 = _rel_l2(hist_states[0], plan_states[0])
            pair_rec["m2"].append({
                "alpha": a, "text": tokenizer.decode(gen2, skip_special_tokens=True).strip(),
                "repeat4": _repeat_n(gen2, 4), "distinct2": _distinct_n(gen2, 2),
                "state_rel_l2_vs_plan": rl2,
                "z_hist_norm": float(Z_hist[:, 0].float().norm()),
                "cos_zhist_zplan": float(torch.nn.functional.cosine_similarity(
                    Z_hist[:, 0].float().flatten(), Z_plan[:, 0].float().flatten(), dim=0)),
            })
        results["pairs"].append(pair_rec)
        print(f"[pair {pi+1}/{len(PAIRS)}] done", flush=True)

    # aggregate
    agg = {"m1": {}, "m2": {}}
    for m in ("m1", "m2"):
        for ai, a in enumerate(ALPHAS):
            r4 = sum(p[m][ai]["repeat4"] for p in results["pairs"]) / len(results["pairs"])
            d2 = sum(p[m][ai]["distinct2"] for p in results["pairs"]) / len(results["pairs"])
            rl = sum(p[m][ai]["state_rel_l2_vs_plan"] for p in results["pairs"]) / len(results["pairs"])
            acc = sum(1 for pi, p in enumerate(results["pairs"]) if GOLD[pi] in p[m][ai]["text"].lower()) / len(results["pairs"])
            agg[m][str(a)] = {"repeat4": round(r4, 4), "distinct2": round(d2, 4),
                              "state_rel_l2_vs_plan": round(rl, 4), "answer_acc": round(acc, 3)}
    results["aggregate"] = agg

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print("\n=== CHAMPION aggregate (mean over pairs) ===")
    for m, label in (("m1", "State Averaging"), ("m2", "Latent Averaging")):
        print(f"\n{label}:")
        print(f"  {'alpha':>5} {'answer_acc':>11} {'repeat4':>8} {'distinct2':>10} {'rel_l2_vs_plan':>15}")
        for a in ALPHAS:
            r = agg[m][str(a)]
            print(f"  {a:>5} {r['answer_acc']:>11} {r['repeat4']:>8} {r['distinct2']:>10} {r['state_rel_l2_vs_plan']:>15}")
    print(f"\nwritten: {args.output}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000")
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--cfg_scale", type=float, default=3.0)
    p.add_argument("--max_new_tokens", type=int, default=48)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="outputs_eval/diag_state_vs_latent_avg_champion.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
