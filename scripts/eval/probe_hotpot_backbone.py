#!/usr/bin/env python3
"""Capability probe: can the frozen RWKV champion answer HotpotQA with full gold context?

This gates the whole HotpotQA two-agent fusion plan (advisory Q2 filter). We measure,
for N examples, three conditions per the advisory:
  full_gold : both gold supporting paragraphs in context  -> backbone ceiling
  shard_A   : only supporting paragraph A                 -> should be lower
  shard_B   : only supporting paragraph B                 -> should be lower
An example is "fusion-suitable" if full_gold is correct AND both single shards are wrong.
If full_gold accuracy is at floor, HotpotQA is a backbone-floor test and must be dropped.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from scripts.eval.relay_utils import load_relay_model  # noqa: E402
from scripts.eval.run_cola_dlm_tasks_prefix_suffix_trajectory_cfg import (  # noqa: E402
    generate_raw_answer,
)


def normalize(s):
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def extract_answer(text):
    for stop in ("\nQuestion", "\nQ:", "Q:", "\n\n", "\nExplanation", "Explanation:", "\nAnswer"):
        idx = text.find(stop)
        if idx > 0:
            text = text[:idx]
    text = text.split("\n")[0]
    text = re.sub(r"^(answer|a)\s*[:.\-]\s*", "", text.strip(), flags=re.I)
    return text.strip()


def em_match(pred, gold):
    p, g = normalize(extract_answer(pred)), normalize(gold)
    if not g:
        return False
    return g in p or p in g


def build_ctx(titles, sents, keep_titles):
    parts = []
    for t, ss in zip(titles, sents):
        if t in keep_titles:
            parts.append(t + ": " + " ".join(ss))
    return "\n".join(parts)


def make_prompt(context, question):
    return (
        "Read the passages and answer the question with a short answer.\n\n"
        f"{context}\n\nQuestion: {question}\nAnswer:"
    )


class Args:
    max_new_tokens = 24
    temperature = 0.0
    top_k = 50
    top_p = 0.9
    repetition_penalty = 1.1


@torch.no_grad()
def run(a):
    data = json.load(open(a.data))[: a.n]
    model, _rwkv, tokenizer, _ckpt, _cfg = load_relay_model(a.ckpt_dir, a.device)
    model.eval()
    gen_args = Args()

    def ask(context, q):
        prompt = make_prompt(context, q)
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(a.device)
        txt, _, _ = generate_raw_answer(model, tokenizer, ids, gen_args)
        return txt

    n = len(data)
    full_ok = shardA_ok = shardB_ok = fusion_suitable = fusion_suitable_relaxed = 0
    suitable_ids = []
    rows = []
    suitable_records = []
    for i, ex in enumerate(data):
        titles, sents = ex["ctx_titles"], ex["ctx_sents"]
        sup = ex["sup_titles"]
        gold = ex["answer"]
        if len(set(sup)) < 2:
            continue
        if a.bridge_only and ex.get("type") != "bridge":
            continue
        a_title, b_title = sup[0], sup[1]
        ctx_full = build_ctx(titles, sents, set(sup))
        ctx_a = build_ctx(titles, sents, {a_title})
        ctx_b = build_ctx(titles, sents, {b_title})
        pf = ask(ctx_full, ex["question"])
        pa = ask(ctx_a, ex["question"])
        pb = ask(ctx_b, ex["question"])
        f_ok, a_ok, b_ok = em_match(pf, gold), em_match(pa, gold), em_match(pb, gold)
        full_ok += f_ok
        shardA_ok += a_ok
        shardB_ok += b_ok
        suit_strict = f_ok and not a_ok and not b_ok
        suit_relaxed = f_ok and not (a_ok and b_ok)
        fusion_suitable += suit_strict
        fusion_suitable_relaxed += suit_relaxed
        rows.append({"id": ex["id"], "type": ex["type"], "gold": gold,
                     "full": pf[:40], "full_ok": f_ok,
                     "A_ok": a_ok, "B_ok": b_ok,
                     "fusion_suitable": suit_strict, "fusion_suitable_relaxed": suit_relaxed})
        if suit_strict:
            suitable_ids.append(ex["id"])
            suitable_records.append(ex)
        print(f"[{i+1}/{n}] full={'1' if f_ok else '.'} A={'1' if a_ok else '.'} "
              f"B={'1' if b_ok else '.'} suit={'1' if suit_strict else '.'} | gold={gold[:24]!r} pred={extract_answer(pf)[:24]!r}",
              flush=True)

    m = len(rows)
    result = {
        "ckpt_dir": a.ckpt_dir, "n": m,
        "full_gold_acc": round(full_ok / m, 3) if m else 0,
        "shardA_acc": round(shardA_ok / m, 3) if m else 0,
        "shardB_acc": round(shardB_ok / m, 3) if m else 0,
        "fusion_suitable_rate": round(fusion_suitable / m, 3) if m else 0,
        "fusion_suitable_count": fusion_suitable,
        "fusion_suitable_relaxed_rate": round(fusion_suitable_relaxed / m, 3) if m else 0,
        "fusion_suitable_relaxed_count": fusion_suitable_relaxed,
        "suitable_ids": suitable_ids,
        "rows": rows,
    }
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(result, indent=2))
    if a.suitable_out:
        Path(a.suitable_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.suitable_out).write_text(json.dumps(suitable_records, indent=2))
        print(f"suitable set ({len(suitable_records)}) written: {a.suitable_out}")
    print("\n=== HotpotQA backbone capability probe (frozen RWKV champion) ===")
    print(f"  full-gold-context acc : {result['full_gold_acc']*100:.0f}%  <- backbone ceiling (GATE)")
    print(f"  shard-A-only acc      : {result['shardA_acc']*100:.0f}%")
    print(f"  shard-B-only acc      : {result['shardB_acc']*100:.0f}%")
    print(f"  fusion-suitable (strict, full+both-shard-wrong)  : {result['fusion_suitable_rate']*100:.0f}%  ({fusion_suitable}/{m})")
    print(f"  fusion-suitable (relaxed, full+>=1-shard-wrong)  : {result['fusion_suitable_relaxed_rate']*100:.0f}%  ({fusion_suitable_relaxed}/{m})")
    print(f"\nwritten: {a.output}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000")
    p.add_argument("--data", default="/tmp/hotpot60.json")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--device", default="cuda")
    p.add_argument("--bridge_only", action="store_true")
    p.add_argument("--suitable_out", default="")
    p.add_argument("--output", default="outputs_eval/probe_hotpot_backbone.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
