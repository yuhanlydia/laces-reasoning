#!/usr/bin/env python3
"""GATE: can our frozen RWKV-7 2.9B (champion) answer ProntoQA with FULL context?

Two conditions:
  - raw     : context + question + options fed directly to the backbone
  - inject  : context read into recurrent state, question decoded with state
              (our full-information ceiling for the distributed version)

If both are ~50%, ProntoQA is a backbone floor and unusable for the
LatentMAS comparison. Target: meaningfully above 50% (labels balanced).
"""
from __future__ import annotations
import argparse, json, re, sys
import torch

sys.path.insert(0, ".")
from scripts.eval.diag_hard_problems_v2 import REPO
from scripts.eval.relay_utils import load_relay_model
from scripts.eval.exp_multiagent_star_fusion import capture_final_state, decode_chunkwise
from scripts.eval.run_cola_dlm_tasks_prefix_suffix_trajectory_cfg import (
    recompute_logits_from_injected_cache, sample_next_token,
)

CKPT = "outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000"


class Args:
    max_new_tokens = 8
    temperature = 0.5
    top_k = 10
    top_p = 0.9
    repetition_penalty = 1.2


def parse_ab(text: str) -> str | None:
    t = text.strip().lower()
    has_true = re.search(r"\btrue\b", t) is not None
    has_false = re.search(r"\bfalse\b", t) is not None
    if has_true and not has_false:
        return "A"
    if has_false and not has_true:
        return "B"
    w = re.findall(r"[a-z]+", t)
    if w:
        if w[0].startswith("true"):
            return "A"
        if w[0].startswith("false"):
            return "B"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--out", type=str, required=True)
    cli = ap.parse_args()

    device = torch.device(cli.device)
    model, _, tokenizer, _, _ = load_relay_model(str(REPO / CKPT), str(device))
    model.eval()
    model._prefix_suffix_trajectory_s2 = True
    H = int(model.trajectory_horizon)

    def enc(t):
        return tokenizer(t, return_tensors="pt").input_ids.to(device)

    args = Args()
    items = [json.loads(l) for l in open("baseline/reasoning_eval/prontoqa.jsonl")][: cli.n]

    res = {"raw": [], "inject": []}
    fails = []
    for ii, it in enumerate(items):
        gold = it["answer"]
        qtext = f"{it['question']}\nAnswer with True or False:"
        ctx_q = it["context"].strip() + "\n" + qtext

        # raw: whole thing through the backbone
        c_ids = enc(ctx_q)
        out = model.rwkv_model(input_ids=c_ids, attention_mask=torch.ones_like(c_ids).bool(),
                               use_cache=True, return_dict=True)
        pk = out.past_key_values
        aids = list(c_ids[0].tolist())
        pk, lb = recompute_logits_from_injected_cache(model, c_ids, torch.ones_like(c_ids), pk)
        logits = lb[0]
        nids = []
        for _ in range(8):
            nid = sample_next_token(logits, aids, args)
            if nid == getattr(tokenizer, "eos_token_id", None):
                break
            nids.append(nid); aids.append(nid)
            o = model.rwkv_model(input_ids=torch.tensor([[nid]], device=device),
                                 past_key_values=pk, use_cache=True, return_dict=True)
            pk = o.past_key_values; logits = o.logits[0, -1]
        pred = parse_ab(tokenizer.decode(nids, skip_special_tokens=True))
        res["raw"].append(pred == gold)

        # inject: context -> state, decode question with state
        s = capture_final_state(model, enc(it["context"].strip()))
        d = decode_chunkwise(model, tokenizer, enc(qtext), lambda h: s, H, args)
        pred = parse_ab(tokenizer.decode(d, skip_special_tokens=True))
        res["inject"].append(pred == gold)

        if pred is None and len(fails) < 5:
            fails.append({"gold": gold, "raw": tokenizer.decode(nids, skip_special_tokens=True)[:80]})
        if (ii + 1) % 20 == 0:
            print(f"[{ii+1}/{len(items)}] raw={sum(res['raw'])/len(res['raw'])*100:.0f}% "
                  f"inject={sum(res['inject'])/len(res['inject'])*100:.0f}%", flush=True)

    acc = {k: round(sum(v) / len(v) * 100, 1) for k, v in res.items()}
    print(json.dumps({"acc": acc, "n": len(items), "parse_fail_samples": fails}, indent=2), flush=True)
    with open(cli.out, "w") as f:
        json.dump({"acc": acc, "n": len(items),
                   "raw_hits": {k: [int(x) for x in v] for k, v in res.items()}}, f, indent=2)
    print(f"wrote {cli.out}", flush=True)


if __name__ == "__main__":
    main()
