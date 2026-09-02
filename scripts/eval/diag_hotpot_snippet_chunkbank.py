#!/usr/bin/env python3
"""Advisory Q6 fixes #3 and #5, stacked on the winning query-conditioned state variant.

Prior diagnostic: qcond_state (query-conditioned memory state) lifted HotpotQA fusion
from 0% to 20%. The remaining gap vs concat (100%) is state averaging blurring exact
answer spans. This tests the advisory's two remaining fixes:

  #5 per-chunk state bank : instead of averaging the two paragraphs' FINAL states, keep
     per-sentence states and select the most question-relevant one per agent, then fuse.
     Avoids blurring entity bindings across a whole paragraph.

  #3 oracle snippet channel : let each agent select its most question-relevant sentence
     and pass ONLY that short raw snippet to the decoder (fixed-size latent plan +
     short exact-span text), preserving the exact answer surface form.

Variants:
  qcond_state       : baseline winner (query-conditioned final-state averaging)
  chunkbank_state   : per-sentence state selection + fusion (fix #5)
  snippet_channel   : latent plan + selected evidence sentences as short text (fix #3)
  snippet_only      : pure text baseline of the two selected sentences (concat control)
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
    recompute_logits_from_injected_cache, sample_next_token, generate_raw_answer,
)
from scripts.eval.relay_utils import load_relay_model  # noqa: E402
from scripts.eval.probe_hotpot_backbone import (  # noqa: E402
    normalize, extract_answer, build_ctx, make_prompt,
)


def em(pred, gold):
    p, g = normalize(extract_answer(pred)), normalize(gold)
    return bool(g) and (g in p or p in g)


class GenArgs:
    max_new_tokens = 24
    temperature = 0.0
    top_k = 50
    top_p = 0.9
    repetition_penalty = 1.1


@torch.no_grad()
def capture_final_state(model, ids):
    out = model.rwkv_model(input_ids=ids, attention_mask=torch.ones_like(ids).bool(),
                           use_cache=True, return_dict=True)
    c = out.past_key_values
    return [c.layers[l].state.get("recurrent_state").float().clone()
            if c.layers[l].state and isinstance(c.layers[l].state.get("recurrent_state"), torch.Tensor)
            else None for l in range(model.num_layers)]


def fuse_states(*weighted):
    L = next(len(sl) for _, sl in weighted if sl is not None)
    out = []
    for l in range(L):
        acc = None
        for w, sl in weighted:
            if sl is None or sl[l] is None:
                continue
            acc = w * sl[l].float() if acc is None else acc + w * sl[l].float()
        out.append(acc)
    return out


@torch.no_grad()
def sentence_relevance(model, tokenizer, q, sentences, dev):
    """Score each sentence by lexical overlap with the question (cheap, deterministic)."""
    qw = set(normalize(q).split())
    scored = []
    for si, sent in enumerate(sentences):
        sw = set(normalize(sent).split())
        overlap = len(qw & sw)
        scored.append((overlap, si, sent))
    scored.sort(reverse=True)
    return scored


@torch.no_grad()
def decode_from_states(model, tokenizer, q_ids, state_fn, H, gen):
    chunk = int(model.trajectory_chunk_size)
    out = model.rwkv_model(input_ids=q_ids, attention_mask=torch.ones_like(q_ids).bool(),
                           use_cache=True, return_dict=True)
    past = out.past_key_values
    logits = out.logits[0, -1]
    all_ids = list(q_ids[0].tolist())
    new_ids = []
    eos = getattr(tokenizer, "eos_token_id", None)
    stop = False
    for h in range(H):
        if stop or len(new_ids) >= gen.max_new_tokens:
            break
        past = model.inject_into_cache(past, state_fn(h))
        ctx = torch.tensor([all_ids], device=q_ids.device)
        past, lb = recompute_logits_from_injected_cache(model, ctx, torch.ones_like(ctx), past)
        logits = lb[0]
        for _ in range(chunk):
            if len(new_ids) >= gen.max_new_tokens:
                stop = True
                break
            nid = sample_next_token(logits, all_ids, gen)
            if eos is not None and nid == eos:
                stop = True
                break
            new_ids.append(nid)
            all_ids.append(nid)
            out = model.rwkv_model(input_ids=torch.tensor([[nid]], device=q_ids.device),
                                   past_key_values=past, use_cache=True, return_dict=True)
            past = out.past_key_values
            logits = out.logits[0, -1]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


@torch.no_grad()
def run(a):
    torch.manual_seed(a.seed)
    dev = a.device
    data = json.load(open(a.data))[: a.n]
    model, _rwkv, tokenizer, _ckpt, _cfg = load_relay_model(a.ckpt_dir, dev)
    model.eval()
    dtype = next(model.alpha_heads.parameters()).dtype
    H = int(model.trajectory_horizon)
    gen = GenArgs()

    def enc(text):
        return tokenizer(text, return_tensors="pt").input_ids.to(dev)

    def enc_z(text):
        ids = enc(text)
        return encode_prefix(model, ids, torch.ones_like(ids))[0].to(dtype)

    def sents_of(t, s, title):
        for ti, ss in zip(t, s):
            if ti == title:
                return ss
        return []

    VARIANTS = ["qcond_state", "chunkbank_state", "snippet_channel", "snippet_only"]
    correct = {v: 0 for v in VARIANTS}
    n = len(data)

    for i, ex in enumerate(data):
        t, s, sup, gold, q = ex["ctx_titles"], ex["ctx_sents"], ex["sup_titles"], ex["answer"], ex["question"]
        ta, tb = sup[0], sup[1]
        sents_a = sents_of(t, s, ta)
        sents_b = sents_of(t, s, tb)
        ctx_a = build_ctx(t, s, {ta})
        ctx_b = build_ctx(t, s, {tb})
        qa = "Question: " + q + "\n" + ctx_a
        qb = "Question: " + q + "\n" + ctx_b
        q_ids = enc(make_prompt("", q))

        # query-conditioned final-state (baseline winner)
        memqa = capture_final_state(model, enc(qa))
        memqb = capture_final_state(model, enc(qb))
        zqa, zqb = enc_z(qa), enc_z(qb)
        cond = 0.5 * zqa + 0.5 * zqb
        Zp = sample_trajectory_cfg(model, cond, a.steps, a.cfg_scale, dev, dtype)
        ps = [model.predict_states(Zp[:, h]) for h in range(H)]
        W = (0.5, 0.25, 0.25)

        preds = {}
        preds["qcond_state"] = decode_from_states(model, tokenizer, q_ids,
            lambda h: fuse_states((W[0], ps[h]), (W[1], memqa), (W[2], memqb)), H, gen)

        # fix #5: per-sentence state bank -> pick most question-relevant sentence per agent
        top_a = sentence_relevance(model, tokenizer, q, sents_a, dev)[:1]
        top_b = sentence_relevance(model, tokenizer, q, sents_b, dev)[:1]
        snip_a = " ".join(x[2] for x in top_a) if top_a else ctx_a
        snip_b = " ".join(x[2] for x in top_b) if top_b else ctx_b
        mem_sa = capture_final_state(model, enc("Question: " + q + "\n" + snip_a))
        mem_sb = capture_final_state(model, enc("Question: " + q + "\n" + snip_b))
        preds["chunkbank_state"] = decode_from_states(model, tokenizer, q_ids,
            lambda h: fuse_states((W[0], ps[h]), (W[1], mem_sa), (W[2], mem_sb)), H, gen)

        # fix #3: latent plan + short evidence snippet text to decoder
        snippet_prompt = make_prompt(snip_a + "\n" + snip_b, q)
        q_ids_snip = enc(snippet_prompt)
        preds["snippet_channel"] = decode_from_states(model, tokenizer, q_ids_snip,
            lambda h: fuse_states((W[0], ps[h]), (W[1], memqa), (W[2], memqb)), H, gen)

        # snippet-only text control (no state, just the two selected sentences)
        txt, _, _ = generate_raw_answer(model, tokenizer, enc(snippet_prompt), gen)
        preds["snippet_only"] = txt

        for v in VARIANTS:
            correct[v] += em(preds[v], gold)
        print(f"[{i+1}/{n}] " + " ".join(f"{v.split('_')[0][:4]}={'1' if em(preds[v],gold) else '.'}" for v in VARIANTS)
              + f" | gold={gold[:18]!r}", flush=True)

    result = {"ckpt_dir": a.ckpt_dir, "n": n, "variants": VARIANTS,
              "accuracy": {v: round(correct[v] / n, 3) for v in VARIANTS}}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(result, indent=2))
    print(f"\n=== Snippet + chunk-bank diagnostic (n={n}) ===")
    for v in VARIANTS:
        print(f"  {v:<18} {result['accuracy'][v]*100:>5.0f}%")
    print(f"\nwritten: {a.output}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000")
    p.add_argument("--data", default="outputs_eval/hotpot_suitable_bridge.json")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--cfg_scale", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="outputs_eval/diag_hotpot_snippet_chunkbank.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
