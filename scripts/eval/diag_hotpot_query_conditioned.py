#!/usr/bin/env python3
"""Advisory Q6 diagnostic: is the HotpotQA fusion failure over-compression (fixable)
or a dead architecture?

The prior experiment encoded each document ALONE (mean-pooled to one R^32), with no
question. The advisory's #1 fix is query-conditioned compression: encode
"question + paragraph" so the pooled latent is already answer-relevant.

This script compares, on the same suitable set, for the dual/latent/state methods:
  base   : encode paragraph alone (reproduces the 7% failure)
  qcond  : encode "question + paragraph" (query-conditioned latent)
  qcond_state : query-conditioned latent AND query-primed memory state

If qcond jumps well above base, the bottleneck is unconditional over-compression
(fixable). If qcond stays at floor, the compression loss is fundamental for this
extractive task.
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
    cache = out.past_key_values
    return [cache.layers[l].state.get("recurrent_state").float().clone()
            if cache.layers[l].state and isinstance(cache.layers[l].state.get("recurrent_state"), torch.Tensor)
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

    VARIANTS = ["base_dual", "qcond_dual", "qcond_state", "base_state"]
    correct = {v: 0 for v in VARIANTS}
    n = len(data)

    for i, ex in enumerate(data):
        t, s, sup, gold, q = ex["ctx_titles"], ex["ctx_sents"], ex["sup_titles"], ex["answer"], ex["question"]
        ctx_a = build_ctx(t, s, {sup[0]})
        ctx_b = build_ctx(t, s, {sup[1]})
        qa = "Question: " + q + "\n" + ctx_a
        qb = "Question: " + q + "\n" + ctx_b
        q_ids = enc(make_prompt("", q))

        # base: paragraph-only encoding
        za, zb = enc_z(ctx_a), enc_z(ctx_b)
        mem_a, mem_b = capture_final_state(model, enc(ctx_a)), capture_final_state(model, enc(ctx_b))
        # query-conditioned: question+paragraph encoding and memory
        zqa, zqb = enc_z(qa), enc_z(qb)
        memqa, memqb = capture_final_state(model, enc(qa)), capture_final_state(model, enc(qb))

        def plan_states(z1, z2):
            cond = 0.5 * z1 + 0.5 * z2
            Zp = sample_trajectory_cfg(model, cond, a.steps, a.cfg_scale, dev, dtype)
            return [model.predict_states(Zp[:, h]) for h in range(H)]

        ps_base = plan_states(za, zb)
        ps_q = plan_states(zqa, zqb)
        W = (0.5, 0.25, 0.25)

        preds = {}
        preds["base_dual"] = decode_from_states(model, tokenizer, q_ids,
            lambda h: fuse_states((W[0], ps_base[h]), (W[1], mem_a), (W[2], mem_b)), H, gen)
        preds["qcond_dual"] = decode_from_states(model, tokenizer, q_ids,
            lambda h: fuse_states((W[0], ps_q[h]), (W[1], mem_a), (W[2], mem_b)), H, gen)
        preds["qcond_state"] = decode_from_states(model, tokenizer, q_ids,
            lambda h: fuse_states((W[0], ps_q[h]), (W[1], memqa), (W[2], memqb)), H, gen)
        preds["base_state"] = decode_from_states(model, tokenizer, q_ids,
            lambda h: fuse_states((0.5, mem_a), (0.5, mem_b)), H, gen)

        for v in VARIANTS:
            correct[v] += em(preds[v], gold)
        print(f"[{i+1}/{n}] " + " ".join(f"{v}={'1' if em(preds[v],gold) else '.'}" for v in VARIANTS)
              + f" | gold={gold[:20]!r} qcd={extract_answer(preds['qcond_dual'])[:20]!r}", flush=True)

    result = {"ckpt_dir": a.ckpt_dir, "n": n, "steps": a.steps, "cfg_scale": a.cfg_scale,
              "variants": VARIANTS, "accuracy": {v: round(correct[v] / n, 3) for v in VARIANTS}}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(result, indent=2))
    print(f"\n=== Query-conditioned compression diagnostic (n={n}) ===")
    for v in VARIANTS:
        print(f"  {v:<14} {result['accuracy'][v]*100:>5.0f}%")
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
    p.add_argument("--output", default="outputs_eval/diag_hotpot_query_conditioned.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
