#!/usr/bin/env python3
"""Main HotpotQA two-agent fusion experiment on the frozen RWKV champion.

Runs on the pre-filtered capability-controlled suitable set (bridge, full-gold
correct, both shards wrong). Compares text-concat baselines against our fixed-size
latent/state fusion, across conditions that reveal WHERE fusion beats concatenation:
clean, +distractors, and byte-budgeted.

Conditions (advisory Q3 baseline set + our stress axes):
  A_only, B_only                : single-shard floors
  concat_clean                  : both gold docs concatenated (threat baseline)
  concat_distract{2,4,8}        : both gold + N distractor paragraphs (noise axis)
  concat_budget                 : both gold truncated to a fixed char budget (budget axis)
  state_avg                     : recurrent memory fusion only
  latent_resid                  : residual-latent plan fusion only
  dual                          : dual fusion (ours)
  raw_latent_avg                : negative control (no residual, no resample-only)
  shuffled_partner              : dual fusion with WRONG partner B (control)

Metrics per condition: EM accuracy, true-fusion-win rate, mean context chars fed.
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
    cache = out.past_key_values
    states = []
    for l in range(model.num_layers):
        st = cache.layers[l].state.get("recurrent_state") if cache.layers[l].state is not None else None
        states.append(st.float().clone() if isinstance(st, torch.Tensor) else None)
    return states


def fuse_states(*weighted):
    L = next(len(sl) for _, sl in weighted if sl is not None)
    out = []
    for l in range(L):
        acc = None
        for w, sl in weighted:
            if sl is None or sl[l] is None:
                continue
            term = w * sl[l].float()
            acc = term if acc is None else acc + term
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

    def raw(context, q):
        ids = enc(make_prompt(context, q))
        txt, _, _ = generate_raw_answer(model, tokenizer, ids, gen)
        return txt

    CONDS = ["A_only", "B_only", "concat_clean", "concat_distract2", "concat_distract4",
             "concat_distract8", "concat_budget", "state_avg", "latent_resid", "dual",
             "raw_latent_avg", "shuffled_partner"]
    correct = {c: 0 for c in CONDS}
    truefus = {c: 0 for c in CONDS}
    ctxchars = {c: 0 for c in CONDS}

    all_z = []
    for ex in data:
        t = ex["ctx_titles"]; s = ex["ctx_sents"]; sup = ex["sup_titles"]
        za = enc_z(build_ctx(t, s, {sup[0]}))
        zb = enc_z(build_ctx(t, s, {sup[1]}))
        all_z += [za[0], zb[0]]
    z_mean = torch.stack(all_z, 0).mean(0, keepdim=True)

    n = len(data)
    for i, ex in enumerate(data):
        t, s, sup, gold, q = ex["ctx_titles"], ex["ctx_sents"], ex["sup_titles"], ex["answer"], ex["question"]
        a_title, b_title = sup[0], sup[1]
        distract_titles = [x for x in t if x not in set(sup)]

        ctx_a = build_ctx(t, s, {a_title})
        ctx_b = build_ctx(t, s, {b_title})
        ctx_clean = build_ctx(t, s, set(sup))
        def concat_with(nd):
            keep = set(sup) | set(distract_titles[:nd])
            return build_ctx(t, s, keep)
        budget = a.budget_chars
        ctx_bud = (ctx_a[: budget // 2] + "\n" + ctx_b[: budget // 2])

        preds = {}
        preds["A_only"] = raw(ctx_a, q)
        preds["B_only"] = raw(ctx_b, q)
        preds["concat_clean"] = raw(ctx_clean, q)
        preds["concat_distract2"] = raw(concat_with(2), q)
        preds["concat_distract4"] = raw(concat_with(4), q)
        preds["concat_distract8"] = raw(concat_with(min(8, len(distract_titles))), q)
        preds["concat_budget"] = raw(ctx_bud, q)
        ctxchars["A_only"] += len(ctx_a); ctxchars["B_only"] += len(ctx_b)
        ctxchars["concat_clean"] += len(ctx_clean)
        ctxchars["concat_distract2"] += len(concat_with(2))
        ctxchars["concat_distract4"] += len(concat_with(4))
        ctxchars["concat_distract8"] += len(concat_with(min(8, len(distract_titles))))
        ctxchars["concat_budget"] += len(ctx_bud)

        mem_a = capture_final_state(model, enc(ctx_a))
        mem_b = capture_final_state(model, enc(ctx_b))
        za = enc_z(ctx_a)
        zb = enc_z(ctx_b)
        q_ids = enc(make_prompt("", q))

        cond_raw = 0.5 * za + 0.5 * zb
        ra, rb = za - z_mean, zb - z_mean
        cond_resid = z_mean + 0.5 * ra + 0.5 * rb
        Zp_resid = sample_trajectory_cfg(model, cond_resid, a.steps, a.cfg_scale, dev, dtype)
        Zp_raw = sample_trajectory_cfg(model, cond_raw, a.steps, a.cfg_scale, dev, dtype)
        ps_resid = [model.predict_states(Zp_resid[:, h]) for h in range(H)]
        ps_raw = [model.predict_states(Zp_raw[:, h]) for h in range(H)]

        W = (0.5, 0.25, 0.25)
        preds["state_avg"] = decode_from_states(model, tokenizer, q_ids,
            lambda h: fuse_states((0.5, mem_a), (0.5, mem_b)), H, gen)
        preds["latent_resid"] = decode_from_states(model, tokenizer, q_ids,
            lambda h: ps_resid[h], H, gen)
        preds["dual"] = decode_from_states(model, tokenizer, q_ids,
            lambda h: fuse_states((W[0], ps_resid[h]), (W[1], mem_a), (W[2], mem_b)), H, gen)
        preds["raw_latent_avg"] = decode_from_states(model, tokenizer, q_ids,
            lambda h: ps_raw[h], H, gen)
        # shuffled partner: use next example's B as wrong partner
        wrong = data[(i + 1) % n]
        wb_ids = enc(build_ctx(wrong["ctx_titles"], wrong["ctx_sents"], {wrong["sup_titles"][1]}))
        mem_wb = capture_final_state(model, wb_ids)
        preds["shuffled_partner"] = decode_from_states(model, tokenizer, q_ids,
            lambda h: fuse_states((W[0], ps_resid[h]), (W[1], mem_a), (W[2], mem_wb)), H, gen)
        for c in ["state_avg", "latent_resid", "dual", "raw_latent_avg", "shuffled_partner"]:
            ctxchars[c] += len(q)

        a_ok = em(preds["A_only"], gold); b_ok = em(preds["B_only"], gold)
        for c in CONDS:
            ok = em(preds[c], gold)
            correct[c] += ok
            if ok and not a_ok and not b_ok:
                truefus[c] += 1
        print(f"[{i+1}/{n}] " + " ".join(
            f"{c.split('_')[0][:4]}={'1' if em(preds[c],gold) else '.'}" for c in
            ["concat_clean","concat_distract8","concat_budget","state_avg","latent_resid","dual","raw_latent_avg","shuffled_partner"]),
            flush=True)

    result = {"ckpt_dir": a.ckpt_dir, "n": n, "steps": a.steps, "cfg_scale": a.cfg_scale,
              "budget_chars": a.budget_chars, "conditions": CONDS,
              "accuracy": {c: round(correct[c] / n, 3) for c in CONDS},
              "true_fusion_win": {c: round(truefus[c] / n, 3) for c in CONDS},
              "mean_ctx_chars": {c: round(ctxchars[c] / n) for c in CONDS}}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(result, indent=2))
    print(f"\n=== HotpotQA two-agent fusion (n={n}, champion) ===")
    print(f"  {'condition':<20} {'EM':>6} {'true-fus':>9} {'ctx-chars':>10}")
    for c in CONDS:
        print(f"  {c:<20} {result['accuracy'][c]*100:>5.0f}% {result['true_fusion_win'][c]*100:>8.0f}% {result['mean_ctx_chars'][c]:>10}")
    print(f"\nwritten: {a.output}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000")
    p.add_argument("--data", default="outputs_eval/hotpot_suitable_bridge.json")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--cfg_scale", type=float, default=3.0)
    p.add_argument("--budget_chars", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="outputs_eval/exp_hotpot_multiagent_fusion.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
