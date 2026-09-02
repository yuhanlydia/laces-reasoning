#!/usr/bin/env python3
"""Sharded-BABILong: many-agent distributed-evidence integration over long contexts.

Splits each BABILong sample's context into M shards of L tokens. Each "agent" reads
one shard (within the model's trained context range); the system must answer a
question requiring evidence buried anywhere in the aggregate M x L tokens.

Conditions per sample:
  first_shard     : shard 0 only + question (lower bound)
  oracle_shard    : shard containing the gold answer string + question (per-shard read quality)
  text_full       : whole context + question, one-shot prefill (reference ceiling)
  text_trunc      : last --trunc_tokens of context + question (fixed-window baseline)
  carryover       : sequential shard prefill, decoder continues from final state
                    (== text_full mathematically; sanity check for state passing)
  state_inject    : decoder sees QUESTION ONLY + injected carryover recurrent state
                    (the paper's claim: query + fixed-size state, no text)
  dual            : state_inject + plan channel (per-shard latents, residual fusion,
                    one diffusion re-sample, fused 0.5/0.5 with carryover state)

Metrics per condition: word-boundary EM, prefill/plan/decode ms, peak GPU mem,
analytic payload bytes (text tokens vs fixed state bytes vs KV reference).

Example smoke:
  python scripts/eval/exp_sharded_babilong.py --babilong_length 2k --tasks qa1 \
    --num_agents 4 --local_tokens 512 --max_samples 5 --device cuda:0 \
    --output outputs_eval/sharded_babilong/smoke_M4_L512.json
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from scripts.eval.sample_prefix_suffix_trajectory_cfg import (  # noqa: E402
    encode_prefix, sample_trajectory_cfg,
)
from scripts.eval.exp_multiagent_star_fusion import (  # noqa: E402
    decode_chunkwise,
)
from scripts.eval.run_cola_dlm_tasks_prefix_suffix_trajectory_cfg import (  # noqa: E402
    recompute_logits_from_injected_cache, sample_next_token,
)
from scripts.eval.relay_utils import load_relay_model  # noqa: E402

ALL_CONDITIONS = ["first_shard", "oracle_shard", "text_full", "text_trunc",
                  "carryover", "state_inject", "dual", "dual_q", "dual_rtp", "dual_rtp2"]


# ----------------------------------------------------------------------------- helpers
@torch.no_grad()
def extract_recurrent_states(model, past):
    states = []
    for l in range(model.num_layers):
        st = past.layers[l].state.get("recurrent_state") if past.layers[l].state is not None else None
        states.append(st.float().clone() if isinstance(st, torch.Tensor) else None)
    return states


@torch.no_grad()
def prefill(model, ids, past=None):
    kwargs = dict(input_ids=ids, attention_mask=torch.ones_like(ids).bool(),
                  use_cache=True, return_dict=True)
    if past is not None:
        kwargs["past_key_values"] = past
    return model.rwkv_model(**kwargs)


@torch.no_grad()
def decode_plain(model, tokenizer, past, all_ids, args):
    """Greedy-ish continuation from an existing cache (no state injection)."""
    logits = None
    new_ids: list[int] = []
    eos_id = getattr(tokenizer, "eos_token_id", None)
    out = None
    # recompute logits for the last position of the prefilled context
    ctx = torch.tensor([all_ids], device=args.device, dtype=torch.long)
    past, lb = recompute_logits_from_injected_cache(model, ctx, torch.ones_like(ctx), past)
    logits = lb[0]
    for _ in range(args.max_new_tokens):
        nid = sample_next_token(logits, all_ids, args)
        if eos_id is not None and nid == eos_id:
            break
        new_ids.append(nid)
        all_ids.append(nid)
        out = model.rwkv_model(input_ids=torch.tensor([[nid]], device=args.device),
                               past_key_values=past, use_cache=True, return_dict=True)
        past = out.past_key_values
        logits = out.logits[0, -1]
    return new_ids


def word_boundary_hit(gen: str, gold: str) -> bool:
    gold = gold.strip().lower()
    if not gold:
        return False
    return re.search(r"\b" + re.escape(gold) + r"\b", gen.strip().lower()) is not None


def _cache_tensor_bytes(past) -> int:
    total = 0
    for layer in past.layers:
        if layer.state is None:
            continue
        for v in layer.state.values():
            if isinstance(v, torch.Tensor):
                total += v.numel() * v.element_size()
    return total


class Timer:
    def __init__(self, device):
        self.device = device

    def __enter__(self):
        torch.cuda.synchronize(self.device)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        torch.cuda.synchronize(self.device)
        self.ms = (time.perf_counter() - self.t0) * 1000.0
        return False


# ----------------------------------------------------------------------------- main
@torch.no_grad()
def run(args):
    torch.manual_seed(args.seed)
    device = args.device
    model, _rwkv, tokenizer, _ckpt, cfg = load_relay_model(args.ckpt_dir, device)
    model.eval()
    model._prefix_suffix_trajectory_s2 = True
    dtype = next(model.alpha_heads.parameters()).dtype
    H = int(model.trajectory_horizon)
    M, L = args.num_agents, args.local_tokens
    conditions = [c for c in ALL_CONDITIONS if c in args.conditions.split(",")]
    print(f"[init] ckpt={args.ckpt_dir}\n[init] H={H} M={M} L={L} conds={conditions}", flush=True)

    class Args:  # decode params (match published babilong protocol)
        max_new_tokens = args.max_new_tokens
        temperature = 0.0
        top_k = 50
        top_p = 0.9
        repetition_penalty = 1.1
        device = args.device
    dec_args = Args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    out_all = {}

    for task in tasks:
        ds = load_dataset("RMT-team/babilong", args.babilong_length,
                          split=f"{task}[:{args.max_samples}]")
        print(f"[{task}] {len(ds)} samples @ {args.babilong_length}", flush=True)

        # ---- first pass: tokenize, shard, collect calibration latents for dual ----
        prepared = []
        all_z = []
        for i, r in enumerate(ds):
            ctx_text = str(r["input"]).strip()
            q_text = str(r["question"]).strip()
            gold = str(r["target"]).strip()
            ctx_ids = tokenizer(ctx_text, return_tensors="pt",
                                add_special_tokens=False).input_ids[0].to(device)
            ctx_ids = ctx_ids[: M * L]
            shards = [ctx_ids[j * L:(j + 1) * L] for j in range(M)]
            shards = [s for s in shards if s.numel() > 0]
            q_ids = tokenizer(f"Question: {q_text}\nAnswer:", return_tensors="pt",
                              add_special_tokens=False).input_ids.to(device)
            # oracle shard = the one containing the gold string (fallback: last shard)
            gold_l = gold.lower()
            oracle_idx = len(shards) - 1
            for si, s in enumerate(shards):
                if gold_l and gold_l in tokenizer.decode(s, skip_special_tokens=True).lower():
                    oracle_idx = si
                    break
            item = {"id": i, "shards": shards, "q_ids": q_ids, "q_text": q_text, "gold": gold,
                    "oracle_idx": oracle_idx, "n_ctx_tokens": int(ctx_ids.numel())}
            if "dual" in conditions:
                item["zs"] = [encode_prefix(model, s.unsqueeze(0),
                              torch.ones(1, s.numel(), dtype=torch.long, device=device))[0].to(dtype)
                              for s in shards]
                for z in item["zs"]:
                    all_z.append(z[0].float())
            prepared.append(item)
        z_mean = None
        if all_z:
            z_mean = torch.stack(all_z, dim=0).mean(dim=0, keepdim=True).to(dtype)
            print(f"[{task}] calibration z_mean ||.||={float(z_mean.norm()):.2f} "
                  f"over {len(all_z)} shards", flush=True)

        # ---- second pass: run conditions ----
        correct = {c: 0 for c in conditions}
        stats = {c: {"prefill_ms": 0.0, "plan_ms": 0.0, "decode_ms": 0.0,
                     "peak_mem_mb": 0.0, "payload_bytes": 0} for c in conditions}
        stats["_shared"] = {"prefill_ms": 0.0, "plan_ms": 0.0, "decode_ms": 0.0,
                            "peak_mem_mb": 0.0, "payload_bytes": 0}
        items_out = []
        state_bytes = None
        full_cache_bytes = None

        for ii, it in enumerate(prepared):
            shards, q_ids, gold = it["shards"], it["q_ids"], it["gold"]
            rec = {"id": it["id"], "gold": gold, "oracle_idx": it["oracle_idx"],
                   "n_ctx_tokens": it["n_ctx_tokens"], "gen": {}}
            q_list = list(q_ids[0].tolist())

            def measure(cond, fn):
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.empty_cache()
                with Timer(device) as t:
                    out = fn()
                st = stats[cond]
                st["peak_mem_mb"] = max(st["peak_mem_mb"],
                                        torch.cuda.max_memory_allocated(device) / 1e6)
                return out, t.ms

            # -- first_shard (lower bound) --
            if "first_shard" in conditions:
                def fn():
                    o = prefill(model, shards[0].unsqueeze(0))
                    return decode_plain(model, tokenizer, o.past_key_values, list(q_list), dec_args)
                gen, ms = measure("first_shard", fn)
                txt = tokenizer.decode(gen, skip_special_tokens=True)
                hit = word_boundary_hit(txt, gold)
                correct["first_shard"] += int(hit)
                stats["first_shard"]["prefill_ms"] += ms
                stats["first_shard"]["payload_bytes"] += 2 * (shards[0].numel() + len(q_list))
                rec["gen"]["first_shard"] = {"text": txt[:100], "hit": hit}

            # -- oracle_shard --
            if "oracle_shard" in conditions:
                os_ids = shards[it["oracle_idx"]]
                def fn():
                    o = prefill(model, os_ids.unsqueeze(0))
                    return decode_plain(model, tokenizer, o.past_key_values, list(q_list), dec_args)
                gen, ms = measure("oracle_shard", fn)
                txt = tokenizer.decode(gen, skip_special_tokens=True)
                hit = word_boundary_hit(txt, gold)
                correct["oracle_shard"] += int(hit)
                stats["oracle_shard"]["prefill_ms"] += ms
                stats["oracle_shard"]["payload_bytes"] += 2 * (os_ids.numel() + len(q_list))
                rec["gen"]["oracle_shard"] = {"text": txt[:100], "hit": hit}

            # -- text_full (one-shot full context) --
            if "text_full" in conditions:
                full_ids = torch.cat(shards, dim=0)
                def fn():
                    o = prefill(model, full_ids.unsqueeze(0))
                    return decode_plain(model, tokenizer, o.past_key_values, list(q_list), dec_args)
                gen, ms = measure("text_full", fn)
                txt = tokenizer.decode(gen, skip_special_tokens=True)
                hit = word_boundary_hit(txt, gold)
                correct["text_full"] += int(hit)
                stats["text_full"]["prefill_ms"] += ms
                stats["text_full"]["payload_bytes"] += 2 * (full_ids.numel() + len(q_list))
                rec["gen"]["text_full"] = {"text": txt[:100], "hit": hit}

            # -- text_trunc (fixed-window baseline: last trunc_tokens) --
            if "text_trunc" in conditions:
                full_ids = torch.cat(shards, dim=0)
                tr_ids = full_ids[-args.trunc_tokens:]
                def fn():
                    o = prefill(model, tr_ids.unsqueeze(0))
                    return decode_plain(model, tokenizer, o.past_key_values, list(q_list), dec_args)
                gen, ms = measure("text_trunc", fn)
                txt = tokenizer.decode(gen, skip_special_tokens=True)
                hit = word_boundary_hit(txt, gold)
                correct["text_trunc"] += int(hit)
                stats["text_trunc"]["prefill_ms"] += ms
                stats["text_trunc"]["payload_bytes"] += 2 * (tr_ids.numel() + len(q_list))
                rec["gen"]["text_trunc"] = {"text": txt[:100], "hit": hit}

            # -- carryover / state_inject / dual: sequential shard prefill --
            if any(c in conditions for c in ("carryover", "state_inject", "dual", "dual_rtp")):
                def fn_prefill_seq():
                    past = None
                    for s in shards:
                        past = prefill(model, s.unsqueeze(0), past=past).past_key_values
                    return past
                past_seq, seq_ms = measure("_shared", fn_prefill_seq)
                seq_state = extract_recurrent_states(model, past_seq)
                if state_bytes is None:
                    state_bytes = sum(s.numel() * 4 for s in seq_state if s is not None)
                if full_cache_bytes is None:
                    full_cache_bytes = _cache_tensor_bytes(past_seq)

                if "carryover" in conditions:
                    # full fixed-size cache state crosses the wire; decoder runs query on a clone
                    def fn():
                        past_g = copy.deepcopy(past_seq)
                        o = model.rwkv_model(input_ids=q_ids,
                                             attention_mask=torch.ones_like(q_ids).bool(),
                                             past_key_values=past_g, use_cache=True, return_dict=True)
                        pk, logits = o.past_key_values, o.logits[0, -1]
                        ids, new = list(q_list), []
                        eos_id = getattr(tokenizer, "eos_token_id", None)
                        for _ in range(dec_args.max_new_tokens):
                            nid = sample_next_token(logits, ids, dec_args)
                            if eos_id is not None and nid == eos_id:
                                break
                            new.append(nid); ids.append(nid)
                            o = model.rwkv_model(input_ids=torch.tensor([[nid]], device=device),
                                                 past_key_values=pk, use_cache=True, return_dict=True)
                            pk, logits = o.past_key_values, o.logits[0, -1]
                        return new
                    gen, ms = measure("carryover", fn)
                    txt = tokenizer.decode(gen, skip_special_tokens=True)
                    hit = word_boundary_hit(txt, gold)
                    correct["carryover"] += int(hit)
                    stats["carryover"]["prefill_ms"] += seq_ms
                    stats["carryover"]["decode_ms"] += ms
                    stats["carryover"]["payload_bytes"] += full_cache_bytes * len(shards) + 2 * len(q_list)
                    rec["gen"]["carryover"] = {"text": txt[:100], "hit": hit}

                if "state_inject" in conditions:
                    def fn():
                        return decode_chunkwise(model, tokenizer, q_ids,
                                                lambda h: seq_state, H, dec_args)
                    gen, ms = measure("state_inject", fn)
                    txt = tokenizer.decode(gen, skip_special_tokens=True)
                    hit = word_boundary_hit(txt, gold)
                    correct["state_inject"] += int(hit)
                    stats["state_inject"]["decode_ms"] += ms
                    stats["state_inject"]["prefill_ms"] += seq_ms
                    stats["state_inject"]["payload_bytes"] += state_bytes * len(shards) + 2 * len(q_list)
                    rec["gen"]["state_inject"] = {"text": txt[:100], "hit": hit}

                if "dual" in conditions:
                    def fn_plan():
                        w = 1.0 / len(it["zs"])
                        resid = [z - z_mean for z in it["zs"]]
                        cond = z_mean + sum((w * r for r in resid), torch.zeros_like(z_mean))
                        Z = sample_trajectory_cfg(model, cond, args.steps, args.cfg_scale,
                                                  device, dtype)
                        return [model.predict_states(Z[:, h]) for h in range(H)]
                    plan_states, plan_ms = measure("dual", fn_plan)
                    stats["dual"]["plan_ms"] += plan_ms

                    # plan-seeded carryover: plan is the initial state, memory
                    # accumulates on top via recurrent dynamics; no tensor fusion
                    def fn_prefill_dual():
                        dummy = prefill(model, q_ids[:, :1])
                        past_d = model.inject_into_cache(dummy.past_key_values, plan_states[0])
                        for s in shards:
                            past_d = prefill(model, s.unsqueeze(0), past=past_d).past_key_values
                        return past_d
                    past_dual, dual_prefill_ms = measure("dual", fn_prefill_dual)
                    stats["dual"]["prefill_ms"] += dual_prefill_ms
                    dual_state = extract_recurrent_states(model, past_dual)

                    gen, ms = measure("dual", lambda: decode_chunkwise(
                        model, tokenizer, q_ids, lambda h: dual_state, H, dec_args))
                    txt = tokenizer.decode(gen, skip_special_tokens=True)
                    hit = word_boundary_hit(txt, gold)
                    correct["dual"] += int(hit)
                    stats["dual"]["decode_ms"] += ms
                    z_bytes = sum(z.numel() * 4 for z in it["zs"])
                    stats["dual"]["payload_bytes"] += (state_bytes + z_bytes) * len(shards) \
                        + state_bytes + 2 * len(q_list)
                    rec["gen"]["dual"] = {"text": txt[:100], "hit": hit}

                if "dual_q" in conditions:
                    # Path 1: blind plan conditioned on the question only (no context)
                    def fn_plan_q():
                        z_q = encode_prefix(model, q_ids, torch.ones_like(q_ids))[0].to(dtype)
                        Z = sample_trajectory_cfg(model, z_q, args.steps, args.cfg_scale,
                                                  device, dtype)
                        return [model.predict_states(Z[:, h]) for h in range(H)]
                    plan_q, plan_q_ms = measure("dual_q", fn_plan_q)
                    stats["dual_q"]["plan_ms"] += plan_q_ms

                    def fn_prefill_q():
                        dummy = prefill(model, q_ids[:, :1])
                        past_d = model.inject_into_cache(dummy.past_key_values, plan_q[0])
                        for s in shards:
                            past_d = prefill(model, s.unsqueeze(0), past=past_d).past_key_values
                        return past_d
                    past_dq, pfq_ms = measure("dual_q", fn_prefill_q)
                    stats["dual_q"]["prefill_ms"] += pfq_ms
                    dq_state = extract_recurrent_states(model, past_dq)

                    gen, ms = measure("dual_q", lambda: decode_chunkwise(
                        model, tokenizer, q_ids, lambda h: dq_state, H, dec_args))
                    txt = tokenizer.decode(gen, skip_special_tokens=True)
                    hit = word_boundary_hit(txt, gold)
                    correct["dual_q"] += int(hit)
                    stats["dual_q"]["decode_ms"] += ms
                    stats["dual_q"]["payload_bytes"] += state_bytes * len(shards) \
                        + state_bytes + 2 * len(q_list)
                    rec["gen"]["dual_q"] = {"text": txt[:100], "hit": hit}

                if "dual_rtp" in conditions:
                    # Path 2: read-then-plan. First read memory and verbalize a draft;
                    # the plan is then conditioned on question+draft (content-aware).
                    draft_ids, draft_ms = measure("dual_rtp", lambda: decode_chunkwise(
                        model, tokenizer, q_ids, lambda h: seq_state, H, dec_args))
                    stats["dual_rtp"]["decode_ms"] += draft_ms
                    draft_txt = tokenizer.decode(draft_ids, skip_special_tokens=True).strip()

                    def fn_plan_rtp():
                        cond_text = f"Question: {it['q_text']}\nDraft answer: {draft_txt}\n"
                        c_ids = tokenizer(cond_text, return_tensors="pt",
                                          add_special_tokens=False).input_ids.to(device)
                        z_c = encode_prefix(model, c_ids, torch.ones_like(c_ids))[0].to(dtype)
                        Z = sample_trajectory_cfg(model, z_c, args.steps, args.cfg_scale,
                                                  device, dtype)
                        return [model.predict_states(Z[:, h]) for h in range(H)]
                    plan_rtp, plan_rtp_ms = measure("dual_rtp", fn_plan_rtp)
                    stats["dual_rtp"]["plan_ms"] += plan_rtp_ms

                    def fn_prefill_rtp():
                        dummy = prefill(model, q_ids[:, :1])
                        past_d = model.inject_into_cache(dummy.past_key_values, plan_rtp[0])
                        for s in shards:
                            past_d = prefill(model, s.unsqueeze(0), past=past_d).past_key_values
                        return past_d
                    past_rtp, pfr_ms = measure("dual_rtp", fn_prefill_rtp)
                    stats["dual_rtp"]["prefill_ms"] += pfr_ms + seq_ms
                    rtp_state = extract_recurrent_states(model, past_rtp)

                    gen, ms = measure("dual_rtp", lambda: decode_chunkwise(
                        model, tokenizer, q_ids, lambda h: rtp_state, H, dec_args))
                    txt = tokenizer.decode(gen, skip_special_tokens=True)
                    hit = word_boundary_hit(txt, gold)
                    correct["dual_rtp"] += int(hit)
                    stats["dual_rtp"]["decode_ms"] += ms
                    stats["dual_rtp"]["payload_bytes"] += state_bytes * len(shards) \
                        + state_bytes + 2 * len(q_list)
                    rec["gen"]["dual_rtp"] = {"text": txt[:100], "hit": hit,
                                              "draft": draft_txt[:60]}

                if "dual_rtp2" in conditions:
                    # Path 2b: draft from the blind-plan-seeded state (strong read-out
                    # vs dual_rtp's raw-inject draft), then a content-aware plan.
                    def fn_plan1():
                        z_q = encode_prefix(model, q_ids, torch.ones_like(q_ids))[0].to(dtype)
                        Z = sample_trajectory_cfg(model, z_q, args.steps, args.cfg_scale,
                                                  device, dtype)
                        return [model.predict_states(Z[:, h]) for h in range(H)]
                    plan1, p1_ms = measure("dual_rtp2", fn_plan1)
                    stats["dual_rtp2"]["plan_ms"] += p1_ms

                    def fn_prefill1():
                        dummy = prefill(model, q_ids[:, :1])
                        past_d = model.inject_into_cache(dummy.past_key_values, plan1[0])
                        for s in shards:
                            past_d = prefill(model, s.unsqueeze(0), past=past_d).past_key_values
                        return past_d
                    past1, pf1_ms = measure("dual_rtp2", fn_prefill1)
                    stats["dual_rtp2"]["prefill_ms"] += pf1_ms
                    s1_state = extract_recurrent_states(model, past1)

                    draft_ids, draft_ms = measure("dual_rtp2", lambda: decode_chunkwise(
                        model, tokenizer, q_ids, lambda h: s1_state, H, dec_args))
                    stats["dual_rtp2"]["decode_ms"] += draft_ms
                    draft_txt = tokenizer.decode(draft_ids, skip_special_tokens=True).strip()

                    def fn_plan2():
                        cond_text = f"Question: {it['q_text']}\nDraft answer: {draft_txt}\n"
                        c_ids = tokenizer(cond_text, return_tensors="pt",
                                          add_special_tokens=False).input_ids.to(device)
                        z_c = encode_prefix(model, c_ids, torch.ones_like(c_ids))[0].to(dtype)
                        Z = sample_trajectory_cfg(model, z_c, args.steps, args.cfg_scale,
                                                  device, dtype)
                        return [model.predict_states(Z[:, h]) for h in range(H)]
                    plan2, p2_ms = measure("dual_rtp2", fn_plan2)
                    stats["dual_rtp2"]["plan_ms"] += p2_ms

                    def fn_prefill2():
                        dummy = prefill(model, q_ids[:, :1])
                        past_d = model.inject_into_cache(dummy.past_key_values, plan2[0])
                        for s in shards:
                            past_d = prefill(model, s.unsqueeze(0), past=past_d).past_key_values
                        return past_d
                    past2, pf2_ms = measure("dual_rtp2", fn_prefill2)
                    stats["dual_rtp2"]["prefill_ms"] += pf2_ms
                    rtp2_state = extract_recurrent_states(model, past2)

                    gen, ms = measure("dual_rtp2", lambda: decode_chunkwise(
                        model, tokenizer, q_ids, lambda h: rtp2_state, H, dec_args))
                    txt = tokenizer.decode(gen, skip_special_tokens=True)
                    hit = word_boundary_hit(txt, gold)
                    correct["dual_rtp2"] += int(hit)
                    stats["dual_rtp2"]["decode_ms"] += ms
                    stats["dual_rtp2"]["payload_bytes"] += 2 * state_bytes * len(shards) \
                        + state_bytes + 2 * len(q_list)
                    rec["gen"]["dual_rtp2"] = {"text": txt[:100], "hit": hit,
                                               "draft": draft_txt[:60]}

            items_out.append(rec)
            if (ii + 1) % max(1, args.print_every) == 0:
                msg = " ".join(f"{c[:9]}={correct[c]/(ii+1)*100:.0f}%" for c in conditions)
                print(f"[{task} {ii+1}/{len(prepared)}] {msg}", flush=True)

        n = len(prepared)
        summary = {
            "ckpt_dir": args.ckpt_dir, "babilong_length": args.babilong_length,
            "task": task, "num_agents": M, "local_tokens": L,
            "aggregate_tokens": M * L, "n": n,
            "trunc_tokens": args.trunc_tokens,
            "steps": args.steps, "cfg_scale": args.cfg_scale,
            "plan_weight": args.plan_weight,
            "state_bytes_fp32": state_bytes,
            "full_cache_bytes": full_cache_bytes,
            "accuracy": {c: round(correct[c] / n, 3) for c in conditions},
            "mean_ms": {c: {k: round(v / n, 1) for k, v in stats[c].items() if k != "payload_bytes"}
                        for c in conditions},
            "peak_mem_mb": {c: round(stats[c]["peak_mem_mb"], 1) for c in conditions},
            "mean_payload_bytes": {c: round(stats[c]["payload_bytes"] / n) for c in conditions},
        }
        out_all[task] = {"summary": summary, "items": items_out}
        print(f"[{task}] FINAL " +
              " ".join(f"{c}={summary['accuracy'][c]*100:.1f}%" for c in conditions), flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_all, indent=2))
    print(f"written: {out_path}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir",
                   default="outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000")
    p.add_argument("--babilong_length", default="2k",
                   help="RMT-team/babilong config; should ~= num_agents x local_tokens")
    p.add_argument("--tasks", default="qa1")
    p.add_argument("--num_agents", type=int, default=4)
    p.add_argument("--local_tokens", type=int, default=512)
    p.add_argument("--max_samples", type=int, default=5)
    p.add_argument("--trunc_tokens", type=int, default=4096)
    p.add_argument("--conditions", default=",".join(ALL_CONDITIONS))
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--cfg_scale", type=float, default=3.0)
    p.add_argument("--plan_weight", type=float, default=0.25,
                   help="dual: recurrent = plan_weight*plan + (1-plan_weight)*carryover")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print_every", type=int, default=5)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", default="outputs_eval/sharded_babilong/smoke.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
