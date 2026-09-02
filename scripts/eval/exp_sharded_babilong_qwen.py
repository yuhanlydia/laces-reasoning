#!/usr/bin/env python3
"""Qwen baseline for Sharded-BABILong: text_full / text_trunc / latentmas.

Compares our RWKV state-passing method against a strong dense Transformer
(Qwen3-4B, non-thinking, plain completion) on the same sharded tasks.

  text_full  : whole aggregate context + question (single-model ceiling)
  text_trunc : last --trunc_tokens of context + question (native-window baseline)
  latentmas  : LatentMAS-style sharded KV relay: per-shard prefill + K latent
               thinking steps (hidden-state feedback), KV cache carried across
               agents, final question decode. KV payload grows with M x L.

Metrics: word-boundary EM, prefill/latent/decode ms, peak GPU mem, KV payload bytes.

Example:
  python scripts/eval/exp_sharded_babilong_qwen.py --babilong_length 32k --tasks qa1 \
    --num_agents 8 --local_tokens 4096 --max_samples 30 --device cuda:0 \
    --output outputs_eval/sharded_babilong_qwen/M8_32k_qa1.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

MODEL_NAME = "Qwen/Qwen3-4B"
ALL_CONDITIONS = ["text_full", "text_trunc", "latentmas"]


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


def word_boundary_hit(gen: str, gold: str) -> bool:
    gold = gold.strip().lower()
    return bool(gold) and re.search(r"\b" + re.escape(gold) + r"\b", gen.strip().lower()) is not None


def kv_bytes(past) -> int:
    total = 0
    if past is None:
        return 0
    layers = past.layers if hasattr(past, "layers") else past
    for layer in layers:
        state = getattr(layer, "state", None)
        tensors = []
        if isinstance(state, dict):
            tensors = [v for v in state.values() if isinstance(v, torch.Tensor)]
        elif isinstance(layer, (tuple, list)):
            tensors = [v for v in layer if isinstance(v, torch.Tensor)]
        elif isinstance(layer, torch.Tensor):
            tensors = [layer]
        for t in tensors:
            total += t.numel() * t.element_size()
    return total


@torch.no_grad()
def prefill_chunked(model, ids, device, chunk=4096):
    past = None
    for j in range(0, ids.numel(), chunk):
        s = ids[j:j + chunk]
        plen = past.get_seq_length() if past is not None else 0
        fm = torch.ones((1, plen + s.numel()), dtype=torch.bool, device=device)
        out = model(input_ids=s.unsqueeze(0), attention_mask=fm,
                    past_key_values=past, use_cache=True, return_dict=True)
        past = out.past_key_values
    return past


@torch.no_grad()
def decode_greedy(model, tokenizer, past, all_ids, max_new, device):
    new_ids = []
    eos = getattr(tokenizer, "eos_token_id", None)
    ctx = torch.tensor([all_ids], device=device)
    plen = past.get_seq_length() if past is not None else 0
    fm = torch.ones((1, plen + ctx.shape[1]), dtype=torch.bool, device=device)
    out = model(input_ids=ctx, attention_mask=fm,
                past_key_values=past, use_cache=True, return_dict=True)
    past = out.past_key_values
    logits = out.logits[0, -1]
    for _ in range(max_new):
        nid = int(logits.argmax())
        if eos is not None and nid == eos:
            break
        new_ids.append(nid)
        all_ids.append(nid)
        out = model(input_ids=torch.tensor([[nid]], device=device),
                    past_key_values=past, use_cache=True, return_dict=True)
        past = out.past_key_values
        logits = out.logits[0, -1]
    return new_ids


@torch.no_grad()
def latent_think(model, past, last_hidden, k, embed_norm_target):
    for _ in range(k):
        emb = last_hidden.unsqueeze(1)
        emb = emb * (embed_norm_target / emb.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        out = model(inputs_embeds=emb, past_key_values=past, use_cache=True,
                    output_hidden_states=True, return_dict=True)
        past = out.past_key_values
        last_hidden = out.hidden_states[-1][:, -1, :]
    return past


@torch.no_grad()
def run(args):
    torch.manual_seed(args.seed)
    device = args.device
    print(f"[init] loading {MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa",
    ).to(device).eval()
    with torch.no_grad():
        emb_w = model.get_input_embeddings().weight
        embed_norm_target = emb_w.norm(dim=-1).mean().item()
    print(f"[init] embed_norm_target={embed_norm_target:.3f}", flush=True)

    M, L = args.num_agents, args.local_tokens
    conditions = [c for c in ALL_CONDITIONS if c in args.conditions.split(",")]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    out_all = {}

    for task in tasks:
        ds = load_dataset("RMT-team/babilong", args.babilong_length,
                          split=f"{task}[:{args.max_samples}]")
        print(f"[{task}] {len(ds)} samples @ {args.babilong_length}", flush=True)
        correct = {c: 0 for c in conditions}
        stats = {c: {"prefill_ms": 0.0, "latent_ms": 0.0, "decode_ms": 0.0,
                     "peak_mem_mb": 0.0, "payload_bytes": 0} for c in conditions}
        items_out = []

        for ii, r in enumerate(ds):
            ctx_text = str(r["input"]).strip()
            q_text = str(r["question"]).strip()
            gold = str(r["target"]).strip()
            ctx_ids = tokenizer(ctx_text, return_tensors="pt",
                                add_special_tokens=False).input_ids[0].to(device)
            ctx_ids = ctx_ids[: M * L]
            shards = [ctx_ids[j * L:(j + 1) * L] for j in range(M)]
            shards = [s for s in shards if s.numel() > 0]
            q_ids = tokenizer(f"\n\nQuestion: {q_text}\nAnswer:", return_tensors="pt",
                              add_special_tokens=False).input_ids[0].to(device)
            q_list = list(q_ids.tolist())
            rec = {"id": ii, "gold": gold, "n_ctx_tokens": int(ctx_ids.numel()), "gen": {}}

            def measure(cond, fn):
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.empty_cache()
                with Timer(device) as t:
                    out = fn()
                st = stats[cond]
                st["peak_mem_mb"] = max(st["peak_mem_mb"],
                                        torch.cuda.max_memory_allocated(device) / 1e6)
                return out, t.ms

            if "text_full" in conditions:
                full_ids = torch.cat(shards, 0)
                def fn():
                    past = prefill_chunked(model, full_ids, device)
                    return decode_greedy(model, tokenizer, past, q_list, args.max_new_tokens, device)
                gen, ms = measure("text_full", fn)
                txt = tokenizer.decode(gen, skip_special_tokens=True)
                hit = word_boundary_hit(txt, gold)
                correct["text_full"] += int(hit)
                stats["text_full"]["prefill_ms"] += ms
                stats["text_full"]["payload_bytes"] += 2 * (full_ids.numel() + len(q_list))
                rec["gen"]["text_full"] = {"text": txt[:100], "hit": hit}

            if "text_trunc" in conditions:
                full_ids = torch.cat(shards, 0)
                tr_ids = full_ids[-args.trunc_tokens:]
                def fn():
                    past = prefill_chunked(model, tr_ids, device)
                    return decode_greedy(model, tokenizer, past, q_list, args.max_new_tokens, device)
                gen, ms = measure("text_trunc", fn)
                txt = tokenizer.decode(gen, skip_special_tokens=True)
                hit = word_boundary_hit(txt, gold)
                correct["text_trunc"] += int(hit)
                stats["text_trunc"]["prefill_ms"] += ms
                stats["text_trunc"]["payload_bytes"] += 2 * (tr_ids.numel() + len(q_list))
                rec["gen"]["text_trunc"] = {"text": txt[:100], "hit": hit}

            if "latentmas" in conditions:
                past = None
                payload = 0
                with Timer(device) as tpf:
                    for s in shards:
                        plen = past.get_seq_length() if past is not None else 0
                        fm = torch.ones((1, plen + s.numel()), dtype=torch.bool, device=device)
                        out = model(input_ids=s.unsqueeze(0), attention_mask=fm,
                                    past_key_values=past, use_cache=True,
                                    output_hidden_states=True, return_dict=True)
                        past = out.past_key_values
                pf_ms = tpf.ms
                last_hidden = out.hidden_states[-1][:, -1, :]
                with Timer(device) as tlt:
                    past = latent_think(model, past, last_hidden, args.latent_steps,
                                        embed_norm_target)
                lt_ms = tlt.ms
                payload = kv_bytes(past)
                gen, dec_ms = measure("latentmas", lambda: decode_greedy(
                    model, tokenizer, past, q_list, args.max_new_tokens, device))
                txt = tokenizer.decode(gen, skip_special_tokens=True)
                hit = word_boundary_hit(txt, gold)
                correct["latentmas"] += int(hit)
                stats["latentmas"]["prefill_ms"] += pf_ms
                stats["latentmas"]["latent_ms"] += lt_ms
                stats["latentmas"]["decode_ms"] += dec_ms
                stats["latentmas"]["payload_bytes"] += payload
                stats["latentmas"]["peak_mem_mb"] = max(
                    stats["latentmas"]["peak_mem_mb"],
                    torch.cuda.max_memory_allocated(device) / 1e6)
                rec["gen"]["latentmas"] = {"text": txt[:100], "hit": hit}

            items_out.append(rec)
            if (ii + 1) % max(1, args.print_every) == 0:
                msg = " ".join(f"{c[:9]}={correct[c]/(ii+1)*100:.0f}%" for c in conditions)
                print(f"[{task} {ii+1}/{len(ds)}] {msg}", flush=True)

        n = len(ds)
        summary = {
            "model": MODEL_NAME, "babilong_length": args.babilong_length, "task": task,
            "num_agents": M, "local_tokens": L, "aggregate_tokens": M * L, "n": n,
            "trunc_tokens": args.trunc_tokens, "latent_steps": args.latent_steps,
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
    p.add_argument("--babilong_length", default="32k")
    p.add_argument("--tasks", default="qa1")
    p.add_argument("--num_agents", type=int, default=8)
    p.add_argument("--local_tokens", type=int, default=4096)
    p.add_argument("--max_samples", type=int, default=30)
    p.add_argument("--trunc_tokens", type=int, default=30720)
    p.add_argument("--latent_steps", type=int, default=10)
    p.add_argument("--conditions", default=",".join(ALL_CONDITIONS))
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print_every", type=int, default=5)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", default="outputs_eval/sharded_babilong_qwen/smoke.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
