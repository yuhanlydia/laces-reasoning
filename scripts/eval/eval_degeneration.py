#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from relay_utils import load_relay_model

PROMPTS = [
    "The history of artificial intelligence",
    "In a small town near the coast,",
    "The economic impact of climate change",
    "She opened the door and",
    "Scientists have recently discovered",
    "The most important lesson I learned",
    "Once upon a time, in a distant land,",
    "The future of renewable energy depends on",
    "He walked into the room and noticed",
    "The relationship between technology and society",
]


def rep_n(tokens, n):
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return (len(grams) - len(set(grams))) / max(1, len(grams))


def distinct_n(tokens, n):
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / max(1, len(grams))


@torch.no_grad()
def build_cache(relay, rwkv, input_ids, mode, sample_steps, device):
    if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
        torch.compiler.cudagraph_mark_step_begin()
    out = rwkv(input_ids=input_ids[:, :1], use_cache=True, return_dict=True)
    cache = out.past_key_values
    if mode == "trajectory":
        dtype = next(relay.trajectory_dit.parameters()).dtype
        z = relay.trajectory_sample(1, num_steps=sample_steps, device=device, dtype=dtype)
        layer_states = relay.predict_trajectory_states(z)
        states = [ls[:, 0] for ls in layer_states]
    else:
        dtype = next(relay.latent_dit.parameters()).dtype
        z = relay.ddpm_sample(1, num_steps=sample_steps, device=device, dtype=dtype)
        states = relay.predict_states(z)
    return relay.inject_into_cache(cache, states)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="raw", choices=("raw", "single", "trajectory"))
    p.add_argument("--ckpt_dir", default=None)
    p.add_argument("--raw_model", default="hf_release/base_models/RWKV7-Goose-World3-2.9B-HF")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--sample_steps", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    device = args.device
    torch.manual_seed(args.seed)

    if args.mode == "raw":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        rwkv = AutoModelForCausalLM.from_pretrained(
            args.raw_model, trust_remote_code=True, torch_dtype=torch.bfloat16, local_files_only=True
        ).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.raw_model, trust_remote_code=True, local_files_only=True)
        relay = None
    else:
        relay, rwkv, tokenizer, _ckpt, _cfg = load_relay_model(args.ckpt_dir, device)

    per_prompt = []
    agg = {"rep4": [], "rep8": [], "distinct1": [], "distinct2": []}
    for prompt in PROMPTS:
        inp = tokenizer(prompt, return_tensors="pt").to(device)
        gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=True, temperature=0.7,
                          top_k=50, top_p=0.9, repetition_penalty=1.0)
        if args.mode == "raw":
            out = rwkv.generate(**inp, **gen_kwargs)[0]
        else:
            cache = build_cache(relay, rwkv, inp.input_ids, args.mode, args.sample_steps, device)
            out = rwkv.generate(input_ids=inp.input_ids, past_key_values=cache, use_cache=True, **gen_kwargs)[0]
        gen_tokens = out[inp.input_ids.shape[-1]:].tolist()
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        words = text.split()
        m = {
            "rep4": round(rep_n(words, 4), 4),
            "rep8": round(rep_n(words, 8), 4),
            "distinct1": round(distinct_n(words, 1), 4),
            "distinct2": round(distinct_n(words, 2), 4),
            "n_words": len(words),
        }
        for k in agg:
            agg[k].append(m[k])
        per_prompt.append({"prompt": prompt, **m, "text_head": text[:120]})

    summary = {
        "mode": args.mode,
        "ckpt_dir": args.ckpt_dir,
        "n_prompts": len(PROMPTS),
        "mean_rep4": round(sum(agg["rep4"]) / len(agg["rep4"]), 4),
        "mean_rep8": round(sum(agg["rep8"]) / len(agg["rep8"]), 4),
        "mean_distinct1": round(sum(agg["distinct1"]) / len(agg["distinct1"]), 4),
        "mean_distinct2": round(sum(agg["distinct2"]) / len(agg["distinct2"]), 4),
        "per_prompt": per_prompt,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.output, "w"), indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_prompt"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
