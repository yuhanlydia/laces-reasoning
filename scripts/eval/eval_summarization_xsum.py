#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from relay_utils import load_relay_model
from rouge_score import rouge_scorer

XSUM_DIR = "preprocessed_data/xsum/validation"
INSTR = "\nTL;DR:"


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
    p.add_argument("--max_samples", type=int, default=100)
    p.add_argument("--max_new_tokens", type=int, default=48)
    p.add_argument("--max_prefix_tokens", type=int, default=400)
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

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    files = sorted(glob.glob(f"{XSUM_DIR}/*.npz"))[: args.max_samples]
    instr_ids = tokenizer(INSTR, return_tensors="pt").input_ids.to(device)

    agg = {"rouge1": [], "rouge2": [], "rougeL": []}
    records = []
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        tstart = int(d["target_start"][0])
        doc_ids = torch.tensor(d["input_ids"][:tstart][: args.max_prefix_tokens], dtype=torch.long, device=device).unsqueeze(0)
        prefix_ids = torch.cat([doc_ids, instr_ids], dim=1)
        ref = str(d["target_text"][0]).strip()

        gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=False, repetition_penalty=1.2)
        if args.mode == "raw":
            out = rwkv.generate(input_ids=prefix_ids, use_cache=True, **gen_kwargs)[0]
        else:
            cache = build_cache(relay, rwkv, prefix_ids, args.mode, args.sample_steps, device)
            out = rwkv.generate(input_ids=prefix_ids, past_key_values=cache, use_cache=True, **gen_kwargs)[0]
        gen_ids = out[prefix_ids.shape[-1]:].tolist()
        decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)
        hyp = next((ln.strip() for ln in decoded.splitlines() if ln.strip()), decoded.strip())

        s = scorer.score(ref, hyp)
        for k in agg:
            agg[k].append(s[k].fmeasure)
        records.append({"ref": ref[:120], "hyp": hyp[:120],
                        "r1": round(s["rouge1"].fmeasure, 4), "rL": round(s["rougeL"].fmeasure, 4)})
        if len(records) % 20 == 0:
            print(f"{len(records)}/{len(files)} r1={np.mean(agg['rouge1']):.4f} rL={np.mean(agg['rougeL']):.4f}", flush=True)

    summary = {
        "mode": args.mode, "ckpt_dir": args.ckpt_dir, "n": len(records),
        "rouge1": round(float(np.mean(agg["rouge1"])), 4),
        "rouge2": round(float(np.mean(agg["rouge2"])), 4),
        "rougeL": round(float(np.mean(agg["rougeL"])), 4),
        "records": records,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.output, "w"), indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
