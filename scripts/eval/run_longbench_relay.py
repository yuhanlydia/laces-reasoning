#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/tmp/LongBench/LongBench")

from relay_utils import load_relay_model
from metrics import qa_f1_score, retrieval_score

DATASET2METRIC = {
    "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "passage_retrieval_en": retrieval_score,
}
DATASET2PROMPT = json.load(open("/tmp/LongBench/LongBench/config/dataset2prompt.json"))
DATASET2MAXGEN = json.load(open("/tmp/LongBench/LongBench/config/dataset2maxlen.json"))
DATA_DIR = "/tmp/LongBench/LongBench/data/data"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default=None, help="RELAY ckpt; omit for raw mode")
    p.add_argument("--raw_model", default="hf_release/base_models/RWKV7-Goose-World3-2.9B-HF")
    p.add_argument("--mode", default="raw", choices=("raw", "single", "trajectory"))
    p.add_argument("--datasets", default="multifieldqa_en,hotpotqa,passage_retrieval_en")
    p.add_argument("--max_length", type=int, default=3500)
    p.add_argument("--sample_steps", type=int, default=100)
    p.add_argument("--max_samples", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", required=True)
    return p.parse_args()


def middle_truncate(tokenizer, prompt, max_length):
    ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(ids) <= max_length:
        return prompt
    half = max_length // 2
    return tokenizer.decode(ids[:half], skip_special_tokens=True) + tokenizer.decode(
        ids[-half:], skip_special_tokens=True
    )


@torch.no_grad()
def inject_cache(relay, rwkv, input_ids, mode, sample_steps, device):
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
    args = parse_args()
    device = args.device
    if args.mode == "raw":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        rwkv = AutoModelForCausalLM.from_pretrained(
            args.raw_model, trust_remote_code=True, torch_dtype=torch.bfloat16,
            local_files_only=True,
        ).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.raw_model, trust_remote_code=True, local_files_only=True)
        relay = None
    else:
        relay, rwkv, tokenizer, _ckpt, _cfg = load_relay_model(args.ckpt_dir, device)

    results = {"mode": args.mode, "ckpt_dir": args.ckpt_dir, "datasets": {}}
    for dataset in args.datasets.split(","):
        prompt_fmt = DATASET2PROMPT[dataset]
        max_gen = DATASET2MAXGEN[dataset]
        metric = DATASET2METRIC[dataset]
        rows = [json.loads(l) for l in open(f"{DATA_DIR}/{dataset}.jsonl")][: args.max_samples]
        scores = []
        for i, row in enumerate(rows):
            prompt = prompt_fmt.format(**row)
            prompt = middle_truncate(tokenizer, prompt, args.max_length)
            inp = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            ctx_len = inp.input_ids.shape[-1]
            gen_kwargs = dict(max_new_tokens=max_gen, num_beams=1, do_sample=False)
            if args.mode == "raw":
                out = rwkv.generate(**inp, **gen_kwargs)[0]
            else:
                cache = inject_cache(relay, rwkv, inp.input_ids, args.mode, args.sample_steps, device)
                out = rwkv.generate(input_ids=inp.input_ids, past_key_values=cache, use_cache=True, **gen_kwargs)[0]
            pred = tokenizer.decode(out[ctx_len:], skip_special_tokens=True)
            best = max(metric(pred, gt, all_classes=row.get("all_classes")) for gt in row["answers"])
            scores.append(best)
            if (i + 1) % 10 == 0:
                print(f"{dataset} {args.mode}: {i+1}/{len(rows)} running_score={sum(scores)/len(scores):.4f}", flush=True)
        avg = sum(scores) / max(1, len(scores))
        results["datasets"][dataset] = {"score": round(avg * 100, 2), "n": len(scores)}
        print(f"=== {dataset} {args.mode} DONE: {round(avg*100,2)} (n={len(scores)}) ===", flush=True)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(args.output, "w"), indent=2)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
