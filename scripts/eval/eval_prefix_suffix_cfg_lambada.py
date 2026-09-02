#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.eval.relay_utils import get_data_dir, list_npz_files, load_relay_model
from scripts.eval.sample_prefix_suffix_cfg import encode_prefix, sample_ddim_cfg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--max_samples", type=int, default=5153)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--scales", default="0.5,1,5,10")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None)
    return p.parse_args()


@torch.no_grad()
def score_raw(model, ids, mask, target_start):
    out = model.rwkv_model(input_ids=ids, attention_mask=mask.bool(), return_dict=True)
    pred_logits = out.logits[0, target_start - 1:ids.shape[1] - 1]
    pred_tokens = pred_logits.argmax(dim=-1)
    target_tokens = ids[0, target_start:ids.shape[1]]
    return int(torch.equal(pred_tokens, target_tokens))


@torch.no_grad()
def score_cfg(model, ids, mask, target_start, scale, steps, device, dtype, seed):
    prefix_ids = ids[:, :target_start]
    prefix_mask = mask[:, :target_start]
    suffix_context = ids[:, target_start - 1:]
    suffix_mask = mask[:, target_start - 1:]
    torch.manual_seed(seed)
    cond = encode_prefix(model, prefix_ids, prefix_mask)
    z = sample_ddim_cfg(model, cond, steps, scale, device, dtype)
    states = model.predict_states(z)
    out_prefix = model.rwkv_model(input_ids=prefix_ids, attention_mask=prefix_mask.bool(), use_cache=True, return_dict=True)
    cache = model.inject_into_cache(out_prefix.past_key_values, states)
    out = model.rwkv_model(input_ids=suffix_context, attention_mask=suffix_mask.bool(), past_key_values=cache, use_cache=False, return_dict=True)
    pred_tokens = out.logits[0, :-1].argmax(dim=-1)
    target_tokens = suffix_context[0, 1:]
    return int(torch.equal(pred_tokens, target_tokens))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dtype = torch.bfloat16
    scales = [float(x.strip()) for x in args.scales.split(",") if x.strip()]
    model, _rwkv, _tokenizer, ckpt, cfg = load_relay_model(args.ckpt_dir, args.device)
    model_any = cast(Any, model)
    cfg_any = cast(Any, cfg)
    model_any._prefix_suffix_s2 = True
    model_any._training_stage = 2
    model_any._cfg_drop_prob = float(cfg_any.training.get("cfg_drop_prob", 0.0))
    data_dir = args.data_dir or get_data_dir("DIFFRWKV_DATA_DIR", "preprocessed_data/lambada/test")
    files = list_npz_files(data_dir, args.max_samples)
    results = {"ckpt_dir": args.ckpt_dir, "step": ckpt.get("step", -1), "n_files": len(files), "steps": args.steps, "scales": scales}
    correct_raw = 0
    t0 = time.time()
    for fp in files:
        d = np.load(fp)
        ids = torch.from_numpy(d["input_ids"].astype(np.int64)).unsqueeze(0).to(args.device)
        mask = torch.from_numpy(d["attention_mask"].astype(np.int64)).unsqueeze(0).to(args.device)
        correct_raw += score_raw(model, ids, mask, int(d["target_start"][0]))
    results["acc_raw_suffix"] = correct_raw / max(1, len(files))
    results["correct_raw_suffix"] = correct_raw
    results["elapsed_raw_suffix_s"] = round(time.time() - t0, 1)
    for i, scale in enumerate(scales):
        correct = 0
        t0 = time.time()
        for j, fp in enumerate(files):
            d = np.load(fp)
            ids = torch.from_numpy(d["input_ids"].astype(np.int64)).unsqueeze(0).to(args.device)
            mask = torch.from_numpy(d["attention_mask"].astype(np.int64)).unsqueeze(0).to(args.device)
            correct += score_cfg(model, ids, mask, int(d["target_start"][0]), scale, args.steps, args.device, dtype, args.seed + 100000 * (i + 1) + j)
        key = f"cfg{scale:g}"
        results[f"acc_{key}"] = correct / max(1, len(files))
        results[f"correct_{key}"] = correct
        results[f"elapsed_{key}_s"] = round(time.time() - t0, 1)
        print(f"{key}: acc={results[f'acc_{key}']:.4f} correct={correct}/{len(files)}", flush=True)
        if args.output:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
    output_json = json.dumps(results, indent=2)
    print(output_json)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(output_json)


if __name__ == "__main__":
    main()
