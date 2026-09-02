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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.eval.relay_utils import list_npz_files, load_relay_model
from scripts.eval.sample_prefix_suffix_cfg import encode_prefix, sample_ddim_cfg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--task_dir", required=True)
    p.add_argument("--max_samples", type=int, default=2000)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--scales", default="0.5,1,5,10")
    p.add_argument("--length_normalize", action="store_true", default=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None)
    return p.parse_args()


@torch.no_grad()
def raw_score(model, ids, mask, choice_start, length_normalize):
    out = model.rwkv_model(input_ids=ids, attention_mask=mask.bool(), return_dict=True)
    return span_nll(out.logits, ids, mask, choice_start, length_normalize)


def span_nll(logits, ids, mask, choice_start, length_normalize):
    if choice_start <= 0:
        choice_start = 1
    pred_logits = logits[0, choice_start - 1:ids.shape[1] - 1].float()
    targets = ids[0, choice_start:ids.shape[1]]
    valid = mask[0, choice_start:ids.shape[1]].bool()
    if valid.sum() == 0:
        return float("inf")
    log_probs = F.log_softmax(pred_logits, dim=-1)
    nll = -log_probs.gather(1, targets.unsqueeze(-1)).squeeze(-1)
    nll = nll[valid]
    total = nll.sum().item()
    if length_normalize:
        return total / max(1, nll.numel())
    return total


@torch.no_grad()
def cfg_score(model, ids, mask, choice_start, scale, steps, device, dtype, seed, length_normalize, shared_z=None):
    if choice_start <= 0:
        choice_start = 1
    prefix_ids = ids[:, :choice_start]
    prefix_mask = mask[:, :choice_start]
    suffix_context = ids[:, choice_start - 1:]
    suffix_mask = mask[:, choice_start - 1:]
    if shared_z is None:
        torch.manual_seed(seed)
        cond = encode_prefix(model, prefix_ids, prefix_mask)
        z = sample_ddim_cfg(model, cond, steps, scale, device, dtype)
    else:
        z = shared_z
    states = model.predict_states(z)
    out_prefix = model.rwkv_model(input_ids=prefix_ids, attention_mask=prefix_mask.bool(), use_cache=True, return_dict=True)
    cache = model.inject_into_cache(out_prefix.past_key_values, states)
    out = model.rwkv_model(input_ids=suffix_context, attention_mask=suffix_mask.bool(), past_key_values=cache, use_cache=False, return_dict=True)
    pred_logits = out.logits[0, :-1].float()
    targets = suffix_context[0, 1:]
    valid = suffix_mask[0, 1:].bool()
    if valid.sum() == 0:
        return float("inf")
    log_probs = F.log_softmax(pred_logits, dim=-1)
    nll = -log_probs.gather(1, targets.unsqueeze(-1)).squeeze(-1)
    nll = nll[valid]
    total = nll.sum().item()
    if length_normalize:
        return total / max(1, nll.numel())
    return total


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
    files = list_npz_files(args.task_dir, args.max_samples)
    task_name = os.path.basename(args.task_dir.rstrip("/"))
    results = {"ckpt_dir": args.ckpt_dir, "step": ckpt.get("step", -1), "task": task_name, "task_dir": args.task_dir, "n_files": len(files), "steps": args.steps, "scales": scales}
    modes = [None] + scales
    for mode_i, mode in enumerate(modes):
        correct = 0
        t0 = time.time()
        for file_i, fp in enumerate(files):
            d = np.load(fp)
            input_ids = d["input_ids"]
            attn_mask = d["attention_mask"]
            choice_start = d["choice_start"]
            label = int(d["label"][0])
            shared_z = None
            if mode is not None:
                first_ids = torch.from_numpy(input_ids[0:1].astype(np.int64)).to(args.device)
                first_mask = torch.from_numpy(attn_mask[0:1].astype(np.int64)).to(args.device)
                first_cs = int(choice_start[0])
                if first_cs <= 0:
                    first_cs = 1
                torch.manual_seed(args.seed + 100000 * mode_i + 1000 * file_i)
                cond = encode_prefix(model, first_ids[:, :first_cs], first_mask[:, :first_cs])
                shared_z = sample_ddim_cfg(model, cond, args.steps, mode, args.device, dtype)

            scores = []
            for j in range(input_ids.shape[0]):
                ids = torch.from_numpy(input_ids[j:j + 1].astype(np.int64)).to(args.device)
                mask = torch.from_numpy(attn_mask[j:j + 1].astype(np.int64)).to(args.device)
                cs = int(choice_start[j])
                if mode is None:
                    score = raw_score(model, ids, mask, cs, args.length_normalize)
                else:
                    score = cfg_score(model, ids, mask, cs, mode, args.steps, args.device, dtype, args.seed + 100000 * mode_i + 1000 * file_i, args.length_normalize, shared_z=shared_z)
                scores.append(score)
            pred = int(np.argmin(scores))
            correct += int(pred == label)
        key = "raw_suffix" if mode is None else f"cfg{mode:g}"
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
