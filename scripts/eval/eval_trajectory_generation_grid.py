#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.eval.relay_utils import load_relay_model


def parse_csv(value, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def apply_repetition_penalty(logits, generated_ids, penalty):
    if penalty == 1.0 or not generated_ids:
        return logits
    token_ids = torch.tensor(list(set(generated_ids)), device=logits.device, dtype=torch.long)
    token_ids = token_ids.clamp(0, logits.size(-1) - 1)
    scores = logits[token_ids]
    logits = logits.clone()
    logits[token_ids] = torch.where(scores < 0, scores * penalty, scores / penalty)
    return logits


def apply_top_p(probs, top_p):
    if top_p <= 0.0 or top_p >= 1.0:
        return probs
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove = cumulative > top_p
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
    filtered = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
    return filtered / filtered.sum().clamp(min=1e-12)


@torch.no_grad()
def sample_next_token(logits, generated_ids, temperature, top_k, top_p, repetition_penalty):
    logits = apply_repetition_penalty(logits.float(), generated_ids, repetition_penalty)
    logits = logits / max(float(temperature), 1e-6)
    probs = torch.softmax(logits, dim=-1)
    if top_k > 0:
        topk_vals, topk_idx = torch.topk(probs, int(top_k))
        probs = torch.zeros_like(probs).scatter(-1, topk_idx, topk_vals)
        probs = probs / probs.sum().clamp(min=1e-12)
    probs = apply_top_p(probs, top_p)
    return int(torch.multinomial(probs, 1).item())


@torch.no_grad()
def sample_text(model, tokenizer, args, temperature, repetition_penalty, seed):
    torch.manual_seed(seed)
    device = args.device
    dtype = torch.bfloat16
    input_ids = torch.zeros(1, 1, dtype=torch.long, device=device)
    out = model.rwkv_model(input_ids=input_ids, use_cache=True, return_dict=True)
    past_kv = out.past_key_values
    z_traj = model.trajectory_sample(
        1, num_steps=args.sample_steps, device=device, dtype=dtype, sampler=args.trajectory_sampler
    )
    z_norm = float(z_traj.norm(dim=-1).mean().item())
    s1_mode = str(getattr(model, "trajectory_s1_mode", model.config.get("trajectory_s1_mode", "independent")))
    if s1_mode in ("transformer", "rwkv", "birwkv"):
        layer_states = model.predict_trajectory_states(z_traj)
        blend = float(model.config.get("trajectory_state_blend", getattr(model, "trajectory_state_blend", 1.0)))
    else:
        layer_states = None
        blend = 1.0
    generated = list(input_ids[0].tolist())
    logits = out.logits[0, -1]
    chunk_size = int(model.trajectory_chunk_size)
    max_chunks = min(z_traj.shape[1], max(1, (args.max_new_tokens + chunk_size - 1) // chunk_size))
    for h in range(max_chunks):
        if s1_mode in ("transformer", "rwkv", "birwkv"):
            states_h = [layer_state[:, h] for layer_state in layer_states]
            past_kv = model.blend_into_cache(past_kv, states_h, blend)
        else:
            states_h = model.predict_states(z_traj[:, h])
            past_kv = model.inject_into_cache(past_kv, states_h)
        for _ in range(chunk_size):
            if len(generated) - 1 >= args.max_new_tokens:
                break
            next_id = sample_next_token(logits, generated, temperature, args.top_k, args.top_p, repetition_penalty)
            generated.append(next_id)
            out = model.rwkv_model(
                input_ids=torch.tensor([[next_id]], device=device),
                past_key_values=past_kv,
                use_cache=True,
                return_dict=True,
            )
            past_kv = out.past_key_values
            logits = out.logits[0, -1]
    return tokenizer.decode(generated, skip_special_tokens=True), z_norm


def repetition_rate(texts, n):
    repeated = 0
    total = 0
    for text in texts:
        words = text.split()
        grams = [tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1))]
        total += len(grams)
        repeated += len(grams) - len(set(grams))
    return repeated / max(1, total)


@torch.no_grad()
def reference_ppl(rwkv_model, tokenizer, texts, device):
    total_nll = 0.0
    total_tok = 0
    for text in texts:
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        if ids.shape[1] < 2:
            continue
        out = rwkv_model(input_ids=ids, return_dict=True)
        logits = out.logits[:, :-1].float()
        labels = ids[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum")
        total_nll += float(loss.item())
        total_tok += int(labels.numel())
    avg_nll = total_nll / max(1, total_tok)
    return math.exp(min(avg_nll, 100.0)), avg_nll, total_tok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--texts_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample_steps", type=int, default=1000)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperatures", default="0.5,0.55,0.6")
    parser.add_argument("--repetition_penalties", default="1.2,1.3,1.35")
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--top_p", type=float, default=0.7)
    parser.add_argument("--trajectory_s1_mode", choices=("independent", "transformer", "rwkv", "birwkv"), default="transformer")
    # default=None preserves the ckpt's trained blend; a fixed default silently mismatches it.
    parser.add_argument("--trajectory_state_blend", type=float, default=None)
    parser.add_argument("--trajectory_sampler", choices=("rf_heun",), default=None)
    args = parser.parse_args()

    model, rwkv, tokenizer, ckpt, cfg = load_relay_model(args.ckpt_dir, args.device)
    model.config.trajectory_s1_mode = args.trajectory_s1_mode
    model.trajectory_s1_mode = args.trajectory_s1_mode
    if args.trajectory_state_blend is not None:
        model.config.trajectory_state_blend = float(args.trajectory_state_blend)
        model.trajectory_state_blend = float(args.trajectory_state_blend)
    effective_blend = float(getattr(model, "trajectory_state_blend", model.config.get("trajectory_state_blend", 1.0)))
    print(f"[eval] trajectory_state_blend = {effective_blend} "
          f"({'CLI override' if args.trajectory_state_blend is not None else 'from checkpoint/config'})")
    Path(args.texts_dir).mkdir(parents=True, exist_ok=True)
    results = {
        "ckpt_dir": args.ckpt_dir,
        "checkpoint_step": ckpt.get("step"),
        "checkpoint_gen_type": cfg.training.get("gen_type"),
        "sample_steps": args.sample_steps,
        "max_new_tokens": args.max_new_tokens,
        "trajectory_sampler": args.trajectory_sampler,
        "trajectory_s1_mode": args.trajectory_s1_mode,
        "trajectory_state_blend": effective_blend,
        "trajectory_state_blend_source": "cli" if args.trajectory_state_blend is not None else "checkpoint",
        "configs": {},
    }
    for temperature in parse_csv(args.temperatures, float):
        for penalty in parse_csv(args.repetition_penalties, float):
            key = f"t{temperature:g}_pen{penalty:g}"
            texts = []
            z_norms = []
            t0 = time.time()
            for seed in parse_csv(args.seeds, int):
                text, z_norm = sample_text(model, tokenizer, args, temperature, penalty, seed)
                texts.append(text)
                z_norms.append(z_norm)
            text_path = os.path.join(args.texts_dir, f"{key}.txt")
            with open(text_path, "w") as f:
                for text in texts:
                    f.write(text.replace("\n", "\\n") + "\n")
            ppl, avg_nll, tokens = reference_ppl(rwkv, tokenizer, texts, args.device)
            results["configs"][key] = {
                "temperature": temperature,
                "repetition_penalty": penalty,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "seeds": parse_csv(args.seeds, int),
                "texts": text_path,
                "ref_ppl_rwkv": ppl,
                "ref_avg_nll_rwkv": avg_nll,
                "ref_tokens": tokens,
                "repeat_4": repetition_rate(texts, 4),
                "repeat_8": repetition_rate(texts, 8),
                "z_norm_mean": sum(z_norms) / max(1, len(z_norms)),
                "z_norms": z_norms,
                "elapsed_s": round(time.time() - t0, 1),
            }
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(json.dumps({key: results["configs"][key]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
