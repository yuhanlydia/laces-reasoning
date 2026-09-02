"""Cross-model single-z teaching eval on champion's generate-then-match tasks.

For each benchmark question: 13.3B S0 encodes the prompt into a global R^32 Z
(mean-pool the per-chunk latents), then the 0.4B backbone + trained cross-S1
projects that Z into WKV state via predict_states and generates the answer.
Writes champion-compatible JSONL for baseline/Cola-DLM/scripts/acc_calc.py.

Modes:
  raw    : 0.4B backbone, no injection (lower bound)
  cross  : 0.4B + cross-S1 reading 13.3B's encoded Z (teaching)
"""
import argparse, json, sys
from pathlib import Path
from typing import Any, Optional, cast
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from scripts.eval.relay_utils import load_relay_model
from scripts.eval.sample_prefix_suffix_cfg import (
    apply_repetition_penalty, apply_top_p, encode_prefix,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder_ckpt", required=True, help="13.3B S0/S2 ckpt that encodes Z")
    p.add_argument("--generator_ckpt", required=True, help="0.4B base ckpt")
    p.add_argument("--cross_s1", default=None, help="trained cross-S1 patch for generator")
    p.add_argument("--task_data_dir", default="baseline/Cola-DLM/generate_task_data")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--tasks", default="obqa,mmlu,race")
    p.add_argument("--mode", choices=["raw", "cross"], default="cross")
    p.add_argument("--max_samples", type=int, default=500)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--top_p", type=float, default=0.75)
    p.add_argument("--repetition_penalty", type=float, default=1.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def sample_next_token(logits, generated_ids, args):
    logits = apply_repetition_penalty(logits.float(), generated_ids, args.repetition_penalty)
    logits = logits / max(args.temperature, 1e-6)
    if args.top_k > 0:
        kth = torch.topk(logits, min(args.top_k, logits.numel())).values[-1]
        logits[logits < kth] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    probs = apply_top_p(probs, args.top_p)
    return int(torch.multinomial(probs, 1).item())


@torch.no_grad()
def generate(gen_model, tokenizer, prompt, states, args):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
    rwkv = gen_model.rwkv_model
    if states is not None:
        out = rwkv(input_ids=input_ids, use_cache=True, return_dict=True)
        cache = gen_model.inject_into_cache(out.past_key_values, states)
        out = rwkv(input_ids=input_ids, past_key_values=cache, use_cache=True, return_dict=True)
    else:
        out = rwkv(input_ids=input_ids, use_cache=True, return_dict=True)
    cache = out.past_key_values
    all_ids = list(input_ids[0].tolist())
    generated_ids: list[int] = []
    logits = out.logits[0, -1]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    for _ in range(args.max_new_tokens):
        nxt = sample_next_token(logits, all_ids, args)
        if eos_id is not None and nxt == eos_id:
            break
        generated_ids.append(nxt); all_ids.append(nxt)
        out = rwkv(input_ids=torch.tensor([[nxt]], device=args.device),
                   past_key_values=cache, use_cache=True, return_dict=True)
        cache = out.past_key_values
        logits = out.logits[0, -1]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def iter_jsonl(path, max_samples):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            line = line.strip()
            if line:
                yield i, json.loads(line)


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    gen_model, _, tokenizer, _, _ = load_relay_model(args.generator_ckpt, args.device)
    gen = cast(Any, gen_model); gen.eval()
    if args.mode == "cross":
        if getattr(gen, "trajectory_s1_mode", "independent") != "independent":
            gen.trajectory_s1_mode = "independent"
            gen.trajectory_state_decoder = None
        if args.cross_s1:
            sd = torch.load(args.cross_s1, map_location=args.device)
            trainable = sd.get("trainable_state", sd)
            miss, unexp = gen.load_state_dict(trainable, strict=False)
            print(f"cross-S1 loaded: applied={len(trainable)} unexpected={len(unexp)}", flush=True)
        enc_model, _, _, _, _ = load_relay_model(args.encoder_ckpt, args.device)
        enc = cast(Any, enc_model); enc.eval(); enc._prefix_suffix_s2 = True
    else:
        enc = None

    summary: dict[str, Any] = {"encoder": args.encoder_ckpt, "generator": args.generator_ckpt,
                               "cross_s1": args.cross_s1, "mode": args.mode, "tasks": {}}
    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        in_path = Path(args.task_data_dir) / f"{task}.jsonl"
        if not in_path.exists():
            print(f"[SKIP] missing {in_path}", flush=True); continue
        out_path = out_dir / f"{task}.jsonl"
        n = 0
        with open(out_path, "w", encoding="utf-8") as out_f:
            for i, item in iter_jsonl(in_path, args.max_samples):
                prompt = item["prompt"]
                states = None
                if args.mode == "cross":
                    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
                    am = torch.ones_like(ids, dtype=torch.float32)
                    z = encode_prefix(enc, ids, am)
                    if z.dim() == 3:
                        z = z.mean(dim=1)
                    z = z.to(dtype)
                    states = gen.predict_states(z)
                generated = generate(gen, tokenizer, prompt, states, args)
                rec = dict(item); rec["generate"] = generated
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                if n % 50 == 0:
                    print(f"{task}: {n}", flush=True)
        summary["tasks"][task] = {"samples": n, "output": str(out_path)}
        print(f"{task}: wrote {n} to {out_path}", flush=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
