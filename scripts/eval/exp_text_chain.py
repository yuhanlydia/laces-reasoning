#!/usr/bin/env python3
"""
Text-based chain experiment: each agent sees all previous text responses.

Agent 1: shard 1 (512 tokens) → response R1
Agent 2: shard 2 (512 tokens) + R1 → response R2
Agent 3: shard 3 (512 tokens) + R1 + R2 → response R3
...
Agent M: shard M (512 tokens) + R1 + ... + R_{M-1} → response R_M

Compare with dual (latent-based chain).
"""
import argparse
import json
import sys
import time
import torch
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.eval.relay_utils import load_relay_model
from scripts.eval.exp_sharded_babilong import (
    prefill, decode_plain, word_boundary_hit, Timer,
    encode_prefix, sample_trajectory_cfg
)
from scripts.eval.run_cola_dlm_tasks_prefix_suffix_trajectory_cfg import sample_next_token
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="outputs_relay/C-LDLM-4096-coadapt-8h200-b5-rnn-20260708/step_00022500")
    parser.add_argument("--babilong_length", type=str, default="16k")
    parser.add_argument("--tasks", type=str, default="qa1")
    parser.add_argument("--num_agents", type=int, default=32)
    parser.add_argument("--local_tokens", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16

    print(f"Loading model from {args.ckpt_dir}...")
    model, rwkv, tokenizer, ckpt, cfg = load_relay_model(args.ckpt_dir, device=str(device))
    model.eval()

    M = args.num_agents
    L = args.local_tokens

    results = {}
    for task in args.tasks.split(","):
        print(f"\n=== Task: {task} ===")
        ds = load_dataset("RMT-team/babilong", args.babilong_length,
                          split=f"{task}[:{args.max_samples}]")
        print(f"[{task}] {len(ds)} samples @ {args.babilong_length}", flush=True)

        dec_args = SimpleNamespace(
            device=str(device),
            max_new_tokens=args.max_new_tokens,
            temperature=0.7,
            top_k=10,
            top_p=0.75,
            repetition_penalty=1.3,
        )

        correct_text_chain = 0
        correct_dual = 0
        time_text_chain = 0.0
        time_dual = 0.0

        for i, r in enumerate(ds):
            ctx_text = str(r["input"]).strip()
            q_text = str(r["question"]).strip()
            gold = str(r["target"]).strip()

            ctx_ids = tokenizer(ctx_text, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)
            ctx_ids = ctx_ids[: M * L]
            shards = [ctx_ids[j * L:(j + 1) * L] for j in range(M)]
            shards = [s for s in shards if s.numel() > 0]

            q_ids = tokenizer(f"Question: {q_text}\nAnswer:", return_tensors="pt",
                              add_special_tokens=False).input_ids.to(device)

            # === Method 1: Text-based chain ===
            torch.cuda.reset_peak_memory_stats(device)
            t0 = time.time()

            past = None
            responses = []
            for j in range(len(shards)):
                agent_input = shards[j]
                if responses:
                    prev_text = " ".join(responses)
                    prev_ids = tokenizer(prev_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)
                    agent_input = torch.cat([prev_ids, shards[j]])

                out = prefill(model, agent_input.unsqueeze(0), past=past)
                past = out.past_key_values

                if j < len(shards) - 1:
                    logits = out.logits[0, -1]
                    new_ids = []
                    all_ids_for_decode = list(shards[j].tolist())
                    for _ in range(dec_args.max_new_tokens):
                        nid = sample_next_token(logits, all_ids_for_decode, dec_args)
                        eos_id = getattr(tokenizer, "eos_token_id", None)
                        if eos_id is not None and nid == eos_id:
                            break
                        new_ids.append(nid)
                        all_ids_for_decode.append(nid)
                        out_next = model.rwkv_model(input_ids=torch.tensor([[nid]], device=device),
                                                    past_key_values=past, use_cache=True, return_dict=True)
                        past = out_next.past_key_values
                        logits = out_next.logits[0, -1]
                    response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                    responses.append(response)
                else:
                    out = prefill(model, q_ids, past=past)
                    past = out.past_key_values
                    new_ids = decode_plain(model, tokenizer, past, list(q_ids[0].tolist()), dec_args)
                    response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

            t1 = time.time()
            text_chain_ms = (t1 - t0) * 1000
            time_text_chain += text_chain_ms

            hit = word_boundary_hit(response, gold.lower())
            correct_text_chain += int(hit)

            # === Method 2: Dual (latent-based chain) ===
            torch.cuda.reset_peak_memory_stats(device)
            t0 = time.time()

            all_z = []
            for s in shards:
                z = encode_prefix(model, s.unsqueeze(0), torch.ones(1, s.numel(), dtype=torch.long, device=device))[0].to(dtype)
                all_z.append(z[0].float())

            z_mean = torch.stack(all_z, dim=0).mean(dim=0, keepdim=True).to(dtype)
            w = 1.0 / len(all_z)
            resid = [z - z_mean for z in all_z]
            cond = z_mean + sum((w * r for r in resid), torch.zeros_like(z_mean))

            H = 16
            Z = sample_trajectory_cfg(model, cond, args.steps, args.cfg_scale, device, dtype)
            plan_states = [model.predict_states(Z[:, h]) for h in range(H)]

            dummy = prefill(model, q_ids[:, :1])
            past_dual = model.inject_into_cache(dummy.past_key_values, plan_states[0])
            for s in shards:
                past_dual = prefill(model, s.unsqueeze(0), past=past_dual).past_key_values

            new_ids = decode_plain(model, tokenizer, past_dual, list(q_ids[0].tolist()), dec_args)
            response_dual = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

            t1 = time.time()
            dual_ms = (t1 - t0) * 1000
            time_dual += dual_ms

            hit_dual = word_boundary_hit(response_dual, gold.lower())
            correct_dual += int(hit_dual)

            if i % 10 == 0:
                print(f"  [{i}/{len(ds)}] text_chain: {correct_text_chain}/{i+1} ({time_text_chain/(i+1):.0f}ms) | dual: {correct_dual}/{i+1} ({time_dual/(i+1):.0f}ms)")

        n = len(ds)
        acc_text_chain = correct_text_chain / n
        acc_dual = correct_dual / n
        avg_time_text_chain = time_text_chain / n
        avg_time_dual = time_dual / n

        print(f"\n=== Summary ===")
        print(f"Text-based chain: acc={acc_text_chain:.1%}, avg_time={avg_time_text_chain:.0f}ms")
        print(f"Dual (latent):    acc={acc_dual:.1%}, avg_time={avg_time_dual:.0f}ms")

        results[task] = {
            "text_chain": {"accuracy": acc_text_chain, "avg_time_ms": avg_time_text_chain},
            "dual": {"accuracy": acc_dual, "avg_time_ms": avg_time_dual},
            "n_samples": n,
        }

    output = {
        "config": vars(args),
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
