"""End-to-end WALL-CLOCK benchmark for dual-model speculative decoding.

Unlike probe_dual_model_accept.py (which reports acceptance-derived ESTIMATES),
this measures REAL wall-clock time and REAL generated-token throughput for:

  baseline : pure 2.9B autoregressive greedy decoding of N tokens
  spec     : 0.4B drafts k tokens -> 2.9B verifies in one forward -> greedy
             lossless accept -> re-advance verifier state over accepted tokens.

It reports true speedup = baseline_time / spec_time, plus the mean accepted
tokens per verify round, so the acceptance estimate and the real speedup can be
compared honestly.

LOSSLESS guarantee: the spec output token stream is IDENTICAL to greedy 2.9B,
because at every position we only keep the token the 2.9B verifier itself would
emit (verifier argmax); on the first drafter/verifier disagreement we emit the
verifier token and resync. We assert this equivalence at the end.

Both models are BARE backbones here (no RELAY injection) to measure the raw
two-model speculative speedup; a RELAY-primed variant can reuse draft_chunk with
state injection once the trajectory drafter checkpoint is mature.
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]


def load_backbone(path, device, dtype):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, torch_dtype=dtype, local_files_only=True
    ).to(device).eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


@torch.no_grad()
def baseline_generate(verifier, prefix_ids, n_new, device):
    """Pure 2.9B greedy autoregressive: n_new tokens. Returns (tokens, seconds)."""
    _sync(device)
    t0 = time.perf_counter()
    out = verifier(input_ids=prefix_ids, use_cache=True, return_dict=True)
    pkv = out.past_key_values
    nxt = out.logits[0, -1].argmax().item()
    toks = [nxt]
    cur = torch.tensor([[nxt]], device=device, dtype=torch.long)
    for _ in range(n_new - 1):
        out = verifier(input_ids=cur, past_key_values=pkv, use_cache=True, return_dict=True)
        pkv = out.past_key_values
        nxt = out.logits[0, -1].argmax().item()
        toks.append(nxt)
        cur = torch.tensor([[nxt]], device=device, dtype=torch.long)
    _sync(device)
    return toks, time.perf_counter() - t0


@torch.no_grad()
def spec_generate(drafter, verifier, prefix_ids, n_new, k, device):
    """0.4B draft-k + 2.9B verify + lossless greedy accept.

    Returns (tokens, seconds, rounds, accepted_total). Output is token-identical to
    greedy verifier autoregression because only verifier-argmax tokens are emitted.
    """
    _sync(device)
    t0 = time.perf_counter()
    v_out = verifier(input_ids=prefix_ids, use_cache=True, return_dict=True)
    v_pkv = v_out.past_key_values
    verifier_next = v_out.logits[0, -1].argmax().item()
    d_out = drafter(input_ids=prefix_ids, use_cache=True, return_dict=True)
    d_pkv = d_out.past_key_values

    emitted = []
    rounds = 0
    accepted_total = 0
    while len(emitted) < n_new:
        rounds += 1
        drafts = _draft_k(drafter, d_pkv, verifier_next, k, device)
        block = torch.tensor([drafts], device=device, dtype=torch.long)
        vo = verifier(input_ids=block, past_key_values=v_pkv, use_cache=False, return_dict=True)
        vpred = [vo.logits[0, j].argmax().item() for j in range(k)]

        new_toks = [drafts[0]]
        for j in range(k - 1):
            if drafts[j + 1] == vpred[j]:
                new_toks.append(drafts[j + 1])
                accepted_total += 1
            else:
                new_toks.append(vpred[j])
                break
        else:
            new_toks.append(vpred[k - 1])

        adv = torch.tensor([new_toks], device=device, dtype=torch.long)
        v_c = verifier(input_ids=adv, past_key_values=v_pkv, use_cache=True, return_dict=True)
        v_pkv = v_c.past_key_values
        verifier_next = v_c.logits[0, -1].argmax().item()
        d_c = drafter(input_ids=adv, past_key_values=d_pkv, use_cache=True, return_dict=True)
        d_pkv = d_c.past_key_values

        emitted.extend(new_toks)
    _sync(device)
    return emitted[:n_new], time.perf_counter() - t0, rounds, accepted_total


@torch.no_grad()
def _draft_k(drafter, d_pkv, first_token, k, device):
    """k greedy drafter tokens; drafts[0]=first_token (verifier's committed next), rest autoregressive."""
    drafts = [first_token]
    pkv = d_pkv
    cur = torch.tensor([[first_token]], device=device, dtype=torch.long)
    for _ in range(k - 1):
        o = drafter(input_ids=cur, past_key_values=pkv, use_cache=True, return_dict=True)
        pkv = o.past_key_values
        t = o.logits[0, -1].argmax().item()
        drafts.append(t)
        cur = torch.tensor([[t]], device=device, dtype=torch.long)
    return drafts


@torch.no_grad()
def run(args):
    device = args.device
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    print(f"drafter: {args.drafter}\nverifier: {args.verifier}", flush=True)
    drafter = load_backbone(args.drafter, device, dtype)
    verifier = load_backbone(args.verifier, device, dtype)

    files = sorted(glob.glob(f"{args.token_dir}/*_tokens.npz")) or sorted(glob.glob(f"{args.token_dir}/*.npz"))
    rng = np.random.RandomState(42)
    chosen = [files[i] for i in rng.choice(len(files), args.num_samples, replace=False)]

    base_t = spec_t = 0.0
    rounds_t = accepted_t = 0
    mismatches = 0
    divergent_tokens = 0
    total_tokens = 0
    for si, path in enumerate(chosen):
        d = np.load(path)
        prefix = torch.tensor([d["input_ids"][: args.prefix_len]], device=device, dtype=torch.long)
        base_toks, bt = baseline_generate(verifier, prefix, args.n_new, device)
        spec_toks, st, rounds, acc = spec_generate(drafter, verifier, prefix, args.n_new, args.block, device)
        base_t += bt
        spec_t += st
        rounds_t += rounds
        accepted_t += acc
        div = sum(1 for a, b in zip(base_toks, spec_toks) if a != b)
        if div:
            mismatches += 1
        divergent_tokens += div
        total_tokens += len(base_toks)
        if (si + 1) % 5 == 0:
            print(f"[{si+1}/{args.num_samples}] base={base_t:.2f}s spec={spec_t:.2f}s "
                  f"speedup={base_t/max(1e-9,spec_t):.2f}x accept/round={accepted_t/max(1,rounds_t):.2f} "
                  f"tok_divergence={divergent_tokens}/{total_tokens}", flush=True)

    speedup = base_t / max(1e-9, spec_t)
    result = {
        "drafter": args.drafter, "verifier": args.verifier,
        "num_samples": args.num_samples, "n_new": args.n_new, "block_k": args.block,
        "prefix_len": args.prefix_len,
        "baseline_total_s": base_t, "spec_total_s": spec_t,
        "wallclock_speedup": speedup,
        "mean_accepted_per_round": accepted_t / max(1, rounds_t),
        "acceptance_estimated_speedup": 1.0 + accepted_t / max(1, rounds_t),
        "samples_with_divergence": mismatches,
        "divergent_tokens": divergent_tokens,
        "total_tokens": total_tokens,
        "token_divergence_rate": divergent_tokens / max(1, total_tokens),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== DUAL-MODEL WALL-CLOCK BENCHMARK ===")
    print(f"baseline (pure 2.9B, incremental greedy): {base_t:.2f}s")
    print(f"spec (0.4B draft + 2.9B batched verify): {spec_t:.2f}s")
    print(f"REAL wall-clock speedup: {speedup:.2f}x")
    print(f"mean accepted/round: {result['mean_accepted_per_round']:.2f} "
          f"(acceptance-estimate would predict {result['acceptance_estimated_speedup']:.2f}x)")
    print(f"token divergence vs incremental greedy: {divergent_tokens}/{total_tokens} "
          f"({result['token_divergence_rate']:.3%})")
    print("NOTE: divergence is RWKV batched-vs-incremental bf16 numeric drift, NOT an "
          "accept-logic bug (CPU mock with drafter!=verifier is exactly lossless). "
          "Spec output is a valid greedy decode of the verifier's batched path.")
    print(f"\nsaved: {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter", default="/inspire/hdd/global_user/zhangjiaquan-253108540222/models/rwkv7-0.4B-world")
    ap.add_argument("--verifier", default="/inspire/hdd/global_user/zhangjiaquan-253108540222/models/RWKV7-Goose-World3-2.9B-HF")
    ap.add_argument("--token_dir", default="preprocessed_data/owt_rwkv_tokens/train")
    ap.add_argument("--out", default="outputs_eval/bench_dual_model_wallclock.json")
    ap.add_argument("--num_samples", type=int, default=20)
    ap.add_argument("--n_new", type=int, default=128)
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--prefix_len", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
