"""Recurrent verification-cost wall benchmark (RWKV speculative white-space).

Motivation: on Transformers, verifying k candidate tokens (or a candidate TREE)
costs ~1 forward via a tree attention mask. On RWKV the recurrent state evolves
token-by-token and each tree branch needs its OWN state copy, so tree/multi-
candidate verification does NOT amortize. This script MEASURES that wall so we
can (a) draw the motivation figure and (b) test our unique escape hatch:
per-chunk INJECTED-state verification, where each future chunk has an
independent planned start state and can therefore be batched.

Three measurements (wall-clock, warmup + repeats, cuda synchronize):

  A. LINEAR verify cost vs depth k (k=1,2,4,8): one path of length k appended to
     a fixed prefix state. Expected: grows with k (serial state evolution).
  B. TREE verify cost vs #branches (separate forward per branch, as the existing
     tree eval does): expected to grow ~linearly in #branches (no amortization).
  C. CHUNK-PARALLEL verify via injected states: B independent chunks, each with
     its own predict_states start, verified as a BATCH in one forward. This is
     the escape hatch only our architecture has. Compare its cost to doing the
     same B chunks as B separate forwards.
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
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import eval.diag_loop1_common as Diag


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


@torch.no_grad()
def timed(fn, device, warmup=3, repeats=10):
    for _ in range(warmup):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) / repeats * 1000.0  # ms/call


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000")
    ap.add_argument("--token_dir", default="preprocessed_data/owt_rwkv_tokens/train")
    ap.add_argument("--out", default="outputs_eval/bench_recurrent_verify.json")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model, tok, dtype, pad = Diag.build_model(args.ckpt_dir, args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    rwkv = model.rwkv_model
    H, C = int(model.trajectory_horizon), int(model.trajectory_chunk_size)

    files = sorted(glob.glob(f"{args.token_dir}/*.npz"))
    d = np.load(files[0])
    ids_full = torch.tensor([d["input_ids"][:H * C]], device=args.device, dtype=torch.long)
    prefix_ids = ids_full[:, :128]

    # build a fixed prefix state once
    prefix_out = rwkv(input_ids=prefix_ids, use_cache=True, return_dict=True)
    results = {"ckpt": args.ckpt_dir, "H": H, "C": C, "repeats": args.repeats}

    # ── A. LINEAR verify cost vs candidate depth k ──
    print("=== A. Linear verify cost vs depth k (separate path, from prefix state) ===", flush=True)
    a = {}
    for k in [1, 2, 4, 8]:
        path = ids_full[:, 128:128 + k]

        def fn(path=path):
            pkv = prefix_out.past_key_values
            rwkv(input_ids=path, past_key_values=pkv, use_cache=False, return_dict=True)
        ms = timed(fn, args.device, repeats=args.repeats)
        a[f"k={k}"] = ms
        print(f"  k={k}: {ms:.2f} ms", flush=True)
    results["A_linear_vs_depth"] = a

    # ── B. TREE verify: N branches as N separate forwards (current approach) ──
    print("\n=== B. Tree verify: N branches, SEPARATE forward each (no amortization) ===", flush=True)
    b = {}
    for n_branch in [1, 2, 4, 8, 16]:
        branches = [ids_full[:, 128:130] for _ in range(n_branch)]  # depth-2 branches

        def fn(branches=branches):
            for br in branches:
                rwkv(input_ids=br, past_key_values=prefix_out.past_key_values,
                     use_cache=False, return_dict=True)
        ms = timed(fn, args.device, repeats=max(3, args.repeats // 2))
        b[f"branches={n_branch}"] = ms
        print(f"  branches={n_branch}: {ms:.2f} ms", flush=True)
    results["B_tree_separate_forwards"] = b

    # ── C. CHUNK-PARALLEL verify via injected states (our escape hatch) ──
    # For B chunks, each has an independent planned start state from predict_states.
    # Compare: (C1) B separate forwards vs (C2) one BATCHED forward of size B.
    print("\n=== C. Chunk-parallel verify: injected-state batch vs separate ===", flush=True)
    Z = model._encode_trajectory_chunks(ids_full, torch.ones_like(ids_full).bool())[0].reshape(1, H, -1)
    c = {}
    for B in [1, 2, 4, 8]:
        # gather B chunk token windows and their planned states
        chunk_tok = torch.stack([ids_full[0, h * C:(h + 1) * C] for h in range(B)], dim=0)  # [B, C]
        z_b = Z[0, :B, :]  # [B, D]
        states_b = model.predict_states(z_b)  # list per layer [B, heads, hd, hd]

        # C1: B separate forwards, each injecting its own state
        def fn_sep(chunk_tok=chunk_tok, states_b=states_b, B=B):
            for i in range(B):
                out = rwkv(input_ids=chunk_tok[i:i + 1], use_cache=True, return_dict=True)
                st_i = [s[i:i + 1] for s in states_b]
                pkv = model.inject_into_cache(out.past_key_values, st_i)
                rwkv(input_ids=chunk_tok[i:i + 1], past_key_values=pkv,
                     use_cache=False, return_dict=True)
        ms_sep = timed(fn_sep, args.device, repeats=max(3, args.repeats // 2))

        # C2: one batched forward of size B, inject all states at once
        def fn_batch(chunk_tok=chunk_tok, states_b=states_b):
            out = rwkv(input_ids=chunk_tok, use_cache=True, return_dict=True)
            pkv = model.inject_into_cache(out.past_key_values, states_b)
            rwkv(input_ids=chunk_tok, past_key_values=pkv, use_cache=False, return_dict=True)
        ms_batch = timed(fn_batch, args.device, repeats=max(3, args.repeats // 2))

        speedup = ms_sep / ms_batch if ms_batch > 0 else 0
        c[f"B={B}"] = {"separate_ms": ms_sep, "batched_ms": ms_batch, "batch_speedup": speedup}
        print(f"  B={B}: separate={ms_sep:.2f}ms  batched={ms_batch:.2f}ms  -> {speedup:.2f}x", flush=True)
    results["C_chunk_parallel_injected"] = c

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    # ── verdicts ──
    print("\n=== VERDICTS ===", flush=True)
    a1, a8 = a["k=1"], a["k=8"]
    print(f"A: depth 1->8 verify cost {a1:.2f}->{a8:.2f} ms ({a8/a1:.2f}x). "
          f"{'serial state cost grows with depth' if a8/a1 > 1.5 else 'near-flat'}", flush=True)
    b1, b16 = b["branches=1"], b["branches=16"]
    print(f"B: 1->16 branches {b1:.2f}->{b16:.2f} ms ({b16/b1:.2f}x). "
          f"{'NO amortization (the wall)' if b16/b1 > 4 else 'partial amortization'}", flush=True)
    best = max(v["batch_speedup"] for v in c.values())
    print(f"C: chunk-parallel injected-state batching best speedup = {best:.2f}x "
          f"{'-> escape hatch WORKS (batched << separate)' if best > 1.5 else '-> limited benefit'}", flush=True)
    print(f"\nsaved: {args.out}", flush=True)


if __name__ == "__main__":
    main()
