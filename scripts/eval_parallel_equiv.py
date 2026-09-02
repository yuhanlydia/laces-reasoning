"""blend=1 parallel-decoding equivalence test.

Question: at blend=1 (recurrent_state fully overwritten by the planned per-chunk
state, no mixing with the previous chunk's cache), is the SERIAL chunk rollout
numerically equal to a PARALLEL independent rollout where each chunk starts ONLY
from its own planned state?

If yes, blend=1 truly decouples chunks -> parallel decoding is valid.
If no, there is residual cross-chunk dependence even at blend=1.

SERIAL:   for h: cache = blend_into_cache(prev_cache, states_h, blend=1);
          out = rwkv(chunk_h, past_key_values=cache); prev_cache = out.cache
PARALLEL: for each h independently: fresh cache injected with states_h ONLY,
          then rwkv(chunk_h) -- no information from other chunks.

We compare the per-chunk teacher-forced logits (max abs diff, cosine).
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import eval.diag_loop1_common as Diag


@torch.no_grad()
def serial_rollout(model, chunks, layer_states, H, blend):
    """chunks: [H, C]; returns per-chunk logits [H, C, V] via serial cache chain."""
    rwkv = model.rwkv_model
    # seed with an empty prefix forward (single BOS-like) to obtain a cache object
    seed = chunks[0:1, 0:1]  # [1,1]
    out0 = rwkv(input_ids=seed, use_cache=True, return_dict=True)
    cache = out0.past_key_values
    logits_by_chunk = []
    for h in range(H):
        states_h = [ls[:, h] for ls in layer_states]  # each [1, heads, hd, hd]
        cache = model.blend_into_cache(cache, states_h, blend)
        out_h = rwkv(input_ids=chunks[h:h + 1], past_key_values=cache,
                     use_cache=True, return_dict=True)
        cache = out_h.past_key_values
        logits_by_chunk.append(out_h.logits[0])  # [C, V]
    return torch.stack(logits_by_chunk, dim=0)  # [H, C, V]


@torch.no_grad()
def parallel_rollout(model, chunks, layer_states, H):
    """Each chunk independently: fresh cache injected ONLY with its planned state.
    Batched into one forward of size H. Returns [H, C, V]."""
    rwkv = model.rwkv_model
    C = chunks.shape[1]
    # one batched forward to get a fresh cache of batch H, then inject all states
    out = rwkv(input_ids=chunks, use_cache=True, return_dict=True)  # [H, C, V] cache batch H
    states_all = [ls[0] for ls in layer_states]  # each [H, heads, hd, hd]  (drop batch dim)
    cache = model.inject_into_cache(out.past_key_values, states_all)
    out2 = rwkv(input_ids=chunks, past_key_values=cache, use_cache=False, return_dict=True)
    return out2.logits  # [H, C, V]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000")
    ap.add_argument("--token_dir", default="preprocessed_data/owt_rwkv_tokens/train")
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--out", default="outputs_eval/parallel_equiv.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model, tok, dtype, pad = Diag.build_model(args.ckpt_dir, args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    H, C = int(model.trajectory_horizon), int(model.trajectory_chunk_size)
    S = H * C

    files = sorted(glob.glob(f"{args.token_dir}/*.npz"))
    np.random.seed(42)
    idxs = np.random.choice(len(files), args.num_samples, replace=False)

    max_abs_diffs, cosines, argmax_agrees = [], [], []
    for si, idx in enumerate(idxs):
        d = np.load(files[idx])
        ids_np = d["input_ids"][:S]
        if len(ids_np) < S:
            continue
        ids = torch.tensor([ids_np], device=args.device, dtype=torch.long)
        am = torch.ones_like(ids, dtype=torch.float32)

        # clean-Z (isolate the equivalence question from sampling noise)
        Z = model._encode_trajectory_chunks(ids, am.bool())[0].reshape(1, H, -1)
        layer_states = model.predict_trajectory_states(Z)  # list per layer [1, H, heads, hd, hd]
        chunks = ids[0].reshape(H, C)  # [H, C]

        L_serial = serial_rollout(model, chunks, layer_states, H, blend=1.0)  # [H,C,V]
        L_parallel = parallel_rollout(model, chunks, layer_states, H)          # [H,C,V]

        diff = (L_serial.float() - L_parallel.float()).abs()
        max_abs = diff.max().item()
        cs = torch.nn.functional.cosine_similarity(
            L_serial.float().reshape(-1), L_parallel.float().reshape(-1), dim=0).item()
        agree = (L_serial.argmax(-1) == L_parallel.argmax(-1)).float().mean().item()
        max_abs_diffs.append(max_abs)
        cosines.append(cs)
        argmax_agrees.append(agree)
        if si % 2 == 0:
            print(f"[{si}/{len(idxs)}] max_abs_diff={max_abs:.4f} cos={cs:.5f} argmax_agree={agree:.4f}", flush=True)

    res = {
        "ckpt": args.ckpt_dir,
        "blend": 1.0,
        "num_samples": int(args.num_samples),
        "mean_max_abs_diff": float(np.mean(max_abs_diffs)),
        "mean_cosine": float(np.mean(cosines)),
        "mean_argmax_agree": float(np.mean(argmax_agrees)),
        "per_sample_argmax_agree": argmax_agrees,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)

    print("\n=== blend=1 serial-vs-parallel equivalence ===")
    print(f"  mean max abs logit diff : {res['mean_max_abs_diff']:.4f}")
    print(f"  mean cosine             : {res['mean_cosine']:.5f}")
    print(f"  mean argmax agreement   : {res['mean_argmax_agree']:.4f}")
    a = res["mean_argmax_agree"]
    verdict = ("DECOUPLED -> parallel valid" if a > 0.98 else
               ("mostly decoupled (minor residual)" if a > 0.90 else
                "NOT decoupled -> residual cross-chunk dependence"))
    print(f"  verdict: {verdict}")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
