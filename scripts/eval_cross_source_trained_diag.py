"""Diagnostic variant of eval_cross_source_trained.py.

Adds a --no_prefix_suffix toggle so we can test whether forcing
`model._prefix_suffix_s2 = True` (the original hardcoded behavior) is what
destabilizes the OWN-encoder PPL, especially for the 0.4B model whose own_ppl
exploded to 822 while raw_ppl was ~54. Also reports per-sample own/cross PPL
distribution (median + p90) so a single pathological sample cannot silently
dominate the mean.

Original scripts/eval_cross_source_trained.py is left untouched.
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

import eval.diag_loop1_common as C
from scripts.eval.sample_prefix_suffix_cfg import encode_prefix


@torch.no_grad()
def ppl_with_states(model, ids, am, states):
    out = model.rwkv_model(input_ids=ids, attention_mask=am.bool(),
                           use_cache=True, return_dict=True)
    pkv = model.inject_into_cache(out.past_key_values, states)
    out2 = model.rwkv_model(input_ids=ids, attention_mask=am.bool(),
                            past_key_values=pkv, use_cache=True, return_dict=True)
    logits = out2.logits[:, :-1]
    tgt = ids[:, 1:]
    vocab = logits.shape[-1]
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab).float(),
                                             tgt.reshape(-1), reduction="mean")
    return float(torch.exp(loss).item())


@torch.no_grad()
def raw_ppl(model, ids, am):
    out = model.rwkv_model(input_ids=ids, attention_mask=am.bool(),
                           use_cache=True, return_dict=True)
    logits = out.logits[:, :-1]
    tgt = ids[:, 1:]
    vocab = logits.shape[-1]
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab).float(),
                                             tgt.reshape(-1), reduction="mean")
    return float(torch.exp(loss).item())


def stats(xs):
    a = np.array(xs, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "max": float("nan")}
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "max": float(a.max()),
    }


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="outputs_relay/test-v6-2.9B-s2-prefix-suffix-cfg/step_00150000")
    ap.add_argument("--trained_s1", default="outputs_relay/cross-source-s1-2.9B/cross_s1_final.pt")
    ap.add_argument("--latent_dir", default="preprocessed_data/owt_13b_s0_latents/train")
    ap.add_argument("--data_dir", default="preprocessed_data/owt_rwkv_tokens/train")
    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--out", default="outputs_eval/cross_source_diag.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_prefix_suffix", action="store_true",
                    help="Do NOT force model._prefix_suffix_s2=True (test the anomaly hypothesis)")
    args = ap.parse_args()

    model, tok, dtype, pad = C.build_model(args.ckpt_dir, args.device)
    model.eval()
    prefix_suffix = not args.no_prefix_suffix
    model._prefix_suffix_s2 = prefix_suffix

    trained = torch.load(args.trained_s1, map_location=args.device, weights_only=False)
    incompatible = model.load_state_dict(trained["trainable_state"], strict=False)
    loaded = len(trained["trainable_state"]) - len(
        [k for k in trained["trainable_state"] if k in incompatible.missing_keys])
    print(f"loaded trained S1: {loaded}/{len(trained['trainable_state'])} tensors, "
          f"step={trained.get('step')}, prefix_suffix_s2={prefix_suffix}", flush=True)

    latent_files = sorted(glob.glob(f"{args.latent_dir}/*.npy"))
    matched = []
    for lf in latent_files:
        stem = Path(lf).stem.replace("_tokens", "").replace("_latent", "")
        tf = Path(args.data_dir) / f"{stem}_tokens.npz"
        if tf.exists():
            matched.append((lf, str(tf)))
        if len(matched) >= args.num_samples:
            break
    print(f"matched {len(matched)} pairs", flush=True)

    raw_p, own_p, cross_p = [], [], []
    for i, (lf, tf) in enumerate(matched):
        d = np.load(tf)
        ids = torch.tensor([d["input_ids"][:512]], device=args.device, dtype=torch.long)
        am = torch.ones_like(ids, dtype=torch.float32)
        raw_p.append(raw_ppl(model, ids, am))
        z_own = encode_prefix(model, ids, am)
        own_p.append(ppl_with_states(model, ids, am, model.predict_states(z_own)))
        z_cross = torch.tensor(np.load(lf), device=args.device, dtype=dtype).unsqueeze(0)
        if z_cross.shape[-1] != 32:
            cross_p.append(float("nan")); continue
        z_flat = z_cross.mean(dim=1) if z_cross.dim() == 3 else z_cross
        cross_p.append(ppl_with_states(model, ids, am, model.predict_states(z_flat)))
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(matched)}] raw={np.nanmean(raw_p):.1f} "
                  f"own={np.nanmean(own_p):.1f} cross(trained)={np.nanmean(cross_p):.1f}", flush=True)

    res = {
        "ckpt": args.ckpt_dir,
        "trained_s1": args.trained_s1,
        "step": trained.get("step"),
        "prefix_suffix_s2": prefix_suffix,
        "num_samples": len(matched),
        "raw_ppl": float(np.nanmean(raw_p)),
        "own_ppl": float(np.nanmean(own_p)),
        "cross_trained_ppl": float(np.nanmean(cross_p)),
        "cross_over_own_ratio": float(np.nanmean(cross_p) / max(0.01, np.nanmean(own_p))),
        "raw_stats": stats(raw_p),
        "own_stats": stats(own_p),
        "cross_stats": stats(cross_p),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("\n=== TRAINED cross-source S1 (diag) ===")
    print(f"  prefix_suffix_s2: {prefix_suffix}")
    print(f"  raw:             {res['raw_ppl']:.2f}  (median {res['raw_stats']['median']:.1f}, p90 {res['raw_stats']['p90']:.1f})")
    print(f"  own (self):      {res['own_ppl']:.2f}  (median {res['own_stats']['median']:.1f}, p90 {res['own_stats']['p90']:.1f}, max {res['own_stats']['max']:.1f})")
    print(f"  cross (trained): {res['cross_trained_ppl']:.2f}  (median {res['cross_stats']['median']:.1f}, p90 {res['cross_stats']['p90']:.1f})")
    print(f"  cross/own ratio: {res['cross_over_own_ratio']:.3f}x")
    print(f"saved: {args.out}")
    print("DIAG_DONE", flush=True)


if __name__ == "__main__":
    main()
