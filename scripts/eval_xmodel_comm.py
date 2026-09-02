"""口径 A: cross-model latent COMMUNICATION eval (training-consistent).

Model A (source) encodes a chunk into a latent z; that z is injected as the
suffix-plan state into the frozen target model B (2.9B) via the trained
cross-model S1 (state_basis / state_scale / alpha_heads). We then measure how
well B continues the REAL suffix tokens (teacher-forced PPL on suffix positions
only) under three latent sources:

  raw   : no injection (frozen B baseline)
  own   : B encodes its own suffix -> z -> predict_states (self-source upper bound)
  cross : external source model's latent for the same chunk -> predict_states

Communication success = cross PPL approaches own PPL (cross/own ratio -> 1),
and both beat raw. This uses the SAME predict_states injection the model was
trained on (single-z prefix/suffix path), and scores ONLY suffix positions so
the prefix (which B always sees raw) does not dilute the signal.
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
from eval.sample_prefix_suffix_cfg import encode_prefix


def stats(xs):
    a = np.array(xs, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan")}
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
    }


@torch.no_grad()
def suffix_ppl(model, ids, am, split_idx, states=None):
    """Teacher-forced PPL on suffix positions only. states=None -> raw baseline."""
    out = model.rwkv_model(
        input_ids=ids, attention_mask=am.bool(), use_cache=True, return_dict=True
    )
    if states is not None:
        pkv = model.inject_into_cache(out.past_key_values, states)
        out = model.rwkv_model(
            input_ids=ids,
            attention_mask=am.bool(),
            past_key_values=pkv,
            use_cache=True,
            return_dict=True,
        )
    logits = out.logits[:, :-1]
    tgt = ids[:, 1:]
    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        tgt.reshape(-1),
        reduction="none",
    ).reshape(tgt.shape)
    # score only suffix positions (targets at index >= split_idx-1 predict suffix)
    pos = torch.arange(tgt.shape[1], device=ids.device).unsqueeze(0)
    suffix_sel = (pos >= (split_idx - 1)).float()
    denom = suffix_sel.sum().clamp(min=1)
    loss = (ce * suffix_sel).sum() / denom
    return float(torch.exp(loss).item())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt_dir",
        default="outputs_relay/test-v6-2.9B-s2-prefix-suffix-cfg/step_00150000",
    )
    ap.add_argument("--trained_s1", required=True, help="xmodel step ckpt model.pt")
    ap.add_argument("--latent_dir", required=True, help="external source latent dir")
    ap.add_argument(
        "--own_latent_dir",
        default=None,
        help="target's own source latent dir (optional; else use encode_prefix on suffix)",
    )
    ap.add_argument("--data_dir", default="preprocessed_data/owt_rwkv_tokens/train")
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--split_idx", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model, tok, dtype, pad = C.build_model(args.ckpt_dir, args.device)
    model.eval()
    model._prefix_suffix_s2 = True
    model._training_stage = 2

    trained = torch.load(args.trained_s1, map_location=args.device, weights_only=False)
    inc = model.load_state_dict(trained["trainable_state"], strict=False)
    loaded = len(trained["trainable_state"]) - len(
        [k for k in trained["trainable_state"] if k in inc.missing_keys]
    )
    print(
        f"loaded S1: {loaded}/{len(trained['trainable_state'])}, step={trained.get('step')}",
        flush=True,
    )

    latent_files = sorted(glob.glob(f"{args.latent_dir}/*_tokens.npy"))
    raw_p, own_p, cross_p = [], [], []
    n = 0
    for lf in latent_files:
        stem = Path(lf).stem.replace("_tokens", "")
        tf = Path(args.data_dir) / f"{stem}_tokens.npz"
        if not tf.exists():
            continue
        d = np.load(tf)
        ids = torch.tensor(
            np.asarray([d["input_ids"][:512]]), device=args.device, dtype=torch.long
        )
        am = torch.ones_like(ids, dtype=torch.float32)
        split = max(1, min(int(args.split_idx), ids.shape[1] - 1))

        # raw baseline (no injection)
        raw_p.append(suffix_ppl(model, ids, am, split, states=None))

        # own: target encodes its own suffix (training-consistent: pool suffix -> z)
        if args.own_latent_dir:
            own_lf = Path(args.own_latent_dir) / f"{stem}_tokens.npy"
            if own_lf.exists():
                z_own = torch.tensor(
                    np.load(str(own_lf)), device=args.device, dtype=dtype
                ).unsqueeze(0)
                z_own = z_own.mean(dim=1) if z_own.dim() == 3 else z_own
            else:
                own_p.append(float("nan"))
                z_own = None
        else:
            suffix_ids = ids[:, split:]
            suffix_am = torch.ones_like(suffix_ids, dtype=torch.float32)
            z_own = encode_prefix(model, suffix_ids, suffix_am)
        if z_own is not None:
            own_p.append(suffix_ppl(model, ids, am, split, model.predict_states(z_own)))

        # cross: external source latent -> inject through trained S1
        z_cross = torch.tensor(
            np.load(lf), device=args.device, dtype=dtype
        ).unsqueeze(0)
        if z_cross.shape[-1] != model.latent_dim:
            cross_p.append(float("nan"))
        else:
            z_flat = z_cross.mean(dim=1) if z_cross.dim() == 3 else z_cross
            cross_p.append(
                suffix_ppl(model, ids, am, split, model.predict_states(z_flat))
            )

        n += 1
        if n % 10 == 0:
            print(
                f"  [{n}/{args.num_samples}] raw={np.nanmean(raw_p):.1f} "
                f"own={np.nanmean(own_p):.1f} cross={np.nanmean(cross_p):.1f}",
                flush=True,
            )
        if n >= args.num_samples:
            break

    res = {
        "ckpt": args.ckpt_dir,
        "trained_s1": args.trained_s1,
        "step": trained.get("step"),
        "num_samples": n,
        "split_idx": args.split_idx,
        "raw_ppl": float(np.nanmean(raw_p)),
        "own_ppl": float(np.nanmean(own_p)),
        "cross_ppl": float(np.nanmean(cross_p)),
        "cross_over_own_ratio": float(
            np.nanmean(cross_p) / max(1e-6, np.nanmean(own_p))
        ),
        "cross_over_raw_ratio": float(
            np.nanmean(cross_p) / max(1e-6, np.nanmean(raw_p))
        ),
        "raw_stats": stats(raw_p),
        "own_stats": stats(own_p),
        "cross_stats": stats(cross_p),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("\n=== xmodel COMMUNICATION (suffix-only teacher-forced PPL) ===")
    print(f"  raw (no inject):  {res['raw_ppl']:.2f}")
    print(f"  own (self src):   {res['own_ppl']:.2f}")
    print(f"  cross (ext src):  {res['cross_ppl']:.2f}")
    print(f"  cross/own ratio:  {res['cross_over_own_ratio']:.3f}  (1.0 = perfect comm)")
    print(f"  cross/raw ratio:  {res['cross_over_raw_ratio']:.3f}  (<1.0 = comm helps)")
    print(f"saved: {args.out}")
    print("XCOMM_DONE", flush=True)


if __name__ == "__main__":
    main()
