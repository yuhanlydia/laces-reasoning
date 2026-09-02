"""Training-consistent cross-model single-z eval.

The old eval_cross_source_trained_diag.py used a trajectory-era path that does
not match how the single-z cross-model S1 was actually trained (it injected the
external latent into a whole-sequence predict_states, while training used the
prefix/suffix diffusion path with z_0_external as the suffix z_0). That mismatch
produced a spurious cross/own=6x.

This eval runs the SAME forward the model was trained on: prefix/suffix
diffusion with z_0_external, so cross (external source latent) and own (the
target model's own suffix encoding) are measured on the identical code path.
It reports the diffusion MSE (the trained objective) and per-sample stats so a
long-tail sample cannot dominate the mean.
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


def stats(xs):
    a = np.array(xs, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan")}
    return {"mean": float(a.mean()), "median": float(np.median(a)), "p90": float(np.percentile(a, 90))}


@torch.no_grad()
def diff_loss_for(model, ids, am, z_ext, device):
    model._prefix_suffix_s2 = True
    model._training_stage = 2
    split = int(ids.shape[1]) // 2
    _, eps_pred, eps_target, _ = model.forward_prefix_suffix_diffusion(
        ids, attention_mask=am, split_idx=split, z_0_external=z_ext
    )
    return float(torch.nn.functional.mse_loss(eps_pred.float(), eps_target.float()).item())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="outputs_relay/test-v6-2.9B-s2-prefix-suffix-cfg/step_00150000")
    ap.add_argument("--trained_s1", required=True)
    ap.add_argument("--latent_dir", required=True, help="external source latent dir")
    ap.add_argument("--own_latent_dir", required=True, help="target's own source latent dir")
    ap.add_argument("--data_dir", default="preprocessed_data/owt_rwkv_tokens/train")
    ap.add_argument("--num_samples", type=int, default=50)
    ap.add_argument("--out", default="outputs_eval/xmodel_singlez.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model, tok, dtype, pad = C.build_model(args.ckpt_dir, args.device)
    model.eval()
    trained = torch.load(args.trained_s1, map_location=args.device, weights_only=False)
    inc = model.load_state_dict(trained["trainable_state"], strict=False)
    loaded = len(trained["trainable_state"]) - len([k for k in trained["trainable_state"] if k in inc.missing_keys])
    print(f"loaded S1: {loaded}/{len(trained['trainable_state'])}, step={trained.get('step')}", flush=True)

    latent_files = sorted(glob.glob(f"{args.latent_dir}/*_tokens.npy"))
    cross_d, own_d = [], []
    n = 0
    for lf in latent_files:
        stem = Path(lf).stem.replace("_tokens", "")
        tf = Path(args.data_dir) / f"{stem}_tokens.npz"
        own_lf = Path(args.own_latent_dir) / f"{stem}_tokens.npy"
        if not (tf.exists() and own_lf.exists()):
            continue
        d = np.load(tf)
        ids = torch.tensor(np.asarray([d["input_ids"][:512]]), device=args.device, dtype=torch.long)
        am = torch.ones_like(ids, dtype=torch.float32)
        z_cross = torch.tensor(np.load(lf), device=args.device, dtype=dtype).unsqueeze(0)
        z_own = torch.tensor(np.load(str(own_lf)), device=args.device, dtype=dtype).unsqueeze(0)
        cross_d.append(diff_loss_for(model, ids, am, z_cross, args.device))
        own_d.append(diff_loss_for(model, ids, am, z_own, args.device))
        n += 1
        if n % 10 == 0:
            print(f"  [{n}/{args.num_samples}] cross_diff={np.nanmean(cross_d):.3f} own_diff={np.nanmean(own_d):.3f}", flush=True)
        if n >= args.num_samples:
            break

    res = {
        "ckpt": args.ckpt_dir, "trained_s1": args.trained_s1, "step": trained.get("step"),
        "num_samples": n,
        "cross_diff_mean": float(np.nanmean(cross_d)), "own_diff_mean": float(np.nanmean(own_d)),
        "cross_over_own_diff": float(np.nanmean(cross_d) / max(1e-6, np.nanmean(own_d))),
        "cross_stats": stats(cross_d), "own_stats": stats(own_d),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n=== xmodel single-z (diffusion MSE, training-consistent) ===")
    print(f"  own_diff (self):  {res['own_diff_mean']:.4f} (median {res['own_stats']['median']:.4f})")
    print(f"  cross_diff:       {res['cross_diff_mean']:.4f} (median {res['cross_stats']['median']:.4f})")
    print(f"  cross/own diff:   {res['cross_over_own_diff']:.3f}")
    print(f"saved: {args.out}")
    print("XEVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
