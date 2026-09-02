"""Quick downstream probe: baseline vs LaDiR on champion trajectory checkpoint.

Measures effective rank and teacher-forced CE on N validation samples.
Outputs per-sample comparison for statistical analysis.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/eval/eval_ladire_downstream.py \
    --ckpt_dir outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000 \
    --num_samples 100 --steps 20 --cfg_scale 3.0
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import eval.diag_loop1_common as Diag
from scripts.eval.sample_prefix_suffix_trajectory_cfg import encode_prefix
from scripts.eval.sample_trajectory_ladire import (
    compute_chunk_ce,
    effective_rank,
    sample_trajectory_ddim_ladire,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt_dir",
        default="outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000",
    )
    ap.add_argument("--token_dir", default="preprocessed_data/owt_rwkv_tokens/train")
    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg_scale", type=float, default=3.0)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--blend", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs_eval/ladire_downstream.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model, _tok, dtype, _pad = Diag.build_model(args.ckpt_dir, args.device)
    model.eval()

    H, C = int(model.trajectory_horizon), int(model.trajectory_chunk_size)
    S = H * C
    prefix_len = (H // 2) * C
    files = sorted(glob.glob(f"{args.token_dir}/*.npz"))
    np.random.seed(args.seed)

    results = []
    for i in range(args.num_samples):
        f = files[np.random.randint(len(files))]
        d = np.load(f)
        ids_np = d["input_ids"][:S]
        if len(ids_np) < S:
            continue

        ids = torch.tensor([ids_np], device=args.device, dtype=torch.long)
        am = torch.ones_like(ids, dtype=torch.float32)
        pre_ids = ids[:, :prefix_len]
        pre_am = am[:, :prefix_len]

        z_prefix, _, _ = encode_prefix(model, pre_ids, pre_am)

        # Baseline (gamma=0)
        torch.manual_seed(args.seed + i * 2)
        z_base = sample_trajectory_ddim_ladire(
            model, z_prefix, args.steps, args.cfg_scale,
            args.device, dtype, gamma_max=0.0,
        )
        ce_base = compute_chunk_ce(model, z_base, ids, am, blend=args.blend)
        rank_base = effective_rank(z_base[0])

        # LaDiR
        torch.manual_seed(args.seed + i * 2 + 1)
        z_ladire = sample_trajectory_ddim_ladire(
            model, z_prefix, args.steps, args.cfg_scale,
            args.device, dtype, gamma_max=args.gamma,
        )
        ce_ladire = compute_chunk_ce(model, z_ladire, ids, am, blend=args.blend)
        rank_ladire = effective_rank(z_ladire[0])

        results.append({
            "i": i,
            "ce_base": ce_base, "ce_ladire": ce_ladire,
            "ce_delta": ce_ladire - ce_base,
            "rank_base": rank_base, "rank_ladire": rank_ladire,
            "rank_delta": rank_ladire - rank_base,
        })

        if (i + 1) % 10 == 0:
            recent = results[-10:]
            avg_ce_delta = np.mean([r["ce_delta"] for r in recent])
            avg_rank_delta = np.mean([r["rank_delta"] for r in recent])
            print(f"[{i+1}/{args.num_samples}] ce_delta={avg_ce_delta:+.3f}  "
                  f"rank_delta={avg_rank_delta:+.2f}", flush=True)

    # Stats
    ces_base = np.array([r["ce_base"] for r in results])
    ces_ladire = np.array([r["ce_ladire"] for r in results])
    ranks_base = np.array([r["rank_base"] for r in results])
    ranks_ladire = np.array([r["rank_ladire"] for r in results])

    ce_delta = ces_ladire - ces_base
    rank_delta = ranks_ladire - ranks_base

    summary = {
        "ckpt": args.ckpt_dir,
        "num_samples": len(results),
        "gamma": args.gamma,
        "ce_base_mean": float(np.mean(ces_base)),
        "ce_base_std": float(np.std(ces_base)),
        "ce_ladire_mean": float(np.mean(ces_ladire)),
        "ce_ladire_std": float(np.std(ces_ladire)),
        "ce_delta_mean": float(np.mean(ce_delta)),
        "ce_delta_std": float(np.std(ce_delta)),
        "ce_delta_pct": float(np.mean(ce_delta) / np.mean(ces_base) * 100),
        "rank_base_mean": float(np.mean(ranks_base)),
        "rank_base_std": float(np.std(ranks_base)),
        "rank_ladire_mean": float(np.mean(ranks_ladire)),
        "rank_ladire_std": float(np.std(ranks_ladire)),
        "rank_delta_mean": float(np.mean(rank_delta)),
        "rank_delta_std": float(np.std(rank_delta)),
        "rank_delta_pct": float(np.mean(rank_delta) / np.mean(ranks_base) * 100),
        "ce_improved_count": int((ce_delta < 0).sum()),
        "ce_worsened_count": int((ce_delta > 0).sum()),
    }
    print(f"\n=== LaDiR downstream (gamma={args.gamma}) ===")
    print(f"  CE:    {summary['ce_base_mean']:.2f} -> {summary['ce_ladire_mean']:.2f}  "
          f"(delta={summary['ce_delta_mean']:+.3f}, {summary['ce_delta_pct']:+.2f}%)")
    print(f"  Rank:  {summary['rank_base_mean']:.2f} -> {summary['rank_ladire_mean']:.2f}  "
          f"(delta={summary['rank_delta_mean']:+.2f}, {summary['rank_delta_pct']:+.1f}%)")
    print(f"  CE improved: {summary['ce_improved_count']}/{len(results)}  "
          f"worsened: {summary['ce_worsened_count']}/{len(results)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "per_sample": results}, f, indent=2)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
