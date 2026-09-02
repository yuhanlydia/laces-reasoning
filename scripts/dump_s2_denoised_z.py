#!/usr/bin/env python3
"""Pre-compute S2-denoised Z for 0.4B S1 training.

Loads 13.3B S2 (frozen), adds diffusion noise to clean Z, runs one-step denoising,
and saves x0_pred for S1 training. This eliminates exposure bias: S1 trains on
the same Z distribution it will see at inference.

Usage:
  python scripts/dump_s2_denoised_z.py \
    --s2_ckpt outputs_relay/owt512-traj32x16-13.3B-basis32-prefix-suffix-blend0p5-s2-rwkv-rf/step_00150000 \
    --latent_dir preprocessed_data/owt_13b_s0_latents/train \
    --save_dir preprocessed_data/owt_13b_s2_denoised_z/train \
    --num_samples 30000
"""
import argparse, glob, sys, os, time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.eval.diag_loop1_common import build_model


def cosine_alpha_bar(t):
    s = 0.008
    return (torch.cos((t + s) / (1.0 + s) * np.pi / 2.0) ** 2) / (
        np.cos(s / (1.0 + s) * np.pi / 2.0) ** 2
    )


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2_ckpt", required=True)
    ap.add_argument("--latent_dir", default="preprocessed_data/owt_13b_s0_latents/train")
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--num_samples", type=int, default=30000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    model, _, _, _ = build_model(args.s2_ckpt, device)
    model.eval()
    # Use single-step x0 prediction from S2 (not full multi-step sampling)
    H = int(model.trajectory_horizon)
    D = int(model.latent_dim)

    latent_files = sorted(glob.glob(f"{args.latent_dir}/*.npy"))[: args.num_samples]
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for i, lf in enumerate(latent_files):
        Z_clean = torch.tensor(np.load(lf), device=device, dtype=dtype).unsqueeze(0)
        B, H_actual, _ = Z_clean.shape
        Z_clean = Z_clean[:, :H, :]

        # Add diffusion noise: random timestep
        t = torch.rand(B, device=device, dtype=dtype)
        ab = cosine_alpha_bar(t)
        view = (B,) + (1,) * (Z_clean.dim() - 1)
        noise = torch.randn_like(Z_clean)
        Z_t = ab.sqrt().view(view) * Z_clean + (1.0 - ab).sqrt().view(view) * noise

        # S2 one-step denoising: predict noise, then recover x0
        eps_pred = model.trajectory_dit(
            Z_t, t.unsqueeze(-1),
            cond=Z_t[:, : H // 2, :].mean(dim=1) if hasattr(model, 'use_cond_boundary') and model.use_cond_boundary else None
        )
        # DDPM x0 recovery: x0 = (Z_t - sqrt(1-ᾱ)·eps) / sqrt(ᾱ)
        x0_pred = (Z_t - (1.0 - ab).sqrt().view(view) * eps_pred) / ab.sqrt().view(view).clamp(min=1e-8)

        out_name = Path(lf).name
        np.save(os.path.join(args.save_dir, out_name), x0_pred.squeeze(0).cpu().float().numpy())

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            sps = (i + 1) / max(elapsed, 0.01)
            eta = (args.num_samples - i - 1) / max(sps, 0.01)
            print(f"[{i+1}/{args.num_samples}] {sps:.1f} samples/s ETA={eta/60:.0f}min", flush=True)

    print(f"Done! {args.num_samples} denoised Z saved to {args.save_dir}", flush=True)


if __name__ == "__main__":
    main()
