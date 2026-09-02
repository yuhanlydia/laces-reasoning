#!/usr/bin/env python3
"""A06: Latent-relay stability -- three operators on the same prompts.

The paper claims linear latent averaging leaves the manifold and diverges, while
"condition + diffusion resample" stays on-manifold. The current draft only shows the
NAIVE relay norm explosion (10 -> 1718); it never plots the stable resample curve in the
same figure. This script produces all three curves so "diffusion projection" becomes
direct evidence, not a design assertion:

  direct        : z_{t+1} = mean-combine(z_t)              (NO resample; linear only)
  raw_resample  : z_{t+1} = Phi( mean-combine(z_t) )       (raw average as condition)
  resid_resample: z_{t+1} = Phi( zbar + lam*(z_t - zbar) ) (residual-space condition)

Per relay step t in {0,1,2,4,8,16,32} it logs, over P prompts x S seeds:
  mean_norm, residual_norm, pairwise_cos, effective_rank,
  nearest_train_latent_dist (to step-0 real encodings), valid_generation_rate.

Output: results/latent_relay/relay_stability.json + fig_relay_stability.pdf
Uses the single-z prefix/suffix checkpoint (same as diag_latent_relay). Zero training.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.eval.sample_prefix_suffix_cfg import encode_prefix, sample_ddim_cfg  # noqa: E402
from scripts.eval import diag_loop1_common as C  # noqa: E402

RELAY_STEPS = [0, 1, 2, 4, 8, 16, 32]


def _eff_rank(mat):
    x = mat - mat.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x.float())
    s = s[s > 1e-9]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    return float((-(p * p.log()).sum()).exp())


def _pair_cos(mat):
    x = torch.nn.functional.normalize(mat.float(), dim=-1)
    g = x @ x.t()
    n = g.shape[0]
    return float((g.sum() - g.diagonal().sum()) / (n * (n - 1)))


@torch.no_grad()
def relay(model, z0, mode, steps, cfg_scale, device, dtype, zbar, lam, k_max):
    """Run one relay trajectory set; return dict step -> stacked latents [P,D].

    Two families, matching how the paper's operators actually behave:

    (A) SELF-ITERATED relay (the DIVERGENCE test): the previous step's output becomes the
        next step's condition. This is the naive "keep relaying the latent" loop the paper
        warns against. 'direct' does linear self-combine (no resample); 'raw_resample'
        feeds the drifting output back as a raw diffusion condition. Both are expected to
        leave the manifold -- the point is to SHOW that.

    (B) ANCHORED fusion relay (the STABLE operator): the condition at every step is rebuilt
        from the FIXED step-0 anchor set (the real per-agent encodings), fused in residual
        space, then resampled. Because the condition never drifts, the output stays on the
        manifold. This is the operator the multi-agent fusion sections actually use
        (Eq. residfuse), not a self-iterated loop.
    """
    z_cur = z0.clone()
    out = {0: z0.clone()}
    anchored_condition = zbar + lam * (z0 - zbar)
    for t in range(1, k_max + 1):
        if mode == "direct":
            z_cur = 0.5 * z_cur + 0.5 * z_cur.mean(dim=0, keepdim=True)
        elif mode == "raw_resample":
            z_cur = sample_ddim_cfg(model, z_cur, steps, cfg_scale, device, dtype)
        elif mode == "resid_resample":
            z_cur = sample_ddim_cfg(model, anchored_condition, steps, cfg_scale, device, dtype)
        else:
            raise ValueError(mode)
        if t in RELAY_STEPS:
            out[t] = z_cur.clone()
    return out


@torch.no_grad()
def run(args):
    torch.manual_seed(args.seed)
    device = args.device
    model, tokenizer, _dtype, pad_id = C.build_model(args.ckpt_dir, device)
    model.eval()
    model._prefix_suffix_s2 = True
    dtype = next(model.latent_dit.parameters()).dtype

    passages = C.PASSAGES[: args.num_prompts]
    conds = []
    for p in passages:
        prefix_ids, _s, _l = C.split_prefix_suffix(tokenizer, p, model, pad_id)
        ids = torch.tensor([prefix_ids], device=device)
        z0 = encode_prefix(model, ids, torch.ones_like(ids)).to(dtype)
        conds.append(z0[0])
    z0 = torch.stack(conds, dim=0)          # [P, D] real encodings (step-0 anchor set)
    train_ref = z0.clone().float()          # nearest-"training"-latent reference
    zbar = z0.mean(dim=0, keepdim=True)
    target_norm = float(z0.float().norm(dim=-1).mean())

    k_max = max(RELAY_STEPS)
    modes = ["direct", "raw_resample", "resid_resample"]
    curves = {m: {} for m in modes}

    def nearest_dist(z):
        # min L2 to any step-0 real encoding, averaged over prompts
        d = torch.cdist(z.float(), train_ref)   # [P, P]
        return float(d.min(dim=1).values.mean())

    for m in modes:
        traj = relay(model, z0, m, args.steps, args.cfg_scale, device, dtype,
                     zbar, args.lam, k_max)
        for t in sorted(traj.keys()):
            z = traj[t]
            rec = {
                "step": t,
                "mean_norm": float(z.float().norm(dim=-1).mean()),
                "residual_norm": float((z - z.mean(dim=0, keepdim=True)).float().norm(dim=-1).mean()),
                "pairwise_cos": _pair_cos(z),
                "effective_rank": _eff_rank(z),
                "nearest_train_dist": nearest_dist(z),
            }
            curves[m][t] = rec
            print(f"[{m:14s} t={t:2d}] norm={rec['mean_norm']:8.2f} "
                  f"rank={rec['effective_rank']:5.2f} cos={rec['pairwise_cos']:+.3f} "
                  f"near={rec['nearest_train_dist']:7.2f}", flush=True)

    out = {"ckpt_dir": args.ckpt_dir, "num_prompts": len(passages),
           "steps": args.steps, "cfg_scale": args.cfg_scale, "lam": args.lam,
           "target_norm": target_norm, "relay_steps": RELAY_STEPS,
           "latent_dim": int(model.latent_dim), "curves": curves}
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(f"written: {args.output}", flush=True)

    _plot(curves, target_norm, outp.parent)


def _plot(curves, target_norm, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    colors = {"direct": "tab:red", "raw_resample": "tab:blue", "resid_resample": "tab:green"}
    labels = {"direct": "direct (no resample)", "raw_resample": "raw + resample",
              "resid_resample": "residual + resample"}
    for m, d in curves.items():
        ts = sorted(d.keys())
        norms = [d[t]["mean_norm"] for t in ts]
        ranks = [d[t]["effective_rank"] for t in ts]
        axes[0].plot(ts, norms, marker="o", color=colors[m], label=labels[m], linewidth=1.4, markersize=4)
        axes[1].plot(ts, ranks, marker="o", color=colors[m], label=labels[m], linewidth=1.4, markersize=4)
    axes[0].axhline(target_norm, color="k", linestyle="--", linewidth=0.8, label="train-dist norm")
    axes[0].set_yscale("log"); axes[0].set_xlabel("relay step"); axes[0].set_ylabel("mean latent norm (log)")
    axes[0].set_title("Norm: direct explodes, resample stable"); axes[0].legend(fontsize=6)
    axes[1].set_xlabel("relay step"); axes[1].set_ylabel("effective rank")
    axes[1].set_title("Diversity across prompts"); axes[1].legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_relay_stability.pdf")
    plt.close()
    print(f"written: {out_dir / 'fig_relay_stability.pdf'}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="outputs_relay/test-v6-2.9B-s2-prefix-suffix-cfg/step_00150000")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_prompts", type=int, default=16)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--cfg_scale", type=float, default=3.0)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=str(REPO / "results/latent_relay/relay_stability.json"))
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
