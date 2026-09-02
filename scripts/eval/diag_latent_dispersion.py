#!/usr/bin/env python3
# pyright: reportAny=false, reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnusedCallResult=false, reportAttributeAccessIssue=false
"""Latent dispersion diagnostic: compare single-z S2 with vs without condboundary.

Measures whether condboundary makes the sampled latent z "spread out more" / learn
better, via three metrics per checkpoint:

  1. sample spread     : across many prompts, sampled z. Report per-dim std, effective
                         rank (participation ratio of covariance eigenvalues), mean
                         pairwise cosine (lower = more diverse / less collapsed).
  2. cond vs uncond    : does the prefix condition actually move z? cosine(z_cond, z_uncond)
                         and ||z_cond - z_uncond|| (higher diff = condition is used).
  3. CFG response      : does raising cfg move z further from uncond? ||z(cfg) - z(uncond)||
                         at cfg in {1,3,5}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.relay_utils import load_relay_model  # noqa: E402
from scripts.eval.sample_prefix_suffix_cfg import encode_prefix, sample_ddim_cfg  # noqa: E402


PROMPTS = [
    "The history of artificial intelligence",
    "In a shocking turn of events, scientists discovered",
    "The recipe calls for two cups of flour and",
    "Climate change is one of the most pressing",
    "Once upon a time in a distant kingdom",
    "The stock market fell sharply today after",
    "To solve this equation, first isolate the variable",
    "The human immune system defends the body by",
    "Ancient Rome was founded according to legend by",
    "Machine learning models require large amounts of",
    "The novel opens with a description of the",
    "Photosynthesis converts sunlight into chemical energy",
    "The Supreme Court ruled today that the law",
    "Basketball is a sport played by two teams",
    "The chemical formula for water is composed of",
    "During the Renaissance, artists began to explore",
]


def _effective_rank(z: torch.Tensor) -> float:
    # participation ratio of covariance eigenvalues: (sum λ)^2 / sum λ^2
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.t() @ zc) / max(1, z.shape[0] - 1)
    ev = torch.linalg.eigvalsh(cov).clamp(min=0)
    s1 = ev.sum()
    s2 = (ev * ev).sum()
    if s2.item() == 0:
        return 0.0
    return float((s1 * s1 / s2).item())


def _mean_pairwise_cosine(z: torch.Tensor) -> float:
    zn = F.normalize(z, dim=-1)
    sim = zn @ zn.t()
    n = z.shape[0]
    off = (sim.sum() - torch.diagonal(sim).sum()) / (n * (n - 1))
    return float(off.item())


@torch.no_grad()
def diagnose(ckpt_dir: str, device: str, steps: int, seed: int) -> dict:
    model, _rwkv, tokenizer, _ckpt, _cfg = load_relay_model(ckpt_dir, device)
    model.eval()
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32

    z_cond_list = []   # cfg=3 sampled z per prompt
    z_uncond_list = []
    cfg_dist = {1.0: [], 3.0: [], 5.0: []}

    for pi, prompt in enumerate(PROMPTS):
        ids = tokenizer(prompt, add_special_tokens=False).input_ids
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        ids = [int(x) for x in ids][:256] or [0]
        input_ids = torch.tensor([ids], device=device, dtype=torch.long)
        attn = torch.ones_like(input_ids)
        cond = encode_prefix(model, input_ids, attn)  # [1, latent_dim]
        zero = torch.zeros_like(cond)

        # fixed seed per prompt so the only variable is the checkpoint / cfg
        for cfg_scale in (1.0, 3.0, 5.0):
            torch.manual_seed(seed + pi)
            if device.startswith("cuda"):
                torch.cuda.manual_seed_all(seed + pi)
            z = sample_ddim_cfg(model, cond, steps, cfg_scale, device, dtype).float()
            cfg_dist[cfg_scale].append(z)
            if cfg_scale == 3.0:
                z_cond_list.append(z)

        # unconditional (cond = zeros), same seed base
        torch.manual_seed(seed + pi)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed + pi)
        z_u = sample_ddim_cfg(model, zero, steps, 1.0, device, dtype).float()
        z_uncond_list.append(z_u)

    z_cond = torch.cat(z_cond_list, dim=0)      # [P, D]
    z_uncond = torch.cat(z_uncond_list, dim=0)  # [P, D]

    # 1. sample spread (on cfg=3 cond samples)
    per_dim_std = z_cond.std(dim=0)
    spread = {
        "n_prompts": z_cond.shape[0],
        "latent_dim": z_cond.shape[1],
        "mean_per_dim_std": float(per_dim_std.mean().item()),
        "min_per_dim_std": float(per_dim_std.min().item()),
        "max_per_dim_std": float(per_dim_std.max().item()),
        "effective_rank": _effective_rank(z_cond),
        "mean_pairwise_cosine": _mean_pairwise_cosine(z_cond),
        "mean_norm": float(z_cond.norm(dim=-1).mean().item()),
    }

    # 2. cond vs uncond
    cos_cu = F.cosine_similarity(z_cond, z_uncond, dim=-1)
    diff_cu = (z_cond - z_uncond).norm(dim=-1)
    cond_effect = {
        "mean_cos_cond_uncond": float(cos_cu.mean().item()),
        "mean_dist_cond_uncond": float(diff_cu.mean().item()),
    }

    # 3. CFG response: dist from uncond at each cfg
    cfg_resp = {}
    for c in (1.0, 3.0, 5.0):
        zc = torch.cat(cfg_dist[c], dim=0)
        d = (zc - z_uncond).norm(dim=-1).mean()
        cfg_resp[f"dist_from_uncond_cfg{c:g}"] = float(d.item())

    return {"spread": spread, "cond_effect": cond_effect, "cfg_response": cfg_resp}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-ckpt", required=True, help="single-z SFT (no condboundary)")
    ap.add_argument("--condboundary-ckpt", required=True, help="single-z SFT + condboundary")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="/tmp/diffrwkv_traj_diag_cleanz/latent_dispersion.json")
    args = ap.parse_args()

    print(f"[diag] baseline (no condboundary): {args.baseline_ckpt}", flush=True)
    base = diagnose(args.baseline_ckpt, args.device, args.steps, args.seed)
    print(f"[diag] condboundary: {args.condboundary_ckpt}", flush=True)
    cb = diagnose(args.condboundary_ckpt, args.device, args.steps, args.seed)

    out = {
        "baseline_no_condboundary": {"ckpt": args.baseline_ckpt, **base},
        "with_condboundary": {"ckpt": args.condboundary_ckpt, **cb},
        "delta_condboundary_minus_baseline": {
            "effective_rank": cb["spread"]["effective_rank"] - base["spread"]["effective_rank"],
            "mean_per_dim_std": cb["spread"]["mean_per_dim_std"] - base["spread"]["mean_per_dim_std"],
            "mean_pairwise_cosine": cb["spread"]["mean_pairwise_cosine"] - base["spread"]["mean_pairwise_cosine"],
            "mean_dist_cond_uncond": cb["cond_effect"]["mean_dist_cond_uncond"] - base["cond_effect"]["mean_dist_cond_uncond"],
            "cfg5_dist_from_uncond": cb["cfg_response"]["dist_from_uncond_cfg5"] - base["cfg_response"]["dist_from_uncond_cfg5"],
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
