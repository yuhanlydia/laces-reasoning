#!/usr/bin/env python3
"""Cross-agent latent separability on the CHAMPION (scratch joint co-adapt trajectory).

Answers the decisive question for two-layer fusion: do DIFFERENT agents (different
contexts) map to DIFFERENT directions in the champion's latent space? If yes, the
latent-layer (condition) fusion can carry distinct agent thoughts; if no, latent
fusion is a no-op and only state-layer fusion works.

Measures, over N distinct agent contexts:
  1. z_prefix separability : encode each context -> z_prefix; report mean pairwise
     cosine + effective rank (LOW cosine / HIGH rank = agents are separable).
  2. Z_plan separability    : sample each context's trajectory plan Z_plan ~ S2(z_prefix);
     report per-chunk mean pairwise cosine + effective rank (does re-sampling keep them apart?).
  3. fused-vs-single check  : for a few pairs, does cond_fused=0.5*(z_i+z_j) re-sampled
     produce a Z_plan DISTINCT from each single-agent Z_plan? (cosine to each parent).

Zero training. Uses the champion checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from scripts.eval.sample_prefix_suffix_trajectory_cfg import (  # noqa: E402
    encode_prefix, sample_trajectory_cfg,
)
from scripts.eval.relay_utils import load_relay_model  # noqa: E402


CONTEXTS = [
    "Agent A knows: the treasure chest is buried under the old oak tree.",
    "Agent B knows: the old oak tree stands in the northern courtyard.",
    "Agent C knows: the antidote is stored in the blue vial in the freezer.",
    "Agent D knows: the stolen painting was hidden in the attic.",
    "Agent E knows: the fastest route uses the mountain tunnel to Denver.",
    "Agent F knows: the rare orchid that blooms at midnight is poisonous.",
    "Agent G knows: the missing key opens the archive room behind the red door.",
    "Agent H knows: the champion athlete who trained in Kenya runs the marathon.",
    "Agent I knows: the secret recipe requires saffron harvested in Kashmir.",
    "Agent J knows: the encrypted file named phoenix contains the launch codes.",
    "Agent K knows: the injured hiker on the stretcher was airlifted to Boston.",
    "Agent L knows: the winning lottery ticket bought Tuesday was sold in Chicago.",
    "Agent M knows: the ancient Greek scroll describes a lost city.",
    "Agent N knows: the prototype hydrogen engine powers the new aircraft.",
    "Agent O knows: the suspect in the green jacket fled toward the harbor.",
    "Agent P knows: the violin melody premiered in Prague.",
]


def _eff_rank(mat: torch.Tensor) -> float:
    x = mat - mat.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x.float())
    s = s[s > 1e-9]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    return float((-(p * p.log()).sum()).exp())


def _mean_pairwise_cos(mat: torch.Tensor) -> float:
    x = F.normalize(mat.float(), dim=-1)
    g = x @ x.t()
    n = g.shape[0]
    return float((g.sum() - g.diagonal().sum()) / (n * (n - 1)))


@torch.no_grad()
def run(args):
    torch.manual_seed(args.seed)
    device = args.device
    model, _rwkv, tokenizer, _ckpt, _cfg = load_relay_model(args.ckpt_dir, device)
    model.eval()
    model._prefix_suffix_trajectory_s2 = True
    dtype = next(model.alpha_heads.parameters()).dtype
    H = int(model.trajectory_horizon)

    def enc(t):
        ids = tokenizer(t, return_tensors="pt").input_ids.to(device)
        return ids, torch.ones_like(ids)

    # 1. z_prefix per agent
    z_list = []
    for c in CONTEXTS:
        ids, am = enc(c)
        z_list.append(encode_prefix(model, ids, am)[0].to(dtype)[0])  # [D]
    Zpref = torch.stack(z_list, dim=0)  # [N, D]

    # 2. Z_plan per agent (sample trajectory), flatten chunks for separability
    plan_list = []
    for c in CONTEXTS:
        ids, am = enc(c)
        zp = encode_prefix(model, ids, am)[0].to(dtype)
        Zp = sample_trajectory_cfg(model, zp, args.steps, args.cfg_scale, device, dtype)  # [1,H,D]
        plan_list.append(Zp[0])  # [H, D]
    Zplans = torch.stack(plan_list, dim=0)  # [N, H, D]
    # separability of the mean-pooled plan per agent
    Zplan_pooled = Zplans.mean(dim=1)  # [N, D]

    result = {
        "ckpt_dir": args.ckpt_dir, "n_agents": len(CONTEXTS), "H": H,
        "steps": args.steps, "cfg_scale": args.cfg_scale,
        "z_prefix": {
            "mean_pairwise_cosine": _mean_pairwise_cos(Zpref),
            "effective_rank": _eff_rank(Zpref),
            "max_possible_rank": min(len(CONTEXTS), Zpref.shape[1]),
        },
        "Z_plan_pooled": {
            "mean_pairwise_cosine": _mean_pairwise_cos(Zplan_pooled),
            "effective_rank": _eff_rank(Zplan_pooled),
            "max_possible_rank": min(len(CONTEXTS), Zplan_pooled.shape[1]),
        },
    }

    # 3. fused-vs-single: for 4 pairs, does fused plan differ from each parent?
    pairs = [(0, 1), (2, 3), (4, 5), (8, 10)]
    fused_checks = []
    for (i, j) in pairs:
        ci, cj = CONTEXTS[i], CONTEXTS[j]
        idi, ami = enc(ci); idj, amj = enc(cj)
        zi = encode_prefix(model, idi, ami)[0].to(dtype)
        zj = encode_prefix(model, idj, amj)[0].to(dtype)
        cond_fused = 0.5 * zi + 0.5 * zj
        Zf = sample_trajectory_cfg(model, cond_fused, args.steps, args.cfg_scale, device, dtype)[0].mean(0)
        Zi = sample_trajectory_cfg(model, zi, args.steps, args.cfg_scale, device, dtype)[0].mean(0)
        Zj = sample_trajectory_cfg(model, zj, args.steps, args.cfg_scale, device, dtype)[0].mean(0)
        fused_checks.append({
            "pair": [i, j],
            "cos_zprefix_i_j": float(F.cosine_similarity(zi.flatten(), zj.flatten(), dim=0)),
            "cos_fusedplan_to_i": float(F.cosine_similarity(Zf.flatten(), Zi.flatten(), dim=0)),
            "cos_fusedplan_to_j": float(F.cosine_similarity(Zf.flatten(), Zj.flatten(), dim=0)),
        })
    result["fused_vs_single"] = fused_checks

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))

    print("\n=== CROSS-AGENT LATENT SEPARABILITY (champion) ===")
    zp = result["z_prefix"]; pl = result["Z_plan_pooled"]
    print(f"\n[z_prefix] (agents' raw thought latents)")
    print(f"  mean pairwise cosine = {zp['mean_pairwise_cosine']:.3f}  (LOW=separable, HIGH=same direction)")
    print(f"  effective rank       = {zp['effective_rank']:.2f} / {zp['max_possible_rank']}  (HIGH=diverse)")
    print(f"\n[Z_plan pooled] (after S2 re-sampling)")
    print(f"  mean pairwise cosine = {pl['mean_pairwise_cosine']:.3f}")
    print(f"  effective rank       = {pl['effective_rank']:.2f} / {pl['max_possible_rank']}")
    print(f"\n[fused vs single] does 0.5*(z_i+z_j) re-sampled differ from each parent plan?")
    for fc in fused_checks:
        print(f"  pair{fc['pair']}: cos(z_i,z_j)={fc['cos_zprefix_i_j']:.3f} | "
              f"cos(fused,i)={fc['cos_fusedplan_to_i']:.3f} cos(fused,j)={fc['cos_fusedplan_to_j']:.3f}")
    print(f"\nwritten: {args.output}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000")
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--cfg_scale", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="outputs_eval/diag_agent_latent_separability.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
