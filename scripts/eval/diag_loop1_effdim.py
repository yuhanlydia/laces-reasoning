# pyright: reportAny=false, reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnusedCallResult=false, reportAttributeAccessIssue=false
"""Loop-1 diagnostic #7: latent effective dimension (trajectory).

Reuses the effective-rank / pairwise-cosine helpers from diag_latent_dispersion.py
but samples TRAJECTORY latents (per-chunk z_h) rather than single-z. Compares the
CLEAN encoder latents vs S2-SAMPLED latents: if sampled latents collapse (lower
effective rank, higher pairwise cosine) relative to clean, that is latent-space
homogenization at inference (the SIM-CoT failure mode).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.eval.diag_loop1_common as C  # noqa: E402
from scripts.eval.diag_latent_dispersion import _effective_rank, _mean_pairwise_cosine  # noqa: E402

DEFAULT_CKPT = "outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-ddpm-condboundary/step_00026000"


@torch.no_grad()
def run(ckpt_dir, device, num_samples, steps, cfg_scale, output):
    model, tokenizer, dtype, pad_id = C.build_model(ckpt_dir, device)
    passages = C.PASSAGES[:num_samples]

    clean_z = []    # each [H, D]
    sampled_z = []
    for i, passage in enumerate(passages):
        pre, suf, _ln = C.split_prefix_suffix(tokenizer, passage, model, pad_id)
        zc = C.get_z_clean(model, suf, device)[0].float().cpu()      # [H, D]
        zs = C.get_z_sampled(model, pre, device, dtype, steps, cfg_scale)[0].float().cpu()
        clean_z.append(zc)
        sampled_z.append(zs)
        print(f"[{i+1}/{len(passages)}] encoded/sampled z")

    # flatten all per-chunk latents into [N*H, D] for global dispersion
    clean_flat = torch.cat(clean_z, dim=0)      # [N*H, D]
    sampled_flat = torch.cat(sampled_z, dim=0)

    def dispersion(mat: torch.Tensor) -> dict:
        return {
            "effective_rank": _effective_rank(mat),
            "mean_pairwise_cosine": _mean_pairwise_cosine(mat),
            "per_dim_std_mean": float(mat.std(dim=0).mean().item()),
            "n_vectors": int(mat.shape[0]),
            "dim": int(mat.shape[1]),
        }

    # also per-position effective rank (across prompts, for a fixed chunk index h)
    H = clean_z[0].shape[0]
    clean_stack = torch.stack(clean_z, dim=0)     # [N, H, D]
    sampled_stack = torch.stack(sampled_z, dim=0)
    clean_pos_er, sampled_pos_er = [], []
    for h in range(H):
        clean_pos_er.append(_effective_rank(clean_stack[:, h]))
        sampled_pos_er.append(_effective_rank(sampled_stack[:, h]))

    result = {
        "diagnostic": "loop1_effdim(#7)",
        "ckpt": ckpt_dir,
        "num_samples": len(passages),
        "steps": steps,
        "cfg_scale": cfg_scale,
        "clean_global": dispersion(clean_flat),
        "sampled_global": dispersion(sampled_flat),
        "clean_per_chunk_effective_rank": clean_pos_er,
        "sampled_per_chunk_effective_rank": sampled_pos_er,
        "clean_per_chunk_er_mean": statistics.mean(clean_pos_er),
        "sampled_per_chunk_er_mean": statistics.mean(sampled_pos_er),
        "interpretation": (
            "sampled effective_rank << clean, or sampled pairwise_cosine >> clean => "
            "the S2 sampler collapses/homogenizes latents relative to the encoder "
            "(SIM-CoT-style latent homogenization). Similar dispersion => no collapse; "
            "the gap is not a latent-diversity problem."
        ),
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    print("\n=== SUMMARY (effdim #7) ===")
    print(f"clean   eff_rank={result['clean_global']['effective_rank']:.3f} "
          f"pairwise_cos={result['clean_global']['mean_pairwise_cosine']:.3f} "
          f"std={result['clean_global']['per_dim_std_mean']:.3f}")
    print(f"sampled eff_rank={result['sampled_global']['effective_rank']:.3f} "
          f"pairwise_cos={result['sampled_global']['mean_pairwise_cosine']:.3f} "
          f"std={result['sampled_global']['per_dim_std_mean']:.3f}")
    print(f"written: {output}")
    print(f"gpu peak MiB: {int(torch.cuda.max_memory_allocated()/1024/1024) if device.startswith('cuda') else 0}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_samples", type=int, default=16)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cfg_scale", type=float, default=3.0)
    ap.add_argument("--output", default="/tmp/diag_loop1_effdim.json")
    args = ap.parse_args()
    run(args.ckpt_dir, args.device, args.num_samples, args.steps, args.cfg_scale, args.output)


if __name__ == "__main__":
    main()
