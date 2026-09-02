"""Stage-0 GLSSD premise check: does sampled/clean rank ratio correlate with the
clean-vs-sampled downstream quality gap?

Advisory's main plot: x = sampled_rank / clean_rank, y = quality gap. If lower
rank ratio => larger quality gap, rank preservation is a real bottleneck and the
rank-band loss is justified. If no correlation, the collapse story is wrong for
the champion and we must pivot.

Per passage: encode clean-Z and sample sampled-Z, measure per-chunk effective
rank for each, and measure the teacher-forced suffix CE under clean-injection vs
sampled-injection (the quality gap proxy). Then report the correlation across
passages.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import eval.diag_loop1_common as C
from scripts.eval.diag_latent_dispersion import _effective_rank


@torch.no_grad()
def suffix_ce_under_z(model, tokenizer, pre_ids, suffix_ids, z_traj, blend, device):
    from scripts.eval.sample_prefix_suffix_trajectory_cfg import encode_prefix
    pre = torch.tensor([pre_ids], device=device, dtype=torch.long)
    pre_am = torch.ones_like(pre, dtype=torch.float32)
    _zp, cache, logits = encode_prefix(model, pre, pre_am)
    C_ = int(model.trajectory_chunk_size)
    H = z_traj.shape[1]
    layer_states = model.predict_trajectory_states(z_traj)
    ce_sum, ce_n, gp = 0.0, 0, 0
    for h in range(H):
        states_h = [ls[:, h] for ls in layer_states]
        cache = model.blend_into_cache(cache, states_h, blend)
        for _c in range(C_):
            if gp >= len(suffix_ids):
                break
            tgt = int(suffix_ids[gp])
            ce_sum += F.cross_entropy(logits.float().unsqueeze(0),
                                      torch.tensor([tgt], device=device)).item()
            ce_n += 1
            out = model.rwkv_model(input_ids=torch.tensor([[tgt]], device=device),
                                   past_key_values=cache, use_cache=True, return_dict=True)
            cache = out.past_key_values
            logits = out.logits[0, -1]
            gp += 1
    return ce_sum / max(1, ce_n)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000")
    ap.add_argument("--num_samples", type=int, default=16)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cfg_scale", type=float, default=3.0)
    ap.add_argument("--blend", type=float, default=0.7)
    ap.add_argument("--out", default="outputs_eval/rank_quality_corr.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model, tokenizer, dtype, pad_id = C.build_model(args.ckpt_dir, args.device)
    passages = C.PASSAGES[:args.num_samples]

    rows = []
    for i, passage in enumerate(passages):
        pre, suf, ln = C.split_prefix_suffix(tokenizer, passage, model, pad_id)
        zc = C.get_z_clean(model, suf, args.device)              # [1,H,D]
        zs = C.get_z_sampled(model, pre, args.device, dtype, args.steps, args.cfg_scale)
        zc_cpu, zs_cpu = zc[0].float().cpu(), zs[0].float().cpu()
        clean_rank = _effective_rank(zc_cpu)
        sampled_rank = _effective_rank(zs_cpu)
        ce_clean = suffix_ce_under_z(model, tokenizer, pre, suf[:ln], zc, args.blend, args.device)
        ce_sampled = suffix_ce_under_z(model, tokenizer, pre, suf[:ln], zs, args.blend, args.device)
        rows.append({
            "idx": i,
            "clean_rank": clean_rank,
            "sampled_rank": sampled_rank,
            "rank_ratio": sampled_rank / clean_rank if clean_rank > 0 else 0.0,
            "ce_clean": ce_clean,
            "ce_sampled": ce_sampled,
            "quality_gap": ce_sampled - ce_clean,
        })
        print(f"[{i+1}/{len(passages)}] rank_ratio={rows[-1]['rank_ratio']:.3f} "
              f"gap={rows[-1]['quality_gap']:.4f} (ce_clean={ce_clean:.3f} ce_sampled={ce_sampled:.3f})",
              flush=True)

    ratios = np.array([r["rank_ratio"] for r in rows])
    gaps = np.array([r["quality_gap"] for r in rows])
    corr = float(np.corrcoef(ratios, gaps)[0, 1]) if len(rows) > 1 else 0.0
    res = {
        "ckpt": args.ckpt_dir, "num_samples": len(rows),
        "rows": rows,
        "mean_rank_ratio": float(ratios.mean()),
        "mean_quality_gap": float(gaps.mean()),
        "corr_rankratio_vs_gap": corr,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)

    print("\n=== Stage-0: rank ratio vs quality gap ===")
    print(f"  mean rank_ratio (sampled/clean): {ratios.mean():.3f}")
    print(f"  mean quality_gap (ce_sampled - ce_clean): {gaps.mean():.4f}")
    print(f"  correlation(rank_ratio, quality_gap): {corr:+.3f}")
    verdict = ("NEG corr -> lower rank => bigger gap => rank preservation IS the bottleneck"
               if corr < -0.3 else
               ("weak/no corr -> rank collapse NOT the general bottleneck, pivot"
                if abs(corr) <= 0.3 else
                "POS corr -> unexpected, investigate"))
    print(f"  verdict: {verdict}")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
