# pyright: reportAny=false, reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnusedCallResult=false, reportAttributeAccessIssue=false
"""Loop-1 diagnostics #5, #6: blend and per-layer sensitivity (sampled-Z injected).

#5 BLEND CURVE: teacher-forced suffix CE vs global blend in {0,.1,.25,.5,.75,1}.
   Shows how much the injected (sampled) state helps/hurts as it overrides the
   running state. A U-shaped or monotone curve localizes the useful blend region.
#6 PER-LAYER SENSITIVITY: inject the planned state at exactly ONE layer (blend=1
   there, 0 elsewhere) and measure suffix-CE change vs a no-injection baseline.
   Ranks which layers actually accept planned state.

Uses the S2-SAMPLED latent (the realistic inference distribution).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.eval.diag_loop1_common as C  # noqa: E402
from scripts.eval.diag_clean_z_trajectory import blend_into_cache_per_layer  # noqa: E402

DEFAULT_CKPT = "outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-ddpm-condboundary/step_00026000"


@torch.no_grad()
def suffix_ce_with_z(model, prefix_ids, suffix_ids, z, per_layer_blends, device):
    """Teacher-forced suffix CE injecting arbitrary latent z with per-layer blends."""
    Cs = int(model.trajectory_chunk_size)
    usable = (len(suffix_ids) // Cs) * Cs
    suffix_ids = suffix_ids[:usable]
    if len(prefix_ids) == 0 or usable < Cs * 2:
        return float("nan")
    prefix = torch.tensor([prefix_ids], device=device, dtype=torch.long)
    suffix = torch.tensor([suffix_ids], device=device, dtype=torch.long)
    chunks, _cm, h_eff, _c = model._trajectory_view(suffix, torch.ones_like(suffix))
    layer_states = model.predict_trajectory_states(z)
    prefix_out = model.rwkv_model(input_ids=prefix, use_cache=True, return_dict=True)
    cache = prefix_out.past_key_values
    logits_by_chunk = []
    for h in range(min(h_eff, z.shape[1])):
        states_h = [ls[:, h] for ls in layer_states]
        cache = blend_into_cache_per_layer(model, cache, states_h, per_layer_blends, False)
        out_h = model.rwkv_model(input_ids=chunks[:, h], past_key_values=cache,
                                 use_cache=True, return_dict=True)
        cache = out_h.past_key_values
        logits_by_chunk.append(out_h.logits)
    logits = torch.stack(logits_by_chunk, dim=1).reshape(1, -1, logits_by_chunk[0].shape[-1])
    tgt_len = logits.shape[1]
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
        suffix[:, 1:tgt_len].reshape(-1),
    )
    return float(loss.item())


@torch.no_grad()
def baseline_ce_no_injection(model, prefix_ids, suffix_ids, device):
    """No state injection: pure prefix->suffix teacher forcing."""
    Cs = int(model.trajectory_chunk_size)
    usable = (len(suffix_ids) // Cs) * Cs
    suffix_ids = suffix_ids[:usable]
    if len(prefix_ids) == 0 or usable < Cs * 2:
        return float("nan")
    prefix = torch.tensor([prefix_ids], device=device, dtype=torch.long)
    suffix = torch.tensor([suffix_ids], device=device, dtype=torch.long)
    prefix_out = model.rwkv_model(input_ids=prefix, use_cache=True, return_dict=True)
    out = model.rwkv_model(input_ids=suffix, past_key_values=prefix_out.past_key_values,
                           use_cache=True, return_dict=True)
    logits = out.logits
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
        suffix[:, 1:].reshape(-1),
    )
    return float(loss.item())


@torch.no_grad()
def run(ckpt_dir, device, num_samples, steps, cfg_scale, output):
    model, tokenizer, dtype, pad_id = C.build_model(ckpt_dir, device)
    num_layers = int(model.num_layers)
    blends = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    passages = C.PASSAGES[:num_samples]

    blend_curve = {f"{b}": [] for b in blends}
    layer_delta_acc = [[] for _ in range(num_layers)]
    base_acc = []

    for i, passage in enumerate(passages):
        pre, suf, _ln = C.split_prefix_suffix(tokenizer, passage, model, pad_id)
        z_sampled = C.get_z_sampled(model, pre, device, dtype, steps, cfg_scale)

        base = baseline_ce_no_injection(model, pre, suf, device)
        base_acc.append(base)

        for b in blends:
            pl = [b] * num_layers
            ce = suffix_ce_with_z(model, pre, suf, z_sampled, pl, device)
            blend_curve[f"{b}"].append(ce)

        # per-layer: blend=1 at one layer only, measure CE delta vs baseline (lower CE = layer accepts planned state usefully)
        for l in range(num_layers):
            pl = [0.0] * num_layers
            pl[l] = 1.0
            ce = suffix_ce_with_z(model, pre, suf, z_sampled, pl, device)
            layer_delta_acc[l].append(ce - base if (ce == ce and base == base) else float("nan"))
        print(f"[{i+1}/{len(passages)}] base_CE={base:.3f} "
              f"blend0.5_CE={blend_curve['0.5'][-1]:.3f} blend1.0_CE={blend_curve['1.0'][-1]:.3f}")

    def ms(x):
        xf = [v for v in x if v == v]
        return {"mean": statistics.mean(xf) if xf else float("nan"),
                "std": statistics.pstdev(xf) if len(xf) > 1 else 0.0}

    blend_curve_ms = {b: ms(v) for b, v in blend_curve.items()}
    layer_delta_mean = []
    for l in range(num_layers):
        vals = [v for v in layer_delta_acc[l] if v == v]
        layer_delta_mean.append(statistics.mean(vals) if vals else float("nan"))
    ranked = sorted(range(num_layers), key=lambda l: layer_delta_mean[l])  # most-helpful (lowest CE delta) first

    result = {
        "diagnostic": "loop1_blend_layer(#5,#6)",
        "ckpt": ckpt_dir,
        "num_samples": len(passages),
        "steps": steps,
        "cfg_scale": cfg_scale,
        "baseline_ce_no_injection": ms(base_acc),
        "blend_sensitivity_curve_ce": blend_curve_ms,
        "per_layer_ce_delta_vs_baseline_mean": layer_delta_mean,
        "layers_ranked_most_helpful_first": ranked[:10],
        "layers_ranked_most_harmful_first": ranked[::-1][:10],
        "interpretation": (
            "#5: CE lower than baseline at some blend => injected sampled state helps; "
            "if best blend is small, sampled state is only partially trustworthy. "
            "#6: layers with negative CE delta accept planned state usefully; layers "
            "with large positive delta reject it (injecting there hurts)."
        ),
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    print("\n=== SUMMARY (blend/layer #5,#6) ===")
    print(f"baseline CE (no inject): {result['baseline_ce_no_injection']['mean']:.4f}")
    for b in ["0.0", "0.25", "0.5", "0.75", "1.0"]:
        print(f"  blend {b}: CE {blend_curve_ms[b]['mean']:.4f}")
    print(f"most-helpful layers: {ranked[:5]}")
    print(f"written: {output}")
    print(f"gpu peak MiB: {int(torch.cuda.max_memory_allocated()/1024/1024) if device.startswith('cuda') else 0}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_samples", type=int, default=12)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cfg_scale", type=float, default=3.0)
    ap.add_argument("--output", default="/tmp/diag_loop1_blend_layer.json")
    args = ap.parse_args()
    run(args.ckpt_dir, args.device, args.num_samples, args.steps, args.cfg_scale, args.output)


if __name__ == "__main__":
    main()
