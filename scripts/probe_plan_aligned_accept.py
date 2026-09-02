"""Plan-aligned dual-model acceptance probe (variants A and B).

Prereq: a TRAJECTORY joint-scratch 0.4B drafter checkpoint (independent S1 +
birwkv S2 + condboundary), e.g.
  outputs_relay/drafter-04b-traj32x16-joint-scratch-coadapt/step_000XXXXX
and the 2.9B champion as verifier:
  outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000

Baseline reference: bare 0.4B -> bare 2.9B argmax agreement = 64% (probe_dual_model_accept.py).

This probe measures whether RELAY plan-priming lifts (or lowers) that acceptance:

  Variant A (single-side primed drafter):
    0.4B drafter is state-INJECTED by its own sampled trajectory plan (S2 -> S1 ->
    inject) before drafting a chunk; verifier is the PLAIN 2.9B (no injection).
    Tests the plan_spec_probe hypothesis: does priming only the drafter HURT
    acceptance (drafter pushed toward a plan the verifier never saw)?

  Variant B (plan-aligned both models):
    BOTH models are state-injected. Each is primed by ITS OWN plan (0.4B by 0.4B's
    plan, 2.9B by 2.9B's plan) so both are steered toward the SAME target region.
    Latent spaces are NOT shared (that hits the cross_model_latent 6.2x wall);
    only the *target region* is shared. Tests whether aligning both sides recovers
    or exceeds base 64%.

For each variant we measure per-chunk greedy argmax agreement between the drafter's
proposed tokens and the verifier's own argmax at the same positions, primed at the
chunk start by the respective planned state (blend applied via blend_into_cache).

NO TRAINING. Read-only model APIs.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.eval.diag_loop1_common import (  # noqa: E402
    PASSAGES,
    build_model,
    get_z_sampled,
    split_prefix_suffix,
)


@torch.no_grad()
def _inject_chunk_state(model, cache, z_plan, chunk_idx, blend):
    """Inject the planned state for chunk `chunk_idx` into `cache` (blend)."""
    states = model.predict_trajectory_states(z_plan)  # list per layer [1,H,heads,hd,hd]
    states_h = [s[:, chunk_idx] for s in states]  # each [1,heads,hd,hd]
    return model.blend_into_cache(cache, states_h, blend)


@torch.no_grad()
def _chunk_agreement(
    drafter, verifier,
    d_prefix_ids, v_prefix_ids,
    chunk_ids,
    d_plan, v_plan,
    chunk_idx, blend,
    prime_drafter, prime_verifier,
    device,
):
    """Return (#agree, #positions) for one chunk between drafter proposals and
    verifier argmax, each primed at chunk start if requested."""
    C = len(chunk_ids)
    # drafter autoregressively GENERATES C tokens (real speculative draft), primed if requested
    d_ids = torch.tensor([d_prefix_ids], device=device, dtype=torch.long)
    d_out = drafter.rwkv_model(input_ids=d_ids, use_cache=True, return_dict=True)
    d_cache = d_out.past_key_values
    if prime_drafter:
        d_cache = _inject_chunk_state(drafter, d_cache, d_plan, chunk_idx, blend)
    d_props = [d_out.logits[0, -1].argmax().item()]
    cur = torch.tensor([[d_props[0]]], device=device, dtype=torch.long)
    for _ in range(C - 1):
        o = drafter.rwkv_model(input_ids=cur, past_key_values=d_cache, use_cache=True, return_dict=True)
        d_cache = o.past_key_values
        d_props.append(o.logits[0, -1].argmax().item())
        cur = torch.tensor([[d_props[-1]]], device=device, dtype=torch.long)

    # verifier verifies the drafter's proposed tokens (its own argmax at each position)
    v_ids = torch.tensor([v_prefix_ids], device=device, dtype=torch.long)
    v_out = verifier.rwkv_model(input_ids=v_ids, use_cache=True, return_dict=True)
    v_cache = v_out.past_key_values
    if prime_verifier:
        v_cache = _inject_chunk_state(verifier, v_cache, v_plan, chunk_idx, blend)
    v_first = v_out.logits[0, -1].argmax().item()
    prop_t = torch.tensor([d_props], device=device, dtype=torch.long)
    v_ver = verifier.rwkv_model(input_ids=prop_t, past_key_values=v_cache,
                                use_cache=False, return_dict=True)
    v_props = [v_first] + [v_ver.logits[0, i].argmax().item() for i in range(C - 1)]

    agree = sum(1 for a, b in zip(d_props, v_props) if a == b)
    return agree, C


@torch.no_grad()
def run(args):
    device = args.device
    print(f"loading drafter (0.4B traj): {args.drafter}", flush=True)
    drafter, tok_d, dtype, pad_d = build_model(args.drafter, device)
    print(f"loading verifier (2.9B champion): {args.verifier}", flush=True)
    verifier, tok_v, _, pad_v = build_model(args.verifier, device)

    H = int(drafter.trajectory_horizon)
    C = int(drafter.trajectory_chunk_size)

    variants = {
        "A_primed_drafter_only": dict(prime_drafter=True, prime_verifier=False),
        "B_plan_aligned_both": dict(prime_drafter=True, prime_verifier=True),
        "ctrl_unprimed": dict(prime_drafter=False, prime_verifier=False),
    }
    stats = {k: {"agree": 0, "total": 0} for k in variants}

    n = min(args.num_passages, len(PASSAGES))
    for pi in range(n):
        passage = PASSAGES[pi]
        d_prefix, d_suffix, _ = split_prefix_suffix(tok_d, passage, drafter, pad_d)
        v_prefix, v_suffix, _ = split_prefix_suffix(tok_v, passage, verifier, pad_v)
        # each model samples ITS OWN plan (latent spaces not shared; target shared)
        d_plan = get_z_sampled(drafter, d_prefix, device, dtype, args.steps, args.cfg_scale)
        v_plan = get_z_sampled(verifier, v_prefix, device, dtype, args.steps, args.cfg_scale)
        # walk chunks over the (drafter) suffix window
        for h in range(H):
            chunk_ids = d_suffix[h * C:(h + 1) * C]
            if len(chunk_ids) < C:
                break
            # prefix seen by each model = original prefix + already-decoded chunks
            d_ctx = d_prefix + d_suffix[: h * C]
            v_ctx = v_prefix + v_suffix[: h * C]
            for name, cfg in variants.items():
                agree, tot = _chunk_agreement(
                    drafter, verifier, d_ctx, v_ctx, chunk_ids,
                    d_plan, v_plan, h, args.blend,
                    cfg["prime_drafter"], cfg["prime_verifier"], device,
                )
                stats[name]["agree"] += agree
                stats[name]["total"] += tot
        print(f"[{pi+1}/{n}] " + " ".join(
            f"{k}={stats[k]['agree']/max(1,stats[k]['total']):.3f}" for k in variants
        ), flush=True)

    result = {
        "drafter": args.drafter,
        "verifier": args.verifier,
        "num_passages": n,
        "blend": args.blend,
        "cfg_scale": args.cfg_scale,
        "steps": args.steps,
        "base_reference_pos0_agreement": 0.643,
        "variants": {
            k: stats[k]["agree"] / max(1, stats[k]["total"]) for k in variants
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== PLAN-ALIGNED ACCEPTANCE ===")
    for k in variants:
        print(f"  {k}: {result['variants'][k]:.3f}")
    print(f"  base reference (bare 0.4B->2.9B): 0.643")
    a = result["variants"]["A_primed_drafter_only"]
    b = result["variants"]["B_plan_aligned_both"]
    c = result["variants"]["ctrl_unprimed"]
    print(f"\n  variant A - ctrl (priming drafter only): {a-c:+.3f} "
          f"({'HURTS as predicted' if a < c else 'helps'})")
    print(f"  variant B - ctrl (plan-aligned both):    {b-c:+.3f} "
          f"({'RECOVERS/helps' if b >= c else 'still hurts'})")
    print(f"\nsaved: {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter", required=True,
                    help="0.4B trajectory joint-scratch drafter checkpoint dir")
    ap.add_argument("--verifier",
                    default="outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000")
    ap.add_argument("--out", default="outputs_eval/probe_plan_aligned_accept.json")
    ap.add_argument("--num_passages", type=int, default=16)
    ap.add_argument("--blend", type=float, default=0.7)
    ap.add_argument("--cfg_scale", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
