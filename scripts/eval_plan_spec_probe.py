"""Plan-conditioned speculative decoding probe (idea 1 + idea 2).

Self-contained; does NOT use the buggy eval_all_spec.py path. All hidden states
are cast to the Medusa head weight dtype before the head, which fixes the
Float-vs-BFloat16 crash that killed eval_all_spec.py.

IDEA 1 (plan-conditioned speculation):
  Does the trained Medusa draft head accept MORE at chunk boundaries (where the
  planned per-chunk state was just injected) than at generic mid-chunk
  positions? If boundary acceptance > non-boundary acceptance, the planned
  future state carries usable draft signal exactly where the plan lands.

IDEA 2 (acceptance as a plan-quality probe):
  Run the SAME Medusa head twice: once with CLEAN-Z injection (encoder latent
  from real text) and once with SAMPLED-Z injection (S2 diffusion sample from
  the prefix). If clean-Z draft acceptance > sampled-Z, speculative acceptance
  is a training-free, decoder-grounded metric of plan executability.

Acceptance here = the frozen RWKV verifier's argmax at the draft position equals
the head's argmax draft token (greedy self-agreement of the executor with the
head's prediction), measured per position, teacher-forced over real text so the
context is identical across conditions.
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

import eval.diag_loop1_common as Diag
from scripts.train_medusa import MedusaHeads
from scripts.eval.sample_prefix_suffix_trajectory_cfg import encode_prefix, sample_trajectory_cfg


@torch.no_grad()
def hidden_with_injection(model, ids, am, Z):
    """Return last-layer hidden [1,S,D] after injecting per-chunk states from Z."""
    states = model.predict_trajectory_states(Z)
    out = model.rwkv_model(input_ids=ids, attention_mask=am.bool(),
                           output_hidden_states=True, use_cache=True, return_dict=True)
    pkv = model.inject_into_cache(out.past_key_values, states)
    out2 = model.rwkv_model(input_ids=ids, attention_mask=am.bool(),
                            past_key_values=pkv, output_hidden_states=True,
                            use_cache=True, return_dict=True)
    return out2.hidden_states[-1], out2.logits


@torch.no_grad()
def verifier_argmax(model, ids, am, Z):
    """Frozen RWKV next-token argmax at each position, under the SAME injection."""
    _, logits = hidden_with_injection(model, ids, am, Z)
    return logits[0].argmax(-1)  # [S]


@torch.no_grad()
def draft_acceptance_by_position(model, medusa, ids, am, Z, head_dtype):
    """Per-position h0 acceptance: does medusa head-0 argmax == verifier argmax?

    Returns a bool tensor [S-1] aligned so index t means 'draft for token t+1'.
    Acceptance = executor greedily agrees with the head's drafted next token.
    """
    hidden, logits = hidden_with_injection(model, ids, am, Z)
    S = hidden.shape[1]
    h_in = hidden[0].to(head_dtype)  # [S, D]
    head0 = medusa([h_in])[0] if False else medusa(h_in)[0]  # [S, vocab]
    draft_tok = head0.argmax(-1)  # [S]  head0[t] drafts token t+1
    verifier_tok = logits[0].argmax(-1)  # [S]  verifier[t] = greedy token t+1
    T = S - 1
    accept = (draft_tok[:T] == verifier_tok[:T])  # [T]
    return accept.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000")
    ap.add_argument("--medusa_path", default="outputs_relay/medusa-compare/medusa_final.pt")
    ap.add_argument("--token_dir", default="preprocessed_data/owt_rwkv_tokens/train")
    ap.add_argument("--num_samples", type=int, default=30)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cfg_scale", type=float, default=3.0)
    ap.add_argument("--out", default="outputs_eval/plan_spec_probe.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model, tok, dtype, pad_id = Diag.build_model(args.ckpt_dir, args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    H, C = int(model.trajectory_horizon), int(model.trajectory_chunk_size)
    S = H * C
    hidden_dim = model.rwkv_model.config.hidden_size

    ckpt = torch.load(args.medusa_path, map_location=args.device, weights_only=False)
    n_heads = ckpt.get("config", {}).get("num_heads", 4)
    medusa = MedusaHeads(hidden_dim, 65536, n_heads).to(args.device)
    medusa.load_state_dict(ckpt["medusa_state"])
    medusa.eval()
    head_dtype = medusa.heads[0][0].weight.dtype  # fix: match head dtype
    print(f"model={args.ckpt_dir}", flush=True)
    print(f"medusa={args.medusa_path} step={ckpt.get('step')} heads={n_heads} dtype={head_dtype}", flush=True)
    print(f"H={H} C={C} S={S}", flush=True)

    files = sorted(glob.glob(f"{args.token_dir}/*.npz"))
    np.random.seed(42)
    idxs = np.random.choice(len(files), args.num_samples, replace=False)

    # chunk boundary positions in the draft-index space [0..S-2]:
    # token position p is a chunk boundary if p % C == 0 (start of a chunk).
    # The draft at index t predicts token t+1, so "drafting INTO a boundary" is
    # t+1 == k*C  =>  t == k*C - 1. We collect both the boundary-entering draft
    # (t = k*C - 1) and generic mid-chunk drafts for contrast.
    boundary_draft_idx = set((k * C - 1) for k in range(1, H) if k * C - 1 >= 0)

    # accumulators
    clean_accept_boundary, clean_accept_nonboundary = [], []
    clean_accept_all, sampled_accept_all = [], []

    for si, idx in enumerate(idxs):
        d = np.load(files[idx])
        ids_np = d["input_ids"][:S]
        am_np = d["attention_mask"][:S]
        if len(ids_np) < S:
            continue
        ids = torch.tensor([ids_np], device=args.device, dtype=torch.long)
        am = torch.tensor([am_np], device=args.device, dtype=torch.float32)

        # CLEAN-Z: encode real text into trajectory latents
        Z_clean = model._encode_trajectory_chunks(ids, am.bool())[0].reshape(1, H, -1)

        # SAMPLED-Z: S2 diffusion sample from the prefix (first half as condition)
        prefix_len = (H // 2) * C
        pre_ids = ids[:, :prefix_len]
        pre_am = am[:, :prefix_len]
        z_prefix, _c, _l = encode_prefix(model, pre_ids, pre_am)
        Z_sampled = sample_trajectory_cfg(model, z_prefix, args.steps, args.cfg_scale, args.device, dtype)

        # IDEA 1 + IDEA 2 acceptance vectors (teacher-forced context = real text)
        acc_clean = draft_acceptance_by_position(model, medusa, ids, am, Z_clean, head_dtype)  # [S-1]
        acc_sampled = draft_acceptance_by_position(model, medusa, ids, am, Z_sampled, head_dtype)  # [S-1]

        T = acc_clean.shape[0]
        for t in range(T):
            v = bool(acc_clean[t].item())
            clean_accept_all.append(v)
            if t in boundary_draft_idx:
                clean_accept_boundary.append(v)
            else:
                clean_accept_nonboundary.append(v)
        sampled_accept_all.extend(bool(x) for x in acc_sampled.tolist())

        if si % 5 == 0:
            cb = np.mean(clean_accept_boundary) if clean_accept_boundary else 0
            cn = np.mean(clean_accept_nonboundary) if clean_accept_nonboundary else 0
            print(f"[{si}/{len(idxs)}] boundary={cb:.3f} nonboundary={cn:.3f} "
                  f"clean_all={np.mean(clean_accept_all):.3f} sampled_all={np.mean(sampled_accept_all):.3f}",
                  flush=True)

    res = {
        "ckpt": args.ckpt_dir,
        "medusa": args.medusa_path,
        "num_samples": int(args.num_samples),
        "cfg_scale": args.cfg_scale,
        "steps": args.steps,
        "idea1_chunk_boundary": {
            "boundary_accept": float(np.mean(clean_accept_boundary)) if clean_accept_boundary else None,
            "nonboundary_accept": float(np.mean(clean_accept_nonboundary)) if clean_accept_nonboundary else None,
            "n_boundary": len(clean_accept_boundary),
            "n_nonboundary": len(clean_accept_nonboundary),
        },
        "idea2_plan_quality_probe": {
            "clean_z_accept": float(np.mean(clean_accept_all)) if clean_accept_all else None,
            "sampled_z_accept": float(np.mean(sampled_accept_all)) if sampled_accept_all else None,
            "n_clean": len(clean_accept_all),
            "n_sampled": len(sampled_accept_all),
        },
    }
    b = res["idea1_chunk_boundary"]
    q = res["idea2_plan_quality_probe"]
    if b["boundary_accept"] is not None and b["nonboundary_accept"] is not None:
        res["idea1_delta_boundary_minus_nonboundary"] = b["boundary_accept"] - b["nonboundary_accept"]
    if q["clean_z_accept"] is not None and q["sampled_z_accept"] is not None:
        res["idea2_delta_clean_minus_sampled"] = q["clean_z_accept"] - q["sampled_z_accept"]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)

    print("\n=== IDEA 1: chunk-boundary draft acceptance (clean-Z) ===")
    print(f"  boundary    : {b['boundary_accept']:.4f}  (n={b['n_boundary']})")
    print(f"  non-boundary: {b['nonboundary_accept']:.4f}  (n={b['n_nonboundary']})")
    if "idea1_delta_boundary_minus_nonboundary" in res:
        dl = res["idea1_delta_boundary_minus_nonboundary"]
        verdict = "SUPPORTS plan-conditioned edge" if dl > 0.02 else ("neutral" if abs(dl) <= 0.02 else "AGAINST")
        print(f"  delta       : {dl:+.4f}  -> {verdict}")
    print("\n=== IDEA 2: acceptance as plan-quality probe ===")
    print(f"  clean-Z  accept: {q['clean_z_accept']:.4f}  (n={q['n_clean']})")
    print(f"  sampled-Z accept: {q['sampled_z_accept']:.4f}  (n={q['n_sampled']})")
    if "idea2_delta_clean_minus_sampled" in res:
        dl = res["idea2_delta_clean_minus_sampled"]
        verdict = "SUPPORTS acceptance-as-probe" if dl > 0.02 else ("neutral" if abs(dl) <= 0.02 else "AGAINST")
        print(f"  delta       : {dl:+.4f}  -> {verdict}")
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
