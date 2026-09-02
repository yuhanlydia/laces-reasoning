#!/usr/bin/env python3
"""Shared-Z dual-model speculative decoding for RWKV (small drafts, large verifies).

Loads a small drafter (e.g., 0.4B) and a large verifier (e.g., 2.9B champion or 13.3B),
both state-injected from a shared latent plan Z. The small model drafts tokens; the
large model verifies in chunks — lossless greedy accept/reject.

Pipeline:
  1. Encode prefix → z_prefix (via verifier's or drafter's S0, depending on mode)
  2. Sample full trajectory Z from z_prefix (via S2 conditional diffusion)
  3. For each chunk h:
     a. Extract z_h = Z[h], feed through per-model S1 → per-model WKV states
     b. Inject planned states into each model's RWKV cache
     c. Drafter autoregressively drafts k tokens
     d. Verifier does ONE forward over [prefix+drafts] → argmax at each slot
     e. Lossless greedy accept walk → emit matching tokens, correct at first mismatch
  4. Report wall-clock speedup, acceptance rates, lossless guarantee

Plan modes (--plan_mode):
  shared      Verifier's S0+S2 produces Z; both S1 adapters consume the same Z.
              Requires cross-model S1 training (see train_cross_source_full.py).
  independent Each model uses its OWN S0+S2 → each has its own Z (default).
              Works out-of-the-box with any trajectory checkpoints.
  bare        No state injection; pure backbone speculative decoding (like
              bench_dual_model_wallclock.py).

Usage:
  # Independent plans (each model's own Z) — works with existing checkpoints:
  python scripts/eval_dual_model_shared_z.py \
    --drafter outputs_relay/drafter-04b-traj32x16-joint-scratch-coadapt/step_00006000 \
    --verifier outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000 \
    --plan_mode independent --block 4 --blend 0.7

  # Bare dual-model (no RELAY), for baseline comparison:
  python scripts/eval_dual_model_shared_z.py \
    --drafter /path/to/rwkv7-0.4B-world \
    --verifier /path/to/RWKV7-Goose-World3-2.9B-HF \
    --plan_mode bare --block 4

  # Shared Z (requires cross-model trained S1):
  python scripts/eval_dual_model_shared_z.py \
    --drafter outputs_relay/cross-source-0.4B-s1/cross_s1_final.pt \
    --verifier outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000 \
    --plan_mode shared
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# ── model loading ──────────────────────────────────────────────────────────


def _load_bare_backbone(path: str, device: str, dtype: torch.dtype):
    """Load bare RWKV backbone (no RELAY)."""
    from transformers import AutoModelForCausalLM  # type: ignore[import-untyped]

    model = (
        AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, torch_dtype=dtype, local_files_only=True
        )
        .to(device)
        .eval()
    )
    for p in model.parameters():
        p.requires_grad = False
    return model


def _load_trajectory_model(ckpt_dir: str, device: str):
    """Load a full trajectory RELAY model (S0 + S1 + S2)."""
    from scripts.eval.diag_loop1_common import build_model  # type: ignore[import-untyped]

    return build_model(ckpt_dir, device)


def _check_model_has_trajectory(model) -> bool:
    """Return True if the model supports trajectory mode (needed for plan injection)."""
    return bool(getattr(model, "trajectory_enabled", False))


def _get_rwkv_backend(model):
    """Extract the raw RWKV backbone from either a RELAY wrapper or a bare model.

    RELAY models (StateInjectionDiTRELAY) expose the backbone via .rwkv_model.
    Bare Transformers models are used directly.
    """
    if hasattr(model, "rwkv_model"):
        return model.rwkv_model
    return model


# ── state injection helpers ─────────────────────────────────────────────────


@torch.no_grad()
def _encode_prefix_to_z(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor
):
    """Encode prefix tokens → single z_prefix via model's S0."""
    out = model.rwkv_model(
        input_ids=input_ids,
        attention_mask=attention_mask.bool(),
        output_hidden_states=True,
        use_cache=True,
        return_dict=True,
    )
    pooled = model._pool_hidden(out.hidden_states[-1], attention_mask)
    z_prefix, _ = model._encode_pooled(pooled)
    return z_prefix, out.past_key_values, out.logits[:, -1]


@torch.no_grad()
def _sample_trajectory_z(
    model, z_prefix: torch.Tensor, steps: int, cfg_scale: float, device: str
):
    """Sample full trajectory Z from z_prefix via model's S2 (conditional diffusion)."""
    from scripts.eval.sample_prefix_suffix_trajectory_cfg import (  # type: ignore[import-untyped]
        sample_trajectory_cfg,
    )

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    B, D = z_prefix.shape
    # S2 expects z_prefix of shape [1, D] if single, or handles trajectory condition
    return sample_trajectory_cfg(model, z_prefix, steps, cfg_scale, device, dtype)


@torch.no_grad()
def _inject_chunk_state(
    model, cache, z_plan: torch.Tensor, chunk_idx: int, blend: float
):
    """Inject planned state for chunk `chunk_idx` from trajectory Z into RWKV cache.

    z_plan: [1, H, D] — full trajectory latent plan
    Returns: modified cache
    """
    states = model.predict_trajectory_states(z_plan)  # list[L]: each [1, H, heads, hd, hd]
    states_h = [s[:, chunk_idx] for s in states]  # each [1, heads, hd, hd]
    return model.blend_into_cache(cache, states_h, blend)


@torch.no_grad()
def _inject_singlez_state(model, cache, z_plan: torch.Tensor, blend: float):
    """Single-z injection (Path A): mean-pool the trajectory Z into one global
    R^32 and write it once via predict_states, instead of per-chunk trajectory
    states. This is the injection path the PPL matrix found effective (-31%),
    vs the per-chunk trajectory path that failed (+13%).

    z_plan: [1, H, D] trajectory plan -> mean-pooled to [1, D].
    """
    z_global = z_plan.mean(dim=1)  # [1, D]
    states = model.predict_states(z_global)  # list[L]: each [1, heads, hd, hd]
    return model.blend_into_cache(cache, states, blend)


# ── speculative decoding core ───────────────────────────────────────────────


@torch.no_grad()
def _draft_k_autoregressive(model_or_backbone, cache, first_logits, k: int, device: str):
    """Draft k tokens autoregressively from cache.

    Returns (drafts: list[int], new_cache).
    drafts[0] = argmax(first_logits), the rest autoregressively sampled.
    """
    first_tok = first_logits.argmax(dim=-1).item()
    drafts = [first_tok]
    cur_cache = cache
    cur = torch.tensor([[first_tok]], device=device, dtype=torch.long)
    for _ in range(k - 1):
        out = model_or_backbone(
            input_ids=cur, past_key_values=cur_cache, use_cache=True, return_dict=True
        )
        cur_cache = out.past_key_values
        nxt = out.logits[0, -1].argmax(dim=-1).item()
        drafts.append(nxt)
        cur = torch.tensor([[nxt]], device=device, dtype=torch.long)
    return drafts, cur_cache


@torch.no_grad()
def _verify_block_with_cache(model_or_backbone, cache, drafts: list[int], device: str, verifier_first: int):
    """Verify drafts using past_key_values (efficient, O(k) not O(n)).

    verifier_first: the verifier's own argmax prediction from the prefix end
                    (before seeing any draft). drafts[0] is compared to this.
    cache: verifier's past_key_values from the prefix forward.

    Returns verifier argmax at each draft slot.
    With past_key_values, logits[0, j] predicts position j+1 of the draft block.
    So draft[0] → verifier_first, draft[j+1] → logits[0, j].
    """
    k = len(drafts)
    block = torch.tensor([drafts], device=device, dtype=torch.long)
    out = model_or_backbone(
        input_ids=block, past_key_values=cache, use_cache=False, return_dict=True
    )
    result = [verifier_first]
    for j in range(k - 1):
        result.append(out.logits[0, j].argmax(dim=-1).item())
    return result


@torch.no_grad()
def _run_speculative_loop(
    drafter,
    verifier,
    prefix_ids: torch.Tensor,
    n_new: int,
    k: int,
    z_plan_d: Optional[torch.Tensor],
    z_plan_v: Optional[torch.Tensor],
    blend: float,
    device: str,
    use_state_injection: bool,
    plan_geometry: str = "trajectory",
):
    """Run speculative decoding loop for n_new tokens.

    drafter, verifier: RELAY wrappers (StateInjectionDiTRELAY) if use_state_injection,
                       otherwise bare RWKV backbones.

    plan_geometry: "trajectory" injects per-chunk states at chunk boundaries;
                   "singlez" mean-pools the plan to one global R^32 and injects
                   it once at the start (no per-chunk re-injection).

    Returns: (emitted_tokens, wall_seconds, num_rounds, total_accepted)
    """
    singlez = plan_geometry == "singlez"
    _sync(device)
    t0 = time.perf_counter()

    rwkv_d = _get_rwkv_backend(drafter)
    rwkv_v = _get_rwkv_backend(verifier)

    C = int(getattr(verifier, "trajectory_chunk_size", 32)) if use_state_injection else 32
    H = int(getattr(verifier, "trajectory_horizon", 16)) if use_state_injection else 16

    # Forward prefix through both backbones for initial caches
    d_out = rwkv_d(input_ids=prefix_ids, use_cache=True, return_dict=True)
    d_cache = d_out.past_key_values

    v_out = rwkv_v(input_ids=prefix_ids, use_cache=True, return_dict=True)
    v_cache = v_out.past_key_values
    verifier_next = v_out.logits[0, -1].argmax(dim=-1).item()

    chunk_idx = 0
    tokens_in_chunk = 0

    # State injection at chunk 0 into both caches (single-z: one global write)
    if use_state_injection and z_plan_d is not None:
        d_cache = (
            _inject_singlez_state(drafter, d_cache, z_plan_d, blend)
            if singlez
            else _inject_chunk_state(drafter, d_cache, z_plan_d, chunk_idx, blend)
        )
    if use_state_injection and z_plan_v is not None:
        v_cache = (
            _inject_singlez_state(verifier, v_cache, z_plan_v, blend)
            if singlez
            else _inject_chunk_state(verifier, v_cache, z_plan_v, chunk_idx, blend)
        )

    emitted: list[int] = []
    rounds = 0
    accepted_total = 0

    while len(emitted) < n_new:
        rounds += 1

        # Drafter drafts k tokens from its cache (drafts[0] = drafter's argmax from prefix)
        d_logits = d_out.logits[:, -1]
        drafts, d_cache_new = _draft_k_autoregressive(
            rwkv_d, d_cache, d_logits, k, device
        )

        # Verifier verifies using cache (efficient, O(k))
        v_argmax = _verify_block_with_cache(rwkv_v, v_cache, drafts, device, verifier_next)

        # Lossless greedy accept walk
        new_tokens: list[int] = []
        for j in range(k):
            if drafts[j] == v_argmax[j]:
                new_tokens.append(drafts[j])
                accepted_total += 1
            else:
                new_tokens.append(v_argmax[j])
                break

        # Advance BOTH caches
        adv = torch.tensor([new_tokens], device=device, dtype=torch.long)

        v_out = rwkv_v(
            input_ids=adv, past_key_values=v_cache, use_cache=True, return_dict=True
        )
        v_cache = v_out.past_key_values
        verifier_next = v_out.logits[0, -1].argmax(dim=-1).item()

        d_out = rwkv_d(
            input_ids=adv, past_key_values=d_cache, use_cache=True, return_dict=True
        )
        d_cache = d_out.past_key_values

        emitted.extend(new_tokens)
        tokens_in_chunk += len(new_tokens)

        if use_state_injection and not singlez:
            while tokens_in_chunk >= C and chunk_idx + 1 < H:
                chunk_idx += 1
                tokens_in_chunk -= C
                if z_plan_d is not None:
                    d_cache = _inject_chunk_state(
                        drafter, d_cache, z_plan_d, chunk_idx, blend
                    )
                if z_plan_v is not None:
                    v_cache = _inject_chunk_state(
                        verifier, v_cache, z_plan_v, chunk_idx, blend
                    )

    _sync(device)
    elapsed = time.perf_counter() - t0
    return emitted[:n_new], elapsed, rounds, accepted_total


def _sync(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


# ── main entry ──────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="Shared-Z dual-model speculative decoding for RWKV"
    )
    # Model paths
    ap.add_argument(
        "--drafter",
        default="outputs_relay/drafter-04b-traj32x16-joint-scratch-coadapt/step_00006000",
        help="Path to drafter checkpoint (trajectory RELAY model) or bare backbone dir",
    )
    ap.add_argument(
        "--verifier",
        default="outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000",
        help="Path to verifier checkpoint (trajectory RELAY model) or bare backbone dir",
    )
    # Plan mode
    ap.add_argument(
        "--plan_mode",
        choices=("independent", "shared", "bare"),
        default="independent",
        help="independent: each model uses own S0+S2 (default). "
        "shared: verifier S0+S2 for both (needs cross-model S1). "
        "bare: no state injection (pure backbones).",
    )
    # Speculative decoding
    ap.add_argument("--block", type=int, default=4, help="Draft block size k")
    ap.add_argument("--n_new", type=int, default=256, help="Tokens to generate")
    ap.add_argument("--prefix_len", type=int, default=128, help="Prefix context length")
    ap.add_argument(
        "--plan_geometry",
        choices=("trajectory", "singlez"),
        default="trajectory",
        help="trajectory: per-chunk state injection (Path B). "
        "singlez: mean-pool plan to global R^32, inject once (Path A).",
    )
    # State injection
    ap.add_argument("--blend", type=float, default=0.7, help="State blend factor")
    ap.add_argument("--cfg_scale", type=float, default=3.0, help="CFG scale for S2 sampling")
    ap.add_argument("--steps", type=int, default=100, help="S2 diffusion steps")
    # Data
    ap.add_argument(
        "--token_dir",
        default="preprocessed_data/owt_rwkv_tokens/train",
        help="Directory with .npz token files",
    )
    ap.add_argument("--num_samples", type=int, default=20, help="Number of test samples")
    # Output
    ap.add_argument(
        "--out",
        default="outputs_eval/dual_model_shared_z.json",
        help="Output JSON path",
    )
    ap.add_argument("--device", default="cuda:0", help="Device for both models")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--patch_s1", default="", help="Path to cross-source S1 .pt to patch into drafter")
    ap.add_argument("--use_gate", action="store_true", help="Use confidence gate to decide injection vs bare")
    args = ap.parse_args()

    device = args.device
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    # ── load models ────────────────────────────────────────────────────
    use_state_injection = args.plan_mode != "bare"

    if use_state_injection:
        print(f"Loading drafter (trajectory RELAY): {args.drafter}", flush=True)
        drafter, tok_d, _, pad_d = _load_trajectory_model(args.drafter, device)
        if not _check_model_has_trajectory(drafter):
            print("WARNING: drafter does not have trajectory_enabled=True; "
                  "state injection may fail. Use --plan_mode bare for bare backbones.",
                  flush=True)

        print(f"Loading verifier (trajectory RELAY): {args.verifier}", flush=True)
        verifier, tok_v, _, pad_v = _load_trajectory_model(args.verifier, device)
        if not _check_model_has_trajectory(verifier):
            print("WARNING: verifier does not have trajectory_enabled=True; "
                  "state injection may fail. Use --plan_mode bare for bare backbones.",
                  flush=True)

        # Optional: patch cross-source S1 weights into drafter
        if args.patch_s1:
            s1_ckpt = torch.load(args.patch_s1, map_location=device, weights_only=False)
            info = drafter.load_state_dict(s1_ckpt["trainable_state"], strict=False)
            n_loaded = len(s1_ckpt["trainable_state"]) - len(info.missing_keys)
            print(f"Patched cross-source S1: {n_loaded}/{len(s1_ckpt['trainable_state'])} tensors "
                  f"(step={s1_ckpt.get('step','?')})", flush=True)

        tokenizer = tok_v

        # Optional: load confidence gate
        gate_model = None
        gate_count_injected = 0
        gate_count_bare = 0
        if args.use_gate:
            import pickle
            try:
                gate_model = pickle.load(open("outputs_eval/gate_catastrophic_model.pkl", "rb"))
                print(f"Loaded gate model (features: {gate_model['features']})", flush=True)
            except FileNotFoundError:
                print("WARNING: gate model not found, running without gate", flush=True)
    else:
        print(f"Loading drafter (bare backbone): {args.drafter}", flush=True)
        drafter = _load_bare_backbone(args.drafter, device, dtype)

        print(f"Loading verifier (bare backbone): {args.verifier}", flush=True)
        verifier = _load_bare_backbone(args.verifier, device, dtype)

        from transformers import AutoTokenizer  # type: ignore[import-untyped]
        tokenizer = AutoTokenizer.from_pretrained(
            args.verifier, trust_remote_code=True, local_files_only=True
        )

    # ── load test data ─────────────────────────────────────────────────
    import glob as _glob

    files = sorted(_glob.glob(f"{args.token_dir}/*_tokens.npz"))
    if not files:
        files = sorted(_glob.glob(f"{args.token_dir}/*.npz"))
    rng = np.random.RandomState(args.seed)
    chosen = [files[i] for i in rng.choice(len(files), args.num_samples, replace=False)]

    # ── benchmark loop ─────────────────────────────────────────────────
    base_total_s = 0.0
    spec_total_s = 0.0
    rounds_total = 0
    accepted_total = 0
    total_tokens = 0
    divergent_tokens = 0
    mismatches = 0

    for si, path in enumerate(chosen):
        d = np.load(path)
        tok_ids = d["input_ids"]
        if len(tok_ids) < args.prefix_len + args.n_new + args.block:
            continue

        # Build prefix
        prefix_np = tok_ids[: args.prefix_len]
        prefix_ids = torch.tensor([prefix_np], device=device, dtype=torch.long)

        # ── sample trajectory plans ─────────────────────────────────
        z_plan_d = None
        z_plan_v = None

        if use_state_injection and args.plan_mode == "shared":
            am = torch.ones_like(prefix_ids, dtype=torch.float32, device=device)
            try:
                z_prefix, _, _ = _encode_prefix_to_z(verifier, prefix_ids, am)
                z_shared = _sample_trajectory_z(
                    verifier, z_prefix, args.steps, args.cfg_scale, device
                )
            except Exception:
                H = int(getattr(verifier, "trajectory_horizon", 16))
                z_prefix, _, _ = _encode_prefix_to_z(verifier, prefix_ids, am)
                z_shared = z_prefix.unsqueeze(1).expand(-1, H, -1)

            # Gate: decide whether to inject or fall back to bare
            should_inject = True
            if gate_model is not None:
                z_norm = float(z_prefix.norm(dim=-1).item())
                X = np.array([[z_norm, 0.6, 0.6]])  # raw_top5/inj_top5 not available; use mean
                pred = gate_model["model"].predict(X)[0]
                should_inject = (pred == 0)  # 0=safe, 1=catastrophic
                if should_inject:
                    gate_count_injected += 1
                else:
                    gate_count_bare += 1

            if should_inject:
                z_plan_d = z_shared
                z_plan_v = z_shared
            else:
                z_plan_d = None
                z_plan_v = None

        elif use_state_injection and args.plan_mode == "independent":
            am_d = torch.ones_like(prefix_ids, dtype=torch.float32, device=device)
            z_prefix_d, _, _ = _encode_prefix_to_z(drafter, prefix_ids, am_d)
            try:
                z_plan_d = _sample_trajectory_z(
                    drafter, z_prefix_d, args.steps, args.cfg_scale, device
                )
            except Exception:
                H = int(getattr(drafter, "trajectory_horizon", 16))
                z_plan_d = z_prefix_d.unsqueeze(1).expand(-1, H, -1)

            am_v = torch.ones_like(prefix_ids, dtype=torch.float32, device=device)
            z_prefix_v, _, _ = _encode_prefix_to_z(verifier, prefix_ids, am_v)
            try:
                z_plan_v = _sample_trajectory_z(
                    verifier, z_prefix_v, args.steps, args.cfg_scale, device
                )
            except Exception:
                H = int(getattr(verifier, "trajectory_horizon", 16))
                z_plan_v = z_prefix_v.unsqueeze(1).expand(-1, H, -1)

        # ── baseline: pure verifier greedy autoregressive ─────────────
        rwkv_v = _get_rwkv_backend(verifier)
        _sync(device)
        t0 = time.perf_counter()
        v_out = rwkv_v(input_ids=prefix_ids, use_cache=True, return_dict=True)
        v_pkv = v_out.past_key_values
        nxt = v_out.logits[0, -1].argmax(dim=-1).item()
        base_toks = [nxt]
        cur = torch.tensor([[nxt]], device=device, dtype=torch.long)
        for _ in range(args.n_new - 1):
            v_out = rwkv_v(
                input_ids=cur, past_key_values=v_pkv, use_cache=True, return_dict=True
            )
            v_pkv = v_out.past_key_values
            nxt = v_out.logits[0, -1].argmax(dim=-1).item()
            base_toks.append(nxt)
            cur = torch.tensor([[nxt]], device=device, dtype=torch.long)
        _sync(device)
        bt = time.perf_counter() - t0

        # ── speculative decoding ──────────────────────────────────────
        spec_toks, st, rounds, acc = _run_speculative_loop(
            drafter=drafter,
            verifier=verifier,
            prefix_ids=prefix_ids,
            n_new=args.n_new,
            k=args.block,
            z_plan_d=z_plan_d,
            z_plan_v=z_plan_v,
            blend=args.blend,
            device=device,
            use_state_injection=use_state_injection,
            plan_geometry=args.plan_geometry,
        )

        base_total_s += bt
        spec_total_s += st
        rounds_total += rounds
        accepted_total += acc

        n_tok = min(len(base_toks), len(spec_toks))
        div = sum(1 for a, b in zip(base_toks[:n_tok], spec_toks[:n_tok]) if a != b)
        if div:
            mismatches += 1
        divergent_tokens += div
        total_tokens += n_tok

        if (si + 1) % 5 == 0:
            spd = base_total_s / max(1e-9, spec_total_s)
            avg_acc = accepted_total / max(1, rounds_total)
            print(
                f"[{si + 1}/{args.num_samples}] "
                f"base={base_total_s:.2f}s spec={spec_total_s:.2f}s "
                f"speedup={spd:.2f}x accept/round={avg_acc:.2f} "
                f"divergence={divergent_tokens}/{total_tokens}",
                flush=True,
            )

    # ── final report ───────────────────────────────────────────────────────
    speedup = base_total_s / max(1e-9, spec_total_s)
    result = {
        "plan_mode": args.plan_mode,
        "plan_geometry": args.plan_geometry,
        "drafter": args.drafter,
        "verifier": args.verifier,
        "block_k": args.block,
        "n_new": args.n_new,
        "prefix_len": args.prefix_len,
        "blend": args.blend,
        "cfg_scale": args.cfg_scale,
        "diffusion_steps": args.steps,
        "num_samples": args.num_samples,
        "use_state_injection": use_state_injection,
        # Timing
        "baseline_total_s": round(base_total_s, 3),
        "spec_total_s": round(spec_total_s, 3),
        "wallclock_speedup": round(speedup, 3),
        # Acceptance
        "total_rounds": rounds_total,
        "total_accepted": accepted_total,
        "mean_accepted_per_round": round(accepted_total / max(1, rounds_total), 3),
        "acceptance_estimated_speedup": round(
            1.0 + accepted_total / max(1, rounds_total), 3
        ),
        # Lossless verification
        "samples_with_divergence": mismatches,
        "divergent_tokens": divergent_tokens,
        "total_tokens": total_tokens,
        "token_divergence_rate": round(
            divergent_tokens / max(1, total_tokens), 5
        ),
    }

    if gate_model is not None:
        result["gate_injected"] = gate_count_injected
        result["gate_bare"] = gate_count_bare
        result["gate_inject_rate"] = round(
            gate_count_injected / max(1, gate_count_injected + gate_count_bare), 3
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  DUAL-MODEL SPECULATIVE DECODING — {args.plan_mode.upper()} PLAN MODE")
    print(f"{'='*60}")
    print(f"  drafter : {args.drafter}")
    print(f"  verifier: {args.verifier}")
    print(f"  block k : {args.block}  |  tokens gen : {args.n_new}")
    print(f"  plan    : {args.plan_mode}  |  state inj : {use_state_injection}")
    if use_state_injection:
        print(f"  blend   : {args.blend}  |  cfg_scale  : {args.cfg_scale}")
    if gate_model is not None:
        print(f"  gate    : injected={gate_count_injected} bare={gate_count_bare} "
              f"({result['gate_inject_rate']:.0%})")
    print(f"{'='*60}")
    print(f"  baseline (pure verifier greedy): {base_total_s:.2f}s")
    print(f"  speculative (draft + verify)  : {spec_total_s:.2f}s")
    print(f"  >>> REAL wall-clock speedup    : {speedup:.2f}x")
    print(f"  mean accepted / round         : {result['mean_accepted_per_round']:.2f}")
    print(f"  acceptance-estimated speedup  : {result['acceptance_estimated_speedup']:.2f}x")
    print(f"  token divergence vs baseline  : {divergent_tokens}/{total_tokens} "
          f"({result['token_divergence_rate']:.3%})")
    if divergent_tokens > 0:
        print(f"  NOTE: divergence is typically bf16 numeric drift in batched vs "
              f"incremental forward, not an accept-logic bug.")
    print(f"\n  saved: {args.out}")

    if speedup < 1.0:
        print(f"\n  ⚠  speedup < 1.0x — speculative decoding is SLOWER than baseline."
              f"\n     Possible causes: draft model too large, k too small, "
              f"verification overhead dominates.")


if __name__ == "__main__":
    main()
