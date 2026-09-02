#!/usr/bin/env python3
"""P1: Behavior-alignment S1 training.

Retrains 0.4B cross-source S1 using CE(verifier_argmax, drafter_logits)
instead of CE(gold_token, drafter_logits). Loads frozen 13.3B verifier
alongside to compute argmax on-the-fly.
"""
import argparse, glob, sys, os, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.eval.diag_loop1_common import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter_ckpt",
                    default="outputs_relay/drafter-04b-traj32x16-joint-scratch-coadapt/step_00006000")
    ap.add_argument("--resume_s1",
                    default="outputs_relay/cross-source-0.4B-s1-denoised-v2/cross_s1_final.pt")
    ap.add_argument("--verifier_ckpt",
                    default="outputs_relay/owt512-traj32x16-13.3B-basis32-prefix-suffix-blend0p5-s2-rwkv-rf/step_00150000")
    ap.add_argument("--latent_dir", default="preprocessed_data/owt_13b_s2_denoised_z/train")
    ap.add_argument("--token_dir", default="preprocessed_data/owt_rwkv_tokens/train")
    ap.add_argument("--save_dir", default="outputs_relay/cross-source-0.4B-s1-behavior")
    ap.add_argument("--num_steps", type=int, default=20000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--save_every", type=int, default=5000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device; dtype = torch.bfloat16

    # Load drafter + patch S1
    print("Loading 0.4B drafter...", flush=True)
    drafter, _, _, _ = build_model(args.drafter_ckpt, device)
    s1 = torch.load(args.resume_s1, map_location=device, weights_only=False)
    drafter.load_state_dict(s1["trainable_state"], strict=False); drafter.eval()
    rwkv_d = drafter.rwkv_model

    # Load verifier (frozen, only for argmax)
    print("Loading 13.3B verifier...", flush=True)
    verifier, _, _, _ = build_model(args.verifier_ckpt, device)
    verifier.eval(); rwkv_v = verifier.rwkv_model
    for p in verifier.parameters():
        p.requires_grad = False

    # Unfreeze S1 only
    s1_names = {"alpha_heads", "state_basis", "state_scale", "state_norm"}
    for name, p in drafter.named_parameters():
        p.requires_grad = (name.split(".")[0] in s1_names)
    n_trainable = sum(p.numel() for p in drafter.parameters() if p.requires_grad)
    print(f"Trainable: {n_trainable/1e6:.1f}M params", flush=True)

    opt = torch.optim.AdamW([p for p in drafter.parameters() if p.requires_grad], lr=args.lr)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Data loading
    latent_files = sorted(glob.glob(f"{args.latent_dir}/*.npy"))
    token_files = {Path(f).stem.replace("_tokens", ""): f
                   for f in sorted(glob.glob(f"{args.token_dir}/*.npz"))}

    H = int(drafter.trajectory_horizon)
    C = int(drafter.trajectory_chunk_size)

    t0 = time.time()
    for step in range(1, args.num_steps + 1):
        # Random batch
        chosen = np.random.choice(latent_files, args.batch_size, replace=False)
        batch_ids, batch_ams, batch_Z = [], [], []
        for lf in chosen:
            stem = Path(lf).stem.replace("_tokens", "").replace("_latent", "")
            tf = token_files.get(stem)
            if tf is None: continue
            d = np.load(tf); lat = np.load(lf)
            ids = d["input_ids"][: H * C]
            am = d.get("attention_mask", np.ones_like(ids))[: H * C]
            batch_ids.append(ids); batch_ams.append(am); batch_Z.append(lat)
            if len(batch_ids) >= args.batch_size: break

        ids = torch.tensor(np.stack(batch_ids), device=device, dtype=torch.long)
        am = torch.tensor(np.stack(batch_ams), device=device, dtype=torch.float32)
        Z = torch.tensor(np.stack(batch_Z), device=device, dtype=dtype)
        B = Z.shape[0]; Z = Z[:, :H, :]

        # ── Verifier argmax (frozen) ──
        with torch.no_grad():
            out_v = rwkv_v(input_ids=ids, attention_mask=am.bool(), use_cache=True, return_dict=True)
            verifier_target = out_v.logits[:, :-1].argmax(-1)  # [B, seq-1]

        # ── Drafter forward (S1 → inject → logits) ──
        states = drafter.predict_trajectory_states(Z)
        out_d = rwkv_d(input_ids=ids, attention_mask=am.bool(), use_cache=True, return_dict=True)
        pkv = drafter.inject_into_cache(out_d.past_key_values, states)
        out_d2 = rwkv_d(input_ids=ids, attention_mask=am.bool(), past_key_values=pkv, use_cache=True, return_dict=True)
        logits = out_d2.logits[:, :-1].float()  # [B, seq-1, V]
        targets = ids[:, 1:]  # gold tokens for comparison

        # ── Behavior-alignment loss ──
        # CE(verifier_argmax, drafter_logits)
        ce_mask = am[:, 1:].bool()
        loss_behavior = F.cross_entropy(
            logits[ce_mask], verifier_target[ce_mask], reduction="mean"
        )
        # Also keep a small PPL loss for stability
        loss_ppl = F.cross_entropy(
            logits[ce_mask], targets[ce_mask], reduction="mean"
        )

        loss = 0.3 * loss_behavior + 1.0 * loss_ppl

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in drafter.parameters() if p.requires_grad], 1.0)
        opt.step()

        if step % 100 == 0:
            elapsed = time.time() - t0
            # Compute top-1 agreement
            drafter_argmax = logits.argmax(-1)
            agree = (drafter_argmax[ce_mask] == verifier_target[ce_mask]).float().mean().item()
            print(f"[S1 step {step:5d}] loss={loss.item():.3f} "
                  f"(beh={loss_behavior.item():.3f} ppl={loss_ppl.item():.3f}) "
                  f"agree_with_verifier={agree:.3f} "
                  f"lr={args.lr:.1e} step/s={step/max(elapsed,0.01):.1f}", flush=True)

        if step % args.save_every == 0:
            ckpt = {"trainable_state": {k: v for k, v in drafter.state_dict().items()
                                         if any(p in k for p in s1_names)}, "step": step}
            torch.save(ckpt, f"{args.save_dir}/cross_s1_step{step}.pt")
            print(f"  saved step {step}", flush=True)

    torch.save({"trainable_state": {k: v for k, v in drafter.state_dict().items()
                                     if any(p in k for p in s1_names)},
                 "step": args.num_steps},
                f"{args.save_dir}/cross_s1_final.pt")
    print(f"Done! {args.num_steps} steps, saved to {args.save_dir}", flush=True)


if __name__ == "__main__":
    main()
