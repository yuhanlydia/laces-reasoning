#!/usr/bin/env python3
"""Continue cross-source S1 training from an existing checkpoint.

Unlike train_cross_source_full.py which reinitializes S1, this resumes from
a saved cross_s1_step*.pt file and continues training.
"""
import argparse, glob, sys, os, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from eval.diag_loop1_common import build_model


def load_batch(token_dir, latent_dir, batch_size, seq_len, device, dtype):
    latent_files = sorted(glob.glob(f"{latent_dir}/*.npy"))
    np.random.shuffle(latent_files)
    batch_tokens, batch_ams, batch_lats = [], [], []
    for lf in latent_files:
        stem = Path(lf).stem.replace("_tokens", "")
        tf = Path(token_dir) / f"{stem}_tokens.npz"
        if not tf.exists():
            continue
        d = np.load(tf)
        lat = np.load(lf)
        ids = d["input_ids"][:seq_len]
        am = d["attention_mask"][:seq_len]
        batch_tokens.append(ids)
        batch_ams.append(am)
        batch_lats.append(lat)
        if len(batch_tokens) >= batch_size:
            break
    ids = torch.tensor(np.stack(batch_tokens), device=device, dtype=torch.long)
    am = torch.tensor(np.stack(batch_ams), device=device, dtype=torch.float32)
    lats = torch.tensor(np.stack(batch_lats), device=device, dtype=dtype)
    return ids, am, lats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template_ckpt",
                    default="outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000")
    ap.add_argument("--resume_s1", required=True)
    ap.add_argument("--token_dir", default="preprocessed_data/owt_rwkv_tokens/train")
    ap.add_argument("--latent_dir", default="preprocessed_data/owt_13b_s0_latents/train")
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--num_steps", type=int, default=30000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--save_every", type=int, default=5000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    model, tok, _, pad_id = build_model(args.template_ckpt, device)
    model.eval()

    if getattr(model, "trajectory_s1_mode", "independent") != "independent":
        model.trajectory_s1_mode = "independent"
        model.trajectory_state_decoder = None

    # Load existing S1 weights
    s1_ckpt = torch.load(args.resume_s1, map_location=device, weights_only=False)
    loaded = model.load_state_dict(s1_ckpt["trainable_state"], strict=False)
    start_step = s1_ckpt.get("step", 0)
    print(f"Resumed S1 from step {start_step}, loaded {len(s1_ckpt['trainable_state'])} tensors", flush=True)

    # Freeze all, unfreeze S1
    s1_param_names = {"alpha_heads", "state_basis", "state_scale", "state_norm"}
    for name, p in model.named_parameters():
        base = name.split(".")[0]
        p.requires_grad = (base in s1_param_names)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {n_trainable/1e6:.1f}M params", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    H = int(model.trajectory_horizon)
    C = int(model.trajectory_chunk_size)
    seq_len = H * C

    param_list = [p for p in model.parameters() if p.requires_grad]

    t0 = time.time()
    for step in range(1, args.num_steps + 1):
        ids, am, Z_0 = load_batch(args.token_dir, args.latent_dir, args.batch_size, seq_len, device, dtype)
        B, H_actual, D = Z_0.shape
        Z_0 = Z_0.to(dtype=dtype)

        states_list = model.predict_trajectory_states(Z_0)

        with torch.no_grad():
            out = model.rwkv_model(input_ids=ids, attention_mask=am.bool(),
                                   use_cache=True, return_dict=True)
        past_kv = model.inject_into_cache(out.past_key_values, states_list)

        out2 = model.rwkv_model(input_ids=ids, attention_mask=am.bool(),
                                past_key_values=past_kv, use_cache=True, return_dict=True)
        logits = out2.logits[:, :-1].float()
        targets = ids[:, 1:]
        ce_mask = am[:, 1:].bool()
        loss = F.cross_entropy(logits[ce_mask], targets[ce_mask])

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(param_list, 1.0)
        opt.step()

        global_step = start_step + step
        if step % 100 == 0:
            ppl = float(torch.exp(loss.detach()).item())
            elapsed = time.time() - t0
            sps = step / max(elapsed, 0.01)
            print(f"[S1 step {global_step:5d}] loss={loss.item():.3f} PPL={ppl:.1f} | "
                  f"z_norm={Z_0.float().norm(dim=-1).mean():.2f} | lr={args.lr:.1e} "
                  f"step/s={sps:.1f}", flush=True)

        if step % args.save_every == 0:
            ckpt = {
                "trainable_state": {k: v for k, v in model.state_dict().items()
                                    if any(p in k for p in s1_param_names)},
                "step": global_step,
            }
            torch.save(ckpt, f"{args.save_dir}/cross_s1_step{global_step}.pt")
            print(f"  saved step {global_step}", flush=True)

    # Final save
    final_step = start_step + args.num_steps
    torch.save({
        "trainable_state": {k: v for k, v in model.state_dict().items()
                            if any(p in k for p in s1_param_names)},
        "step": final_step,
    }, f"{args.save_dir}/cross_s1_final.pt")
    print(f"Done! Final step: {final_step}, saved to {args.save_dir}/cross_s1_final.pt", flush=True)


if __name__ == "__main__":
    main()
