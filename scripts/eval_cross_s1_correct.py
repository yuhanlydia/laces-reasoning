"""Correctly-paired cross-model S1 eval.

Replicates the EXACT training setup from train_cross_source_full.py:
  base = traj32x16-0.4B-s0/step_00050000, forced independent S1 mode,
  inject via predict_trajectory_states + inject_into_cache.
Compares raw vs cross-injected PPL for a given S1 patch and latent dir.
"""
import argparse, glob, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from eval.diag_loop1_common import build_model


@torch.no_grad()
def raw_ppl(rwkv, ids, am):
    out = rwkv(input_ids=ids, attention_mask=am.bool(), use_cache=True, return_dict=True)
    loss = torch.nn.functional.cross_entropy(
        out.logits[:, :-1].reshape(-1, 65536).float(), ids[:, 1:].reshape(-1), reduction="mean"
    )
    return float(torch.exp(loss).item())


@torch.no_grad()
def cross_ppl(model, ids, am, Z):
    H = int(model.trajectory_horizon)
    Z = Z[:, :H, :]
    states = model.predict_trajectory_states(Z)
    out = model.rwkv_model(input_ids=ids, attention_mask=am.bool(), use_cache=True, return_dict=True)
    pkv = model.inject_into_cache(out.past_key_values, states)
    out2 = model.rwkv_model(input_ids=ids, attention_mask=am.bool(),
                            past_key_values=pkv, use_cache=True, return_dict=True)
    loss = torch.nn.functional.cross_entropy(
        out2.logits[:, :-1].reshape(-1, 65536).float(), ids[:, 1:].reshape(-1), reduction="mean"
    )
    return float(torch.exp(loss).item())


@torch.no_grad()
def run(base_ckpt, patch, latent_dir, data_dir, device, num_samples, shuffle):
    model, tok, dtype, pad = build_model(base_ckpt, device)
    model.eval()
    if getattr(model, "trajectory_s1_mode", "independent") != "independent":
        model.trajectory_s1_mode = "independent"
        model.trajectory_state_decoder = None
    sd = torch.load(patch, map_location=device)
    trainable = sd.get("trainable_state", sd)
    missing, unexpected = model.load_state_dict(trainable, strict=False)
    print(f"base={base_ckpt.split('/')[-2]} patch={patch.split('/')[-2]} "
          f"applied={len(trainable)} unexpected={len(unexpected)}", flush=True)

    latent_files = sorted(glob.glob(f"{latent_dir}/*.npy"))
    matched = []
    for lf in latent_files:
        stem = Path(lf).stem.replace("_tokens", "").replace("_latent", "")
        tf = Path(data_dir) / f"{stem}_tokens.npz"
        if tf.exists():
            matched.append((lf, str(tf)))
        if len(matched) >= num_samples:
            break

    raws, crosses = [], []
    for lf, tf in matched:
        d = np.load(tf)
        ids = torch.tensor([d["input_ids"][:512]], device=device, dtype=torch.long)
        am = torch.ones_like(ids, dtype=torch.float32)
        Z = torch.tensor(np.load(lf), device=device, dtype=dtype).unsqueeze(0)
        if shuffle:
            perm = torch.randperm(Z.shape[1], device=device)
            Z = Z[:, perm, :]
        raws.append(raw_ppl(model.rwkv_model, ids, am))
        crosses.append(cross_ppl(model, ids, am, Z))

    r, c = float(np.mean(raws)), float(np.mean(crosses))
    tag = "shuffle" if shuffle else "clean"
    print(f"  raw={r:.2f}  cross({tag})={c:.2f}  delta={100*(c-r)/r:+.1f}%", flush=True)
    return {"raw": r, "cross": c, "shuffle": shuffle, "n": len(matched)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="outputs_relay/traj32x16-0.4B-s0/step_00050000")
    ap.add_argument("--patch", required=True)
    ap.add_argument("--latent_dir", default="preprocessed_data/owt_13b_s2_denoised_z/train")
    ap.add_argument("--data_dir", default="preprocessed_data/owt_rwkv_tokens/train")
    ap.add_argument("--num_samples", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shuffle", action="store_true")
    a = ap.parse_args()
    run(a.base, a.patch, a.latent_dir, a.data_dir, a.device, a.num_samples, a.shuffle)
