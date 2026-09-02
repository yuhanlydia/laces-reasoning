# pyright: reportAny=false, reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnusedCallResult=false, reportAttributeAccessIssue=false
"""Dump per-chunk trajectory latents from one backbone's frozen S0 encoder.

For Direction-2 (shared latent across backbones), one strong backbone (e.g. 13.3B)
encodes text into a canonical R^32 latent trajectory offline. Other backbones then
train S1+S2 on these precomputed latents WITHOUT loading the big encoder backbone,
avoiding OOM (13.3B ~27GB + 2.9B ~26GB won't co-reside on one 49GB GPU).

Reads OWT token .npz (input_ids/attention_mask), runs each through the backbone's
frozen S0 (rwkv -> pool -> encode), saves z[H, latent_dim] as .npy matching the
token file stem, so data_simple.py can load them via use_external_latents.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.relay_utils import load_relay_model  # noqa: E402


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description="Dump S0 trajectory latents from a frozen backbone.")
    ap.add_argument("--ckpt_dir", required=True, help="S0 checkpoint dir (encoder source, e.g. 13.3B S0)")
    ap.add_argument("--token_dir", default="preprocessed_data/owt_rwkv_tokens/train")
    ap.add_argument("--out_dir", required=True, help="Where to save {stem}.npy latents")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_files", type=int, default=0, help="0 = all")
    ap.add_argument("--log_every", type=int, default=200)
    args = ap.parse_args()

    model, _rwkv, _tok, _ckpt, _cfg = load_relay_model(args.ckpt_dir, args.device)
    model.eval()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    token_dirs = [p.strip() for p in str(args.token_dir).split(",") if p.strip()]
    per_dir = [sorted(Path(d).glob("*.npz")) for d in token_dirs]
    files = []
    if len(per_dir) > 1:
        i = 0
        while any(i < len(g) for g in per_dir):
            for g in per_dir:
                if i < len(g):
                    files.append(g[i])
            i += 1
    else:
        files = per_dir[0] if per_dir else []
    if args.max_files > 0:
        files = files[: args.max_files]
    print(f"encoding {len(files)} token files -> {out} (latent_dim={model.latent_dim}, horizon={int(model.trajectory_horizon)})", flush=True)

    done = 0
    for f in files:
        stem = f.stem
        dst = out / f"{stem}.npy"
        if dst.exists():
            done += 1
            continue
        d = np.load(f)
        ids = torch.tensor(d["input_ids"], dtype=torch.long, device=args.device).unsqueeze(0)
        am = d.get("attention_mask")
        am_t = torch.tensor(am, dtype=torch.long, device=args.device).unsqueeze(0) if am is not None else torch.ones_like(ids)
        z, _kl, _h, _c = model._encode_trajectory_chunks(ids, am_t)  # [1, H, latent_dim]
        np.save(dst, z[0].float().cpu().numpy().astype(np.float32))
        done += 1
        if done % args.log_every == 0:
            print(f"  {done}/{len(files)} encoded", flush=True)
    print(f"DONE: {done} latents in {out}", flush=True)


if __name__ == "__main__":
    main()
