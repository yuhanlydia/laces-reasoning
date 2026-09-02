"""Merge a distilled trajectory_dit state_dict into a base champion checkpoint.

The serial->parallel distillation saves only the trajectory_dit weights. To
evaluate it, overlay those weights onto the base champion checkpoint's
trainable_state (which uses the trajectory_dit.* prefix) and write a full
model.pt the eval harness can load.
"""
import argparse
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument("--distill_ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    base = torch.load(Path(args.base_ckpt) / "model.pt", map_location="cpu", weights_only=False)
    distill = torch.load(args.distill_ckpt, map_location="cpu", weights_only=False)
    dit = distill["trainable"]

    ts = base["trainable_state"]
    n_replaced = 0
    for k in list(ts.keys()):
        if k.startswith("trajectory_dit."):
            sub = k[len("trajectory_dit."):]
            if sub in dit:
                ts[k] = dit[sub]
                n_replaced += 1
    print(f"replaced {n_replaced} trajectory_dit tensors (distill has {len(dit)})")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(base, out / "model.pt")
    print(f"saved merged checkpoint: {out / 'model.pt'}")


if __name__ == "__main__":
    main()
