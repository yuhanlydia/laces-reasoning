"""Transplant 13.3B's trained S2 (latent_dit) weights into a target-backbone shell.

Cross-model direction-2 setup: S2 lives in R^32 latent space and is
backbone-independent (verified: latent_dit tensors match shape 55/55 between
2.9B and 13.3B). So we can freeze a SHARED S2 from 13.3B and only train a new
per-backbone S1. This script produces the merged starting checkpoint:

  base  = target backbone ckpt (keeps its backbone-bound S1 + S0 encoder shell)
  S2    = overwritten byte-for-byte with the 13.3B latent_dit weights

The result is loadable by C.build_model and is byte-identical to the 13.3B S2
on every latent_dit.* tensor, while keeping the target backbone's dims.
"""
import argparse
import glob
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def latest_ckpt(spec):
    p = Path(spec)
    if p.name.endswith(".pt"):
        return str(p)
    if (p / "model.pt").exists():
        return str(p / "model.pt")
    cands = sorted(glob.glob(str(p / "step_*" / "model.pt")))
    if not cands:
        raise FileNotFoundError(f"no model.pt under {spec}")
    return cands[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base_ckpt",
        default="outputs_relay/test-v6-2.9B-s2-prefix-suffix-cfg/step_00150000",
        help="target-backbone ckpt (provides backbone + S1 shell + S0 encoder)",
    )
    ap.add_argument(
        "--s2_source_ckpt",
        default="outputs_relay/test-v6-13.3B-s2-prefix-suffix-cfg",
        help="13.3B ckpt whose latent_dit (S2) weights get transplanted",
    )
    ap.add_argument("--out", required=True, help="output merged ckpt dir (writes model.pt)")
    args = ap.parse_args()

    base_p = latest_ckpt(args.base_ckpt)
    s2_p = latest_ckpt(args.s2_source_ckpt)
    print(f"base   : {base_p}", flush=True)
    print(f"s2 src : {s2_p}", flush=True)

    base = torch.load(base_p, map_location="cpu", weights_only=False)
    s2 = torch.load(s2_p, map_location="cpu", weights_only=False)
    bt = base["trainable_state"]
    st = s2.get("trainable_state", s2)

    s2_keys = [k for k in bt if k.startswith("latent_dit")]
    copied, shape_mismatch, missing = 0, [], []
    for k in s2_keys:
        if k not in st:
            missing.append(k)
            continue
        if bt[k].shape != st[k].shape:
            shape_mismatch.append((k, tuple(bt[k].shape), tuple(st[k].shape)))
            continue
        bt[k] = st[k].clone()
        copied += 1

    print(f"latent_dit(S2) copied: {copied}/{len(s2_keys)}", flush=True)
    print(f"  missing in source: {len(missing)}  shape_mismatch: {len(shape_mismatch)}", flush=True)
    if shape_mismatch:
        for k, a, b in shape_mismatch[:5]:
            print("   MISMATCH", k, a, b, flush=True)
        raise SystemExit("S2 shape mismatch: cannot transplant losslessly")
    if missing:
        raise SystemExit(f"S2 keys missing in source: {missing[:5]}")

    # step reset to 0 so S1 training starts a fresh schedule
    base["step"] = 0
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model.pt"
    torch.save(base, out_path)
    print(f"saved merged ckpt: {out_path}", flush=True)

    # verify byte-identical latent_dit vs 13.3B source
    chk = torch.load(out_path, map_location="cpu", weights_only=False)["trainable_state"]
    max_diff = 0.0
    for k in s2_keys:
        max_diff = max(max_diff, (chk[k].float() - st[k].float()).abs().max().item())
    print(f"verify: max|S2_merged - S2_13.3B| = {max_diff:.3e} (expect 0.0)", flush=True)
    print("MERGE_DONE", flush=True)


if __name__ == "__main__":
    main()
