"""Splice a fine-tuned trajectory_dit state_dict into a full champion checkpoint.

Method 1/2/5 scripts save only {"trajectory_dit": <module state_dict>}. The eval
runner (run_cola_dlm_tasks_prefix_suffix_trajectory_cfg.py) needs a full ckpt_dir
with a model.pt whose trainable_state carries flat "trajectory_dit.*" keys. This
overwrites those keys with the fine-tuned weights and writes a new ckpt_dir.

Usage:
  python scripts/tools/merge_trained_dit.py \
    --base outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000 \
    --dit outputs_relay/s2-logit-js/logit_js_final.pt \
    --out outputs_relay/s2-logit-js-merged/step_00000000
"""
import argparse, shutil, sys
from pathlib import Path
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--dit", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base_dir = Path(args.base)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(base_dir / "model.pt", map_location="cpu", weights_only=False)
    ts = ckpt["trainable_state"]
    dit_sd = torch.load(args.dit, map_location="cpu", weights_only=False)["trajectory_dit"]

    n_over, n_miss = 0, 0
    for k, v in dit_sd.items():
        full = f"trajectory_dit.{k}"
        if full in ts:
            ts[full] = v
            n_over += 1
        else:
            n_miss += 1
    ckpt["trainable_state"] = ts
    torch.save(ckpt, out_dir / "model.pt")

    for extra in ("config.yaml", "config.json"):
        src = base_dir / extra
        if src.exists():
            shutil.copy(src, out_dir / extra)

    n_dit_keys = sum(1 for k in ts if k.startswith("trajectory_dit."))
    print(f"overwrote {n_over}/{n_dit_keys} trajectory_dit keys, {n_miss} unmatched")
    print(f"saved: {out_dir/'model.pt'}")
    if n_miss > 0:
        print("WARNING: some fine-tuned keys did not match base; verify architecture", file=sys.stderr)


if __name__ == "__main__":
    main()
