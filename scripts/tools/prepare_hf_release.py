#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


VARIANTS = {
    "0p4b-uncond-ddpm": {
        "title": "DiffRWKV State-Hijacking RELAY 0.4B Unconditional DDPM",
        "checkpoint": "outputs_relay/test-v6-0.4B-s2/step_00150000/model.pt",
        "config": "configs/rwkv_relay_0.4B_state_hijack_dit_vae32.yaml",
        "sampler": None,
        "base_model": "fla-hub/rwkv7-0.4B-world",
        "sample_command": "CUDA_VISIBLE_DEVICES=0 python train_state_hijacking_dit.py --sample --ckpt_dir . --prompt '' --sample_steps 1000 --max_len 256 --temperature 0.5 --top_k 10 --top_p 0.75 --repetition_penalty 1.2",
    },
    "2p9b-prefix": {
        "title": "DiffRWKV State-Hijacking RELAY 2.9B Prefix/Suffix CFG",
        "checkpoint": "outputs_relay/test-v6-2.9B-s2-prefix-suffix-cfg/step_00150000/model.pt",
        "config": "configs/rwkv_relay_2.9B_state_hijack_dit_vae32_prefix_suffix_s2.yaml",
        "sampler": "scripts/eval/sample_prefix_suffix_cfg.py",
        "base_model": "RWKV/RWKV7-Goose-World3-2.9B-HF",
        "sample_command": "CUDA_VISIBLE_DEVICES=0 python scripts/eval/sample_prefix_suffix_cfg.py --ckpt_dir . --prompt 'The history of artificial intelligence' --output sample.json --steps 100 --cfg_scale 3 --max_new_tokens 128 --temperature 0.5 --top_k 5 --top_p 0.7 --repetition_penalty 1.2",
    },
    "13p3b-prefix": {
        "title": "DiffRWKV State-Hijacking RELAY 13.3B Prefix/Suffix CFG",
        "checkpoint": "outputs_relay/test-v6-13.3B-s2-prefix-suffix-cfg/step_00100000/model.pt",
        "config": "configs/rwkv_relay_13.3B_state_hijack_dit_vae32_prefix_suffix_s2.yaml",
        "sampler": "scripts/eval/sample_prefix_suffix_cfg.py",
        "base_model": "local RWKV7-G1f-13.3B-HF conversion",
        "sample_command": "CUDA_VISIBLE_DEVICES=0 python scripts/eval/sample_prefix_suffix_cfg.py --ckpt_dir . --prompt 'The history of artificial intelligence' --output sample.json --steps 100 --cfg_scale 3 --max_new_tokens 128 --temperature 0.5 --top_k 5 --top_p 0.7 --repetition_penalty 1.2",
    },
    "2p9b-uncond-ddpm": {
        "title": "DiffRWKV State-Hijacking RELAY 2.9B Unconditional DDPM",
        "checkpoint": "outputs_relay/test-v6-2.9B-s2/step_00150000/model.pt",
        "config": "configs/rwkv_relay_2.9B_state_hijack_dit_vae32.yaml",
        "sampler": None,
        "base_model": "RWKV/RWKV7-Goose-World3-2.9B-HF",
        "sample_command": "CUDA_VISIBLE_DEVICES=0 python train_state_hijacking_dit.py --sample --ckpt_dir . --prompt '' --sample_steps 1000 --max_len 256 --temperature 0.5 --top_k 10 --top_p 0.75 --repetition_penalty 1.2",
    },
    "13p3b-uncond-ddpm": {
        "title": "DiffRWKV State-Hijacking RELAY 13.3B Unconditional DDPM",
        "checkpoint": "outputs_relay/test-v6-13.3B-s2/step_00150000/model.pt",
        "config": "configs/rwkv_relay_13.3B_state_hijack_dit_vae32.yaml",
        "sampler": None,
        "base_model": "local RWKV7-G1f-13.3B-HF conversion",
        "sample_command": "CUDA_VISIBLE_DEVICES=0 python train_state_hijacking_dit.py --sample --ckpt_dir . --prompt '' --sample_steps 1000 --max_len 256 --temperature 0.5 --top_k 10 --top_p 0.75 --repetition_penalty 1.2",
    },
    "2p9b-traj-transformer-ddpm": {
        "title": "DiffRWKV State-Hijacking RELAY 2.9B Trajectory Transformer DDPM",
        "checkpoint": "outputs_relay/traj32x16-2.9B-s2-transformer-s1-256d4-denoiser-768d8/step_00150000/model.pt",
        "config": "configs/rwkv_relay_2.9B_state_hijack_dit_vae32_traj32x16.yaml",
        "sampler": "scripts/eval/eval_trajectory_generation_grid.py",
        "base_model": "RWKV/RWKV7-Goose-World3-2.9B-HF",
        "sample_command": "CUDA_VISIBLE_DEVICES=0 python scripts/eval/eval_trajectory_generation_grid.py --ckpt_dir . --output traj_grid.json --texts_dir traj_grid_texts --sample_steps 1000 --max_new_tokens 512 --trajectory_s1_mode transformer --trajectory_state_blend 0.7 --temperatures 0.5 --repetition_penalties 1.2 --seeds 42 --top_k 5 --top_p 0.7",
    },
    "2p9b-traj-transformer-rf": {
        "title": "DiffRWKV State-Hijacking RELAY 2.9B Trajectory Transformer RF",
        "checkpoint": "outputs_relay/traj32x16-2.9B-s2-transformer-s1-256d4-denoiser-768d8-rf/step_00150000/model.pt",
        "config": "configs/rwkv_relay_2.9B_state_hijack_dit_vae32_traj32x16.yaml",
        "sampler": "scripts/eval/eval_trajectory_generation_grid.py",
        "base_model": "RWKV/RWKV7-Goose-World3-2.9B-HF",
        "sample_command": "CUDA_VISIBLE_DEVICES=0 python scripts/eval/eval_trajectory_generation_grid.py --ckpt_dir . --output traj_grid.json --texts_dir traj_grid_texts --sample_steps 1000 --max_new_tokens 512 --trajectory_s1_mode transformer --trajectory_state_blend 0.7 --trajectory_sampler rf_heun --temperatures 0.5 --repetition_penalties 1.2 --seeds 42 --top_k 5 --top_p 0.7",
    },
    "2p9b-512-pretrained-champion": {
        "title": "DiffRWKV State-Hijacking RELAY 2.9B 512 Trajectory Champion (from-scratch joint co-adapt + condboundary, pretrained/no-SFT, 57.67)",
        "checkpoint": "outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000/model.pt",
        "config": "configs/rwkv_relay_2.9B_state_hijack_dit_vae32_traj32x16.yaml",
        "sampler": "scripts/eval/sample_prefix_suffix_trajectory_cfg.py",
        "base_model": "RWKV/RWKV7-Goose-World3-2.9B-HF",
        "sample_command": "CUDA_VISIBLE_DEVICES=0 python scripts/eval/sample_prefix_suffix_trajectory_cfg.py --ckpt_dir . --prompt 'The history of artificial intelligence' --output sample.json --steps 100 --cfg_scale 3 --max_new_tokens 512 --trajectory_s1_mode independent --trajectory_state_blend 0.7 --temperature 0.5 --top_k 5 --top_p 0.7 --repetition_penalty 1.2",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--out_root", default="hf_release")
    parser.add_argument("--repo_id", default=None)
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def read_step(checkpoint: Path) -> int | None:
    import torch

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    step = ckpt.get("step")
    return int(step) if step is not None else None


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    spec = VARIANTS[args.variant]
    out_dir = root / args.out_root / args.variant
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    checkpoint = root / spec["checkpoint"]
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    files = [
        (checkpoint, out_dir / "model.pt"),
        (root / spec["config"], out_dir / spec["config"]),
        (root / "models" / "state_hijacking_dit.py", out_dir / "models" / "state_hijacking_dit.py"),
        (root / "models" / "__init__.py", out_dir / "models" / "__init__.py"),
        (root / "scripts" / "eval" / "relay_utils.py", out_dir / "scripts" / "eval" / "relay_utils.py"),
        (root / "train_state_hijacking_dit.py", out_dir / "train_state_hijacking_dit.py"),
        (root / "data_simple.py", out_dir / "data_simple.py"),
        (root / "utils.py", out_dir / "utils.py"),
    ]
    if spec["sampler"]:
        files.append((root / spec["sampler"], out_dir / spec["sampler"]))
    for src, dst in files:
        copy_file(src, dst)

    requirements = "\n".join([
        "torch",
        "transformers==5.3.0",
        "hydra-core==1.3.2",
        "omegaconf==2.3.0",
        "safetensors==0.5.3",
        "numpy==1.26.4",
        "huggingface_hub",
        "",
    ])
    write_text(out_dir / "requirements.txt", requirements)

    manifest = {
        "variant": args.variant,
        "title": spec["title"],
        "repo_id": args.repo_id,
        "private": bool(args.private),
        "checkpoint_source": spec["checkpoint"],
        "checkpoint_file": "model.pt",
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint),
        "step": read_step(checkpoint),
        "config": spec["config"],
        "base_model": spec["base_model"],
        "sampler": spec["sampler"],
    }
    write_text(out_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")

    repo_line = args.repo_id or "YOUR_USERNAME/YOUR_PRIVATE_REPO"
    readme = f"""---
license: other
library_name: pytorch
private: {str(bool(args.private)).lower()}
tags:
- rwkv
- diffusion
- state-hijacking
- diffrwkv
---

# {spec['title']}

This private release contains a DiffRWKV State-Hijacking RELAY checkpoint plus the minimal inference code needed to load it.

## Files

- `model.pt`: trainable RELAY checkpoint only. It does not include the frozen RWKV backbone weights.
- `{spec['config']}`: training/inference config used by the checkpoint.
- `models/state_hijacking_dit.py`: RELAY model implementation.
- `scripts/eval/relay_utils.py`: checkpoint loader.
- `{spec['sampler'] or 'train_state_hijacking_dit.py'}`: inference entrypoint.
- `manifest.json`: size, SHA256, source path, and step metadata.

## Base Model

{spec['base_model']}

The loader expects the base RWKV path in the checkpoint config to exist locally. If your path differs, edit `cfg.model.rwkv_local_path` in `scripts/eval/relay_utils.py` after loading or patch the config before constructing the model.

## Install

```bash
pip install -r requirements.txt
pip install flash-linear-attention fla-core
```

Use the same FLA build as the training environment when reporting numbers.

## Inference

```bash
{spec['sample_command']}
```

## Upload

```bash
huggingface-cli repo create {repo_line} --type model --private
huggingface-cli upload {repo_line} . . --repo-type model
```
"""
    write_text(out_dir / "README.md", readme)
    print(json.dumps({"out_dir": str(out_dir), **manifest}, indent=2))


if __name__ == "__main__":
    main()
