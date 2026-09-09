#!/usr/bin/env python3
"""Evaluate a trained recurrent LACES adapter on cached ARC records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from models.arc_feature_cache import load_feature_record
from scripts.eval.train_arc_recurrent_reasoner import build_models, evaluate_paths


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature_root", type=Path, required=True)
    parser.add_argument("--split", choices=["training", "evaluation"], default="evaluation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    paths = sorted((args.feature_root / args.split).glob("*_t0_q*.pt"))
    if not paths:
        raise ValueError(f"no original-orientation records for split {args.split}")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = SimpleNamespace(**checkpoint["config"])
    reasoner, decoder = build_models(load_feature_record(paths[0]), model_args, device)
    reasoner.load_state_dict(checkpoint["reasoner"])
    decoder.load_state_dict(checkpoint["decoder"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    metrics = evaluate_paths(reasoner, decoder, paths, device, tuple(args.depths))
    result = {
        "checkpoint": str(args.checkpoint), "split": args.split,
        "record_count": len(paths), "metrics": metrics,
        "peak_gpu_memory_gib": (
            torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
