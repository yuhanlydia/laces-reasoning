#!/usr/bin/env python3
"""Download and strictly validate the official ARC-AGI-1 400/400 splits."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from models.arc_data import load_arc_split, prepare_official_arc, verify_split_counts

DEFAULT_ROOT = REPO / "preprocessed_data" / "arc_agi1"
DEFAULT_URL = "https://codeload.github.com/fchollet/ARC-AGI/tar.gz/refs/heads/master"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source-url", default=DEFAULT_URL)
    parser.add_argument("--source-ref", default="fchollet/ARC-AGI@master")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.verify_only:
        training = load_arc_split(args.root / "training")
        evaluation = load_arc_split(args.root / "evaluation")
        verify_split_counts(training, evaluation)
        print(f"verified training={len(training)} evaluation={len(evaluation)}")
        return
    manifest = prepare_official_arc(args.root, args.source_url, args.source_ref)
    print(
        f"prepared training={manifest['training_tasks']} "
        f"evaluation={manifest['evaluation_tasks']} at {args.root}"
    )


if __name__ == "__main__":
    main()
