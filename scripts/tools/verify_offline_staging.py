#!/usr/bin/env python3
"""
verify_offline_staging.py — Verify offline staging completeness for DiffRwkv.

Checks that all required files, model weights, data, and dependencies
are present and functional for offline training on a 2xH200 machine.

Usage:
    python scripts/tools/verify_offline_staging.py --root /path/to/DiffRwkv
    python scripts/tools/verify_offline_staging.py  # uses DIFFRWKV_ROOT or cwd
"""

import argparse
import glob
import os
import subprocess
import sys


# ---------------------------------------------------------------------------
# Manifest: (label, relative_path, type, expected_count_or_none)
#   type: "dir"       — directory must exist, file count checked if expected given
#         "dir_ne"    — directory must exist and be non-empty
#         "file"      — single file must exist
#         "whl_dir"   — vendor dir with .whl files
# ---------------------------------------------------------------------------

EVAL_DATASETS = [
    ("wikitext2_full/tokens/val", 999),
    ("lambada/test", 5153),
    ("hellaswag/validation", 10042),
    ("piqa/validation", 1838),
    ("arc_easy/validation", 570),
    ("arc_challenge/validation", 299),
    ("winogrande/validation", 1267),
    ("openbookqa/validation", 500),
]

MANIFEST = [
    # --- Code ---
    ("Code root (./)", ".", "dir_min", 100),
    ("configs/", "configs", "dir_min", 30),
    ("models/ (Python package)", "models", "dir_min", 5),
    ("scripts/", "scripts", "dir_min", 10),
    ("tests/", "tests", "dir_min", 1),
    # --- Model Weights ---
    ("Model weights (rwkv7-0.4B-world)", "models/rwkv7-0.4B-world", "dir_ne", None),
    # --- Training Data ---
    ("Training data (owt_rwkv_tokens/train)", "preprocessed_data/owt_rwkv_tokens/train", "dir_count", 304052),
    # --- Eval Data ---
] + [
    (f"Eval: {name}", f"preprocessed_data/{name}", "dir_count", count)
    for name, count in EVAL_DATASETS
] + [
    # --- Dependencies ---
    ("Vendor wheels (FLA 0.5.0)", "vendor", "whl_dir", 2),
]


def count_files(path: str) -> int:
    """Count all files recursively under path."""
    total = 0
    for _, _, files in os.walk(path):
        total += len(files)
    return total


def check_item(label: str, path: str, check_type: str, expected) -> tuple:
    """
    Check a single manifest item.
    Returns (passed: bool, detail: str).
    """
    if not os.path.exists(path):
        return False, f"MISSING: {path}"

    if check_type == "dir_min":
        if not os.path.isdir(path):
            return False, f"NOT A DIR: {path}"
        n = count_files(path)
        if n < expected:
            return False, f"too few files: {n} < {expected}"
        return True, f"{n} files (>= {expected})"

    if check_type == "dir_ne":
        if not os.path.isdir(path):
            return False, f"NOT A DIR: {path}"
        n = count_files(path)
        if n == 0:
            return False, "directory is empty"
        return True, f"{n} files (non-empty)"

    if check_type == "dir_count":
        if not os.path.isdir(path):
            return False, f"NOT A DIR: {path}"
        npz_files = [f for f in glob.glob(os.path.join(path, "**"), recursive=True)
                     if os.path.isfile(f) and f.endswith(".npz")]
        n = len(npz_files)
        # Allow 1% tolerance for file count
        tolerance = max(1, int(expected * 0.01))
        if abs(n - expected) > tolerance:
            return False, f"file count mismatch: {n} (expected {expected} +/- {tolerance})"
        return True, f"{n} npz files (expected {expected})"

    if check_type == "whl_dir":
        if not os.path.isdir(path):
            return False, f"NOT A DIR: {path}"
        whls = glob.glob(os.path.join(path, "*.whl"))
        if len(whls) != expected:
            return False, f"wheel count: {len(whls)} (expected {expected})"
        names = [os.path.basename(w) for w in whls]
        return True, f"{len(whls)} wheels: {', '.join(names)}"

    return False, f"unknown check type: {check_type}"


def check_pip_dry_run(root: str) -> tuple:
    """Check that vendor wheels are installable via pip --dry-run."""
    vendor_dir = os.path.join(root, "vendor")
    whls = glob.glob(os.path.join(vendor_dir, "*.whl"))
    if not whls:
        return False, "no .whl files found in vendor/"

    cmd = [sys.executable, "-m", "pip", "install", "--dry-run", "--no-deps"] + whls
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            return False, f"pip dry-run failed:\n{result.stderr.strip()}"
        return True, "pip install --dry-run succeeded"
    except subprocess.TimeoutExpired:
        return False, "pip dry-run timed out (60s)"
    except Exception as e:
        return False, f"pip dry-run error: {e}"


def check_model_loads(root: str) -> tuple:
    """Try loading the RWKV-7 0.4B model from local path."""
    model_path = os.path.join(root, "models", "rwkv7-0.4B-world")
    if not os.path.isdir(model_path):
        return False, f"model dir missing: {model_path}"

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True,
            torch_dtype=torch.bfloat16, local_files_only=True,
        )
        n_params = sum(p.numel() for p in model.parameters())
        return True, f"loaded OK ({n_params / 1e6:.0f}M params)"
    except ImportError as e:
        return False, f"import error (torch/transformers): {e}"
    except Exception as e:
        return False, f"load failed: {e}"


def check_fla_sanity(root: str) -> tuple:
    """
    Forward pass sanity check: load frozen backbone, run a short sequence,
    verify cross-entropy loss < 5.0 (random baseline for 65K vocab is ~11).
    """
    model_path = os.path.join(root, "models", "rwkv7-0.4B-world")
    if not os.path.isdir(model_path):
        return False, f"model dir missing: {model_path}"

    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True,
            torch_dtype=torch.bfloat16, local_files_only=True,
        ).to(device).eval()

        # Short test sequence
        text = "The quick brown fox jumps over the lazy dog."
        ids = tok(text, return_tensors="pt")["input_ids"].to(device)

        with torch.no_grad():
            outputs = model(ids, labels=ids)
            ce_loss = outputs.loss.item()

        if ce_loss >= 5.0:
            return False, f"CE loss = {ce_loss:.3f} (expected < 5.0)"
        return True, f"CE loss = {ce_loss:.3f} (< 5.0, device={device})"
    except ImportError as e:
        return False, f"import error: {e}"
    except Exception as e:
        return False, f"sanity check failed: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Verify offline staging completeness for DiffRwkv",
    )
    parser.add_argument(
        "--root", default=None,
        help="Root directory of DiffRwkv (default: $DIFFRWKV_ROOT or cwd)",
    )
    parser.add_argument(
        "--skip-model", action="store_true",
        help="Skip model loading and FLA sanity checks (useful if no GPU)",
    )
    args = parser.parse_args()

    # Resolve root directory
    if args.root:
        root = os.path.abspath(args.root)
    else:
        root = os.environ.get("DIFFRWKV_ROOT", os.getcwd())
        root = os.path.abspath(root)

    print(f"{'=' * 70}")
    print(f"DiffRwkv Offline Staging Verification")
    print(f"Root: {root}")
    print(f"{'=' * 70}")
    print()

    if not os.path.isdir(root):
        print(f"FATAL: root directory does not exist: {root}")
        sys.exit(1)

    passed = 0
    failed = 0
    results = []

    # --- Manifest checks ---
    print("--- File/Directory Checks ---")
    for label, rel_path, check_type, expected in MANIFEST:
        full_path = os.path.join(root, rel_path)
        ok, detail = check_item(label, full_path, check_type, expected)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}: {detail}")
        results.append((label, ok))
        if ok:
            passed += 1
        else:
            failed += 1

    # --- Pip dry-run check ---
    print()
    print("--- Dependency Checks ---")
    ok, detail = check_pip_dry_run(root)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] pip install --dry-run: {detail}")
    results.append(("pip install --dry-run", ok))
    if ok:
        passed += 1
    else:
        failed += 1

    # --- Model loading check ---
    if not args.skip_model:
        print()
        print("--- Model Checks ---")
        ok, detail = check_model_loads(root)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] Model loads: {detail}")
        results.append(("Model loads", ok))
        if ok:
            passed += 1
        else:
            failed += 1

        # --- FLA sanity check ---
        ok, detail = check_fla_sanity(root)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] FLA sanity (CE < 5): {detail}")
        results.append(("FLA sanity (CE < 5)", ok))
        if ok:
            passed += 1
        else:
            failed += 1
    else:
        print()
        print("--- Model Checks (SKIPPED) ---")
        print("  [SKIP] Model loads (--skip-model)")
        print("  [SKIP] FLA sanity (--skip-model)")

    # --- Summary ---
    total = passed + failed
    print()
    print(f"{'=' * 70}")
    print(f"SUMMARY: {passed}/{total} checks passed, {failed} failed")
    print(f"{'=' * 70}")

    if failed > 0:
        print()
        print("Failed items:")
        for label, ok in results:
            if not ok:
                print(f"  - {label}")
        sys.exit(1)
    else:
        print()
        print("All checks passed. Staging is complete.")
        sys.exit(0)


if __name__ == "__main__":
    main()
