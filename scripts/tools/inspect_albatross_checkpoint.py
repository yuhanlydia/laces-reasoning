"""Albatross checkpoint schema inspector — report-only CLI.

Loads a ``.pth`` state dict and produces a JSON report describing the Albatross
checkpoint structure.  The tool does **not** convert, save, or write any
checkpoint file; its only output is the requested JSON report.

Usage::

    python scripts/tools/inspect_albatross_checkpoint.py \\
        --input albatross_valid.pth \\
        --output-json report.json \\
        --expected-layers 2 --expected-embd 64

    python scripts/tools/inspect_albatross_checkpoint.py \\
        --input albatross_missing_keys.pth \\
        --output-json report_missing.json \\
        --strict
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Required keys for a well-formed Albatross checkpoint
# ---------------------------------------------------------------------------
REQUIRED_TOP_KEYS: tuple[str, ...] = (
    "emb.weight",
    "ln_out.weight",
    "head.weight",
)

REQUIRED_PER_LAYER: tuple[str, ...] = (
    "blocks.{layer}.ffn.key.weight",
)


# ---------------------------------------------------------------------------
# Key-group categorisation (recognized groups)
# ---------------------------------------------------------------------------

_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.(.+)$")
_ATT_RE = re.compile(r"^att\.(.+)$")
_FFN_RE = re.compile(r"^ffn\.(.+)$")
_LN_RE = re.compile(r"^ln\d+\.(.+)$")  # ln0 / ln1 / ln2


def _classify_key(key: str) -> str | None:
    """Return a recognised group label or *None* for unrecognised keys."""
    if key.startswith("emb."):
        return "embedding"
    if key.startswith("ln_out."):
        return "ln_out"
    if key.startswith("head."):
        return "head"
    m = _BLOCK_RE.match(key)
    if m:
        block_id, rest = m.group(1), m.group(2)
        if _ATT_RE.match(rest):
            return f"blocks.{block_id}.attention"
        if _FFN_RE.match(rest):
            return f"blocks.{block_id}.ffn"
        if _LN_RE.match(rest):
            return f"blocks.{block_id}.layernorm"
        return f"blocks.{block_id}.other"
    return None


# ---------------------------------------------------------------------------
# DiffRwkv conceptual mapping preview (Albatross → DiffRwkv)
# ---------------------------------------------------------------------------

def _diff_rwkv_mapping() -> dict[str, str]:
    """Return a static conceptual mapping from Albatross key groups to
    DiffRwkv / RWKV‑7 field names.

    This is a **preview only** — it does not convert tensors or write files.
    """
    return {
        "emb.weight": "embeddings.token_embedding.weight",
        "head.weight": "lm_head.weight  (tied-unless-untied)",
        "ln_out.weight": "ln_out.weight",
        "ln_out.bias": "ln_out.bias",
        # Attention
        "blocks.*.att.x_r": "blocks.*.att.time_x_r  (RWKV-7 wkv time-mixing vector)",
        "blocks.*.att.w0": "blocks.*.att.time_decay_w0",
        "blocks.*.att.w1": "blocks.*.att.time_decay_w1",
        "blocks.*.att.w2": "blocks.*.att.time_decay_w2",
        "blocks.*.att.a0": "blocks.*.att.time_first_a0",
        "blocks.*.att.a1": "blocks.*.att.time_first_a1",
        "blocks.*.att.a2": "blocks.*.att.time_first_a2",
        "blocks.*.att.g0": "blocks.*.att.time_first_g0",
        "blocks.*.att.g1": "blocks.*.att.time_first_g1",
        "blocks.*.att.g2": "blocks.*.att.time_first_g2",
        "blocks.*.att.k_k": "blocks.*.att.key_k  (wkv key mixing scalar)",
        "blocks.*.att.k_a": "blocks.*.att.key_a  (wkv key alpha scalar)",
        "blocks.*.att.r_k": "blocks.*.att.receptance_k  (wkv receptance mixing scalar)",
        "blocks.*.att.receptance.weight": "blocks.*.att.receptance.weight",
        "blocks.*.att.key.weight": "blocks.*.att.key.weight",
        "blocks.*.att.value.weight": "blocks.*.att.value.weight",
        "blocks.*.att.gate.weight": "blocks.*.att.gate.weight",
        "blocks.*.att.ln_x.weight": "blocks.*.att.ln_x.weight",
        "blocks.*.att.ln_x.bias": "blocks.*.att.ln_x.bias",
        # FFN
        "blocks.*.ffn.key.weight": "blocks.*.ffn.key.weight  (time-mix)",
        "blocks.*.ffn.value.weight": "blocks.*.ffn.value.weight",
        "blocks.*.ffn.receptance.weight": "blocks.*.ffn.receptance.weight",
        # Layer norms
        "blocks.*.ln0.weight": "blocks.*.ln0.weight  (pre-attention)",
        "blocks.*.ln0.bias": "blocks.*.ln0.bias",
        "blocks.*.ln1.weight": "blocks.*.ln1.weight  (pre-ffn)",
        "blocks.*.ln1.bias": "blocks.*.ln1.bias",
        "blocks.*.ln2.weight": "blocks.*.ln2.weight  (post-ffn)",
        "blocks.*.ln2.bias": "blocks.*.ln2.bias",
    }


# ---------------------------------------------------------------------------
# Shape serialisation
# ---------------------------------------------------------------------------

def _serialise_shape(tensor: torch.Tensor) -> str:
    """Return a compact, deterministic shape string e.g. ``(128, 64)``."""
    return str(tuple(tensor.shape))


# ---------------------------------------------------------------------------
# Embedding-dimension shape validation
# ---------------------------------------------------------------------------

# Patterns for per-layer Albatross keys whose shape should match expected_embd.
_FFN_KEY_RE = re.compile(r"^blocks\.\d+\.ffn\.key\.weight$")
_FFN_VALUE_RE = re.compile(r"^blocks\.\d+\.ffn\.value\.weight$")
_SQUARE_PROJ_RE = re.compile(
    r"^blocks\.\d+\.(att\.(receptance|key|value|gate)|ffn\.receptance)\.weight$"
)
_BLOCK_LN_RE = re.compile(r"^blocks\.\d+\.ln\d\.(weight|bias)$")
_ATT_LN_X_RE = re.compile(r"^blocks\.\d+\.att\.ln_x\.(weight|bias)$")
_ALBATROSS_VEC_RE = re.compile(
    r"^blocks\.\d+\.att\.(x_r|w\d|a\d|g\d|k_k|k_a|r_k)$"
)


def _validate_embd(
    state: dict[str, torch.Tensor],
    expected_embd: int,
) -> list[dict[str, object]]:
    """Return a list of shape-mismatch objects when a tensor dimension
    disagrees with *expected_embd*.

    Each entry::

        {"key": str, "dimension": str, "expected": int, "actual": int, "shape": str}

    Only keys whose shape contract mentions *expected_embd* are checked.
    """
    mismatches: list[dict[str, object]] = []

    for key, tensor in sorted(state.items()):
        shape = tuple(tensor.shape)
        ndim = len(shape)

        # ---- dispatch: which slice(s) should equal expected_embd ----------
        if key in ("emb.weight", "head.weight"):
            # [vocab, embd]  →  last dim == embd
            checks: list[tuple[int, str]] = [(ndim - 1, "last")]
        elif key in ("ln_out.weight", "ln_out.bias"):
            # [embd]  →  dim 0 == embd
            checks = [(0, "size")]
        elif _BLOCK_LN_RE.match(key):
            # blocks.N.ln{0,1,2}.{weight,bias}  →  [embd]
            checks = [(0, "size")]
        elif _ATT_LN_X_RE.match(key):
            # blocks.N.att.ln_x.{weight,bias}  →  [embd]
            checks = [(0, "size")]
        elif _SQUARE_PROJ_RE.match(key):
            # [embd, embd]  →  both dims == embd
            checks = [(0, "dim-0"), (1, "dim-1")]
        elif _FFN_KEY_RE.match(key):
            # [embd*4, embd]  →  last dim == embd
            checks = [(ndim - 1, "last")]
        elif _FFN_VALUE_RE.match(key):
            # [embd, embd*4]  →  first dim == embd
            checks = [(0, "dim-0")]
        elif _ALBATROSS_VEC_RE.match(key):
            # (1, 1, embd)  →  last dim == embd
            checks = [(ndim - 1, "last")]
        else:
            continue  # key not in the shape contract

        for dim_idx, dim_label in checks:
            if dim_idx < ndim and shape[dim_idx] != expected_embd:
                mismatches.append({
                    "key": key,
                    "dimension": dim_label,
                    "expected": expected_embd,
                    "actual": shape[dim_idx],
                    "shape": str(shape),
                })

    return mismatches


# ---------------------------------------------------------------------------
# Main inspection logic
# ---------------------------------------------------------------------------

def _inspect(
    input_path: str,
    expected_layers: int | None,
    expected_embd: int | None,
    strict: bool,
) -> dict[str, Any]:
    """Load *input_path*, produce a report dict.

    Returns the report; does **not** handle file I/O or exit codes.
    """
    state: dict[str, torch.Tensor] = torch.load(input_path, map_location="cpu", weights_only=True)

    # -- classify every key --------------------------------------------------
    recognized: list[str] = []
    unrecognized: list[str] = []
    for key in sorted(state.keys()):
        group = _classify_key(key)
        if group is not None:
            recognized.append(group)
        else:
            unrecognized.append(key)

    # deduplicate group names while preserving order
    seen: set[str] = set()
    recognized_groups: list[str] = []
    for g in recognized:
        if g not in seen:
            seen.add(g)
            recognized_groups.append(g)

    # -- required-key presence -----------------------------------------------
    missing_required: list[str] = []
    for rk in REQUIRED_TOP_KEYS:
        if rk not in state:
            missing_required.append(rk)

    # If expected_layers is given, check per-layer required keys
    if expected_layers is not None:
        for layer in range(expected_layers):
            template = f"blocks.{layer}.ffn.key.weight"
            if template not in state:
                missing_required.append(f"blocks.{layer}.ffn.key.weight")

    # -- shape summary -------------------------------------------------------
    shape_summary: dict[str, str] = {}
    for key, tensor in sorted(state.items()):
        group = _classify_key(key)
        if group is not None:
            shape_summary[key] = _serialise_shape(tensor)

    # -- unexpected -----------------------------------------------------------
    unexpected: list[str] = sorted(unrecognized)

    # -- DiffRwkv mapping preview --------------------------------------------
    mapping = _diff_rwkv_mapping()
    diff_rwkv_mapping_preview: dict[str, str] = {}
    for key in sorted(state.keys()):
        # Find the closest mapping key (exact or wildcard)
        if key in mapping:
            diff_rwkv_mapping_preview[key] = mapping[key]
        else:
            # Try wildcard match: replace block number with *
            for pattern, mapped in mapping.items():
                if "*" in pattern:
                    regex = "^" + re.escape(pattern).replace(r"\*", r"\d+") + "$"
                    if re.match(regex, key):
                        diff_rwkv_mapping_preview[key] = mapped
                        break

    # -- shape-mismatch validation (expected-embd) ---------------------------
    shape_mismatches: list[dict[str, object]] = []
    if expected_embd is not None:
        shape_mismatches = _validate_embd(state, expected_embd)

    return {
        "recognized": recognized_groups,
        "missing_required": missing_required,
        "unexpected": unexpected,
        "shape_summary": shape_summary,
        "diff_rwkv_mapping_preview": diff_rwkv_mapping_preview,
        "shape_mismatches": shape_mismatches,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inspect an Albatross checkpoint schema and produce a JSON report.",
    )
    p.add_argument("--input", required=True, help="Path to the .pth checkpoint file.")
    p.add_argument(
        "--output-json",
        required=True,
        help="Path where the JSON report will be written.",
    )
    p.add_argument(
        "--expected-layers",
        type=int,
        default=None,
        help="Expected number of layers (optional). If given, each layer must "
        "have a ffn.key.weight key.",
    )
    p.add_argument(
        "--expected-embd",
        type=int,
        default=None,
        help="Expected embedding dimension (optional, recorded in report).",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when required keys are missing or shapes are "
        "incompatible.  The JSON report is still written.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        return 1

    report = _inspect(
        input_path=args.input,
        expected_layers=args.expected_layers,
        expected_embd=args.expected_embd,
        strict=args.strict,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    has_missing = len(report["missing_required"]) > 0
    has_mismatches = len(report.get("shape_mismatches", [])) > 0

    if args.strict and (has_missing or has_mismatches):
        if has_missing:
            missing_str = ", ".join(report["missing_required"])
            print(
                f"Strict mode: required keys missing: {missing_str}",
                file=sys.stderr,
            )
        if has_mismatches:
            mismatches = report["shape_mismatches"]
            print(
                f"Strict mode: {len(mismatches)} shape mismatches detected "
                f"against expected-embd={args.expected_embd}",
                file=sys.stderr,
            )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
