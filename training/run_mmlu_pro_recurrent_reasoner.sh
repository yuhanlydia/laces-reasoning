#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CKPT_DIR="${CKPT_DIR:-/root/laces-reasoning/outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03/step_00030000}"
RWKV_PATH="${RWKV_PATH:-/root/laces-reasoning/models/RWKV7-Goose-World3-2.9B-HF-fla-v2}"
FEATURE_ROOT="${FEATURE_ROOT:-$REPO_ROOT/preprocessed_data/mmlu_pro_rwkv}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs_mmlu_pro/mmlu_pro_r32_6h}"
DEVICE="${DEVICE:-cuda:0}"

PYTHONPATH=. python3 scripts/preprocess/cache_mmlu_pro_rwkv_features.py \
  --output_root "$FEATURE_ROOT" --ckpt_dir "$CKPT_DIR" --rwkv_path "$RWKV_PATH" \
  --device "$DEVICE" --batch_size 8 --max_length 1024

PYTHONPATH=. python3 scripts/eval/train_mmlu_pro_recurrent_reasoner.py \
  --feature_root "$FEATURE_ROOT" --output_dir "$OUTPUT_DIR" --device "$DEVICE" \
  --max_steps 500000 --max_hours 6 --eval_every 5000 --save_every 10000 \
  --grad_accum 4 --depths 1 2 4 8 16 --writer_rank 32 --base_logit_scale 0.25
