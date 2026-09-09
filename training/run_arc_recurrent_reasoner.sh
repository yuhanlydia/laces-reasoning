#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03/step_00030000}"
RWKV_PATH="${RWKV_PATH:-$REPO_ROOT/models/RWKV7-Goose-World3-2.9B-HF-fla-v2}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/preprocessed_data/arc_agi1}"
FEATURE_ROOT="${FEATURE_ROOT:-$REPO_ROOT/preprocessed_data/arc_agi1_features}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs_arc/arc_agi1_r32}"
DEVICE="${DEVICE:-cuda:0}"
STAGE="${1:-all}"

if [[ "$STAGE" == "prepare" || "$STAGE" == "all" ]]; then
  python3 scripts/preprocess/prepare_arc_agi1.py --output_root "$DATA_ROOT"
fi

if [[ "$STAGE" == "cache" || "$STAGE" == "all" ]]; then
  PYTHONPATH=. python3 scripts/preprocess/cache_arc_rwkv_features.py \
    --data_root "$DATA_ROOT" --output_root "$FEATURE_ROOT" \
    --ckpt_dir "$CKPT_DIR" --rwkv_path "$RWKV_PATH" --split both --device "$DEVICE"
fi

if [[ "$STAGE" == "overfit" || "$STAGE" == "all" ]]; then
  PYTHONPATH=. python3 scripts/eval/train_arc_recurrent_reasoner.py \
    --feature_root "$FEATURE_ROOT" --output_dir "$OUTPUT_ROOT/overfit8" \
    --overfit_tasks 8 --max_steps 5000 --eval_every 100 --save_every 500 \
    --grad_accum 4 --depths 1 2 4 8 16 --writer_rank 32 --device "$DEVICE"
  test -f "$OUTPUT_ROOT/overfit8/gate_passed.json"
fi

if [[ "$STAGE" == "train" || "$STAGE" == "all" ]]; then
  test -f "$OUTPUT_ROOT/overfit8/gate_passed.json"
  PYTHONPATH=. python3 scripts/eval/train_arc_recurrent_reasoner.py \
    --feature_root "$FEATURE_ROOT" --output_dir "$OUTPUT_ROOT/full360" \
    --dev_count 40 --max_steps 50000 --eval_every 1000 --save_every 5000 \
    --grad_accum 4 --depths 1 2 4 8 16 --writer_rank 32 --device "$DEVICE"
fi

if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
  PYTHONPATH=. python3 scripts/eval/eval_arc_recurrent_reasoner.py \
    --checkpoint "$OUTPUT_ROOT/full360/best.pt" --feature_root "$FEATURE_ROOT" \
    --split evaluation --depths 1 2 4 8 16 --device "$DEVICE" \
    --output "$OUTPUT_ROOT/evaluation_400.json"
fi
