#!/usr/bin/env bash
# Causal 16-block GRPO for the ORIGINAL LACES S2 on GSM8K/MATH.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${CKPT_DIR:?Set CKPT_DIR to the existing step_00030000 directory}"
: "${DATA_DIR:?Set DATA_DIR to a prepared GSM8K or MATH bundle}"
PYTHON="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
args=(
  -m laces_posttrain.run_math_block_grpo
  --data "$DATA_DIR"
  --ckpt-dir "$CKPT_DIR"
  --device "${DEVICE:-cuda:0}"
  --output "${OUTPUT_DIR:-results/math_block_grpo}"
  --mode "${MODE:-preflight}"
  --steps "${NUM_STEPS:-1000}"
  --diffusion-steps "${DIFFUSION_STEPS:-32}"
  --eta "${ETA:-0.3}"
  --min-std "${MIN_STD:-0.02}"
  --cfg-scale "${CFG_SCALE:-2.0}"
  --blend "${STATE_BLEND:-0.7}"
  --lr "${LR:-3e-7}"
  --weight-decay "${WEIGHT_DECAY:-0.0}"
  --group-size "${GROUP_SIZE:-8}"
  --inner-epochs "${INNER_EPOCHS:-1}"
  --advantage-mode "${ADVANTAGE_MODE:-center}"
  --kl-coef "${KL_COEF:-0.05}"
  --clip-range "${CLIP_RANGE:-0.2}"
  --max-grad-norm "${MAX_GRAD_NORM:-1.0}"
  --max-log-ratio "${MAX_LOG_RATIO:-20.0}"
  --max-blocks "${MAX_BLOCKS:-16}"
  --tokens-per-block "${TOKENS_PER_BLOCK:-32}"
  --exact-weight "${EXACT_WEIGHT:-1.0}"
  --format-weight "${FORMAT_WEIGHT:-0.1}"
  --eval-samples "${EVAL_SAMPLES:-1}"
  --save-every "${SAVE_EVERY:-100}"
  --eval-every "${EVAL_EVERY:-250}"
  --seed "${SEED:-42}"
)
if [[ -n "${RWKV_PATH:-}" ]]; then args+=(--rwkv-path "$RWKV_PATH"); fi
if [[ -n "${TRAIN_LIMIT:-}" ]]; then args+=(--train-limit "$TRAIN_LIMIT"); fi
if [[ -n "${DEV_LIMIT:-}" ]]; then args+=(--dev-limit "$DEV_LIMIT"); fi
if [[ -n "${TEST_LIMIT:-}" ]]; then args+=(--test-limit "$TEST_LIMIT"); fi
if [[ -n "${RESUME:-}" ]]; then args+=(--resume "$RESUME"); fi
if [[ -n "${S2_CHECKPOINT:-}" ]]; then args+=(--s2-checkpoint "$S2_CHECKPOINT"); fi
if [[ "${MODE:-preflight}" == "eval" ]]; then
  args+=(--split "${SPLIT:-dev}")
  if [[ "${ACKNOWLEDGE_TEST:-0}" == "1" ]]; then args+=(--acknowledge-test); fi
fi
if [[ -n "${BLOCK_BUDGETS:-}" ]]; then
  read -r -a budget_args <<< "$BLOCK_BUDGETS"
  args+=(--block-budgets "${budget_args[@]}")
fi
exec "$PYTHON" "${args[@]}" "$@"
