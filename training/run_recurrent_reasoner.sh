#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

: "${CKPT_DIR:?Set CKPT_DIR to a trained dynamic-basis LACES checkpoint directory}"
: "${TASKS:=2agent 3agent 4agent}"
: "${TRAIN_MAX_STEPS:=8}"
: "${EVAL_MAX_STEPS:=8}"
: "${EVAL_DEPTHS:=1 2 4 8}"
: "${RANK:=32}"
: "${EPOCHS:=100}"
: "${N_TRAIN:=60}"
: "${N_TEST:=25}"
: "${LR:=1e-3}"
: "${WEIGHT_DECAY:=1e-4}"
: "${GRAD_ACCUM:=1}"
: "${EARLY_STOP:=0}"
: "${OUTPUT:=results/capacity_audit/e10_variable_depth.json}"

read -r -a TASK_ARGS <<< "${TASKS}"
read -r -a DEPTH_ARGS <<< "${EVAL_DEPTHS}"

EXTRA_ARGS=()
if [[ "${EARLY_STOP}" == "1" ]]; then
  EXTRA_ARGS+=(--early_stop)
fi

print_header "Variable-depth recurrent latent reasoner"
run_cmd env CUDA_VISIBLE_DEVICES="${GPU}" python -u scripts/eval/train_recurrent_reasoner.py \
  --ckpt_dir "${CKPT_DIR}" \
  --tasks "${TASK_ARGS[@]}" \
  --n_train "${N_TRAIN}" \
  --n_test "${N_TEST}" \
  --device cuda:0 \
  --train_max_steps "${TRAIN_MAX_STEPS}" \
  --eval_max_steps "${EVAL_MAX_STEPS}" \
  --eval_depths "${DEPTH_ARGS[@]}" \
  --R_mode sweep \
  --r_s "${RANK}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --grad_accum "${GRAD_ACCUM}" \
  --output "${OUTPUT}" \
  "${EXTRA_ARGS[@]}"
