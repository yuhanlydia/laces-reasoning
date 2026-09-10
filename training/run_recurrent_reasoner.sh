#!/usr/bin/env bash
# Corrected default entry. Preflight is deliberately the default, not a long job.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${CKPT_DIR:=outputs_dynamic_basis/laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03/step_00030000}"
: "${GPU:=0}"
: "${MODE:=preflight}"
: "${OBJECTIVE:=answer_pg}"
: "${PROTOCOL:=aligned}"
: "${EPOCHS:=1}"
: "${N_TRAIN:=4}"
: "${N_VALIDATION:=2}"
: "${N_TEST:=2}"
: "${TRAIN_DEPTHS:=1 2 4 8}"
: "${EVAL_DEPTHS:=0 1 2 4 8}"
: "${PLAN_STEPS:=1000}"
: "${CFG_SCALE:=2.0}"
: "${STATE_BLEND:=0.7}"
: "${GROUP_SIZE:=4}"
: "${LR:=1e-4}"
: "${MAX_NEW_TOKENS:=32}"
: "${OUTPUT_DIR:=outputs_eval/laces_pretrained_reasoner}"
: "${CACHE_DIR:=outputs_cache/laces_pretrained_reasoner}"
read -r -a train_depths <<< "$TRAIN_DEPTHS"
read -r -a eval_depths <<< "$EVAL_DEPTHS"
extra=()
[[ -z "${RWKV_PATH:-}" ]] || extra+=(--rwkv_path "$RWKV_PATH")
[[ -z "${RESUME:-}" ]] || extra+=(--resume "$RESUME")
if [[ -n "${TRAIN_JSONL:-}${VALIDATION_JSONL:-}${TEST_JSONL:-}" ]]; then
  : "${TRAIN_JSONL:?All three data splits are required}"
  : "${VALIDATION_JSONL:?All three data splits are required}"
  : "${TEST_JSONL:?All three data splits are required}"
  extra+=(--train_jsonl "$TRAIN_JSONL" --validation_jsonl "$VALIDATION_JSONL" --test_jsonl "$TEST_JSONL")
fi
exec env CUDA_VISIBLE_DEVICES="$GPU" "${PYTHON:-python}" -u scripts/eval/train_recurrent_reasoner.py \
  --ckpt_dir "$CKPT_DIR" --device cuda:0 --mode "$MODE" --objective "$OBJECTIVE" --protocol "$PROTOCOL" \
  --epochs "$EPOCHS" --n_train "$N_TRAIN" --n_validation "$N_VALIDATION" --n_test "$N_TEST" \
  --train_depths "${train_depths[@]}" --eval_depths "${eval_depths[@]}" \
  --plan_steps "$PLAN_STEPS" --cfg_scale "$CFG_SCALE" --blend "$STATE_BLEND" \
  --group_size "$GROUP_SIZE" --lr "$LR" --max_new_tokens "$MAX_NEW_TOKENS" \
  --output_dir "$OUTPUT_DIR" --cache_dir "$CACHE_DIR" "${extra[@]}" "$@"
