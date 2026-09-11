#!/usr/bin/env bash
# Direct post-training of the original S2. No new writer, GRU or latent refiner.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${CKPT_DIR:?Set CKPT_DIR to the existing step_00030000 directory}"
: "${DATA_DIR:?Prepare a MMLU-Pro data bundle and set DATA_DIR}"
PYTHON="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
args=(--data "$DATA_DIR" --ckpt-dir "$CKPT_DIR" --device "${DEVICE:-cuda:0}"
  --output "${OUTPUT_DIR:-results/mmlu_pro/direct_s2}"
  --mode "${MODE:-preflight}" --objective "${OBJECTIVE:-distill}"
  --distill-source "${DISTILL_SOURCE:-candidates}" --reward "${REWARD:-gold_logprob}"
  --steps "${NUM_STEPS:-200}" --diffusion-steps "${DIFFUSION_STEPS:-32}"
  --native-control-steps "${NATIVE_CONTROL_STEPS:-1000}"
  --eta "${ETA:-0.3}" --min-std "${MIN_STD:-0.02}" --cfg-scale "${CFG_SCALE:-2.0}"
  --blend "${STATE_BLEND:-0.7}" --lr "${LR:-1e-6}" --weight-decay "${WEIGHT_DECAY:-0}"
  --group-size "${GROUP_SIZE:-4}" --inner-epochs "${INNER_EPOCHS:-1}"
  --kl-coef "${KL_COEF:-0.01}" --clip-range "${CLIP_RANGE:-0.2}"
  --distill-temperature "${DISTILL_TEMPERATURE:-0.5}"
  --seed "${SEED:-42}" --eval-samples "${EVAL_SAMPLES:-1}"
  --train-limit "${TRAIN_LIMIT:-0}" --dev-limit "${DEV_LIMIT:-0}" --test-limit "${TEST_LIMIT:-0}"
  --eval-every "${EVAL_EVERY:-25}" --save-every "${SAVE_EVERY:-25}"
  --split "${SPLIT:-dev}" --answer-mode "${ANSWER_MODE:-direct}"
  --max-new-tokens "${MAX_NEW_TOKENS:-256}" --eval-sampler "${EVAL_SAMPLER:-stochastic}"
  --expected-step "${EXPECTED_STEP:-30000}" --expected-writer "${EXPECTED_WRITER:-dynlowrank}")
[[ -z "${RWKV_PATH:-}" ]] || args+=(--rwkv-path "$RWKV_PATH")
[[ -z "${RESUME:-}" ]] || args+=(--resume "$RESUME")
[[ -z "${INIT_S2:-}" ]] || args+=(--init-s2 "$INIT_S2")
[[ -z "${S2_CHECKPOINT:-}" ]] || args+=(--s2-checkpoint "$S2_CHECKPOINT")
[[ "${ACKNOWLEDGE_TEST:-0}" != 1 ]] || args+=(--acknowledge-test)
printf 'Direct S2 run: mode=%s objective=%s checkpoint=%s\n' "${MODE:-preflight}" "${OBJECTIVE:-distill}" "$CKPT_DIR"
exec "$PYTHON" -m laces_posttrain.run "${args[@]}" "$@"
