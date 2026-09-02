#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${SAVE_EVERY:=5000}"
source "${SCRIPT_DIR}/common.sh"

: "${SFT_TOKEN_DIR:?Set SFT_TOKEN_DIR to a response-mask token directory}"
: "${SFT_RESUME:?Set SFT_RESUME to the S0/S1/S2 checkpoint to continue from}"
: "${CONTEXT_LENGTH:=512}"
: "${BASIS:=32}"
: "${SFT_STEPS:=20000}"
: "${SFT_STAGE:=0}"

BACKBONE="${BACKBONE:-2.9B}"
CONFIG="$(single_config_for "${BACKBONE}")"
CONTEXT_LABEL="$(context_label_for "${CONTEXT_LENGTH}")"
RUN_NAME="${RUN_NAME:-${CONTEXT_LABEL}-${BACKBONE}-basis${BASIS}-response-sft}"

print_header "${CONTEXT_LENGTH} ${BACKBONE} response-only SFT basis=${BASIS} stage=${SFT_STAGE}"

ARGS=(
  train_state_hijacking_dit.py
  --config-name "${CONFIG}"
  data.token_dir="${SFT_TOKEN_DIR}" data.latent_dir=null data.use_external_latents=false
  data.max_length="${CONTEXT_LENGTH}" data.num_workers="${NUM_WORKERS}"
  +data.val_ratio=0.0
  model.n_basis="${BASIS}"
  training.stage="${SFT_STAGE}" +training.sft_response_only=true
  loss.diff_loss_weight=0.0 loss.kl_weight=0.0
  training.num_train_steps="${SFT_STEPS}" training.save_every_n_steps="${SAVE_EVERY}"
  training.train_batch_size="${BATCH_SIZE}"
  logging.run_name="${RUN_NAME}"
  training.resume="${SFT_RESUME}"
)

cuda_python "${GPU}" "${ARGS[@]}"
