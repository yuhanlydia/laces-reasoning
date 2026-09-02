#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${SAVE_EVERY:=50000}"
source "${SCRIPT_DIR}/common.sh"

: "${SFT_TOKEN_DIR:?Set SFT_TOKEN_DIR to an SFT token directory with prompt_lengths}"
: "${SFT_RESUME:?Set SFT_RESUME to the single-z S1/S2 checkpoint to continue from}"
: "${CONTEXT_LENGTH:=512}"
: "${BASIS:=32}"
: "${OBJECTIVE:=ddpm}"
: "${CFG_DROP_PROB:=0.1}"
: "${SFT_STEPS:=20000}"

BACKBONE="${BACKBONE:-2.9B}"
validate_basis "${BASIS}"
validate_objective "${OBJECTIVE}"

CONFIG="$(single_config_for "${BACKBONE}")"
CONTEXT_LABEL="$(context_label_for "${CONTEXT_LENGTH}")"
RUN_NAME="${RUN_NAME:-${CONTEXT_LABEL}-${BACKBONE}-basis${BASIS}-singlez-${OBJECTIVE}-s2-sft}"

print_header "${CONTEXT_LENGTH} ${BACKBONE} single-z S2 SFT basis=${BASIS} objective=${OBJECTIVE}"

ARGS=(
  train_state_hijacking_dit.py
  --config-name "${CONFIG}"
  data.token_dir="${SFT_TOKEN_DIR}" data.latent_dir=null data.use_external_latents=false
  data.max_length="${CONTEXT_LENGTH}" data.num_workers="${NUM_WORKERS}"
  +data.val_ratio=0.0
  model.n_basis="${BASIS}"
  training.stage=2 training.gen_type="${OBJECTIVE}"
  +training.prefix_suffix_s2=true +training.s2_single_z_sft=true +training.cfg_drop_prob="${CFG_DROP_PROB}"
  +training.prefix_suffix_min_prefix=1 +training.prefix_suffix_min_suffix=1
  loss.diff_loss_weight=1.0 loss.kl_weight=0.0
  training.num_train_steps="${SFT_STEPS}" training.save_every_n_steps="${SAVE_EVERY}"
  training.train_batch_size="${BATCH_SIZE}"
  logging.run_name="${RUN_NAME}"
  training.resume="${SFT_RESUME}"
)

cuda_python "${GPU}" "${ARGS[@]}"
