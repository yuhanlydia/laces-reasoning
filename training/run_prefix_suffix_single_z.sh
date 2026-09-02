#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

: "${CONTEXT_LENGTH:?Set CONTEXT_LENGTH=512 or 4096}"
: "${BASIS:?Set BASIS=8,16,32,64}"
: "${OBJECTIVE:?Set OBJECTIVE=ddpm,rf,flow}"
: "${CFG_DROP_PROB:=0.1}"

validate_basis "${BASIS}"
validate_objective "${OBJECTIVE}"

CONFIG="$(single_config_for "${BACKBONE}")"
CONTEXT_LABEL="$(context_label_for "${CONTEXT_LENGTH}")"
TOKEN_DIR="${TOKEN_DIR:-$(single_token_dir_for "${CONTEXT_LENGTH}")}"

NAME="${CONTEXT_LABEL}-${BACKBONE}-singlez-basis${BASIS}-prefix-suffix"
S0_NAME="${CONTEXT_LABEL}-${BACKBONE}-singlez-basis${BASIS}-s0"
S1_NAME="${NAME}-s1"
S2_NAME="${NAME}-s2-${OBJECTIVE}"
S0_CKPT="${S0_CKPT:-outputs_relay/${S0_NAME}/step_00050000}"
S1_CKPT="${S1_CKPT:-outputs_relay/${S1_NAME}/step_00050000}"

print_header "${CONTEXT_LENGTH} prefix/suffix single-z basis=${BASIS} objective=${OBJECTIVE}"
wait_for_ckpt "${S0_CKPT}"

cuda_python "${GPU}" train_state_hijacking_dit.py \
  --config-name "${CONFIG}" \
  data.token_dir="${TOKEN_DIR}" data.max_length="${CONTEXT_LENGTH}" \
  model.n_basis="${BASIS}" \
  training.stage=1 training.num_train_steps="${S1_STEPS}" \
  training.resume="${S0_CKPT}" \
  +training.prefix_suffix_s1=true \
  +training.prefix_suffix_min_prefix=32 +training.prefix_suffix_min_suffix=32 \
  training.save_every_n_steps="${SAVE_EVERY}" data.num_workers="${NUM_WORKERS}" \
  training.train_batch_size="${BATCH_SIZE}" logging.run_name="${S1_NAME}"
wait_for_ckpt "${S1_CKPT}"

cuda_python "${GPU}" train_state_hijacking_dit.py \
  --config-name "${CONFIG}" \
  data.token_dir="${TOKEN_DIR}" data.max_length="${CONTEXT_LENGTH}" \
  model.n_basis="${BASIS}" \
  training.stage=2 training.gen_type="${OBJECTIVE}" training.num_train_steps="${S2_STEPS}" \
  training.resume="${S1_CKPT}" \
  +training.prefix_suffix_s2=true +training.cfg_drop_prob="${CFG_DROP_PROB}" \
  +training.prefix_suffix_min_prefix=32 +training.prefix_suffix_min_suffix=32 \
  training.save_every_n_steps="${SAVE_EVERY}" data.num_workers="${NUM_WORKERS}" \
  training.train_batch_size="${BATCH_SIZE}" logging.run_name="${S2_NAME}"
