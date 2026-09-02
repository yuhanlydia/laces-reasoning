#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "${SCRIPT_DIR}/../../../../../../../" && pwd)"
USER_GPU_SET="${GPU+x}"
USER_NUM_WORKERS_SET="${NUM_WORKERS+x}"
source "${TRAINING_DIR}/common.sh"

if [[ -z "${USER_GPU_SET}" ]]; then
  GPU=1
fi
if [[ -z "${USER_NUM_WORKERS_SET}" ]]; then
  NUM_WORKERS=0
fi
: "${S1_BATCH_SIZE:=1}"
: "${S2_BATCH_SIZE:=1}"
: "${STATE_BLEND:=0.5}"
: "${CFG_DROP_PROB:=0.1}"
: "${BASIS:=32}"
: "${BACKBONE:=2.9B}"
: "${CONTEXT_LENGTH:=4096}"
: "${S2_DENOISER_HIDDEN:=768}"
: "${S2_DENOISER_DEPTH:=8}"
: "${TRAJ_DELTA:=0.02}"

if [[ "${BACKBONE}" != "2.9B" ]]; then
  echo "This launcher is for BACKBONE=2.9B only." >&2
  exit 2
fi
if [[ "${CONTEXT_LENGTH}" != "4096" ]]; then
  echo "This launcher is for CONTEXT_LENGTH=4096 only." >&2
  exit 2
fi
validate_basis "${BASIS}"

CONFIG="$(trajectory_config_for "${BACKBONE}")"
TOKEN_DIR="${TOKEN_DIR:-$(trajectory_token_dir_for "${CONTEXT_LENGTH}")}"
CHUNK="$(trajectory_chunk_for "${CONTEXT_LENGTH}")"
HORIZON="$(trajectory_horizon_for "${CONTEXT_LENGTH}")"
CONTEXT_LABEL="$(context_label_for "${CONTEXT_LENGTH}")"
STATE_BLEND_LABEL="${STATE_BLEND//./p}"

BASE_NAME="${CONTEXT_LABEL}-traj${CHUNK}x${HORIZON}-${BACKBONE}-basis${BASIS}"
S0_CKPT="${S0_CKPT:-outputs_relay/${CONTEXT_LABEL}-traj${CHUNK}x${HORIZON}-${BACKBONE}-s0/step_$(printf '%08d' "${S0_STEPS}")}"
S1_NAME="${BASE_NAME}-prefix-suffix-blend${STATE_BLEND_LABEL}-s1-rwkv"
S2_NAME="${BASE_NAME}-prefix-suffix-blend${STATE_BLEND_LABEL}-s2-rwkv-s1-birwkv-rf"
S1_CKPT="${S1_CKPT:-outputs_relay/${S1_NAME}/step_$(printf '%08d' "${S1_STEPS}")}"

print_header "${CONTEXT_LENGTH} ${BACKBONE} prefix/suffix mixed trajectory S1=RWKV S2=BiRWKV RF basis=${BASIS} blend=${STATE_BLEND}"

S1_ARGS=(
  train_state_hijacking_dit.py
  --config-name "${CONFIG}"
  data.token_dir="${TOKEN_DIR}" data.max_length="${CONTEXT_LENGTH}"
  model.n_basis="${BASIS}"
  model.trajectory_chunk_size="${CHUNK}" model.trajectory_horizon="${HORIZON}"
  training.stage=1 training.num_train_steps="${S1_STEPS}"
  training.resume="${S0_CKPT}"
  +training.prefix_suffix_trajectory_s1=true
  +training.prefix_suffix_min_prefix="${CHUNK}" +training.prefix_suffix_min_suffix="${CHUNK}"
  model.trajectory_s1_mode=rwkv model.trajectory_state_hidden=256
  model.trajectory_state_depth=4 model.trajectory_state_blend="${STATE_BLEND}"
  training.save_every_n_steps="${SAVE_EVERY}" data.num_workers="${NUM_WORKERS}"
  training.train_batch_size="${S1_BATCH_SIZE}"
  logging.run_name="${S1_NAME}"
)
if [[ "${DRY_RUN}" != "1" && "${FORCE_S1:-0}" != "1" && -f "${S1_CKPT}/model.pt" ]]; then
  echo "Found existing S1 checkpoint: ${S1_CKPT}/model.pt"
  echo "Skipping S1. Set FORCE_S1=1 to rerun S1 from ${S0_CKPT}."
else
  cuda_python "${GPU}" "${S1_ARGS[@]}"
fi

wait_for_ckpt "${S1_CKPT}"

S2_ARGS=(
  train_state_hijacking_dit.py
  --config-name "${CONFIG}"
  data.token_dir="${TOKEN_DIR}" data.max_length="${CONTEXT_LENGTH}"
  model.n_basis="${BASIS}"
  model.trajectory_chunk_size="${CHUNK}" model.trajectory_horizon="${HORIZON}"
  training.stage=2 training.gen_type=rf training.num_train_steps="${S2_STEPS}"
  training.resume="${S1_CKPT}"
  +training.prefix_suffix_trajectory_s2=true +training.cfg_drop_prob="${CFG_DROP_PROB}"
  +training.prefix_suffix_min_prefix="${CHUNK}" +training.prefix_suffix_min_suffix="${CHUNK}"
  loss.trajectory_delta_loss_weight="${TRAJ_DELTA}"
  model.trajectory_s1_mode=rwkv model.trajectory_denoiser_type=birwkv
  model.dit_hidden="${S2_DENOISER_HIDDEN}" model.dit_depth="${S2_DENOISER_DEPTH}"
  model.trajectory_state_hidden=256 model.trajectory_state_depth=4
  model.trajectory_state_blend="${STATE_BLEND}"
  training.save_every_n_steps="${SAVE_EVERY}" data.num_workers="${NUM_WORKERS}"
  training.train_batch_size="${S2_BATCH_SIZE}"
  logging.run_name="${S2_NAME}"
)
cuda_python "${GPU}" "${S2_ARGS[@]}"
