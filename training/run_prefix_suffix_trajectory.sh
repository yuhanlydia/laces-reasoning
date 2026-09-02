#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

: "${CONTEXT_LENGTH:?Set CONTEXT_LENGTH=512 or 4096}"
: "${ARCH:?Set ARCH=dit,rwkv,birwkv}"
: "${BASIS:?Set BASIS=8,16,32,64}"
: "${OBJECTIVE:?Set OBJECTIVE=ddpm,rf,flow}"
: "${CFG_DROP_PROB:=0.1}"
: "${STATE_BLEND:=0.5}"
: "${DENOISER_HIDDEN:=768}"
: "${DENOISER_DEPTH:=8}"
: "${DENOISER_HEADS:=8}"

BACKBONE="${BACKBONE:-2.9B}"

validate_arch "${ARCH}"
validate_basis "${BASIS}"
validate_objective "${OBJECTIVE}"

CONFIG="$(trajectory_config_for "${BACKBONE}")"
CONTEXT_LABEL="$(context_label_for "${CONTEXT_LENGTH}")"
TOKEN_DIR="${TOKEN_DIR:-$(trajectory_token_dir_for "${CONTEXT_LENGTH}")}"
CHUNK="$(trajectory_chunk_for "${CONTEXT_LENGTH}")"
HORIZON="$(trajectory_horizon_for "${CONTEXT_LENGTH}")"
S1_MODE="$(trajectory_s1_mode_for_arch "${ARCH}")"
DENOISER="$(trajectory_denoiser_for_arch "${ARCH}")"
TRAJ_DELTA_DEFAULT="0.02"
if [[ "${OBJECTIVE}" == "ddpm" ]]; then
  TRAJ_DELTA_DEFAULT="0.0"
fi
TRAJ_DELTA="${TRAJ_DELTA:-${TRAJ_DELTA_DEFAULT}}"

BASE_NAME="${CONTEXT_LABEL}-traj${CHUNK}x${HORIZON}-${BACKBONE}-basis${BASIS}"
S0_NAME="${BASE_NAME}-s0"
S0_CKPT="${S0_CKPT:-outputs_relay/${S0_NAME}/step_00050000}"
STATE_BLEND_LABEL="${STATE_BLEND//./p}"
S1_NAME="${BASE_NAME}-prefix-suffix-blend${STATE_BLEND_LABEL}-s1-${ARCH}"
S2_NAME="${BASE_NAME}-prefix-suffix-blend${STATE_BLEND_LABEL}-s2-${ARCH}-${OBJECTIVE}"
S1_CKPT="${S1_CKPT:-outputs_relay/${S1_NAME}/step_00050000}"

print_header "${CONTEXT_LENGTH} ${BACKBONE} prefix/suffix trajectory arch=${ARCH} basis=${BASIS} objective=${OBJECTIVE} state_blend=${STATE_BLEND}"
wait_for_ckpt "${S0_CKPT}"

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
  model.trajectory_s1_mode="${S1_MODE}" model.trajectory_state_hidden=256
  model.trajectory_state_depth=4 model.trajectory_state_blend="${STATE_BLEND}"
  training.save_every_n_steps="${SAVE_EVERY}" data.num_workers="${NUM_WORKERS}"
  training.train_batch_size="${BATCH_SIZE}" logging.run_name="${S1_NAME}"
)
cuda_python "${GPU}" "${S1_ARGS[@]}"
wait_for_ckpt "${S1_CKPT}"

S2_ARGS=(
  train_state_hijacking_dit.py
  --config-name "${CONFIG}"
  data.token_dir="${TOKEN_DIR}" data.max_length="${CONTEXT_LENGTH}"
  model.n_basis="${BASIS}"
  model.trajectory_chunk_size="${CHUNK}" model.trajectory_horizon="${HORIZON}"
  training.stage=2 training.gen_type="${OBJECTIVE}" training.num_train_steps="${S2_STEPS}"
  training.resume="${S1_CKPT}"
  +training.prefix_suffix_trajectory_s2=true +training.cfg_drop_prob="${CFG_DROP_PROB}"
  +training.prefix_suffix_min_prefix="${CHUNK}" +training.prefix_suffix_min_suffix="${CHUNK}"
  loss.trajectory_delta_loss_weight="${TRAJ_DELTA}"
  model.trajectory_s1_mode="${S1_MODE}" model.trajectory_denoiser_type="${DENOISER}"
  model.dit_hidden="${DENOISER_HIDDEN}" model.dit_depth="${DENOISER_DEPTH}"
  model.trajectory_state_hidden=256 model.trajectory_state_depth=4
  model.trajectory_state_blend="${STATE_BLEND}" training.save_every_n_steps="${SAVE_EVERY}"
  training.train_batch_size="${BATCH_SIZE}" data.num_workers="${NUM_WORKERS}"
  logging.run_name="${S2_NAME}"
)
if [[ "${ARCH}" == "dit" ]]; then
  S2_ARGS+=(model.dit_num_heads="${DENOISER_HEADS}")
fi
cuda_python "${GPU}" "${S2_ARGS[@]}"
