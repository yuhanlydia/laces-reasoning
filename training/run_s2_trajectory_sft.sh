#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${SAVE_EVERY:=50000}"
source "${SCRIPT_DIR}/common.sh"

: "${CONTEXT_LENGTH:?Set CONTEXT_LENGTH=512 or 4096}"
: "${SFT_TOKEN_DIR:?Set SFT_TOKEN_DIR to a prompt/response token directory with prompt_lengths}"
: "${SFT_RESUME:?Set SFT_RESUME to an existing S2 checkpoint directory or model.pt}"
: "${ARCH:=dit}"
: "${BASIS:=32}"
: "${OBJECTIVE:=ddpm}"
: "${CFG_DROP_PROB:=0.1}"
: "${STATE_BLEND:=0.5}"
: "${DENOISER_HIDDEN:=768}"
: "${DENOISER_DEPTH:=8}"
: "${DENOISER_HEADS:=8}"
: "${SFT_STEPS:=20000}"

validate_arch "${ARCH}"
validate_basis "${BASIS}"
validate_objective "${OBJECTIVE}"

CONFIG="$(trajectory_config_for "${BACKBONE}")"
CONTEXT_LABEL="$(context_label_for "${CONTEXT_LENGTH}")"
CHUNK="$(trajectory_chunk_for "${CONTEXT_LENGTH}")"
HORIZON="$(trajectory_horizon_for "${CONTEXT_LENGTH}")"
S1_MODE="$(trajectory_s1_mode_for_arch "${ARCH}")"
DENOISER="$(trajectory_denoiser_for_arch "${ARCH}")"
STATE_BLEND_LABEL="${STATE_BLEND//./p}"
RUN_NAME="${RUN_NAME:-${CONTEXT_LABEL}-traj${CHUNK}x${HORIZON}-${BACKBONE}-basis${BASIS}-s2-${ARCH}-${OBJECTIVE}-trajectory-sft-blend${STATE_BLEND_LABEL}}"

TRAJ_DELTA_DEFAULT="0.02"
if [[ "${OBJECTIVE}" == "ddpm" ]]; then
  TRAJ_DELTA_DEFAULT="0.0"
fi
TRAJ_DELTA="${TRAJ_DELTA:-${TRAJ_DELTA_DEFAULT}}"

print_header "${CONTEXT_LENGTH} ${BACKBONE} S2 trajectory SFT arch=${ARCH} basis=${BASIS} objective=${OBJECTIVE}"

ARGS=(
  train_state_hijacking_dit.py
  --config-name "${CONFIG}"
  data.token_dir="${SFT_TOKEN_DIR}" data.latent_dir=null data.use_external_latents=false
  data.max_length="${CONTEXT_LENGTH}" data.num_workers="${NUM_WORKERS}"
  +data.val_ratio=0.0
  model.n_basis="${BASIS}"
  model.trajectory_chunk_size="${CHUNK}" model.trajectory_horizon="${HORIZON}"
  training.stage=2 training.gen_type="${OBJECTIVE}" training.num_train_steps="${SFT_STEPS}"
  training.resume="${SFT_RESUME}"
  +training.prefix_suffix_trajectory_s2=true +training.s2_trajectory_sft=true
  +training.cfg_drop_prob="${CFG_DROP_PROB}"
  +training.prefix_suffix_min_prefix="${CHUNK}" +training.prefix_suffix_min_suffix="${CHUNK}"
  loss.trajectory_delta_loss_weight="${TRAJ_DELTA}"
  model.trajectory_s1_mode="${S1_MODE}" model.trajectory_denoiser_type="${DENOISER}"
  model.dit_hidden="${DENOISER_HIDDEN}" model.dit_depth="${DENOISER_DEPTH}"
  model.trajectory_state_hidden=256 model.trajectory_state_depth=4
  model.trajectory_state_blend="${STATE_BLEND}"
  training.save_every_n_steps="${SAVE_EVERY}" training.train_batch_size="${BATCH_SIZE}"
  logging.run_name="${RUN_NAME}"
)
if [[ "${ARCH}" == "dit" ]]; then
  ARGS+=(model.dit_num_heads="${DENOISER_HEADS}")
fi

cuda_python "${GPU}" "${ARGS[@]}"
