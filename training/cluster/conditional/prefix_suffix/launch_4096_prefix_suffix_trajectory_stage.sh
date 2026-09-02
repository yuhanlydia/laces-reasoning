#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${TRAINING_DIR}/common.sh"

: "${BACKBONE:=2.9B}"
: "${ARCH:=dit}"
: "${BASIS:=16}"
: "${OBJECTIVE:=rf}"
: "${STAGE:=1}"
: "${CONTEXT_LENGTH:=4096}"
: "${CFG_DROP_PROB:=0.1}"
: "${STATE_BLEND:=0.5}"
: "${DENOISER_HIDDEN:=768}"
: "${DENOISER_DEPTH:=8}"
: "${DENOISER_HEADS:=8}"
: "${NPROC_PER_NODE:=8}"
: "${NNODES:=1}"
if [[ "${NNODES}" == "1" ]]; then
  : "${NODE_RANK:=0}"
  : "${MASTER_ADDR:=127.0.0.1}"
else
  : "${NODE_RANK:=}"
  : "${MASTER_ADDR:=}"
fi
: "${MASTER_PORT:=29500}"

require_multinode_env

if [[ "${CONTEXT_LENGTH}" != "4096" ]]; then
  echo "This launcher is for CONTEXT_LENGTH=4096 only." >&2
  exit 2
fi
if [[ "${STAGE}" == "all" || "${STAGE}" == "both" || "${STAGE}" == "1,2" ]]; then
  if [[ "${NNODES}" != "1" ]]; then
    echo "STAGE=${STAGE} is only supported for single-node H200x8 runs; use separate STAGE=1 and STAGE=2 commands for NNODES=${NNODES}." >&2
    exit 2
  fi
  echo "Running prefix/suffix STAGE=1 then STAGE=2 with one command."
  STAGE=1 "${BASH_SOURCE[0]}"
  STAGE=2 "${BASH_SOURCE[0]}"
  exit 0
fi
if [[ "${STAGE}" != "1" && "${STAGE}" != "2" ]]; then
  echo "Unsupported STAGE=${STAGE}; use 1, 2, or all for prefix/suffix S1 then S2." >&2
  exit 2
fi

validate_arch "${ARCH}"
validate_basis "${BASIS}"
validate_objective "${OBJECTIVE}"

CONFIG="$(trajectory_config_for "${BACKBONE}")"
TOKEN_DIR="${TOKEN_DIR:-$(trajectory_token_dir_for "${CONTEXT_LENGTH}")}"
CHUNK="$(trajectory_chunk_for "${CONTEXT_LENGTH}")"
HORIZON="$(trajectory_horizon_for "${CONTEXT_LENGTH}")"
S1_MODE="$(trajectory_s1_mode_for_arch "${ARCH}")"
DENOISER="$(trajectory_denoiser_for_arch "${ARCH}")"
CONTEXT_LABEL="$(context_label_for "${CONTEXT_LENGTH}")"
TRAJ_DELTA_DEFAULT="0.02"
if [[ "${OBJECTIVE}" == "ddpm" ]]; then
  TRAJ_DELTA_DEFAULT="0.0"
fi
TRAJ_DELTA="${TRAJ_DELTA:-${TRAJ_DELTA_DEFAULT}}"

BASE_NAME="${CONTEXT_LABEL}-traj${CHUNK}x${HORIZON}-${BACKBONE}-basis${BASIS}"
S0_NAME="${BASE_NAME}-s0"
STATE_BLEND_LABEL="${STATE_BLEND//./p}"
S1_NAME="${BASE_NAME}-prefix-suffix-blend${STATE_BLEND_LABEL}-s1-${ARCH}"
S2_NAME="${BASE_NAME}-prefix-suffix-blend${STATE_BLEND_LABEL}-s2-${ARCH}-${OBJECTIVE}"

S0_CKPT="${S0_CKPT:-outputs_relay/${S0_NAME}/step_$(printf '%08d' "${S0_STEPS}")}"
S1_CKPT="${S1_CKPT:-outputs_relay/${S1_NAME}/step_$(printf '%08d' "${S1_STEPS}")}"

checkpoint_exists() {
  local ckpt="$1"
  [[ -f "${ckpt}" || -f "${ckpt}/model.pt" ]]
}

require_checkpoint() {
  local label="$1"
  local ckpt="$2"
  local hint="$3"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  if ! checkpoint_exists "${ckpt}"; then
    echo "Missing ${label}: ${ckpt}" >&2
    echo "${hint}" >&2
    exit 2
  fi
}

COMMON_ARGS=(
  --config-name "${CONFIG}"
  data.token_dir="${TOKEN_DIR}"
  data.max_length="${CONTEXT_LENGTH}"
  model.n_basis="${BASIS}"
  model.trajectory_chunk_size="${CHUNK}"
  model.trajectory_horizon="${HORIZON}"
  training.train_batch_size="${BATCH_SIZE}"
  data.num_workers="${NUM_WORKERS}"
  training.save_every_n_steps="${SAVE_EVERY}"
)

case "${STAGE}" in
  1)
    RUN_NAME="${S1_NAME}"
    STAGE_ARGS=(
      training.stage=1
      training.num_train_steps="${S1_STEPS}"
      training.resume="${S0_CKPT}"
      +training.prefix_suffix_trajectory_s1=true
      +training.prefix_suffix_min_prefix="${CHUNK}"
      +training.prefix_suffix_min_suffix="${CHUNK}"
      model.trajectory_s1_mode="${S1_MODE}"
      model.trajectory_state_hidden=256
      model.trajectory_state_depth=4
      model.trajectory_state_blend="${STATE_BLEND}"
      logging.run_name="${RUN_NAME}"
    )
    ;;
  2)
    RUN_NAME="${S2_NAME}"
    STAGE_ARGS=(
      training.stage=2
      training.gen_type="${OBJECTIVE}"
      training.num_train_steps="${S2_STEPS}"
      training.resume="${S1_CKPT}"
      +training.prefix_suffix_trajectory_s2=true
      +training.cfg_drop_prob="${CFG_DROP_PROB}"
      +training.prefix_suffix_min_prefix="${CHUNK}"
      +training.prefix_suffix_min_suffix="${CHUNK}"
      loss.trajectory_delta_loss_weight="${TRAJ_DELTA}"
      model.trajectory_s1_mode="${S1_MODE}"
      model.trajectory_denoiser_type="${DENOISER}"
      model.dit_hidden="${DENOISER_HIDDEN}"
      model.dit_depth="${DENOISER_DEPTH}"
      model.trajectory_state_hidden=256
      model.trajectory_state_depth=4
      model.trajectory_state_blend="${STATE_BLEND}"
      logging.run_name="${RUN_NAME}"
    )
    if [[ "${ARCH}" == "dit" ]]; then
      STAGE_ARGS+=(model.dit_num_heads="${DENOISER_HEADS}")
    fi
    ;;
esac

case "${STAGE}" in
  1)
    require_checkpoint "S0_CKPT" "${S0_CKPT}" \
      "Run the matching trajectory STAGE=0 first, or pass S0_CKPT=<existing-s0-step>."
    ;;
  2)
    require_checkpoint "S1_CKPT" "${S1_CKPT}" \
      "Run prefix/suffix STAGE=1 first, or pass S1_CKPT=<existing-s1-step>."
    ;;
esac

CMD=(
  torchrun
  --nnodes="${NNODES}"
  --nproc_per_node="${NPROC_PER_NODE}"
  --node_rank="${NODE_RANK}"
  --master_addr="${MASTER_ADDR}"
  --master_port="${MASTER_PORT}"
  train_state_hijacking_dit.py
  "${COMMON_ARGS[@]}"
  "${STAGE_ARGS[@]}"
)

echo "========================================="
echo "4096 prefix/suffix trajectory cluster stage launcher"
echo "BACKBONE=${BACKBONE} ARCH=${ARCH} BASIS=${BASIS} OBJECTIVE=${OBJECTIVE} STAGE=${STAGE} STATE_BLEND=${STATE_BLEND}"
echo "NNODES=${NNODES} NPROC_PER_NODE=${NPROC_PER_NODE} NODE_RANK=${NODE_RANK} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "BATCH_SIZE=${BATCH_SIZE} RUN_NAME=${RUN_NAME} S0_CKPT=${S0_CKPT} S1_CKPT=${S1_CKPT} DRY_RUN=${DRY_RUN}"
echo "========================================="

run_cmd "${CMD[@]}"
