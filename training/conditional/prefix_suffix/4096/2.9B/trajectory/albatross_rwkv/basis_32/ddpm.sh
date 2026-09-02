#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "${SCRIPT_DIR}/../../../../../../../" && pwd)"
USER_GPU_SET="${GPU+x}"
USER_NUM_WORKERS_SET="${NUM_WORKERS+x}"
source "${TRAINING_DIR}/common.sh"

if [[ -z "${USER_GPU_SET}" ]]; then
  GPU=3
fi
if [[ -z "${USER_NUM_WORKERS_SET}" ]]; then
  NUM_WORKERS=0
fi
: "${BATCH_SIZE:=1}"
: "${STATE_BLEND:=0.5}"
: "${CFG_DROP_PROB:=0.1}"
: "${BASIS:=32}"
: "${BACKBONE:=2.9B}"
: "${CONTEXT_LENGTH:=4096}"
: "${SAVE_EVERY:=50000}"

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

BASE_NAME="albatross-goose-${BACKBONE}-${CONTEXT_LABEL}-traj${CHUNK}x${HORIZON}-basis${BASIS}"
S2_NAME="${BASE_NAME}-prefix-suffix-blend${STATE_BLEND_LABEL}-s2-rwkv-ddpm"

# Ordered S1_CKPT fallback: explicit env > dai-resubmit > default.
# If S1_CKPT is explicitly set, use it directly.
# Otherwise try each default in order; use the first one that exists.
# If none exists and S1_CKPT is unset, exit nonzero.
if [[ -n "${S1_CKPT:-}" ]]; then
  :  # explicit S1_CKPT provided — use it as-is
else
  _S1_DEFAULT_1="outputs_relay/fineweb4096-traj64x64-2.9B-basis32-prefix-suffix-blend0p5-s1-rwkv-h200x4-dai-resubmit-save1k-20260624/step_00010000"
  _S1_DEFAULT_2="outputs_relay/fineweb4096-traj64x64-2.9B-basis32-prefix-suffix-blend0p5-s1-rwkv/step_00010000"
  if [[ -f "${_S1_DEFAULT_1}/model.pt" ]]; then
    S1_CKPT="${_S1_DEFAULT_1}"
  elif [[ -f "${_S1_DEFAULT_2}/model.pt" ]]; then
    S1_CKPT="${_S1_DEFAULT_2}"
  else
    echo "S1_CKPT required: no default checkpoint found and S1_CKPT is not set." >&2
    echo "Tried:" >&2
    echo "  ${_S1_DEFAULT_1}" >&2
    echo "  ${_S1_DEFAULT_2}" >&2
    echo "Set S1_CKPT=<path> to an existing S1 checkpoint with model.pt." >&2
    exit 2
  fi
fi

# S2_STEPS: use NUM_TRAIN_STEPS if provided, otherwise default 150000.
S2_STEPS="${NUM_TRAIN_STEPS:-${S2_STEPS:-150000}}"

# RUN_NAME override: explicit RUN_NAME env wins; otherwise use S2_NAME.
RUN_NAME="${RUN_NAME:-${S2_NAME}}"

print_header "${CONTEXT_LENGTH} ${BACKBONE} prefix/suffix trajectory S2 Albatross-RWKV DDPM basis=${BASIS} blend=${STATE_BLEND}"

S2_ARGS=(
  train_state_hijacking_dit.py
  --config-name "${CONFIG}"
  data.token_dir="${TOKEN_DIR}" data.max_length="${CONTEXT_LENGTH}"
  model.n_basis="${BASIS}"
  model.trajectory_chunk_size="${CHUNK}" model.trajectory_horizon="${HORIZON}"
  training.stage=2 training.gen_type=ddpm training.num_train_steps="${S2_STEPS}"
  training.resume="${S1_CKPT}"
  +training.prefix_suffix_trajectory_s2=true +training.cfg_drop_prob="${CFG_DROP_PROB}"
  +training.prefix_suffix_min_prefix="${CHUNK}" +training.prefix_suffix_min_suffix="${CHUNK}"
  loss.trajectory_delta_loss_weight=0.0
  model.trajectory_s1_mode=rwkv model.trajectory_denoiser_type=rwkv
  +model.latent_rwkv_variant=albatross_goose
  model.trajectory_state_hidden=256 model.trajectory_state_depth=4
  model.trajectory_state_blend="${STATE_BLEND}"
  training.save_every_n_steps="${SAVE_EVERY}" data.num_workers="${NUM_WORKERS}"
  training.train_batch_size="${BATCH_SIZE}"
  logging.run_name="${RUN_NAME}"
)
cuda_python "${GPU}" "${S2_ARGS[@]}"
