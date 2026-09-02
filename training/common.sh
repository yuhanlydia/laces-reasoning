#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

TRAINING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${TRAINING_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

: "${BACKBONE:=2.9B}"
: "${GPU:=0}"
: "${DRY_RUN:=0}"
: "${BATCH_SIZE:=1}"
: "${NUM_WORKERS:=8}"
: "${S0_STEPS:=50000}"
: "${S1_STEPS:=50000}"
: "${S2_STEPS:=150000}"
: "${SAVE_EVERY:=50000}"

single_config_for() {
  case "$1" in
    0.4B) echo "rwkv_relay_0.4B_state_hijack_dit_vae32" ;;
    2.9B) echo "rwkv_relay_2.9B_state_hijack_dit_vae32" ;;
    13.3B) echo "rwkv_relay_13.3B_state_hijack_dit_vae32" ;;
    *) echo "Unsupported BACKBONE=$1; use 0.4B, 2.9B, or 13.3B" >&2; exit 2 ;;
  esac
}

trajectory_config_for() {
  case "$1" in
    0.4B) echo "rwkv_relay_0.4B_state_hijack_dit_vae32_traj32x16" ;;
    2.9B) echo "rwkv_relay_2.9B_state_hijack_dit_vae32_traj32x16" ;;
    13.3B) echo "rwkv_relay_13.3B_state_hijack_dit_vae32_traj32x16" ;;
    *) echo "Unsupported BACKBONE=$1; use 0.4B, 2.9B, or 13.3B" >&2; exit 2 ;;
  esac
}

run_cmd() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1: command was printed only; no training process started and 0 steps ran. Use DRY_RUN=0 to launch." >&2
    return 0
  fi
  "$@"
}

require_multinode_env() {
  if [[ "${NNODES}" -le 1 ]]; then
    return 0
  fi

  if [[ -z "${NODE_RANK:-}" ]]; then
    echo "NNODES=${NNODES} requires NODE_RANK from the cluster launcher (0..NNODES-1)." >&2
    echo "QZ multi-node jobs must pass the platform-injected NODE_RANK; do not let it default to 0 on every node." >&2
    exit 2
  fi
  if [[ -z "${MASTER_ADDR:-}" || "${MASTER_ADDR}" == "127.0.0.1" || "${MASTER_ADDR}" == "localhost" ]]; then
    echo "NNODES=${NNODES} requires a non-local MASTER_ADDR shared by all nodes; got MASTER_ADDR=${MASTER_ADDR:-<empty>}." >&2
    echo "Use the node-0 address provided by QZ, otherwise each node starts its own localhost rendezvous and the job fails before training." >&2
    exit 2
  fi
}

cuda_python() {
  local gpu="$1"
  shift
  run_cmd env CUDA_VISIBLE_DEVICES="${gpu}" python -u "$@"
}

wait_for_ckpt() {
  local ckpt_dir="$1"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  echo "Waiting for ${ckpt_dir}/model.pt ..."
  while [[ ! -f "${ckpt_dir}/model.pt" ]]; do
    sleep 10
  done
}

print_header() {
  echo "========================================="
  echo "$1"
  echo "BACKBONE=${BACKBONE} GPU=${GPU} BATCH_SIZE=${BATCH_SIZE} DRY_RUN=${DRY_RUN}"
  echo "Project: ${PROJECT_ROOT}"
  echo "========================================="
}

context_label_for() {
  case "$1" in
    512) echo "owt512" ;;
    4096) echo "fineweb4096" ;;
    *) echo "Unsupported CONTEXT_LENGTH=$1; use 512 or 4096" >&2; exit 2 ;;
  esac
}

single_token_dir_for() {
  case "$1" in
    512) echo "preprocessed_data/owt_rwkv_tokens/train" ;;
    4096) echo "preprocessed_data/fineweb_4096_full" ;;
  esac
}

trajectory_token_dir_for() {
  case "$1" in
    512) echo "preprocessed_data/owt_rwkv_tokens/train" ;;
    4096) echo "preprocessed_data/fineweb_4096_packed_full" ;;
  esac
}

trajectory_chunk_for() {
  case "$1" in
    512) echo "32" ;;
    4096) echo "64" ;;
  esac
}

trajectory_horizon_for() {
  case "$1" in
    512) echo "16" ;;
    4096) echo "64" ;;
  esac
}

validate_basis() {
  case "$1" in
    8|16|32|64) ;;
    *) echo "Unsupported BASIS=$1; use 8, 16, 32, or 64" >&2; exit 2 ;;
  esac
}

validate_objective() {
  case "$1" in
    ddpm|rf|flow) ;;
    *) echo "Unsupported OBJECTIVE=$1; use ddpm, rf, or flow" >&2; exit 2 ;;
  esac
}

validate_arch() {
  case "$1" in
    dit|rwkv|birwkv) ;;
    *) echo "Unsupported ARCH=$1; use dit, rwkv, or birwkv" >&2; exit 2 ;;
  esac
}

trajectory_s1_mode_for_arch() {
  case "$1" in
    dit) echo "transformer" ;;
    rwkv) echo "rwkv" ;;
    birwkv) echo "birwkv" ;;
  esac
}

trajectory_denoiser_for_arch() {
  case "$1" in
    dit) echo "dit" ;;
    rwkv) echo "rwkv" ;;
    birwkv) echo "birwkv" ;;
  esac
}
