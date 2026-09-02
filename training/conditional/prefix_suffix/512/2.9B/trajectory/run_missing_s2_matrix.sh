#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"
cd "${REPO_ROOT}"

: "${GPU:=0}"
: "${BATCH_SIZE:=4}"
: "${NUM_WORKERS:=0}"
: "${S2_STEPS:=150000}"
: "${SAVE_EVERY:=50000}"
: "${CFG_DROP_PROB:=0.1}"
: "${TOKEN_DIR:=preprocessed_data/owt_rwkv_tokens/train}"
: "${DRY_RUN:=0}"
: "${FORCE:=0}"
: "${QUEUE:=all}"

print_command() {
  printf 'CUDA_VISIBLE_DEVICES=%q' "${GPU}"
  printf ' %q' "$@"
  printf '\n'
}

run_s2() {
  local arch="$1"
  local blend="$2"
  local objective="$3"
  local blend_label="${blend//./p}"
  local s1_ckpt="outputs_relay/owt512-traj32x16-2.9B-basis32-prefix-suffix-blend${blend_label}-s1-${arch}/step_00050000"
  local run_name="owt512-traj32x16-2.9B-basis32-prefix-suffix-blend${blend_label}-s2-${arch}-${objective}"
  local final_ckpt="outputs_relay/${run_name}/step_00150000/model.pt"
  local delta="0.02"

  if [[ "${objective}" == "ddpm" ]]; then
    delta="0.0"
  fi

  if [[ ! -f "${s1_ckpt}/model.pt" ]]; then
    echo "Missing S1 checkpoint: ${s1_ckpt}/model.pt" >&2
    return 1
  fi

  if [[ "${FORCE}" != "1" && -f "${final_ckpt}" ]]; then
    echo "Skip existing final checkpoint: ${final_ckpt}"
    return 0
  fi

  local cmd=(
    python -u train_state_hijacking_dit.py
    --config-name rwkv_relay_2.9B_state_hijack_dit_vae32_traj32x16
    data.token_dir="${TOKEN_DIR}" data.max_length=512
    model.n_basis=32 model.trajectory_chunk_size=32 model.trajectory_horizon=16
    training.stage=2 training.gen_type="${objective}" training.num_train_steps="${S2_STEPS}"
    training.resume="${s1_ckpt}"
    +training.prefix_suffix_trajectory_s2=true +training.cfg_drop_prob="${CFG_DROP_PROB}"
    +training.prefix_suffix_min_prefix=32 +training.prefix_suffix_min_suffix=32
    loss.trajectory_delta_loss_weight="${delta}"
    model.trajectory_s1_mode="${arch}" model.trajectory_denoiser_type="${arch}"
    model.dit_hidden=768 model.dit_depth=8
    model.trajectory_state_hidden=256 model.trajectory_state_depth=4
    model.trajectory_state_blend="${blend}"
    training.save_every_n_steps="${SAVE_EVERY}" data.num_workers="${NUM_WORKERS}"
    training.train_batch_size="${BATCH_SIZE}" logging.run_name="${run_name}"
  )

  echo "==> ${run_name}"
  print_command "${cmd[@]}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" "${cmd[@]}"
}

case "${QUEUE}" in
  0)
    run_s2 rwkv 1.0 rf
    run_s2 rwkv 1.0 ddpm
    ;;
  1)
    run_s2 birwkv 0.5 rf
    ;;
  2)
    run_s2 birwkv 0.5 ddpm
    ;;
  3)
    run_s2 birwkv 1.0 rf
    run_s2 birwkv 1.0 ddpm
    ;;
  all)
    run_s2 rwkv 1.0 rf
    run_s2 rwkv 1.0 ddpm
    run_s2 birwkv 0.5 rf
    run_s2 birwkv 0.5 ddpm
    run_s2 birwkv 1.0 rf
    run_s2 birwkv 1.0 ddpm
    ;;
  *)
    echo "Unsupported QUEUE=${QUEUE}; use 0, 1, 2, 3, or all" >&2
    exit 2
    ;;
esac
