#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

: "${CONTEXT_LENGTH:?Set CONTEXT_LENGTH=512 or 4096}"
: "${ROUTE:?Set ROUTE=s2_alignment,planner_aware_s2,stochastic_continuation_s2,self_forcing_s2}"
: "${BASIS:=16}"
: "${ARCH:=dit}"
: "${OBJECTIVE:=rf}"
: "${ALLOW_UNIMPLEMENTED:=0}"
: "${STATE_BLEND:=0.5}"
: "${CFG_DROP_PROB:=0.1}"
: "${DENOISER_HIDDEN:=768}"
: "${DENOISER_DEPTH:=8}"
: "${DENOISER_HEADS:=8}"
: "${S2_ALIGN_WEIGHT:=0.01}"
: "${S2_ALIGN_START:=0}"
: "${S2_ALIGN_WARMUP:=1000}"
: "${S2_ALIGN_MAX_CHUNKS:=2}"
: "${S2_ALIGN_TEMPERATURE:=1.0}"

BACKBONE="${BACKBONE:-2.9B}"

validate_basis "${BASIS}"
validate_arch "${ARCH}"
validate_objective "${OBJECTIVE}"

CONFIG="$(trajectory_config_for "${BACKBONE}")"
CONTEXT_LABEL="$(context_label_for "${CONTEXT_LENGTH}")"
TOKEN_DIR="${TOKEN_DIR:-$(trajectory_token_dir_for "${CONTEXT_LENGTH}")}"
CHUNK="$(trajectory_chunk_for "${CONTEXT_LENGTH}")"
HORIZON="$(trajectory_horizon_for "${CONTEXT_LENGTH}")"
BASE_NAME="${CONTEXT_LABEL}-traj${CHUNK}x${HORIZON}-${BACKBONE}-basis${BASIS}"
STATE_BLEND_LABEL="${STATE_BLEND//./p}"
S1_NAME="${BASE_NAME}-prefix-suffix-blend${STATE_BLEND_LABEL}-s1-${ARCH}"
S2_NAME="${BASE_NAME}-prefix-suffix-blend${STATE_BLEND_LABEL}-s2-${ARCH}-${OBJECTIVE}"
S1_CKPT="${S1_CKPT:-outputs_relay/${S1_NAME}/step_00050000}"
S2_CKPT="${S2_CKPT:-outputs_relay/${S2_NAME}/step_00150000}"
RESUME_CKPT="${S1_CKPT}"
S1_MODE="$(trajectory_s1_mode_for_arch "${ARCH}")"
DENOISER="$(trajectory_denoiser_for_arch "${ARCH}")"
ROUTE_IMPLEMENTED=0

case "${ROUTE}" in
  s2_alignment)
    ROUTE_ARGS=(+loss.s2_align_loss_weight="${S2_ALIGN_WEIGHT}" +loss.s2_align_start_steps="${S2_ALIGN_START}" +loss.s2_align_warmup_steps="${S2_ALIGN_WARMUP}" +loss.s2_align_max_chunks="${S2_ALIGN_MAX_CHUNKS}" +loss.s2_align_temperature="${S2_ALIGN_TEMPERATURE}")
    RUN_SUFFIX="s2-align-kl"
    REQUIREMENT=""
    RESUME_CKPT="${S2_CKPT}"
    ROUTE_IMPLEMENTED=1
    ;;
  planner_aware_s2)
    ROUTE_ARGS=(+loss.planner_aware_weight=0.2 +loss.planner_weight_source=chunk_nll_repeat)
    RUN_SUFFIX="planner-aware"
    REQUIREMENT="loss.planner_aware_weight / loss.planner_weight_source must be implemented"
    ;;
  stochastic_continuation_s2)
    ROUTE_ARGS=(+training.rolling_trajectory_s2=true +training.rolling_windows=2 +model.trajectory_condition_prev_z=true +model.trajectory_condition_state=true +model.continuation_rho=0.7)
    RUN_SUFFIX="stochastic-cont"
    REQUIREMENT="rolling trajectory conditioning must be implemented"
    ;;
  self_forcing_s2)
    ROUTE_ARGS=(+training.self_forcing_trajectory_s2=true +training.self_forcing_windows=2 +training.self_forcing_stopgrad_prev=true +training.self_forcing_prefix_truncate="${CONTEXT_LENGTH}")
    RUN_SUFFIX="self-forcing"
    REQUIREMENT="self-forcing rollout training must be implemented"
    ;;
  *)
    echo "Unsupported ROUTE=${ROUTE}" >&2
    exit 2
    ;;
esac

if [[ "${ROUTE_IMPLEMENTED}" != "1" && "${ALLOW_UNIMPLEMENTED}" != "1" && "${DRY_RUN}" != "1" ]]; then
  echo "Refusing to launch ${ROUTE}: ${REQUIREMENT}." >&2
  echo "Use DRY_RUN=1 to inspect the command, or ALLOW_UNIMPLEMENTED=1 after implementing the route." >&2
  exit 2
fi

print_header "${CONTEXT_LENGTH} ${BACKBONE} Section 6 post-training route=${ROUTE} arch=${ARCH} basis=${BASIS} objective=${OBJECTIVE}"
wait_for_ckpt "${RESUME_CKPT}"

cuda_python "${GPU}" train_state_hijacking_dit.py \
  --config-name "${CONFIG}" \
  data.token_dir="${TOKEN_DIR}" data.max_length="${CONTEXT_LENGTH}" \
  model.n_basis="${BASIS}" \
  model.trajectory_chunk_size="${CHUNK}" model.trajectory_horizon="${HORIZON}" \
  training.stage=2 training.gen_type="${OBJECTIVE}" training.num_train_steps="${S2_STEPS}" \
  training.resume="${RESUME_CKPT}" \
  +training.prefix_suffix_trajectory_s2=true +training.cfg_drop_prob="${CFG_DROP_PROB}" \
  +training.prefix_suffix_min_prefix="${CHUNK}" +training.prefix_suffix_min_suffix="${CHUNK}" \
  loss.trajectory_delta_loss_weight=0.02 \
  "${ROUTE_ARGS[@]}" \
  model.trajectory_s1_mode="${S1_MODE}" model.trajectory_denoiser_type="${DENOISER}" \
  model.trajectory_state_blend="${STATE_BLEND}" \
  model.dit_hidden="${DENOISER_HIDDEN}" model.dit_depth="${DENOISER_DEPTH}" model.dit_num_heads="${DENOISER_HEADS}" \
  model.trajectory_state_hidden=256 model.trajectory_state_depth=4 \
  training.save_every_n_steps="${SAVE_EVERY}" data.num_workers="${NUM_WORKERS}" \
  training.train_batch_size="${BATCH_SIZE}" \
  logging.run_name="${BASE_NAME}-post-${ARCH}-${OBJECTIVE}-${RUN_SUFFIX}" \
  "$@"
