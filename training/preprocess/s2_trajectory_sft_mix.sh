#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

: "${CONTEXT_LENGTH:?Set CONTEXT_LENGTH=512 or 4096}"
: "${MIX_MODE:=safe}"

case "${CONTEXT_LENGTH}" in
  512|4096) ;;
  *) echo "Unsupported CONTEXT_LENGTH=${CONTEXT_LENGTH}; use 512 or 4096" >&2; exit 2 ;;
esac

case "${MIX_MODE}" in
  safe|maximal) ;;
  *) echo "Unsupported MIX_MODE=${MIX_MODE}; use safe or maximal" >&2; exit 2 ;;
esac

CHUNK="$(trajectory_chunk_for "${CONTEXT_LENGTH}")"
SFT_PROMPT_TOKENS="${SFT_PROMPT_TOKENS:-}"
OUT_DIR="${OUT_DIR:-preprocessed_data/s2_traj_sft_${CONTEXT_LENGTH}_${MIX_MODE}}"
MODEL_PATH="${MODEL_PATH:-}"
FORCE_REPROCESS="${FORCE_REPROCESS:-0}"

if [[ -n "${SFT_PROMPT_TOKENS}" ]]; then
  if [[ ! "${SFT_PROMPT_TOKENS}" =~ ^[0-9]+$ ]]; then
    echo "Unsupported SFT_PROMPT_TOKENS=${SFT_PROMPT_TOKENS}; use a positive integer multiple of ${CHUNK}" >&2
    exit 2
  fi
  SFT_PROMPT_TOKENS=$((10#${SFT_PROMPT_TOKENS}))
  if (( SFT_PROMPT_TOKENS < CHUNK || SFT_PROMPT_TOKENS >= CONTEXT_LENGTH || SFT_PROMPT_TOKENS % CHUNK != 0 )); then
    echo "Unsupported SFT_PROMPT_TOKENS=${SFT_PROMPT_TOKENS}; use a multiple of ${CHUNK} in [${CHUNK}, $((CONTEXT_LENGTH - 1))]" >&2
    exit 2
  fi
fi
MIN_PROMPT_TOKENS="${SFT_PROMPT_TOKENS:-${CHUNK}}"
if [[ -z "${MODEL_PATH}" ]]; then
  case "${BACKBONE}" in
    0.4B) MODEL_PATH="/inspire/hdd/global_user/zhangjiaquan-253108540222/models/rwkv7-0.4B-world" ;;
    2.9B) MODEL_PATH="/inspire/hdd/global_user/zhangjiaquan-253108540222/models/RWKV7-Goose-World3-2.9B-HF" ;;
    13.3B) MODEL_PATH="/inspire/hdd/global_user/zhangjiaquan-253108540222/models/rwkv7-13.3B-world" ;;
    *) echo "Unsupported BACKBONE=${BACKBONE}" >&2; exit 2 ;;
  esac
fi

if [[ "${MIX_MODE}" == "maximal" ]]; then
  : "${TULU3_MAX:=0}"
  : "${TULU3_IF_MAX:=0}"
  : "${OPENTHOUGHTS3_MAX:=0}"
  : "${OPENR1_MOT_MAX:=0}"
  : "${NEMOTRON_MATH_MAX:=0}"
  : "${NEMOTRON_OPENCODE_MAX:=0}"
  : "${OT_AGENT_MAX:=0}"
  : "${FABLE5_MAX:=0}"
  : "${AGENTTROVE_MAX:=0}"
  : "${NEMOTRON_POST_MAX:=0}"
else
  : "${TULU3_MAX:=300000}"
  : "${TULU3_IF_MAX:=30000}"
  : "${OPENTHOUGHTS3_MAX:=300000}"
  : "${OPENR1_MOT_MAX:=200000}"
  : "${NEMOTRON_MATH_MAX:=150000}"
  : "${NEMOTRON_OPENCODE_MAX:=150000}"
  : "${OT_AGENT_MAX:=100000}"
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${OUT_DIR}"
fi

preprocess_dataset() {
  local prefix="$1"
  local dataset="$2"
  local max_samples="$3"
  shift 3
  local manifest_path="${OUT_DIR}/manifest_${prefix}.json"
  if [[ "${FORCE_REPROCESS}" != "1" && -f "${manifest_path}" ]]; then
    echo "Skipping ${prefix}: found ${manifest_path}. Set FORCE_REPROCESS=1 to regenerate." >&2
    return 0
  fi
  local args=(
    scripts/preprocess/preprocess_response_sft.py
    --dataset "${dataset}"
    --output_dir "${OUT_DIR}"
    --model_path "${MODEL_PATH}"
    --max_length "${CONTEXT_LENGTH}"
    --min_response_tokens "${CHUNK}"
    --filename_prefix "${prefix}"
    --local_files_only
    --streaming
    "$@"
  )
  if [[ -n "${SFT_PROMPT_TOKENS}" ]]; then
    args+=(--min_prompt_tokens "${SFT_PROMPT_TOKENS}" --fixed_prompt_tokens "${SFT_PROMPT_TOKENS}")
  else
    args+=(--min_prompt_tokens "${MIN_PROMPT_TOKENS}")
  fi
  if [[ "${max_samples}" != "0" ]]; then
    args+=(--max_samples "${max_samples}")
  fi
  run_cmd python "${args[@]}"
}

print_header "preprocess S2 trajectory SFT mix=${MIX_MODE} context=${CONTEXT_LENGTH} out=${OUT_DIR}"

preprocess_dataset tulu3_core "allenai/tulu-3-sft-mixture" "${TULU3_MAX}"
preprocess_dataset tulu3_if "allenai/tulu-3-sft-personas-instruction-following" "${TULU3_IF_MAX}"
preprocess_dataset openthoughts3 "open-thoughts/OpenThoughts3-1.2M" "${OPENTHOUGHTS3_MAX}"
preprocess_dataset openr1_mot "open-r1/Mixture-of-Thoughts" "${OPENR1_MOT_MAX}" --config all
preprocess_dataset nemotron_math_v4 "nvidia/Nemotron-SFT-Math-v4" "${NEMOTRON_MATH_MAX}"
preprocess_dataset nemotron_opencode_v1 "nvidia/Nemotron-SFT-OpenCode-v1" "${NEMOTRON_OPENCODE_MAX}"
preprocess_dataset ot_agent_100k "open-thoughts/OpenThoughts-Agent-SFT-100K" "${OT_AGENT_MAX}"

if [[ "${MIX_MODE}" == "maximal" ]]; then
  preprocess_dataset fable5_traces "Glint-Research/Complete-FABLE.5-traces-2M" "${FABLE5_MAX}"
  preprocess_dataset agenttrove "open-thoughts/AgentTrove" "${AGENTTROVE_MAX}"
  preprocess_dataset nemotron_post_code "nvidia/Nemotron-Post-Training-Dataset-v1" "${NEMOTRON_POST_MAX}" --split code
  preprocess_dataset nemotron_post_math "nvidia/Nemotron-Post-Training-Dataset-v1" "${NEMOTRON_POST_MAX}" --split math
  preprocess_dataset nemotron_post_tool "nvidia/Nemotron-Post-Training-Dataset-v1" "${NEMOTRON_POST_MAX}" --split tool_calling
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  cat >"${OUT_DIR}/mix_manifest_${MIX_MODE}_${CONTEXT_LENGTH}.json" <<JSON
{
  "mix_mode": "${MIX_MODE}",
  "context_length": ${CONTEXT_LENGTH},
  "min_response_tokens": ${CHUNK},
  "min_prompt_tokens": ${MIN_PROMPT_TOKENS},
  "fixed_prompt_tokens": ${SFT_PROMPT_TOKENS:-null},
  "prompt_split": "dataset_native_sft_boundary",
  "model_path": "${MODEL_PATH}",
  "safe_core": [
    "allenai/tulu-3-sft-mixture",
    "allenai/tulu-3-sft-personas-instruction-following",
    "open-thoughts/OpenThoughts3-1.2M",
    "open-r1/Mixture-of-Thoughts",
    "nvidia/Nemotron-SFT-Math-v4",
    "nvidia/Nemotron-SFT-OpenCode-v1",
    "open-thoughts/OpenThoughts-Agent-SFT-100K"
  ],
  "maximal_additions": [
    "Glint-Research/Complete-FABLE.5-traces-2M",
    "open-thoughts/AgentTrove",
    "nvidia/Nemotron-Post-Training-Dataset-v1:code",
    "nvidia/Nemotron-Post-Training-Dataset-v1:math",
    "nvidia/Nemotron-Post-Training-Dataset-v1:tool_calling"
  ],
  "license_note": "MIX_MODE=maximal intentionally includes mixed-license and high-risk agent trace sources. Keep safe and maximal outputs separate."
}
JSON
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "S2 trajectory SFT mix dry-run complete: ${OUT_DIR}"
else
  echo "S2 trajectory SFT mix preprocessing complete: ${OUT_DIR}"
fi
