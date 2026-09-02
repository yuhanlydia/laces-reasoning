#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BATCH_SIZE="${BATCH_SIZE:-1}" STATE_BLEND=0.5 exec "${SCRIPT_DIR}/h200x8_4096_2.9B_prefix_suffix_traj_rwkv_ddpm_basis32.sh" "$@"
