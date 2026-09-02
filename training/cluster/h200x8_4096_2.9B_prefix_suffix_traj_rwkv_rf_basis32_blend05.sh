#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/conditional/prefix_suffix/h200x8_4096_2.9B_prefix_suffix_traj_rwkv_rf_basis32_blend05.sh" "$@"
