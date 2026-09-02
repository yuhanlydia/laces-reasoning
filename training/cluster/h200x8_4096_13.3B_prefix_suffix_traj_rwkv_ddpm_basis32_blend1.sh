#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/conditional/prefix_suffix/h200x8_4096_13.3B_prefix_suffix_traj_rwkv_ddpm_basis32_blend1.sh" "$@"
