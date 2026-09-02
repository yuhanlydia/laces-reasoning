#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/conditional/prefix_suffix/launch_4096_prefix_suffix_trajectory_stage.sh" "$@"
