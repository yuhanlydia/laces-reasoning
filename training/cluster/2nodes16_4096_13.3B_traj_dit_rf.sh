#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/unconditional/2nodes16_4096_13.3B_traj_dit_rf.sh" "$@"
