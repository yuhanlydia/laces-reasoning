#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTEXT_LENGTH=512 BACKBONE=2.9B exec "${SCRIPT_DIR}/../../../../../run_s2_trajectory_sft.sh" "$@"
