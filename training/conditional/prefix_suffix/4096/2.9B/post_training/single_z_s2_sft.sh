#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTEXT_LENGTH=4096 BACKBONE=2.9B exec "${SCRIPT_DIR}/../../../../../run_single_z_s2_sft.sh" "$@"
