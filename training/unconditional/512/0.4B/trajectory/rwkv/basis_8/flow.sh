#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTEXT_LENGTH=512 BACKBONE=0.4B ARCH=rwkv BASIS=8 OBJECTIVE=flow exec "${SCRIPT_DIR}/../../../../../../run_trajectory.sh" "$@"
