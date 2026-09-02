#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTEXT_LENGTH=4096 BACKBONE=2.9B ARCH=birwkv BASIS=32 OBJECTIVE=flow exec "${SCRIPT_DIR}/../../../../../../run_trajectory.sh" "$@"
