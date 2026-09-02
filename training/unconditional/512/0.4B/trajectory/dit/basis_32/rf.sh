#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTEXT_LENGTH=512 BACKBONE=0.4B ARCH=dit BASIS=32 OBJECTIVE=rf exec "${SCRIPT_DIR}/../../../../../../run_trajectory.sh" "$@"
