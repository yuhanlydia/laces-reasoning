#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTEXT_LENGTH=512 BACKBONE=13.3B BASIS=8 OBJECTIVE=flow exec "${SCRIPT_DIR}/../../../../../run_single_z.sh" "$@"
