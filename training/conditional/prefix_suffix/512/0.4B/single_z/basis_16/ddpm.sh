#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTEXT_LENGTH=512 BACKBONE=0.4B BASIS=16 OBJECTIVE=ddpm exec "${SCRIPT_DIR}/../../../../../../run_prefix_suffix_single_z.sh" "$@"
