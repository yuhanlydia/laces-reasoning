#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=pack_existing exec "${SCRIPT_DIR}/../../run_preprocess_4096.sh" "$@"
