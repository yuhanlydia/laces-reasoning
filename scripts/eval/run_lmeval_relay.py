#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lmeval_relay_adapter  # noqa: F401  (registers the "relay" model)

from lm_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    cli_evaluate()
