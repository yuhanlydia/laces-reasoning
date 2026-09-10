#!/usr/bin/env python3
"""Default: refine latents through the pretrained LACES S0/S2/S1 interface.

Historical standalone state regression is available ONLY via --legacy_standalone.
Old checkpoints and old CLI flags must not be silently interpreted as the new method.
"""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

if __name__ == '__main__':
    if '--legacy_standalone' in sys.argv:
        sys.argv.remove('--legacy_standalone')
        print('WARNING: legacy standalone writer bypasses pretrained LACES S0/S1/S2.', file=sys.stderr)
        runpy.run_path(str(Path(__file__).with_name('train_standalone_state_reasoner.py')), run_name='__main__')
    else:
        from scripts.eval.train_laces_reasoner import main
        main()
