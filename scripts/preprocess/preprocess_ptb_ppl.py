#!/usr/bin/env python3
"""Preprocess PTB (ptb_text_only) test split as per-line variable-length npz files,
matching the lm1b preprocessing layout consumed by eval_state_hijack_ppl_matrix.py.
"""
import os
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

OUT_DIR = Path("preprocessed_data/ptb/test")
MODEL_NAME = "RWKV/RWKV7-Goose-World3-1.5B-HF"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    n = 0
    with open("/tmp/opencode/ptb_test.txt", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            ids = tokenizer.encode(text)
            if len(ids) < 2:
                continue
            np.savez(
                OUT_DIR / f"{n:06d}.npz",
                input_ids=np.array(ids, dtype=np.int32),
                attention_mask=np.ones(len(ids), dtype=bool),
            )
            n += 1
    print(f"saved {n} files to {OUT_DIR}")


if __name__ == "__main__":
    main()
