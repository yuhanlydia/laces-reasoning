#!/usr/bin/env python3
"""Preprocess LAMBADA for RELAY cloze evaluation.

LAMBADA: given a passage, predict the LAST WORD. Cloze accuracy.
Saves passage + target word separately so eval can score the last token.

Output: preprocessed_data/lambada/{validation,test}/{idx:06d}.npz
  - input_ids: int32 [L]   full passage including target
  - attention_mask: bool [L]
  - target_start: int      position where target word begins (in tokens)
"""
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

BASE_DIR = Path("preprocessed_data/lambada")
MODEL_NAME = os.path.join(os.environ.get("DIFFRWKV_MODEL_DIR", "./models"), "rwkv7-0.4B-world")
MAX_LEN = 256


def process_split(split_name, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Use the openai variant — standard for LM evaluation.
    ds = load_dataset("EleutherAI/lambada_openai", "en", split=split_name)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    saved = 0
    for i, ex in enumerate(ds):
        text = ex["text"].strip()
        # Last word is what the model must predict.
        last_space = text.rfind(" ")
        if last_space <= 0:
            continue
        prefix = text[:last_space + 1]   # include trailing space
        target_word = text[last_space + 1:]

        prefix_ids = tok.encode(prefix, add_special_tokens=False)
        target_ids = tok.encode(target_word, add_special_tokens=False)
        if not target_ids or len(prefix_ids) + len(target_ids) > MAX_LEN:
            continue

        full = prefix_ids + target_ids
        target_start = len(prefix_ids)
        L = len(full)
        ids = np.array(full, dtype=np.int32)
        mask = np.ones(L, dtype=bool)

        np.savez(
            out_dir / f"{i:06d}.npz",
            input_ids=ids,
            attention_mask=mask,
            target_start=np.array([target_start], dtype=np.int32),
        )
        saved += 1

    print(f"  {split_name}: {saved} examples → {out_dir}")
    return saved


if __name__ == "__main__":
    # LAMBADA has only test split in the openai variant.
    process_split("test", BASE_DIR / "test")
