#!/usr/bin/env python3
"""Preprocess multiple-choice benchmarks for RELAY: HellaSwag, PIQA,
ARC-Easy, ARC-Challenge, WinoGrande, OpenBookQA.

For each example we save N candidate sequences (context + each choice).
Eval scores the per-token NLL of the choice continuation, picks min NLL
as prediction. Standard zero-shot LM eval protocol.

Output: preprocessed_data/{benchmark}/{split}/{idx:06d}.npz
  - input_ids: int32 [N, L]   N candidates, padded to max len
  - attention_mask: bool [N, L]
  - choice_start: int32 [N]   token position where each choice begins
  - label: int                gold choice index

Usage:
    python scripts/preprocess/preprocess_multichoice.py --benchmark hellaswag
    python scripts/preprocess/preprocess_multichoice.py --benchmark piqa
    python scripts/preprocess/preprocess_multichoice.py --benchmark arc_easy
    python scripts/preprocess/preprocess_multichoice.py --benchmark arc_challenge
    python scripts/preprocess/preprocess_multichoice.py --benchmark winogrande
    python scripts/preprocess/preprocess_multichoice.py --benchmark openbookqa
"""
import argparse
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = os.path.join(os.environ.get("DIFFRWKV_MODEL_DIR", "./models"), "rwkv7-0.4B-world")
MAX_LEN = 384

# Each benchmark needs its own format function.
# Returns: (context_str, list_of_choice_strs, gold_label_int)


def fmt_hellaswag(ex):
    ctx = ex["activity_label"] + ": " + ex["ctx"]
    choices = ex["endings"]
    return ctx, choices, int(ex["label"])


def fmt_piqa(ex):
    ctx = "Question: " + ex["goal"] + "\nAnswer:"
    choices = [" " + ex["sol1"], " " + ex["sol2"]]
    return ctx, choices, int(ex["label"])


def fmt_arc(ex):
    ctx = "Question: " + ex["question"] + "\nAnswer:"
    choices = [" " + c for c in ex["choices"]["text"]]
    label_text = ex["answerKey"]
    labels = ex["choices"]["label"]
    gold = labels.index(label_text) if label_text in labels else 0
    return ctx, choices, gold


def fmt_winogrande(ex):
    # Replace _ with each option, score whole sentence.
    sent = ex["sentence"]
    op1, op2 = ex["option1"], ex["option2"]
    choices = [sent.replace("_", op1), sent.replace("_", op2)]
    ctx = ""
    return ctx, choices, int(ex["answer"]) - 1  # answer is "1" or "2"


def fmt_openbookqa(ex):
    ctx = "Question: " + ex["question_stem"] + "\nAnswer:"
    choices = [" " + c for c in ex["choices"]["text"]]
    label = ex["answerKey"]
    labels = ex["choices"]["label"]
    gold = labels.index(label) if label in labels else 0
    return ctx, choices, gold


CONFIGS = {
    "hellaswag":     {"hf": ("Rowan/hellaswag", None),               "split": "validation", "fmt": fmt_hellaswag},
    # Original `piqa` is script-based; use ybisk/piqa parquet mirror.
    "piqa":          {"hf": ("ybisk/piqa", None),                    "split": "validation", "fmt": fmt_piqa},
    "arc_easy":      {"hf": ("allenai/ai2_arc", "ARC-Easy"),         "split": "validation", "fmt": fmt_arc},
    "arc_challenge": {"hf": ("allenai/ai2_arc", "ARC-Challenge"),    "split": "validation", "fmt": fmt_arc},
    "winogrande":    {"hf": ("winogrande", "winogrande_xl"),         "split": "validation", "fmt": fmt_winogrande},
    "openbookqa":    {"hf": ("openbookqa", "main"),                  "split": "validation", "fmt": fmt_openbookqa},
}


def process(name, out_dir):
    cfg = CONFIGS[name]
    ds_name, ds_subset = cfg["hf"]
    if ds_subset is not None:
        ds = load_dataset(ds_name, ds_subset, split=cfg["split"], trust_remote_code=True)
    else:
        ds = load_dataset(ds_name, split=cfg["split"], trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    fmt = cfg["fmt"]

    out_dir = out_dir / name / cfg["split"]
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i, ex in enumerate(ds):
        try:
            ctx, choices, label = fmt(ex)
        except Exception as exc:
            print(f"  skip ex {i}: {exc}")
            continue

        ctx_ids = tok.encode(ctx, add_special_tokens=False) if ctx else []
        seqs, starts = [], []
        for c in choices:
            c_ids = tok.encode(c, add_special_tokens=False)
            full = ctx_ids + c_ids
            if len(full) > MAX_LEN:
                full = full[:MAX_LEN]
            seqs.append(full)
            starts.append(min(len(ctx_ids), MAX_LEN))

        L = max(len(s) for s in seqs)
        N = len(seqs)
        input_ids = np.zeros((N, L), dtype=np.int32)
        attn = np.zeros((N, L), dtype=bool)
        for j, s in enumerate(seqs):
            input_ids[j, :len(s)] = s
            attn[j, :len(s)] = True

        np.savez(
            out_dir / f"{i:06d}.npz",
            input_ids=input_ids,
            attention_mask=attn,
            choice_start=np.array(starts, dtype=np.int32),
            label=np.array([label], dtype=np.int32),
        )
        saved += 1

    print(f"  {name}/{cfg['split']}: {saved} examples → {out_dir}")
    return saved


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True, choices=list(CONFIGS.keys()))
    p.add_argument("--out_root", default="preprocessed_data")
    args = p.parse_args()
    process(args.benchmark, Path(args.out_root))
