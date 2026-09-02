#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True, choices=("mmlu", "mmlu_pro", "race", "siqa", "commonsenseqa", "story_cloze", "squad", "xsum", "wmt14_de_en", "lm1b"))
    p.add_argument("--split", default=None)
    p.add_argument("--out_root", default="preprocessed_data")
    p.add_argument("--model_path", default="/inspire/hdd/global_user/zhangjiaquan-253108540222/models/RWKV7-Goose-World3-2.9B-HF")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--max_samples", type=int, default=None)
    return p.parse_args()


def tokenizer_from(path):
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)


def encode(tok, text, max_len):
    return tok.encode(text, add_special_tokens=False)[:max_len]


def save_choice(out_dir, idx, tok, context, choices, label, max_len):
    ctx_ids = encode(tok, context, max_len)
    seqs = []
    starts = []
    for choice in choices:
        choice_ids = encode(tok, choice, max_len)
        full = (ctx_ids + choice_ids)[:max_len]
        seqs.append(full)
        starts.append(min(len(ctx_ids), max_len - 1))
    length = max(len(s) for s in seqs)
    input_ids = np.zeros((len(seqs), length), dtype=np.int32)
    attention_mask = np.zeros((len(seqs), length), dtype=bool)
    for i, seq in enumerate(seqs):
        input_ids[i, :len(seq)] = seq
        attention_mask[i, :len(seq)] = True
    np.savez_compressed(out_dir / f"{idx:06d}.npz", input_ids=input_ids, attention_mask=attention_mask, choice_start=np.asarray(starts, dtype=np.int32), label=np.asarray([label], dtype=np.int32))


def save_target(out_dir, idx, tok, prompt, target, max_len, extra=None):
    prompt_ids = encode(tok, prompt, max_len)
    remaining = max(1, max_len - len(prompt_ids))
    target_ids = encode(tok, target, remaining)
    full = (prompt_ids + target_ids)[:max_len]
    if len(full) < 2:
        return False
    input_ids = np.asarray(full, dtype=np.int32)
    attention_mask = np.ones_like(input_ids, dtype=bool)
    payload = {"input_ids": input_ids, "attention_mask": attention_mask, "target_start": np.asarray([min(len(prompt_ids), len(full) - 1)], dtype=np.int32), "target_text": np.asarray([target])}
    if extra:
        payload.update(extra)
    np.savez_compressed(out_dir / f"{idx:06d}.npz", **payload)
    return True


def iter_limited(ds, max_samples):
    for i, ex in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break
        yield i, ex


def preprocess_mmlu(tok, out_root, split, max_len, max_samples):
    ds = load_dataset("cais/mmlu", "all", split=split)
    out_dir = out_root / "mmlu" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, ex in iter_limited(ds, max_samples):
        choices = [" " + str(c) for c in ex["choices"]]
        context = "Question: " + ex["question"] + "\nAnswer:"
        save_choice(out_dir, i, tok, context, choices, int(ex["answer"]), max_len)
        saved += 1
    return out_dir, saved


def preprocess_mmlu_pro(tok, out_root, split, max_len, max_samples):
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
    out_dir = out_root / "mmlu_pro" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, ex in iter_limited(ds, max_samples):
        choices = [" " + str(c) for c in ex["options"]]
        context = "Question: " + ex["question"] + "\nAnswer:"
        label = int(ex["answer_index"])
        save_choice(out_dir, i, tok, context, choices, label, max_len)
        saved += 1
    return out_dir, saved


def preprocess_race(tok, out_root, split, max_len, max_samples):
    ds = load_dataset("race", "all", split=split, trust_remote_code=True)
    out_dir = out_root / "race" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, ex in iter_limited(ds, max_samples):
        context = "Article: " + ex["article"] + "\nQuestion: " + ex["question"] + "\nAnswer:"
        choices = [" " + str(c) for c in ex["options"]]
        label = "ABCD".index(ex["answer"])
        save_choice(out_dir, i, tok, context, choices, label, max_len)
        saved += 1
    return out_dir, saved


def preprocess_siqa(tok, out_root, split, max_len, max_samples):
    ds = load_dataset("social_i_qa", split=split, trust_remote_code=True)
    out_dir = out_root / "siqa" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, ex in iter_limited(ds, max_samples):
        context = "Context: " + ex["context"] + "\nQuestion: " + ex["question"] + "\nAnswer:"
        choices = [" " + ex["answerA"], " " + ex["answerB"], " " + ex["answerC"]]
        save_choice(out_dir, i, tok, context, choices, int(ex["label"]) - 1, max_len)
        saved += 1
    return out_dir, saved


def preprocess_commonsenseqa(tok, out_root, split, max_len, max_samples):
    ds = load_dataset("commonsense_qa", split=split, trust_remote_code=True)
    out_dir = out_root / "commonsenseqa" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, ex in iter_limited(ds, max_samples):
        context = "Question: " + ex["question"] + "\nAnswer:"
        labels = list(ex["choices"]["label"])
        choices = [" " + str(c) for c in ex["choices"]["text"]]
        label = labels.index(ex["answerKey"])
        save_choice(out_dir, i, tok, context, choices, label, max_len)
        saved += 1
    return out_dir, saved


def preprocess_story_cloze(tok, out_root, split, max_len, max_samples):
    tried = []
    for args in (("story_cloze", "2016"), ("MoE-UNC/story_cloze", None), ("LSDSem/story_cloze", "2016"), ("tau/story_cloze", None), ("TimoImhof/Story-Cloze-Test", None)):
        try:
            ds = load_dataset(args[0], args[1], split=split, trust_remote_code=True) if args[1] else load_dataset(args[0], split=split, trust_remote_code=True)
            break
        except Exception as exc:
            tried.append(repr(exc))
    else:
        raise RuntimeError("Could not load Story Cloze dataset: " + " | ".join(tried))
    out_dir = out_root / "story_cloze" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, ex in iter_limited(ds, max_samples):
        keys = set(ex.keys())
        if {"sentence1", "sentence2", "sentence3", "sentence4", "sentence5", "sentence_quiz1", "sentence_quiz2"}.issubset(keys):
            context = " ".join(str(ex[f"sentence{k}"]) for k in range(1, 5))
            choices = [" " + str(ex["sentence_quiz1"]), " " + str(ex["sentence_quiz2"])]
            label = int(ex.get("answer_right_ending", ex.get("label", 1))) - 1
        elif {"input_sentence_1", "input_sentence_2", "input_sentence_3", "input_sentence_4", "sentence_quiz1", "sentence_quiz2"}.issubset(keys):
            context = " ".join(str(ex[f"input_sentence_{k}"]) for k in range(1, 5))
            choices = [" " + str(ex["sentence_quiz1"]), " " + str(ex["sentence_quiz2"])]
            label = int(ex.get("answer_right_ending", ex.get("label", 1))) - 1
        else:
            context = str(ex.get("input_sentence_1", ex.get("context", "")))
            choices = [" " + str(ex.get("sentence_quiz1", ex.get("ending0", ""))), " " + str(ex.get("sentence_quiz2", ex.get("ending1", "")))]
            label = int(ex.get("label", 0))
        save_choice(out_dir, i, tok, context, choices, label, max_len)
        saved += 1
    return out_dir, saved


def preprocess_squad(tok, out_root, split, max_len, max_samples):
    ds = load_dataset("squad", split=split)
    out_dir = out_root / "squad" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, ex in iter_limited(ds, max_samples):
        answers = ex["answers"]["text"]
        if not answers:
            continue
        prompt = "Context: " + ex["context"] + "\nQuestion: " + ex["question"] + "\nAnswer:"
        if save_target(out_dir, i, tok, prompt, " " + answers[0], max_len, {"all_answers": np.asarray(answers)}):
            saved += 1
    return out_dir, saved


def preprocess_xsum(tok, out_root, split, max_len, max_samples):
    ds = load_dataset("EdinburghNLP/xsum", split=split, trust_remote_code=True)
    out_dir = out_root / "xsum" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, ex in iter_limited(ds, max_samples):
        prompt = "Summarize:\n" + ex["document"] + "\nSummary:"
        if save_target(out_dir, i, tok, prompt, " " + ex["summary"], max_len):
            saved += 1
    return out_dir, saved


def preprocess_wmt14(tok, out_root, split, max_len, max_samples):
    ds = load_dataset("wmt14", "de-en", split=split, trust_remote_code=True)
    out_dir = out_root / "wmt14_de_en" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, ex in iter_limited(ds, max_samples):
        tr = ex["translation"]
        prompt = "Translate German to English:\nGerman: " + tr["de"] + "\nEnglish:"
        if save_target(out_dir, i, tok, prompt, " " + tr["en"], max_len):
            saved += 1
    return out_dir, saved


def preprocess_wmt14_de_en(tok, out_root, split, max_len, max_samples):
    return preprocess_wmt14(tok, out_root, split, max_len, max_samples)


def preprocess_lm1b(tok, out_root, split, max_len, max_samples):
    ds = load_dataset("lm1b", split=split, trust_remote_code=True)
    out_dir = out_root / "lm1b" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, ex in iter_limited(ds, max_samples):
        text = ex.get("text", ex.get("sentence", ""))
        ids = np.asarray(encode(tok, text, max_len), dtype=np.int32)
        if len(ids) < 2:
            continue
        np.savez_compressed(out_dir / f"{i:06d}.npz", input_ids=ids, attention_mask=np.ones_like(ids, dtype=bool))
        saved += 1
    return out_dir, saved


def main():
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tok = tokenizer_from(args.model_path)
    out_root = Path(args.out_root)
    defaults = {
        "mmlu": "validation",
        "mmlu_pro": "test",
        "race": "validation",
        "siqa": "validation",
        "commonsenseqa": "validation",
        "story_cloze": "validation",
        "squad": "validation",
        "xsum": "validation",
        "wmt14_de_en": "validation",
        "lm1b": "test",
    }
    split = args.split or defaults[args.benchmark]
    fn = globals()[f"preprocess_{args.benchmark}"]
    out_dir, saved = fn(tok, out_root, split, args.max_len, args.max_samples)
    print(f"{args.benchmark}/{split}: {saved} examples -> {out_dir}")


if __name__ == "__main__":
    main()
