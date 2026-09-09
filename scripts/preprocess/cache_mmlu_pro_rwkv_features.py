#!/usr/bin/env python3
"""Cache frozen RWKV features and base answer logits for MMLU-Pro."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import urllib.parse
import urllib.request

import pyarrow.parquet as pq
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from models.arc_feature_cache import deterministic_projection, pooled_state_features
from models.multichoice_reasoning import (
    MultipleChoiceFeatureRecord,
    format_multiple_choice_prompt,
    save_feature_record,
    stratified_three_way_split,
)
from models.state_hijacking_dit import _cache_layer_state
from scripts.eval.relay_utils import load_relay_model


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, default=REPO / "preprocessed_data/mmlu_pro_rwkv")
    parser.add_argument("--ckpt_dir", type=Path, required=True)
    parser.add_argument("--rwkv_path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--projection_seed", type=int, default=20260909)
    parser.add_argument("--split_seed", type=int, default=20260909)
    parser.add_argument("--pool_size", type=int, default=4)
    parser.add_argument("--max_examples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _states(cache, num_layers):
    values = []
    for layer in range(num_layers):
        state = _cache_layer_state(cache, layer).get("recurrent_state")
        if not isinstance(state, torch.Tensor):
            raise ValueError(f"missing recurrent state at layer {layer}")
        values.append(state)
    return values


def _token_ids(tokenizer, text, max_length):
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if not ids:
        raise ValueError("tokenizer produced an empty sequence")
    return ids[:max_length]


def _load_mmlu_pro_test(cache_root: Path, max_examples: int | None = None) -> list[dict]:
    parquet_path = cache_root / "source" / "mmlu_pro_test.parquet"
    if not parquet_path.exists():
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        url = (
            "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/"
            "refs%2Fconvert%2Fparquet/default/test/0000.parquet"
        )
        temporary = parquet_path.with_suffix(".parquet.tmp")
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(parquet_path)
    table = pq.read_table(parquet_path)
    if max_examples is not None:
        table = table.slice(0, min(max_examples, table.num_rows))
    return table.to_pylist()


def main(argv=None):
    args = parse_args(argv)
    model, rwkv, tokenizer, _, _ = load_relay_model(
        str(args.ckpt_dir), args.device, rwkv_path=str(args.rwkv_path)
    )
    for parameter in rwkv.parameters():
        parameter.requires_grad_(False)
    rwkv.eval()
    projection = deterministic_projection(
        int(model.hidden_size), args.feature_dim, seed=args.projection_seed
    ).to(args.device)

    dataset = _load_mmlu_pro_test(args.output_root, args.max_examples)
    categories = [str(row["category"]) for row in dataset]
    train, validation, test = stratified_three_way_split(categories, seed=args.split_seed)
    assignments = {index: "train" for index in train}
    assignments.update({index: "validation" for index in validation})
    assignments.update({index: "test" for index in test})
    label_token_ids = []
    for label in "ABCDEFGHIJ":
        candidate = tokenizer(" " + label, add_special_tokens=False).input_ids
        if not candidate:
            raise ValueError(f"empty answer tokenization for {label}")
        label_token_ids.append(int(candidate[0]))

    records = []
    pending = []
    for index, row in enumerate(dataset):
        output_split = assignments[index]
        output_path = args.output_root / output_split / f"{int(row['question_id']):06d}.pt"
        records.append({"index": index, "question_id": int(row["question_id"]), "split": output_split})
        if output_path.exists() and not args.overwrite:
            continue
        options = [str(value) for value in row["options"]]
        evidence_text, query_text = format_multiple_choice_prompt(
            str(row["question"]), options, category=str(row["category"])
        )
        query_ids = _token_ids(tokenizer, query_text, args.max_length)
        evidence_ids = _token_ids(tokenizer, evidence_text, args.max_length - len(query_ids))
        pending.append((index, row, output_path, options, evidence_ids, query_ids))

    pending.sort(key=lambda item: len(item[4]) + len(item[5]))
    pad_id = int(tokenizer.pad_token_id or 0)
    for batch_start in range(0, len(pending), args.batch_size):
        batch = pending[batch_start : batch_start + args.batch_size]
        lengths = [len(item[4]) + len(item[5]) for item in batch]
        width = max(lengths)
        input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long, device=args.device)
        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        offsets = []
        for batch_index, item in enumerate(batch):
            ids = item[4] + item[5]
            offset = width - len(ids)
            input_ids[batch_index, offset:] = torch.tensor(ids, device=args.device)
            attention_mask[batch_index, offset:] = True
            offsets.append(offset)
        with torch.no_grad():
            output = rwkv(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
            hidden = output.hidden_states[-1]
            state_features = pooled_state_features(
                _states(output.past_key_values, int(model.num_layers)),
                pool_size=args.pool_size,
            ).half().cpu()

        for batch_index, (index, row, output_path, options, evidence_ids, query_ids) in enumerate(batch):
            offset = offsets[batch_index]
            query_start = offset + len(evidence_ids)
            base_logits = output.logits[batch_index, -1, label_token_ids[: len(options)]].float()
            record = MultipleChoiceFeatureRecord(
                example_id=str(row["question_id"]),
                category=str(row["category"]),
                evidence=(hidden[batch_index, offset:query_start].float() @ projection).half().cpu(),
                query=(hidden[batch_index, query_start:].float() @ projection).half().cpu(),
                base_state_features=state_features[batch_index:batch_index + 1],
                base_choice_logits=base_logits.half().cpu(),
                num_choices=len(options),
                label=int(row["answer_index"]),
            )
            if not all(torch.isfinite(value).all() for value in (
                record.evidence, record.query, record.base_state_features,
                record.base_choice_logits,
            )):
                raise FloatingPointError(f"non-finite feature at MMLU-Pro row {index}")
            save_feature_record(output_path, record)
        completed = batch_start + len(batch)
        if batch_start == 0 or completed % 100 < args.batch_size:
            print(f"cached {completed}/{len(pending)} pending records", flush=True)

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": "TIGER-Lab/MMLU-Pro",
        "source_split": "test",
        "split_seed": args.split_seed,
        "counts": {name: sum(record["split"] == name for record in records) for name in ("train", "validation", "test")},
        "label_token_ids": label_token_ids,
        "feature_dim": args.feature_dim,
        "pool_size": args.pool_size,
        "records": records,
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["counts"]), flush=True)


if __name__ == "__main__":
    main()
