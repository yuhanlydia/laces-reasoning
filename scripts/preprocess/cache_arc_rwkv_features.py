#!/usr/bin/env python3
"""Build compact ARC features with the frozen 30k RWKV backbone."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from models.arc_data import load_arc_split, serialize_arc_context, transform_task
from models.arc_feature_cache import (
    ArcFeatureRecord,
    deterministic_projection,
    pooled_state_features,
    run_rwkv_chunks,
    save_feature_record,
)
from models.state_hijacking_dit import _cache_layer_state
from scripts.eval.relay_utils import load_relay_model


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=REPO / "preprocessed_data/arc_agi1")
    parser.add_argument("--output_root", type=Path, default=REPO / "preprocessed_data/arc_agi1_features")
    parser.add_argument("--ckpt_dir", type=Path, required=True)
    parser.add_argument("--rwkv_path", type=Path)
    parser.add_argument("--split", choices=["training", "evaluation", "both"], default="both")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--projection_seed", type=int, default=20260909)
    parser.add_argument("--pool_size", type=int, default=4)
    parser.add_argument("--max_tasks", type=int)
    parser.add_argument("--transforms", type=int, nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _states_from_cache(cache, num_layers: int):
    states = []
    for layer_index in range(num_layers):
        state = _cache_layer_state(cache, layer_index).get("recurrent_state")
        if not isinstance(state, torch.Tensor):
            raise ValueError(f"cache layer {layer_index} has no recurrent_state tensor")
        states.append(state)
    return states


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _record_metadata(tokenizer, task, transform_id: int, query_index: int, output_path: Path, output_root: Path):
    transformed = transform_task(task, transform_id)
    context = serialize_arc_context(transformed, query_index)
    return {
        "path": str(output_path.relative_to(output_root)),
        "task_id": task.task_id,
        "transform_id": transform_id,
        "query_index": query_index,
        "characters": len(context),
        "tokens": len(tokenizer(context).input_ids),
    }


def _feature_record(
    rwkv,
    tokenizer,
    task,
    *,
    transform_id: int,
    query_index: int,
    projection: torch.Tensor,
    chunk_size: int,
    pool_size: int,
    num_layers: int,
    device: str,
):
    transformed = transform_task(task, transform_id)
    context = serialize_arc_context(transformed, query_index)
    query_offset = context.index("[Q]")
    evidence_text = context[:query_offset]
    query_text = context[query_offset:]
    if tokenizer.decode(tokenizer(context).input_ids) != context:
        raise ValueError(f"tokenizer does not round-trip ARC context for {task.task_id}")

    evidence_ids = tokenizer(evidence_text, return_tensors="pt").input_ids.to(device)
    query_ids = tokenizer(query_text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        evidence_hidden, cache = run_rwkv_chunks(
            rwkv, evidence_ids, chunk_size=chunk_size
        )
        query_hidden, cache = run_rwkv_chunks(
            rwkv, query_ids, chunk_size=chunk_size, past_key_values=cache
        )
        projection = projection.to(device=device)
        evidence = torch.matmul(evidence_hidden[0].float(), projection).half().cpu()
        query = torch.matmul(query_hidden[0].float(), projection).half().cpu()
        state_features = pooled_state_features(
            _states_from_cache(cache, num_layers), pool_size=pool_size
        ).half().cpu()
    return ArcFeatureRecord(
        task_id=task.task_id,
        transform_id=transform_id,
        query_index=query_index,
        evidence=evidence,
        query=query,
        base_state_features=state_features,
        target_grid=torch.tensor(transformed.test[query_index].output, dtype=torch.long),
    ), len(context), int(evidence_ids.numel() + query_ids.numel())


def main(argv=None):
    args = parse_args(argv)
    if args.chunk_size <= 0 or args.feature_dim <= 0 or args.pool_size <= 0:
        raise ValueError("chunk_size, feature_dim, and pool_size must be positive")
    model, rwkv, tokenizer, _, _ = load_relay_model(
        str(args.ckpt_dir), args.device,
        rwkv_path=str(args.rwkv_path) if args.rwkv_path else None,
    )
    for parameter in rwkv.parameters():
        parameter.requires_grad_(False)
    rwkv.eval()
    projection = deterministic_projection(
        int(model.hidden_size), args.feature_dim, seed=args.projection_seed
    )

    splits = ["training", "evaluation"] if args.split == "both" else [args.split]
    manifest_records = []
    for split in splits:
        tasks = load_arc_split(args.data_root / split)
        if args.max_tasks is not None:
            tasks = tasks[: args.max_tasks]
        transforms = args.transforms
        if transforms is None:
            transforms = list(range(8)) if split == "training" else [0]
        if split == "evaluation" and transforms != [0]:
            raise ValueError("evaluation cache must use transform 0 only")
        split_root = args.output_root / split
        split_root.mkdir(parents=True, exist_ok=True)
        for task_number, task in enumerate(tasks, start=1):
            for transform_id in transforms:
                for query_index in range(len(task.test)):
                    output_path = split_root / f"{task.task_id}_t{transform_id}_q{query_index}.pt"
                    if output_path.exists() and not args.overwrite:
                        manifest_records.append(
                            _record_metadata(
                                tokenizer, task, transform_id, query_index, output_path, args.output_root
                            )
                        )
                        continue
                    record, chars, tokens = _feature_record(
                        rwkv,
                        tokenizer,
                        task,
                        transform_id=transform_id,
                        query_index=query_index,
                        projection=projection,
                        chunk_size=args.chunk_size,
                        pool_size=args.pool_size,
                        num_layers=int(model.num_layers),
                        device=args.device,
                    )
                    save_feature_record(output_path, record)
                    manifest_records.append(
                        {
                            "path": str(output_path.relative_to(args.output_root)),
                            "task_id": task.task_id,
                            "transform_id": transform_id,
                            "query_index": query_index,
                            "characters": chars,
                            "tokens": tokens,
                        }
                    )
            print(f"[{split}] {task_number}/{len(tasks)} {task.task_id}", flush=True)

    checkpoint_hash = _sha256_file(args.ckpt_dir / "model.pt")
    manifest = {
        "schema_version": 1,
        "checkpoint": str(args.ckpt_dir),
        "checkpoint_sha256": checkpoint_hash,
        "rwkv_path": str(args.rwkv_path) if args.rwkv_path else None,
        "feature_dim": args.feature_dim,
        "projection_seed": args.projection_seed,
        "pool_size": args.pool_size,
        "chunk_size": args.chunk_size,
        "record_count": len(manifest_records),
        "records": manifest_records,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"indexed {len(manifest_records)} records in {args.output_root}")


if __name__ == "__main__":
    main()
