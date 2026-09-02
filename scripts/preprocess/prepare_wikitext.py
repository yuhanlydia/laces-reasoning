#!/usr/bin/env python3
"""
WikiText-103 data preparation pipeline for DiffRwkv.

Downloads WikiText-103, tokenizes with RWKV tokenizer, and computes 32-d latent
embeddings using multi-GPU parallel processing.

Usage:
    # Single GPU tokenization + latent computation:
    python scripts/preprocess/prepare_wikitext.py --stage all

    # Multi-GPU latent computation only (after tokenization):
    torchrun --nproc_per_node=4 scripts/preprocess/prepare_wikitext.py --stage latents

    # Tokenize only (CPU, uses all cores):
    python scripts/preprocess/prepare_wikitext.py --stage tokenize
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------------------------------------------------------
# Stage 1: Download & split WikiText-103
# ---------------------------------------------------------------------------


def download_wikitext(output_dir: Path):
    """Download WikiText-103 via HuggingFace datasets and save as JSON."""
    from datasets import load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "validation", "test"):
        out_path = output_dir / f"wikitext103_{split}.json"
        if out_path.exists():
            print(f"[download] {out_path} already exists, skipping")
            continue

        print(f"[download] Loading wikitext-103-raw-v1 split={split} ...")
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)

        # Filter empty / whitespace-only lines and very short texts
        records = []
        for row in ds:
            text = row["text"].strip()
            if len(text) < 20:
                continue
            records.append({"text": text})

        with open(out_path, "w") as f:
            json.dump(records, f)
        print(f"[download] Saved {len(records):,} records -> {out_path}")

    return output_dir


# ---------------------------------------------------------------------------
# Stage 2: Tokenize with RWKV tokenizer (CPU parallel)
# ---------------------------------------------------------------------------


def _tokenize_chunk(args):
    """Worker for multiprocessing tokenization."""
    chunk, tokenizer_name, max_length, cache_dir, trust_remote_code = args
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, cache_dir=cache_dir, trust_remote_code=trust_remote_code
        )
    except (OSError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )

    results = []
    for item in chunk:
        enc = tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="np",
        )
        results.append(
            {
                "input_ids": enc["input_ids"][0].astype(np.int32),
                "attention_mask": enc["attention_mask"][0].astype(np.bool_),
            }
        )
    return results


def tokenize_split(
    json_path: Path,
    output_dir: Path,
    tokenizer_name: str = "BlinkDL/rwkv7-g1",
    max_length: int = 512,
    num_workers: int = 16,
    chunk_size: int = 500,
    cache_dir: str | None = None,
    trust_remote_code: bool = False,
):
    """Tokenize a single JSON split into per-sample .npz files."""
    import multiprocessing as mp

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path) as f:
        data = json.load(f)
    print(f"[tokenize] Loaded {len(data):,} records from {json_path}")

    # Chunk data for parallel workers
    chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]
    worker_args = [
        (chunk, tokenizer_name, max_length, cache_dir, trust_remote_code)
        for chunk in chunks
    ]

    all_results = []
    t0 = time.time()
    with mp.Pool(processes=num_workers, maxtasksperchild=50) as pool:
        for batch_results in pool.imap(_tokenize_chunk, worker_args, chunksize=1):
            all_results.extend(batch_results)

    # Save individual .npz files (matches DirectFileDataset expectations)
    for idx, item in enumerate(all_results):
        npz_path = output_dir / f"{idx:08d}_tokens.npz"
        np.savez_compressed(
            npz_path,
            input_ids=item["input_ids"],
            attention_mask=item["attention_mask"],
        )

    elapsed = time.time() - t0
    print(
        f"[tokenize] Saved {len(all_results):,} .npz files -> {output_dir} "
        f"({elapsed:.1f}s, {len(all_results) / elapsed:.0f} samples/s)"
    )
    return output_dir


# ---------------------------------------------------------------------------
# Stage 3: Compute 32-d latent embeddings (multi-GPU)
# ---------------------------------------------------------------------------


def compute_latents_for_split(
    token_dir: Path,
    output_dir: Path,
    model_name: str = "Qwen/Qwen2.5-0.5B",
    latent_dim: int = 32,
    batch_size: int = 64,
    max_length: int = 512,
    cache_dir: str | None = None,
):
    """
    Compute sentence-level latent embeddings using a pretrained LM.

    Supports DDP: when launched with torchrun, each rank processes its shard.
    Latents are mean-pooled hidden states projected to `latent_dim` via PCA
    (fitted on rank 0, broadcast to all ranks).
    """
    # DDP setup
    distributed = all(k in os.environ for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        device = torch.device("cuda", local_rank)
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Gather all token files
    token_files = sorted(list(token_dir.glob("*.npz")))
    if not token_files:
        raise FileNotFoundError(f"No .npz files in {token_dir}")

    # Shard across ranks
    shard_files = token_files[rank::world_size]
    if rank == 0:
        print(
            f"[latents] {len(token_files):,} total files, {world_size} GPUs, ~{len(shard_files):,} per GPU"
        )

    # Load encoder model
    from transformers import AutoModel, AutoTokenizer

    encoder = (
        AutoModel.from_pretrained(
            model_name, cache_dir=cache_dir, torch_dtype=torch.float16
        )
        .to(device)
        .eval()
    )

    hidden_size = encoder.config.hidden_size

    # Random projection matrix (deterministic seed so all ranks agree)
    rng = np.random.RandomState(42)
    proj_matrix = torch.tensor(
        rng.randn(hidden_size, latent_dim).astype(np.float32) / np.sqrt(hidden_size),
        device=device,
        dtype=torch.float16,
    )

    # Process in batches
    t0 = time.time()
    processed = 0

    for batch_start in range(0, len(shard_files), batch_size):
        batch_files = shard_files[batch_start : batch_start + batch_size]

        # Load token data
        all_ids = []
        all_masks = []
        for f in batch_files:
            data = np.load(f)
            all_ids.append(torch.from_numpy(data["input_ids"].astype(np.int64)))
            all_masks.append(
                torch.from_numpy(data["attention_mask"].astype(np.float32))
            )

        input_ids = torch.stack(all_ids).to(device)
        attention_mask = torch.stack(all_masks).to(device)

        # Truncate to max_length
        input_ids = input_ids[:, :max_length]
        attention_mask = attention_mask[:, :max_length]

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
            # Mean pool over non-padding tokens
            hidden = outputs.last_hidden_state  # [B, L, H]
            mask_expanded = attention_mask.unsqueeze(-1).half()
            pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(
                dim=1
            ).clamp(min=1)

            # Project to latent_dim
            latents = pooled @ proj_matrix  # [B, latent_dim]

        # Save individual .npy files
        latents_np = latents.float().cpu().numpy()
        for i, f in enumerate(batch_files):
            stem = f.stem.replace("_tokens", "")
            out_path = output_dir / f"{stem}.npy"
            np.save(out_path, latents_np[i])

        processed += len(batch_files)
        if rank == 0 and (batch_start // batch_size) % 10 == 0:
            elapsed = time.time() - t0
            print(
                f"[latents] rank 0: {processed:,}/{len(shard_files):,} "
                f"({elapsed:.0f}s, {processed / elapsed:.0f} files/s)"
            )

    if distributed:
        torch.distributed.barrier()

    if rank == 0:
        total_time = time.time() - t0
        print(f"[latents] Done. {len(token_files):,} files in {total_time:.1f}s")

    # Cleanup
    del encoder
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Prepare WikiText-103 for DiffRwkv")
    parser.add_argument(
        "--stage",
        choices=["download", "tokenize", "latents", "all"],
        default="all",
        help="Which stage(s) to run",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="preprocessed_data/wikitext103",
        help="Base output directory",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="BlinkDL/rwkv7-g1",
        help="Tokenizer name or path",
    )
    parser.add_argument(
        "--encoder_model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Encoder model for latent computation",
    )
    parser.add_argument(
        "--latent_dim", type=int, default=32, help="Latent embedding dimension"
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="Max sequence length"
    )
    parser.add_argument(
        "--tokenize_workers", type=int, default=16, help="CPU workers for tokenization"
    )
    parser.add_argument(
        "--latent_batch_size",
        type=int,
        default=64,
        help="Batch size per GPU for latent computation",
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None, help="HuggingFace cache directory"
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code for tokenizer/model",
    )

    args = parser.parse_args()

    base = Path(args.base_dir)
    raw_dir = base / "raw"
    splits = {"train": "train", "validation": "val", "test": "test"}

    if args.stage in ("download", "all"):
        download_wikitext(raw_dir)

    if args.stage in ("tokenize", "all"):
        for hf_split, local_split in splits.items():
            json_path = raw_dir / f"wikitext103_{hf_split}.json"
            if not json_path.exists():
                print(f"[tokenize] {json_path} not found, skipping")
                continue
            token_out = base / "tokens" / local_split
            tokenize_split(
                json_path=json_path,
                output_dir=token_out,
                tokenizer_name=args.tokenizer,
                max_length=args.max_length,
                num_workers=args.tokenize_workers,
                cache_dir=args.cache_dir,
                trust_remote_code=args.trust_remote_code,
            )

    if args.stage in ("latents", "all"):
        for hf_split, local_split in splits.items():
            token_dir = base / "tokens" / local_split
            if not token_dir.exists():
                print(f"[latents] {token_dir} not found, skipping")
                continue
            latent_out = base / "latents" / local_split
            compute_latents_for_split(
                token_dir=token_dir,
                output_dir=latent_out,
                model_name=args.encoder_model,
                latent_dim=args.latent_dim,
                batch_size=args.latent_batch_size,
                max_length=args.max_length,
                cache_dir=args.cache_dir,
            )

    print("[prepare_wikitext] All done.")


if __name__ == "__main__":
    main()
