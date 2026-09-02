#!/usr/bin/env python3
"""
Unified dataset preparation for DiffRwkv.

Downloads any HuggingFace dataset, tokenizes (per-row or sliding window),
and computes 32-d latent embeddings — producing .npz + .npy file pairs
compatible with DirectFileDataset.

Usage:
    # All stages, per-row tokenization:
    python scripts/preprocess/prepare_dataset.py \
      --dataset HuggingFaceFW/fineweb-edu \
      --output_dir preprocessed_data/fineweb_edu \
      --max_samples 1000

    # Sliding window tokenization (2.5x more chunks):
    python scripts/preprocess/prepare_dataset.py \
      --dataset HuggingFaceFW/fineweb-edu \
      --output_dir preprocessed_data/fineweb_edu \
      --mode sliding \
      --seq_len 512 --stride 256 \
      --max_samples 50000

    # Tokenize only (skip latent computation):
    python scripts/preprocess/prepare_dataset.py \
      --dataset wikitext --config wikitext-103-raw-v1 \
      --output_dir preprocessed_data/wikitext103 \
      --stages download tokenize

    # Latent computation only (after tokenization):
    torchrun --nproc_per_node=4 scripts/preprocess/prepare_dataset.py \
      --dataset HuggingFaceFW/fineweb-edu \
      --output_dir preprocessed_data/fineweb_edu \
      --stages latents
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

HF_CACHE = os.environ.get("HF_HOME", "./data/huggingface")
RWKV7_1_5B_PATH = os.path.join(HF_CACHE, "models--RWKV--RWKV7-Goose-World3-1.5B-HF/snapshots/7e046eaf58c0cfec2da891f86220d4037cbacd44")

os.environ["HF_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE


# ---------------------------------------------------------------------------
# Stage 1: Download dataset from HuggingFace → JSON
# ---------------------------------------------------------------------------


def download_dataset(
    dataset_name: str,
    config_name: str | None,
    split: str,
    output_dir: Path,
    max_samples: int | None = None,
    text_column: str = "text",
    min_text_length: int = 20,
):
    """Download a HF dataset and save as JSON list of {text, latent_path}."""
    from datasets import load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "train_data.json"

    if json_path.exists():
        print(f"[download] {json_path} already exists, skipping")
        return json_path

    print(f"[download] Loading {dataset_name} (config={config_name}, split={split})...")
    load_kwargs = {"split": split}
    if config_name:
        load_kwargs["name"] = config_name

    ds = load_dataset(dataset_name, **load_kwargs)

    records = []
    skipped = 0
    for row in ds:
        text = row.get(text_column, "").strip()
        if len(text) < min_text_length:
            skipped += 1
            continue
        records.append({"text": text})
        if max_samples and len(records) >= max_samples:
            break

    with open(json_path, "w") as f:
        json.dump(records, f)

    print(
        f"[download] Saved {len(records):,} records -> {json_path} ({skipped:,} skipped)"
    )
    return json_path


# ---------------------------------------------------------------------------
# Stage 2a: Per-row tokenization → .npz files
# ---------------------------------------------------------------------------


def _tokenize_chunk(args):
    """Worker for multiprocessing tokenization (per-row mode)."""
    chunk, tokenizer_name, max_length, cache_dir, trust_remote_code = args
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
    except (OSError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            use_fast=False,
            local_files_only=True,
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


def tokenize_per_row(
    json_path: Path,
    output_dir: Path,
    tokenizer_name: str = "BlinkDL/rwkv7-g1",
    max_length: int = 512,
    num_workers: int = 16,
    chunk_size: int = 500,
    cache_dir: str | None = None,
    trust_remote_code: bool = False,
):
    """Tokenize each JSON row into a separate .npz file."""
    import multiprocessing as mp

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path) as f:
        data = json.load(f)
    print(f"[tokenize] Loaded {len(data):,} records from {json_path}")

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

    for idx, item in enumerate(all_results):
        npz_path = output_dir / f"chunk_{idx:06d}_tokens.npz"
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
    return len(all_results)


# ---------------------------------------------------------------------------
# Stage 2b: Sliding window tokenization → .npz files
# ---------------------------------------------------------------------------


def tokenize_sliding_window(
    json_path: Path,
    output_dir: Path,
    tokenizer_name: str = "BlinkDL/rwkv7-g1",
    seq_len: int = 512,
    stride: int = 256,
    cache_dir: str | None = None,
    trust_remote_code: bool = False,
):
    """Concatenate all text, tokenize once, then chunk with sliding window."""
    from transformers import AutoTokenizer

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path) as f:
        data = json.load(f)
    print(f"[tokenize] Loaded {len(data):,} records from {json_path}")

    # Concatenate all text
    full_text = "\n\n".join(
        item["text"] for item in data if item.get("text", "").strip()
    )
    print(f"[tokenize] Concatenated text: {len(full_text):,} chars")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
    except (OSError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            use_fast=False,
            local_files_only=True,
        )

    all_ids = tokenizer.encode(full_text)
    print(f"[tokenize] Total tokens: {len(all_ids):,}")

    # Sliding window
    chunks = []
    for start in range(0, len(all_ids) - seq_len + 1, stride):
        chunk = all_ids[start : start + seq_len]
        if len(chunk) == seq_len:
            chunks.append(chunk)

    print(f"[tokenize] Chunks (stride={stride}): {len(chunks):,}")

    for i, chunk in enumerate(chunks):
        ids = np.array(chunk, dtype=np.int32)
        mask = np.ones(seq_len, dtype=bool)
        np.savez_compressed(
            output_dir / f"chunk_{i:06d}_tokens.npz",
            input_ids=ids,
            attention_mask=mask,
        )

    print(f"[tokenize] Saved {len(chunks):,} .npz files -> {output_dir}")
    return len(chunks)


# ---------------------------------------------------------------------------
# Stage 3: Compute 32-d latent embeddings (multi-GPU compatible)
# ---------------------------------------------------------------------------


def compute_latents(
    token_dir: Path,
    output_dir: Path,
    model_name: str = "Qwen/Qwen2.5-1.5B",
    latent_dim: int = 32,
    batch_size: int = 64,
    max_length: int = 512,
    cache_dir: str | None = None,
):
    """
    Compute latent embeddings via mean-pooled hidden states + random projection.

    Supports DDP: when launched with torchrun, each rank processes its shard.
    """
    import torch

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

    token_files = sorted(list(token_dir.glob("*.npz")))
    if not token_files:
        raise FileNotFoundError(f"No .npz files in {token_dir}")

    shard_files = token_files[rank::world_size]
    if rank == 0:
        print(
            f"[latents] {len(token_files):,} total files, {world_size} GPUs, "
            f"~{len(shard_files):,} per GPU"
        )

    from transformers import AutoModel

    encoder = (
        AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            local_files_only=True,
        )
        .to(device)
        .eval()
    )

    hidden_size = encoder.config.hidden_size

    # Random projection matrix (deterministic seed)
    rng = np.random.RandomState(42)
    proj_matrix = torch.tensor(
        rng.randn(hidden_size, latent_dim).astype(np.float32) / np.sqrt(hidden_size),
        device=device,
        dtype=torch.float32,
    )

    t0 = time.time()
    processed = 0

    for batch_start in range(0, len(shard_files), batch_size):
        batch_files = shard_files[batch_start : batch_start + batch_size]

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

        input_ids = input_ids[:, :max_length]
        attention_mask = attention_mask[:, :max_length]

        with torch.no_grad():
            outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1)
            pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(
                dim=1
            ).clamp(min=1)
            latents = pooled @ proj_matrix

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

    del encoder
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Unified dataset preparation for DiffRwkv"
    )

    # Dataset
    parser.add_argument("--dataset", type=str, required=True, help="HF dataset name")
    parser.add_argument(
        "--config", type=str, default=None, help="HF dataset config name"
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument(
        "--text_column", type=str, default="text", help="Column name for text"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Max samples to download"
    )
    parser.add_argument(
        "--min_text_length", type=int, default=20, help="Min chars per sample"
    )

    # Output
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Base output directory"
    )

    # Processing mode
    parser.add_argument(
        "--mode",
        choices=["per_row", "sliding"],
        default="per_row",
        help="Tokenization mode: per_row (one chunk per JSON row) or sliding (concatenate + stride)",
    )
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    parser.add_argument("--stride", type=int, default=256, help="Sliding window stride")

    # Tokenizer
    parser.add_argument(
        "--tokenizer", type=str, default=RWKV7_1_5B_PATH, help="Tokenizer local path"
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="Trust remote code",
    )
    parser.add_argument(
        "--cache_dir", type=str, default=HF_CACHE, help="HF cache directory"
    )
    parser.add_argument(
        "--tokenize_workers", type=int, default=32, help="CPU workers for tokenization"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=500, help="Samples per tokenize chunk"
    )

    # Latents
    parser.add_argument(
        "--encoder_model",
        type=str,
        default=RWKV7_1_5B_PATH,
        help="Encoder model local path for latents",
    )
    parser.add_argument("--latent_dim", type=int, default=32, help="Latent dimension")
    parser.add_argument(
        "--latent_batch_size",
        type=int,
        default=64,
        help="Batch size per GPU for latents",
    )

    # Stages
    parser.add_argument(
        "--stages",
        type=str,
        default="all",
        help="Comma-separated stages: download,tokenize,latents (default: all)",
    )

    args = parser.parse_args()

    base = Path(args.output_dir)
    stages = (
        [s.strip() for s in args.stages.split(",")]
        if args.stages != "all"
        else ["download", "tokenize", "latents"]
    )
    if "all" in stages:
        stages = ["download", "tokenize", "latents"]

    json_path = base / "train_data.json"
    token_dir = base / "tokens" / "train"
    latent_dir = base / "latents" / "train"

    # Stage 1: Download
    if "download" in stages:
        json_path = download_dataset(
            dataset_name=args.dataset,
            config_name=args.config,
            split=args.split,
            output_dir=base,
            max_samples=args.max_samples,
            text_column=args.text_column,
            min_text_length=args.min_text_length,
        )

    # Stage 2: Tokenize
    if "tokenize" in stages:
        if args.mode == "sliding":
            n_chunks = tokenize_sliding_window(
                json_path=json_path,
                output_dir=token_dir,
                tokenizer_name=args.tokenizer,
                seq_len=args.seq_len,
                stride=args.stride,
                cache_dir=args.cache_dir,
                trust_remote_code=args.trust_remote_code,
            )
        else:
            n_chunks = tokenize_per_row(
                json_path=json_path,
                output_dir=token_dir,
                tokenizer_name=args.tokenizer,
                max_length=args.seq_len,
                num_workers=args.tokenize_workers,
                chunk_size=args.chunk_size,
                cache_dir=args.cache_dir,
                trust_remote_code=args.trust_remote_code,
            )
        print(f"[tokenize] Total chunks: {n_chunks:,}")

    # Stage 3: Latents
    if "latents" in stages:
        compute_latents(
            token_dir=token_dir,
            output_dir=latent_dir,
            model_name=args.encoder_model,
            latent_dim=args.latent_dim,
            batch_size=args.latent_batch_size,
            max_length=args.seq_len,
            cache_dir=args.cache_dir,
        )

    print(f"\n[prepare_dataset] Done. Output dirs:")
    print(f"  Tokens:  {token_dir}")
    print(f"  Latents: {latent_dir}")
    print(f"\nTo train:")
    print(f"  python train_state_hijacking_dit.py --config-name rwkv_relay_2.9B_state_hijack_dit_vae32 \\")
    print(f"    data.token_dir={token_dir} data.latent_dir={latent_dir}")


if __name__ == "__main__":
    main()
