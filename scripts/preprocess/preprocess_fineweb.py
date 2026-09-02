import argparse
import json
import os
import pickle
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize FineWeb for RELAY training")
    parser.add_argument("--model", default="RWKV/RWKV7-Goose-World3-1.5B-HF")
    parser.add_argument("--cache_dir", default="./data/huggingface")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb")
    parser.add_argument("--dataset_config", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=32000000, help="0 means stream the whole split")
    parser.add_argument("--save_every", type=int, default=100000)
    parser.add_argument("--shard_size", type=int, default=100000, help="Samples per output pkl shard; 0 writes one file")
    parser.add_argument("--out_dir", default="preprocessed_data/fineweb")
    parser.add_argument("--out_file", default="fineweb_tokenized.pkl")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--pack_existing_token_dir",
        default="",
        help="Repack existing tokenized pkl shards from this directory into full max_length chunks; skips dataset streaming and tokenization.",
    )
    parser.add_argument(
        "--pack_sequences",
        action="store_true",
        help="Concatenate documents and emit only full max_length chunks with all-true attention masks.",
    )
    parser.add_argument(
        "--max_output_samples",
        type=int,
        default=0,
        help="Stop after writing this many output samples; 0 means no output-sample cap.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(args.out_dir, args.out_file)
    max_samples = None if args.max_samples <= 0 else args.max_samples

    stem, ext = os.path.splitext(args.out_file)
    ext = ext or ".pkl"
    results = []
    shard_paths = []
    total_saved = 0
    token_buffer = []
    max_output_samples = None if args.max_output_samples <= 0 else args.max_output_samples

    def flush_shard(final=False):
        nonlocal results, total_saved
        if not results:
            return
        if args.shard_size > 0:
            shard_idx = len(shard_paths)
            path = os.path.join(args.out_dir, f"{stem}_{shard_idx:06d}{ext}")
        else:
            path = out_file
        with open(path, "wb") as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        shard_paths.append({"path": os.path.basename(path), "samples": len(results)})
        total_saved += len(results)
        label = "final shard" if final else "shard"
        print(f"  Saved {label}: {len(results)} samples to {path} (total={total_saved})")
        results = []

    def append_result(input_ids, attention_mask):
        nonlocal total_saved
        if max_output_samples is not None and total_saved + len(results) >= max_output_samples:
            return False
        results.append(
            {
                "input_ids": np.asarray(input_ids, dtype=np.int32),
                "attention_mask": np.asarray(attention_mask, dtype=bool),
            }
        )
        if args.shard_size > 0 and len(results) >= args.shard_size:
            flush_shard()
        return True

    def emit_full_chunks():
        while len(token_buffer) >= args.max_length:
            chunk = token_buffer[: args.max_length]
            del token_buffer[: args.max_length]
            if not append_result(chunk, np.ones(args.max_length, dtype=bool)):
                return False
        return True

    if args.pack_existing_token_dir:
        in_dir = args.pack_existing_token_dir
        pkl_files = sorted(
            os.path.join(in_dir, name)
            for name in os.listdir(in_dir)
            if name.endswith(".pkl")
        )
        if not pkl_files:
            raise ValueError(f"No .pkl shards found in {in_dir}")
        print(f"Repacking {len(pkl_files)} existing pkl shard(s) from {in_dir}...")
        seen_samples = 0
        for pkl_path in tqdm(pkl_files, desc="Repacking shards"):
            with open(pkl_path, "rb") as f:
                shard = pickle.load(f)
            for sample in shard:
                ids = np.asarray(sample["input_ids"], dtype=np.int32)
                mask = sample.get("attention_mask")
                if mask is not None:
                    ids = ids[np.asarray(mask, dtype=bool)]
                token_buffer.extend(ids.tolist())
                seen_samples += 1
                if not emit_full_chunks():
                    break
                if max_samples is not None and seen_samples >= max_samples:
                    break
            if max_output_samples is not None and total_saved + len(results) >= max_output_samples:
                break
            if max_samples is not None and seen_samples >= max_samples:
                break
    else:
        print(f"Loading tokenizer {args.model}...")
        tok = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=True,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        )

        print(f"Streaming {args.dataset}/{args.dataset_config}...")
        ds = load_dataset(args.dataset, args.dataset_config, split=args.split, streaming=True)

        total = max_samples if max_samples is not None else None
        for i, sample in enumerate(tqdm(ds, total=total)):
            text = sample["text"]
            if args.pack_sequences:
                ids = tok.encode(text, add_special_tokens=False)
                if tok.eos_token_id is not None:
                    ids.append(tok.eos_token_id)
                token_buffer.extend(ids)
                if not emit_full_chunks():
                    break
            else:
                encoded = tok(
                    text,
                    truncation=True,
                    max_length=args.max_length,
                    padding="max_length",
                    return_tensors="np",
                )
                if not append_result(
                    encoded["input_ids"].squeeze(0),
                    encoded["attention_mask"].squeeze(0),
                ):
                    break
            if max_output_samples is not None and total_saved + len(results) >= max_output_samples:
                break
            if max_samples is not None and i + 1 >= max_samples:
                break

    flush_shard(final=True)
    manifest = {
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "model": args.model,
        "max_length": args.max_length,
        "max_samples": args.max_samples,
        "max_output_samples": args.max_output_samples,
        "pack_sequences": args.pack_sequences,
        "pack_existing_token_dir": args.pack_existing_token_dir,
        "shard_size": args.shard_size,
        "total_samples": total_saved,
        "dropped_tail_tokens": len(token_buffer) if (args.pack_sequences or args.pack_existing_token_dir) else 0,
        "shards": shard_paths,
    }
    manifest_path = os.path.join(args.out_dir, f"{stem}_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Done: {total_saved} samples -> {args.out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
