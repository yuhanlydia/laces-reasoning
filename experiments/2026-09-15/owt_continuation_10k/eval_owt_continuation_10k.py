"""Throwaway paired OWT continuation probe for the native 10k LACES checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from laces_posttrain.native import load_native
from laces_posttrain.policy import PolicyConfig, ddim_sample


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--docs", type=Path, required=True)
    p.add_argument("--ckpt-dir", type=Path, required=True)
    p.add_argument("--rwkv-path", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--prefix-tokens", type=int, default=128)
    p.add_argument("--target-tokens", type=int, default=32)
    p.add_argument("--diffusion-steps", type=int, default=32)
    p.add_argument("--cfg-scale", type=float, default=1.5)
    p.add_argument("--blend", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=20260915)
    return p.parse_args()


def select_rows(tokenizer, root: Path, count: int, prefix_len: int, target_len: int, seed: int):
    paths = sorted(root.glob("*.txt"))
    random.Random(seed).shuffle(paths)
    rows = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) < prefix_len + target_len:
            continue
        prefix_ids = ids[:prefix_len]
        target_ids = ids[prefix_len:prefix_len + target_len]
        prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=False)
        target_text = tokenizer.decode(target_ids, skip_special_tokens=False)
        # Ensure the text-facing S0 API and direct token scoring see identical tokens.
        if tokenizer(prefix_text, add_special_tokens=False).input_ids != prefix_ids:
            continue
        if tokenizer(target_text, add_special_tokens=False).input_ids != target_ids:
            continue
        rows.append(dict(path=str(path), prefix_text=prefix_text, target_text=target_text,
                         prefix_ids=prefix_ids, target_ids=target_ids))
        if len(rows) == count:
            break
    if len(rows) != count:
        raise RuntimeError(f"Only found {len(rows)} token-roundtrippable long documents")
    return rows


def main():
    args = parse_args()
    started = time.time()
    native = load_native(args.ckpt_dir, "cuda", rwkv_path=args.rwkv_path,
                         expected_step=10000, expected_writer="dynlowrank", blend=args.blend)
    cfg = PolicyConfig(steps=args.diffusion_steps, eta=0.3, min_std=0.02,
                       cfg_scale=args.cfg_scale)
    try:
        rows = select_rows(native.tokenizer, args.docs, args.rows, args.prefix_tokens,
                           args.target_tokens, args.seed)
        latents = []
        for i, row in enumerate(rows):
            prefix, cond = native.encode_prompt(row["prefix_text"], seed=args.seed + i)
            assert prefix[0].tolist() == row["prefix_ids"]
            generator = torch.Generator(device=native.device).manual_seed(args.seed + 1000 + i)
            z = ddim_sample(native.s2, cond, native.horizon, cfg, generator=generator)
            latents.append(z)
            row["prefix"] = prefix
            print(f"latent {i + 1}/{len(rows)}", flush=True)

        results = []
        for i, row in enumerate(rows):
            target = torch.tensor([row["target_ids"]], device=native.device)
            teacher_z, teacher_mask = native.encode_teacher(row["target_text"], seed=args.seed + 2000 + i)
            assert int(teacher_mask.sum()) == 1
            scores = {
                "matched": float(native.score_tokens(row["prefix"], target, latents[i], raw=False)),
                "mismatched": float(native.score_tokens(row["prefix"], target,
                                                        latents[(i + 1) % len(rows)], raw=False)),
                "teacher": float(native.score_tokens(row["prefix"], target, teacher_z, raw=False)),
            }
            nll = {k: -v / args.target_tokens for k, v in scores.items()}
            item = {
                "source_file": Path(row["path"]).name,
                "prefix_sha256": hashlib.sha256(row["prefix_text"].encode()).hexdigest(),
                "gold_continuation_sha256": hashlib.sha256(row["target_text"].encode()).hexdigest(),
                "nll_per_token": nll,
                "matched_advantage_vs_mismatch": nll["mismatched"] - nll["matched"],
                "teacher_advantage_vs_matched": nll["matched"] - nll["teacher"],
            }
            if i < 2:
                _, item["matched_greedy_64"] = native.generate(row["prefix"], latents[i], 64, raw=False)
                _, item["mismatched_greedy_64"] = native.generate(
                    row["prefix"], latents[(i + 1) % len(rows)], 64, raw=False)
            results.append(item)
            print(f"scored {i + 1}/{len(rows)} matched={nll['matched']:.4f} "
                  f"mismatch={nll['mismatched']:.4f} teacher={nll['teacher']:.4f}", flush=True)

        means = {arm: sum(x["nll_per_token"][arm] for x in results) / len(results)
                 for arm in ("matched", "mismatched", "teacher")}
        deltas = [x["matched_advantage_vs_mismatch"] for x in results]
        payload = {
            "experiment": "native LACES paired OpenWebText continuation probe",
            "checkpoint_step": 10000,
            "checkpoint": str(args.ckpt_dir.resolve()),
            "dataset": "stas/openwebtext-10k",
            "dataset_note": "Deterministic evaluation slice; no train-set decontamination claim.",
            "settings": vars(args) | {"docs": str(args.docs), "ckpt_dir": str(args.ckpt_dir),
                                      "rwkv_path": str(args.rwkv_path), "output": str(args.output)},
            "mean_nll_per_token": means,
            "perplexity": {k: math.exp(min(v, 50)) for k, v in means.items()},
            "mean_matched_advantage_vs_mismatch": sum(deltas) / len(deltas),
            "matched_win_rate_vs_mismatch": sum(x > 0 for x in deltas) / len(deltas),
            "elapsed_seconds": time.time() - started,
            "audit": native.audit,
            "rows": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(json.dumps({k: payload[k] for k in ("mean_nll_per_token", "perplexity",
              "mean_matched_advantage_vs_mismatch", "matched_win_rate_vs_mismatch",
              "elapsed_seconds")}, indent=2), flush=True)
    finally:
        native.close()


if __name__ == "__main__":
    main()
