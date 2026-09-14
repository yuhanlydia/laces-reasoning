"""Prepare leakage-guarded GSM8K and Hendrycks-MATH bundles for block GRPO."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata
from typing import Iterable

from .math_verify import VERIFIER_VERSION, canonical_numeric, extract_final_answer

SCHEMA = "laces_math_reasoning_v1"
MATH_CONFIGS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


def _norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def problem_hash(task: str, problem: str) -> str:
    payload = json.dumps([task.lower(), _norm(problem)], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_no}: expected JSON object")
                rows.append(row)
    return rows


def canonicalize_record(row: dict, task: str, *, source: str, revision: str, source_split: str = "train") -> dict:
    task = task.lower()
    if task == "gsm8k":
        problem = row.get("question")
        rationale = row.get("answer")
        category = "gsm8k"
    elif task == "math":
        problem = row.get("problem")
        rationale = row.get("solution")
        category = str(row.get("type", row.get("category", "unknown")))
    else:
        raise ValueError("task must be gsm8k or math")
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("Missing problem")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("Missing reference rationale/answer")
    extracted = extract_final_answer(rationale)
    if extracted is None:
        raise ValueError("Could not extract an explicit verified answer from reference rationale")
    numeric = canonical_numeric(extracted)
    answer = numeric if numeric is not None else extracted.strip().strip("$")
    ph = problem_hash(task, problem)
    return {
        "problem_id": str(row.get("problem_id", f"{task}-{ph[:16]}")),
        "problem": problem.strip(),
        "answer": answer,
        "reference_rationale": rationale.strip(),
        "task": task,
        "category": category,
        "source": source,
        "source_revision": revision,
        "source_split": source_split,
        "problem_hash": ph,
    }


def _ensure_unique(rows: list[dict], label: str) -> set[str]:
    keys = [row["problem_hash"] for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f"Duplicate normalized problems in {label}")
    return set(keys)


def _partition_train(rows: list[dict], *, seed: int, dev_fraction: float) -> tuple[list[dict], list[dict]]:
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be in (0,1)")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    train: list[dict] = []
    dev: list[dict] = []
    rng = random.Random(seed)
    for category in sorted(grouped):
        group = sorted(grouped[category], key=lambda x: x["problem_hash"])
        rng.shuffle(group)
        if len(group) < 2:
            raise ValueError(f"Need at least two training problems in category {category!r}")
        n_dev = max(1, min(len(group) - 1, int(round(len(group) * dev_fraction))))
        dev.extend(group[:n_dev])
        train.extend(group[n_dev:])
    return train, dev


def prepare_bundle(train_rows: list[dict], test_rows: list[dict], output: Path | str, *, task: str,
                   source: str, revision: str, seed: int = 42, dev_fraction: float = .05) -> dict:
    output = Path(output)
    if (output / "manifest.json").exists():
        raise ValueError("Output bundle already exists; use a fresh directory")
    canonical_train = [canonicalize_record(r, task, source=source, revision=revision, source_split="train") for r in train_rows]
    canonical_test = [canonicalize_record(r, task, source=source, revision=revision, source_split="test") for r in test_rows]
    train_keys = _ensure_unique(canonical_train, "source train")
    test_keys = _ensure_unique(canonical_test, "source test")
    if train_keys & test_keys:
        raise ValueError("Problem overlap between official/source train and sealed test")
    train, dev = _partition_train(canonical_train, seed=seed, dev_fraction=dev_fraction)
    sets = {"train": _ensure_unique(train, "train"), "dev": _ensure_unique(dev, "dev"), "test": test_keys}
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if sets[a] & sets[b]:
            raise ValueError(f"Problem overlap between {a} and {b}")
    output.mkdir(parents=True, exist_ok=True)
    parts = {"train": train, "dev": dev, "test": canonical_test}
    for name, rows in parts.items():
        _write_jsonl(output / f"{name}.jsonl", rows)
    manifest = {
        "schema": SCHEMA,
        "task": task.lower(),
        "dataset": source,
        "source_revision": revision,
        "seed": seed,
        "dev_fraction": dev_fraction,
        "counts": {name: len(rows) for name, rows in parts.items()},
        "files": {name: {"path": f"{name}.jsonl", "sha256": digest(output / f"{name}.jsonl")} for name in parts},
        "verifier_version": VERIFIER_VERSION,
        "test_policy": "sealed test is excluded from training, reward, development selection, and resume contracts",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _load_manifest(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Wrong math bundle schema")
    for split in ("train", "dev", "test"):
        entry = manifest["files"][split]
        path = directory / entry["path"]
        if digest(path) != entry["sha256"]:
            raise ValueError(f"{split} file hash changed")
    return manifest


def read_training_bundle(directory: Path | str):
    directory = Path(directory)
    manifest = _load_manifest(directory)
    train, dev = _read_jsonl(directory / "train.jsonl"), _read_jsonl(directory / "dev.jsonl")
    if _ensure_unique(train, "train") & _ensure_unique(dev, "dev"):
        raise ValueError("train/dev overlap")
    if any(r.get("source_split") == "test" for r in train + dev):
        raise ValueError("sealed test row entered training bundle")
    return train, dev, manifest


def read_evaluation_bundle(directory: Path | str, split: str, *, acknowledge_test: bool = False):
    if split not in {"dev", "test"}:
        raise ValueError("split must be dev or test")
    if split == "test" and not acknowledge_test:
        raise ValueError("Must explicitly acknowledge final sealed test evaluation")
    directory = Path(directory)
    manifest = _load_manifest(directory)
    return _read_jsonl(directory / f"{split}.jsonl"), manifest


def _resolve_revision(dataset: str, revision: str | None) -> str:
    from huggingface_hub import HfApi
    info = HfApi().dataset_info(dataset, revision=revision)
    if not info.sha:
        raise RuntimeError(f"Could not resolve immutable revision for {dataset}")
    return str(info.sha)


def _load_hf(task: str, dataset: str, revision: str) -> tuple[list[dict], list[dict]]:
    from datasets import load_dataset
    if task == "gsm8k":
        ds = load_dataset(dataset, "main", revision=revision)
        return list(ds["train"]), list(ds["test"])
    train: list[dict] = []
    test: list[dict] = []
    for config in MATH_CONFIGS:
        ds = load_dataset(dataset, config, revision=revision)
        for split, target in (("train", train), ("test", test)):
            for row in ds[split]:
                rec = dict(row)
                rec.setdefault("type", config)
                target.append(rec)
    return train, test


def _read_local(path: str | None) -> list[dict] | None:
    if path is None:
        return None
    return _read_jsonl(Path(path))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["gsm8k", "math"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dataset")
    p.add_argument("--revision")
    p.add_argument("--train-jsonl")
    p.add_argument("--test-jsonl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dev-fraction", type=float, default=.05)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if bool(args.train_jsonl) != bool(args.test_jsonl):
        raise ValueError("Provide both --train-jsonl and --test-jsonl, or neither")
    dataset = args.dataset or ("openai/gsm8k" if args.task == "gsm8k" else "EleutherAI/hendrycks_math")
    local_train, local_test = _read_local(args.train_jsonl), _read_local(args.test_jsonl)
    if local_train is not None:
        revision = args.revision or "local-files"
        train, test = local_train, local_test
    else:
        revision = _resolve_revision(dataset, args.revision)
        train, test = _load_hf(args.task, dataset, revision)
    manifest = prepare_bundle(train, test, args.output, task=args.task, source=dataset, revision=revision,
                              seed=args.seed, dev_fraction=args.dev_fraction)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
