#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
from datasets import DownloadConfig, load_dataset
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="HuggingFace dataset name")
    source.add_argument("--input_file", help="Local .json or .jsonl file")
    p.add_argument("--config", default=None, help="HuggingFace dataset config")
    p.add_argument("--split", default="train")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_path", default="/inspire/hdd/global_user/zhangjiaquan-253108540222/models/RWKV7-Goose-World3-2.9B-HF")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--filename_prefix", default="sft")
    p.add_argument("--start_index", type=int, default=None)
    p.add_argument(
        "--min_response_tokens",
        type=int,
        default=1,
        help="Skip examples whose tokenized response is shorter than this. Use the trajectory chunk size for S2 trajectory SFT.",
    )
    p.add_argument(
        "--min_prompt_tokens",
        type=int,
        default=1,
        help="Skip examples whose tokenized prompt is shorter than this. Use the trajectory chunk size for S2 trajectory SFT.",
    )
    p.add_argument(
        "--fixed_prompt_tokens",
        type=int,
        default=None,
        help="If set, truncate every prompt to exactly this many tokens so S2 trajectory SFT batches have uniform prompt_lengths.",
    )
    p.add_argument("--prompt_field", default="prompt")
    p.add_argument("--response_field", default="response")
    p.add_argument("--instruction_field", default="instruction")
    p.add_argument("--input_field", default="input")
    p.add_argument("--output_field", default="output")
    p.add_argument("--messages_field", default="messages")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument(
        "--streaming",
        action="store_true",
        help="Use HuggingFace streaming iteration. This avoids Arrow schema casting failures on mixed-schema JSON/Parquet datasets.",
    )
    return p.parse_args()


def validate_filename_prefix(prefix: str) -> str:
    if ".." in prefix or not re.fullmatch(r"[A-Za-z0-9_.-]+", prefix):
        raise ValueError("--filename_prefix may only contain letters, numbers, '_', '-', and '.', and must not contain '..'")
    return prefix


def iter_json_file(path: Path) -> Iterable[dict[str, object]]:
    if path.suffix == ".jsonl":
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    with path.open() as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        for key in ("data", "train", "examples"):
            if isinstance(payload.get(key), list):
                yield from payload[key]
                return
        yield payload
        return
    yield from payload


def role_content(message: dict[str, object]) -> tuple[str, str]:
    role = str(message.get("role", message.get("from", "user"))).lower()
    content = str(message.get("content", message.get("value", "")))
    if role in ("human", "user"):
        role = "user"
    elif role in ("gpt", "assistant", "model"):
        role = "assistant"
    elif role == "system":
        role = "system"
    return role, content


def messages_to_prompt_response(messages: list[dict[str, object]]) -> tuple[str, str] | None:
    if not messages:
        return None
    last_assistant = None
    for idx in range(len(messages) - 1, -1, -1):
        role, content = role_content(messages[idx])
        if role == "assistant" and content.strip():
            last_assistant = idx
            break
    if last_assistant is None:
        return None

    prompt_parts = []
    for message in messages[:last_assistant]:
        role, content = role_content(message)
        if not content.strip():
            continue
        if role == "system":
            prompt_parts.append(f"System:\n{content}\n\n")
        elif role == "assistant":
            prompt_parts.append(f"Assistant:\n{content}\n\n")
        else:
            prompt_parts.append(f"User:\n{content}\n\n")
    prompt_parts.append("Assistant:\n")
    _, response = role_content(messages[last_assistant])
    return "".join(prompt_parts), response


def message_fields(args) -> list[str]:
    fields = [args.messages_field, "messages", "conversations", "conversation", "dialogue", "chat"]
    return list(dict.fromkeys(fields))


def choices_from_row(row: dict[str, object]) -> list[str] | None:
    choices = row.get("options", row.get("choices"))
    if isinstance(choices, dict):
        text = choices.get("text")
        if isinstance(text, list):
            return [str(item) for item in text if str(item).strip()]
    if isinstance(choices, list):
        return [str(item) for item in choices if str(item).strip()]
    return None


def answer_letter(row: dict[str, object], choices: list[str]) -> str | None:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for key in ("answer_index", "answer_idx", "label"):
        value = row.get(key)
        if isinstance(value, int) and 0 <= value < len(choices):
            return labels[value]
        if isinstance(value, str):
            value_text = value.strip()
            if value_text.isdigit():
                idx = int(value_text)
                if 0 <= idx < len(choices):
                    return labels[idx]
            if len(value_text) == 1 and value_text.upper() in labels[:len(choices)]:
                return value_text.upper()
    answer = row.get("answer", row.get("answerKey"))
    if answer is None:
        return None
    if isinstance(answer, int) and 0 <= answer < len(choices):
        return labels[answer]
    answer_text = str(answer).strip()
    if len(answer_text) == 1 and answer_text.upper() in labels[:len(choices)]:
        return answer_text.upper()
    labels_obj = row.get("labels")
    if isinstance(labels_obj, list):
        labels_text = [str(item).strip() for item in labels_obj]
        if answer_text in labels_text:
            return labels[labels_text.index(answer_text)]
    for idx, choice in enumerate(choices):
        if answer_text == choice.strip():
            return labels[idx]
    return None


def multiple_choice_prompt_response(row: dict[str, object]) -> tuple[str, str] | None:
    question = row.get("question")
    if question is None:
        return None
    choices = choices_from_row(row)
    if not choices:
        return None
    letter = answer_letter(row, choices)
    if letter is None:
        return None
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    option_lines = [f"({labels[idx]}) {choice}" for idx, choice in enumerate(choices)]
    prompt = "Question:\n" + str(question).strip() + "\n\nOptions:\n" + "\n".join(option_lines) + "\n\nAnswer with the option letter only.\nAnswer:"
    return prompt, " " + letter


def decoded_row_json(row: dict[str, object]) -> dict[str, object] | None:
    value = row.get("row_json")
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def row_to_prompt_response(row: dict[str, object], args, parse_row_json: bool = True) -> tuple[str, str] | None:
    if parse_row_json:
        decoded = decoded_row_json(row)
        if decoded is not None:
            parsed = row_to_prompt_response(decoded, args, parse_row_json=False)
            if parsed is not None:
                return parsed

    for field in message_fields(args):
        messages = row.get(field)
        if isinstance(messages, list) and all(isinstance(message, dict) for message in messages):
            parsed = messages_to_prompt_response(messages)
            if parsed is not None:
                return parsed

    parsed = multiple_choice_prompt_response(row)
    if parsed is not None:
        return parsed

    if args.prompt_field in row and args.response_field in row:
        prompt = str(row.get(args.prompt_field, ""))
        response = str(row.get(args.response_field, ""))
        if prompt.strip() and response.strip():
            return prompt, response

    instruction = str(row.get(args.instruction_field, ""))
    output = str(row.get(args.output_field, row.get(args.response_field, "")))
    if instruction.strip() and output.strip():
        extra_input = str(row.get(args.input_field, ""))
        if extra_input.strip():
            prompt = f"Instruction:\n{instruction}\n\nInput:\n{extra_input}\n\nResponse:\n"
        else:
            prompt = f"Instruction:\n{instruction}\n\nResponse:\n"
        return prompt, output
    return None


def encode_example(
    tokenizer,
    prompt: str,
    response: str,
    max_length: int,
    pad_id: int,
    min_response_tokens: int = 1,
    min_prompt_tokens: int = 1,
    fixed_prompt_tokens: int | None = None,
):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    if not prompt_ids or not response_ids:
        return None
    if fixed_prompt_tokens is not None:
        if fixed_prompt_tokens < min_prompt_tokens:
            return None
        min_prompt_tokens = fixed_prompt_tokens
    if len(prompt_ids) < min_prompt_tokens:
        return None

    response_budget = max_length - min_prompt_tokens
    if response_budget <= 0:
        return None
    response_ids = response_ids[:response_budget]
    if len(response_ids) < min_response_tokens:
        return None
    prompt_budget = max_length - len(response_ids)
    if prompt_budget <= 0:
        return None
    if fixed_prompt_tokens is not None:
        prompt_ids = prompt_ids[-fixed_prompt_tokens:]
    elif len(prompt_ids) > prompt_budget:
        prompt_ids = prompt_ids[-prompt_budget:]
    if len(prompt_ids) < min_prompt_tokens:
        return None

    full = prompt_ids + response_ids
    if len(full) < 2:
        return None

    input_ids = np.full(max_length, pad_id, dtype=np.int32)
    attention_mask = np.zeros(max_length, dtype=bool)
    response_mask = np.zeros(max_length, dtype=bool)
    input_ids[:len(full)] = np.asarray(full, dtype=np.int32)
    attention_mask[:len(full)] = True
    response_mask[len(prompt_ids):len(full)] = True
    return input_ids, attention_mask, response_mask, len(prompt_ids)


def load_hf_rows(args):
    load_kwargs = {"split": args.split}
    if args.config:
        load_kwargs["name"] = args.config
    if args.local_files_only:
        load_kwargs["download_config"] = DownloadConfig(local_files_only=True)
    if args.streaming:
        load_kwargs["streaming"] = True
        return load_dataset(args.dataset, **load_kwargs)
    try:
        return load_dataset(args.dataset, **load_kwargs)
    except Exception as exc:
        print(
            f"load_dataset failed for {args.dataset} with {type(exc).__name__}; "
            "retrying with streaming=True to bypass dataset schema casting.",
            flush=True,
        )
        load_kwargs["streaming"] = True
        return load_dataset(args.dataset, **load_kwargs)


def cleanup_corrupt_tail(out_dir: Path, filename_prefix: str) -> None:
    existing = sorted(out_dir.glob(f"{filename_prefix}_*_tokens.npz"))
    while existing:
        path = existing[-1]
        try:
            with np.load(path) as shard:
                for key in ("input_ids", "attention_mask", "response_mask", "prompt_lengths"):
                    _ = shard[key].shape
            return
        except Exception as exc:
            print(f"removing corrupt tail shard {path}: {type(exc).__name__}: {exc}", flush=True)
            path.unlink(missing_ok=True)
            existing.pop()


def save_npz_atomic(path: Path, **arrays) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("wb") as f:
        np.savez_compressed(f, **arrays)
    tmp_path.replace(path)


def main():
    args = parse_args()
    args.filename_prefix = validate_filename_prefix(args.filename_prefix)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cleanup_corrupt_tail(out_dir, args.filename_prefix)
    if args.dataset:
        rows = load_hf_rows(args)
    else:
        rows = iter_json_file(Path(args.input_file))

    auto_resume = args.start_index is None
    if auto_resume:
        existing = sorted(out_dir.glob(f"{args.filename_prefix}_*_tokens.npz"))
        indices = []
        for path in existing:
            stem = path.name.removeprefix(f"{args.filename_prefix}_").removesuffix("_tokens.npz")
            if stem.isdigit():
                indices.append(int(stem))
        start_index = max(indices) + 1 if indices else 0
    else:
        start_index = args.start_index
    resume_remaining = start_index if auto_resume else 0
    write_limit = None if args.max_samples is None else max(args.max_samples - start_index, 0)
    saved = 0
    skipped = 0
    resumed = 0
    for row in rows:
        if write_limit is not None and saved >= write_limit:
            break
        parsed = row_to_prompt_response(row, args)
        if parsed is None:
            skipped += 1
            continue
        encoded = encode_example(
            tokenizer,
            parsed[0],
            parsed[1],
            args.max_length,
            pad_id,
            min_response_tokens=args.min_response_tokens,
            min_prompt_tokens=args.min_prompt_tokens,
            fixed_prompt_tokens=args.fixed_prompt_tokens,
        )
        if encoded is None:
            skipped += 1
            continue
        if resume_remaining > 0:
            resume_remaining -= 1
            resumed += 1
            continue
        input_ids, attention_mask, response_mask, prompt_len = encoded
        save_npz_atomic(
            out_dir / f"{args.filename_prefix}_{start_index + saved:08d}_tokens.npz",
            input_ids=input_ids,
            attention_mask=attention_mask,
            response_mask=response_mask,
            prompt_lengths=np.asarray([prompt_len], dtype=np.int32),
        )
        saved += 1

    manifest = {
        "saved": saved,
        "skipped": skipped,
        "resumed": resumed,
        "filename_prefix": args.filename_prefix,
        "start_index": start_index,
        "max_length": args.max_length,
        "min_response_tokens": args.min_response_tokens,
        "min_prompt_tokens": args.min_prompt_tokens,
        "fixed_prompt_tokens": args.fixed_prompt_tokens,
        "source": args.dataset or args.input_file,
    }
    (out_dir / f"manifest_{args.filename_prefix}.json").write_text(json.dumps(manifest, indent=2))
    print(f"response SFT: saved={saved} skipped={skipped} -> {out_dir}")


if __name__ == "__main__":
    main()
