#!/usr/bin/env python3
"""Fine-tune the OWT-pretrained LACES writer with recurrent MMLU-Pro reasoning."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from models.multichoice_reasoning import format_multiple_choice_prompt
from models.owt_writer_reasoning import (
    OWTLatentTransition,
    choice_token_logits,
    inject_residual_into_cache,
    native_reasoning_query_ids,
    select_pretrained_writer_parameters,
)
from scripts.eval.relay_utils import load_relay_model


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=Path, required=True)
    parser.add_argument("--rwkv_path", type=Path, required=True)
    parser.add_argument("--feature_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--max_steps", type=int, default=500000)
    parser.add_argument("--max_hours", type=float, default=6.0)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--writer_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--eval_examples", type=int, default=256)
    parser.add_argument("--context_dim", type=int, default=128)
    parser.add_argument("--initial_residual_scale", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args(argv)


def load_examples(feature_root: Path) -> dict[str, list[dict]]:
    manifest = json.loads((feature_root / "manifest.json").read_text())
    rows = pq.read_table(feature_root / "source" / "mmlu_pro_test.parquet").to_pylist()
    splits = {name: [] for name in ("train", "validation", "test")}
    for item in manifest["records"]:
        row = dict(rows[int(item["index"])])
        row["_split"] = item["split"]
        splits[item["split"]].append(row)
    return splits


def tokenize_examples(examples, tokenizer, max_length):
    tokenized = []
    query_ids, _ = native_reasoning_query_ids(tokenizer, min_tokens=64)
    for row in examples:
        evidence, _ = format_multiple_choice_prompt(
            str(row["question"]), [str(x) for x in row["options"]],
            category=str(row["category"]),
        )
        evidence_ids = tokenizer(
            evidence, add_special_tokens=False, truncation=True, max_length=max_length
        ).input_ids
        if not evidence_ids or not query_ids:
            raise ValueError(f"empty prompt tokens for question {row['question_id']}")
        tokenized.append({
            "id": int(row["question_id"]), "category": str(row["category"]),
            "evidence_ids": evidence_ids, "query_ids": query_ids,
            "num_choices": len(row["options"]), "label": int(row["answer_index"]),
        })
    return tokenized


@torch.no_grad()
def encode_mean(relay, pooled):
    if relay.encoder_type == "variational":
        value = pooled.to(next(relay.encoder_trunk.parameters()).dtype)
        if getattr(relay, "s0_input_adapter", None) is not None:
            value = relay.s0_input_adapter(value)
        return relay.mu_head(relay.encoder_trunk(value))
    if relay.encoder_type == "mlp":
        return relay.encoder(pooled.to(next(relay.encoder.parameters()).dtype))
    value = pooled.to(relay.latent_mu.dtype)
    return ((value - relay.latent_mu) / relay.latent_sigma).to(relay.state_scale.dtype)


@torch.no_grad()
def encode_evidence(relay, example, device):
    ids = torch.tensor([example["evidence_ids"]], device=device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    output = relay.rwkv_model(
        input_ids=ids, attention_mask=mask, output_hidden_states=True,
        use_cache=True, return_dict=True,
    )
    hidden = output.hidden_states[-1].detach()
    pooled = relay._pool_hidden(hidden, mask)
    return hidden, encode_mean(relay, pooled).detach(), output.past_key_values


def score_query(relay, cache, example, label_token_ids, device):
    query = torch.tensor([example["query_ids"]], device=device)
    output = relay.rwkv_model(
        input_ids=query, past_key_values=cache, use_cache=False, return_dict=True
    )
    counts = torch.tensor([example["num_choices"]], device=device)
    return choice_token_logits(output.logits[:, -1].float(), label_token_ids, counts)


def written_logits(relay, transition, residual_logit, example, label_token_ids, depth, device):
    hidden, z0, cache = encode_evidence(relay, example, device)
    latent = transition(hidden, z0, steps=depth)[depth]
    residuals = relay.predict_states(latent)
    scale = torch.sigmoid(residual_logit)
    inject_residual_into_cache(cache, residuals, scale=scale)
    return score_query(relay, cache, example, label_token_ids, device), scale


@torch.no_grad()
def evaluate(relay, transition, residual_logit, examples, label_token_ids, depths, device):
    transition.eval(); relay.eval()
    correct = {0: 0, **{int(depth): 0 for depth in depths}}
    for example in examples:
        hidden, z0, evidence_cache = encode_evidence(relay, example, device)
        base = score_query(
            relay, copy.deepcopy(evidence_cache), example, label_token_ids, device
        )
        correct[0] += int(base.argmax(-1).item() == example["label"])
        trace = transition(hidden, z0, steps=max(depths))
        for depth in depths:
            cache = copy.deepcopy(evidence_cache)
            residuals = relay.predict_states(trace[int(depth)])
            inject_residual_into_cache(cache, residuals, scale=torch.sigmoid(residual_logit))
            logits = score_query(relay, cache, example, label_token_ids, device)
            correct[int(depth)] += int(logits.argmax(-1).item() == example["label"])
    total = max(1, len(examples))
    transition.train(); relay.eval()
    return {
        "count": len(examples), "base_accuracy": correct[0] / total,
        "depths": {str(depth): correct[int(depth)] / total for depth in depths},
        "residual_scale": float(torch.sigmoid(residual_logit).detach()),
    }


def checkpoint_payload(transition, residual_logit, writer, optimizer, step, epoch, best, args):
    return {
        "schema_version": 1, "source_checkpoint": str(args.ckpt_dir),
        "transition": transition.state_dict(),
        "residual_logit": residual_logit.detach().cpu(),
        "writer": {name: parameter.detach().cpu() for name, parameter in writer.items()},
        "optimizer": optimizer.state_dict(), "step": step, "epoch": epoch,
        "best_accuracy": best, "config": vars(args),
    }


def save_checkpoint(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary); temporary.replace(path)


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    relay, _, tokenizer, source, _ = load_relay_model(
        str(args.ckpt_dir), args.device, rwkv_path=str(args.rwkv_path)
    )
    if relay.s1_writer_type != "dynlowrank":
        raise ValueError("OWT reasoning requires a dynlowrank checkpoint writer")
    writer = select_pretrained_writer_parameters(relay)
    transition = OWTLatentTransition(
        hidden_dim=relay.hidden_size, z_dim=relay.latent_dim,
        context_dim=args.context_dim,
    ).to(device)
    initial = min(max(args.initial_residual_scale, 1e-5), 1 - 1e-5)
    residual_logit = torch.nn.Parameter(
        torch.tensor(math.log(initial / (1 - initial)), device=device)
    )
    optimizer = torch.optim.AdamW([
        {"params": transition.parameters(), "lr": args.lr},
        {"params": [residual_logit], "lr": args.lr},
        {"params": list(writer.values()), "lr": args.writer_lr},
    ], weight_decay=args.weight_decay)
    splits = load_examples(args.feature_root)
    for name in splits:
        splits[name] = tokenize_examples(splits[name], tokenizer, args.max_length)
    label_token_ids = torch.tensor([
        int(tokenizer(" " + label, add_special_tokens=False).input_ids[0])
        for label in "ABCDEFGHIJ"
    ], device=device)
    step = epoch = 0; best = 0.0
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        transition.load_state_dict(saved["transition"])
        residual_logit.data.copy_(saved["residual_logit"].to(device))
        named = dict(relay.named_parameters())
        for name, value in saved["writer"].items(): named[name].data.copy_(value.to(device))
        optimizer.load_state_dict(saved["optimizer"])
        step, epoch, best = saved["step"], saved["epoch"], saved["best_accuracy"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "history.jsonl"
    order = list(range(len(splits["train"]))); random.shuffle(order); position = 0
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic(); deadline = started + args.max_hours * 3600
    while step < args.max_steps and time.monotonic() < deadline:
        step += 1
        if position >= len(order):
            epoch += 1; random.shuffle(order); position = 0
        example = splits["train"][order[position]]; position += 1
        depth = random.choice(args.depths)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, scale = written_logits(
                relay, transition, residual_logit, example, label_token_ids,
                depth, device,
            )
            target = torch.tensor([example["label"]], device=device)
            loss = F.cross_entropy(logits, target)
        (loss / args.grad_accum).backward()
        if step == 1:
            if not any(parameter.grad is not None for parameter in writer.values()):
                raise RuntimeError("answer CE did not reach the pretrained OWT writer")
        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [*transition.parameters(), residual_logit, *writer.values()], 1.0
            )
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 20 == 0:
            event = {
                "step": step, "epoch": epoch, "depth": depth,
                "loss": float(loss.detach()),
                "correct": int(logits.argmax(-1).item() == example["label"]),
                "residual_scale": float(scale.detach()),
            }
            print(json.dumps(event), flush=True)
            with history_path.open("a") as handle: handle.write(json.dumps(event) + "\n")
        if step % args.eval_every == 0:
            sample = splits["validation"][:args.eval_examples]
            metrics = evaluate(
                relay, transition, residual_logit, sample, label_token_ids,
                tuple(args.depths), device,
            )
            score = metrics["depths"][str(max(args.depths))]
            print(json.dumps({"step": step, "validation": metrics}), flush=True)
            if score >= best:
                best = score
                save_checkpoint(args.output_dir / "best.pt", checkpoint_payload(
                    transition, residual_logit, writer, optimizer, step, epoch, best, args
                ))
        if step % args.save_every == 0:
            save_checkpoint(args.output_dir / f"step_{step:08d}.pt", checkpoint_payload(
                transition, residual_logit, writer, optimizer, step, epoch, best, args
            ))
    final = checkpoint_payload(
        transition, residual_logit, writer, optimizer, step, epoch, best, args
    )
    save_checkpoint(args.output_dir / "last.pt", final)
    validation = evaluate(
        relay, transition, residual_logit, splits["validation"], label_token_ids,
        tuple(args.depths), device,
    )
    test = evaluate(
        relay, transition, residual_logit, splits["test"], label_token_ids,
        tuple(args.depths), device,
    )
    summary = {
        "source_checkpoint_step": source.get("step"), "step": step, "epoch": epoch,
        "elapsed_hours": (time.monotonic() - started) / 3600,
        "best_validation_accuracy": best, "validation": validation, "test": test,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"complete": summary}), flush=True)


if __name__ == "__main__":
    main()
