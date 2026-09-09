#!/usr/bin/env python3
"""Train and evaluate recurrent LACES state writing on MMLU-Pro choices."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
import time

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from models.multichoice_reasoning import (
    MultipleChoiceResidualHead,
    load_feature_record,
    recurrent_multichoice_objective,
    retarget_cosine_schedule,
)
from models.recurrent_latent_reasoner import RecurrentReasoner


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--max_steps", type=int, default=200000)
    parser.add_argument("--max_hours", type=float, default=0.0)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--eval_every", type=int, default=5000)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--z_dim", type=int, default=32)
    parser.add_argument("--context_dim", type=int, default=128)
    parser.add_argument("--writer_rank", type=int, default=32)
    parser.add_argument("--writer_hidden", type=int, default=128)
    parser.add_argument("--head_hidden", type=int, default=256)
    parser.add_argument("--state_pool_size", type=int, default=4)
    parser.add_argument("--base_logit_scale", type=float, default=0.25)
    return parser.parse_args(argv)


def _inputs(record, device, base_logit_scale):
    base = torch.zeros(1, 10, device=device)
    base[0, :record.num_choices] = record.base_choice_logits.to(device).float() * base_logit_scale
    return (
        record.evidence.to(device).float(), record.query.to(device).float(),
        record.base_state_features.to(device).float(), base,
        torch.tensor([record.num_choices], device=device),
        torch.tensor([record.label], device=device),
    )


def build_models(sample, args, device):
    state_dim = int(sample.base_state_features.shape[-1])
    reasoner = RecurrentReasoner(
        hidden_dim=int(sample.evidence.shape[-1]), num_layers=32, num_heads=40,
        head_dim=64, z_dim=args.z_dim, context_dim=args.context_dim,
        writer_rank=args.writer_rank, writer_hidden=args.writer_hidden,
        state_summary_dim=state_dim, state_pool_size=args.state_pool_size,
    ).to(device)
    head = MultipleChoiceResidualHead(
        state_dim=state_dim, hidden_dim=args.head_hidden, max_choices=10
    ).to(device)
    return reasoner, head


@torch.no_grad()
def evaluate(reasoner, head, paths, device, depths, base_logit_scale):
    reasoner.eval(); head.eval()
    correct = {0: 0, **{depth: 0 for depth in depths}}
    category = {}
    elapsed = {depth: 0.0 for depth in depths}
    for path in paths:
        record = load_feature_record(path)
        evidence, query, base_state, base, counts, labels = _inputs(
            record, device, base_logit_scale
        )
        base_prediction = int(base[0, :record.num_choices].argmax())
        correct[0] += int(base_prediction == record.label)
        cat = category.setdefault(record.category, {"count": 0, "base": 0, **{str(d): 0 for d in depths}})
        cat["count"] += 1; cat["base"] += int(base_prediction == record.label)
        for depth in depths:
            if device.type == "cuda": torch.cuda.synchronize(device)
            started = time.perf_counter()
            trace = reasoner(
                evidence, query, steps=depth, base_state_features=base_state,
                materialize_states=False,
            )
            logits = head(trace.pooled_corrections[depth], base, counts)
            if device.type == "cuda": torch.cuda.synchronize(device)
            elapsed[depth] += time.perf_counter() - started
            hit = int(logits.argmax(-1).item() == record.label)
            correct[depth] += hit; cat[str(depth)] += hit
    total = max(len(paths), 1)
    result = {
        "count": len(paths), "base_accuracy": correct[0] / total,
        "depths": {
            str(depth): {
                "accuracy": correct[depth] / total,
                "mean_runtime_seconds": elapsed[depth] / total,
            } for depth in depths
        },
        "categories": category,
    }
    return result


def _checkpoint(reasoner, head, optimizer, scheduler, step, epoch, best, config):
    return {
        "schema_version": 1, "reasoner": copy.deepcopy(reasoner.state_dict()),
        "head": copy.deepcopy(head.state_dict()), "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()), "step": step, "epoch": epoch,
        "best_accuracy": best, "config": config, "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary); os.replace(temporary, path)


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_paths = sorted((args.feature_root / "train").glob("*.pt"))
    validation_paths = sorted((args.feature_root / "validation").glob("*.pt"))
    test_paths = sorted((args.feature_root / "test").glob("*.pt"))
    if not train_paths or not validation_paths or not test_paths:
        raise ValueError("feature root must contain non-empty train/validation/test splits")
    sample = load_feature_record(train_paths[0])
    reasoner, head = build_models(sample, args, device)
    parameters = [*reasoner.parameters(), *head.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.max_steps // args.grad_accum)
    )
    step = epoch = 0; best = 0.0
    if args.resume:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        reasoner.load_state_dict(payload["reasoner"]); head.load_state_dict(payload["head"])
        optimizer.load_state_dict(payload["optimizer"]); scheduler.load_state_dict(payload["scheduler"])
        step, epoch, best = int(payload["step"]), int(payload["epoch"]), float(payload["best_accuracy"])
        random.setstate(payload["python_rng"]); torch.set_rng_state(payload["torch_rng"].cpu())
        if payload.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([value.cpu() for value in payload["cuda_rng"]])
        retarget_cosine_schedule(
            scheduler, total_steps=args.max_steps, grad_accum=args.grad_accum
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = args.output_dir / "history.jsonl"
    order = train_paths[:]; random.shuffle(order); position = 0
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic(); stop_at = started + args.max_hours * 3600 if args.max_hours > 0 else None
    interrupted = False
    try:
        while step < args.max_steps:
            if stop_at is not None and time.monotonic() >= stop_at:
                break
            step += 1
            if position >= len(order):
                epoch += 1; random.shuffle(order); position = 0
            record = load_feature_record(order[position]); position += 1
            evidence, query, base_state, base, counts, labels = _inputs(
                record, device, args.base_logit_scale
            )
            depth = random.choice(args.depths)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                trace = reasoner(
                    evidence, query, steps=depth, base_state_features=base_state,
                    materialize_states=False,
                )
                loss, metrics = recurrent_multichoice_objective(
                    head, trace, base_choice_logits=base, choice_counts=counts,
                    labels=labels, depths=[depth],
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            (loss / args.grad_accum).backward()
            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); scheduler.step()
            event = {"step": step, "epoch": epoch, "depth": depth, **metrics,
                     "lr": optimizer.param_groups[0]["lr"]}
            if step == 1 or step % 20 == 0:
                print(json.dumps(event), flush=True)
                with history.open("a") as handle: handle.write(json.dumps(event) + "\n")
            if step % args.eval_every == 0:
                metrics_eval = evaluate(
                    reasoner, head, validation_paths, device, tuple(args.depths),
                    args.base_logit_scale,
                )
                score = metrics_eval["depths"][str(max(args.depths))]["accuracy"]
                print(json.dumps({"step": step, "validation": metrics_eval}), flush=True)
                reasoner.train(); head.train()
                if score >= best:
                    best = score
                    _save(args.output_dir / "best.pt", _checkpoint(
                        reasoner, head, optimizer, scheduler, step, epoch, best, vars(args)
                    ))
            if step % args.save_every == 0:
                _save(args.output_dir / f"step_{step:08d}.pt", _checkpoint(
                    reasoner, head, optimizer, scheduler, step, epoch, best, vars(args)
                ))
    except KeyboardInterrupt:
        interrupted = True
    last = _checkpoint(reasoner, head, optimizer, scheduler, step, epoch, best, vars(args))
    _save(args.output_dir / "last.pt", last)
    final_validation = evaluate(
        reasoner, head, validation_paths, device, tuple(args.depths), args.base_logit_scale
    )
    final_test = evaluate(
        reasoner, head, test_paths, device, tuple(args.depths), args.base_logit_scale
    )
    summary = {
        "step": step, "epoch": epoch, "elapsed_hours": (time.monotonic() - started) / 3600,
        "interrupted": interrupted, "best_validation_accuracy": best,
        "train_count": len(train_paths), "validation_count": len(validation_paths),
        "test_count": len(test_paths), "validation": final_validation, "test": final_test,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"complete": summary}), flush=True)


if __name__ == "__main__":
    main()
