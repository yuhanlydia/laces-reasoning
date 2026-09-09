#!/usr/bin/env python3
"""Train the recurrent LACES ARC adapter from frozen RWKV feature shards."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Iterable

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from models.arc_feature_cache import ArcFeatureRecord, load_feature_record
from models.arc_grid_adapter import ArcGridDecoder, decode_grid, recurrent_arc_objective
from models.arc_metrics import evaluate_arc_predictions
from models.recurrent_latent_reasoner import RecurrentReasoner, recurrent_state_summary


def split_training_task_ids(
    task_ids: Iterable[str], *, dev_count: int = 40, seed: int = 20260909,
    evaluation_ids: set[str] | None = None,
) -> tuple[set[str], set[str]]:
    unique = sorted(set(task_ids))
    if evaluation_ids and set(unique) & set(evaluation_ids):
        raise ValueError("official evaluation IDs must never enter the training split")
    if not 0 < dev_count < len(unique):
        raise ValueError("dev_count must leave at least one training task")
    random.Random(seed).shuffle(unique)
    return set(unique[dev_count:]), set(unique[:dev_count])


def _rng_state() -> dict:
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def make_checkpoint(
    *, reasoner, decoder, optimizer, scheduler, scaler, step: int, epoch: int,
    best_metric: float, config: dict,
) -> dict:
    return {
        "schema_version": 1,
        "reasoner": copy.deepcopy(reasoner.state_dict()),
        "decoder": copy.deepcopy(decoder.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "scaler": copy.deepcopy(scaler.state_dict()),
        "step": int(step),
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "config": config,
        "rng": _rng_state(),
    }


def restore_checkpoint(checkpoint, *, reasoner, decoder, optimizer, scheduler, scaler) -> dict:
    reasoner.load_state_dict(checkpoint["reasoner"])
    decoder.load_state_dict(checkpoint["decoder"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    random.setstate(checkpoint["rng"]["python"])
    torch.set_rng_state(checkpoint["rng"]["torch"])
    if "cuda" in checkpoint["rng"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["rng"]["cuda"])
    return {
        "step": int(checkpoint["step"]),
        "epoch": int(checkpoint["epoch"]),
        "best_metric": float(checkpoint["best_metric"]),
        "config": checkpoint["config"],
    }


def save_checkpoint(path: Path, checkpoint: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def _task_id(path: Path) -> str:
    return path.stem.split("_t", 1)[0]


def feature_paths(root: Path, task_ids: set[str], *, original_only: bool = False) -> list[Path]:
    paths = []
    for path in sorted((root / "training").glob("*.pt")):
        if _task_id(path) in task_ids and (not original_only or "_t0_" in path.name):
            paths.append(path)
    return paths


def _to_device(record: ArcFeatureRecord, device: torch.device):
    return (
        record.evidence.to(device=device, dtype=torch.float32),
        record.query.to(device=device, dtype=torch.float32),
        record.base_state_features.to(device=device, dtype=torch.float32),
        [record.target_grid.to(device=device)],
    )


def _state_at_depth(trace, base_state: torch.Tensor, depth: int, pool_size: int) -> torch.Tensor:
    correction = recurrent_state_summary(
        trace.cumulative_states[depth], pool_size=pool_size
    ).to(dtype=base_state.dtype)
    return base_state + correction


@torch.no_grad()
def evaluate_paths(reasoner, decoder, paths: list[Path], device, depths=(1, 2, 4, 8)):
    reasoner.eval()
    decoder.eval()
    predictions = {depth: {} for depth in depths}
    gold: dict[str, list[torch.Tensor]] = {}
    elapsed = 0.0
    for path in paths:
        record = load_feature_record(path)
        evidence, query, base, targets = _to_device(record, device)
        started = time.perf_counter()
        trace = reasoner(evidence, query, steps=max(depths), base_state_features=base)
        elapsed += time.perf_counter() - started
        task_gold = gold.setdefault(record.task_id, [])
        while len(task_gold) <= record.query_index:
            task_gold.append(torch.empty(0))
        task_gold[record.query_index] = record.target_grid.cpu()
        for depth in depths:
            state = _state_at_depth(trace, base, depth, reasoner.state_pool_size)
            output = decoder(trace.latents[depth], query.unsqueeze(0) if query.ndim == 2 else query, state)
            task_predictions = predictions[depth].setdefault(record.task_id, [])
            while len(task_predictions) <= record.query_index:
                task_predictions.append([])
            task_predictions[record.query_index] = [decode_grid(output)[0]]
    metrics = {depth: evaluate_arc_predictions(gold, value) for depth, value in predictions.items()}
    for value in metrics.values():
        value["mean_runtime_seconds"] = elapsed / max(len(paths), 1)
    return metrics


def build_models(sample: ArcFeatureRecord, args, device):
    state_dim = int(sample.base_state_features.shape[-1])
    reasoner = RecurrentReasoner(
        hidden_dim=int(sample.evidence.shape[-1]), num_layers=args.num_layers,
        num_heads=args.num_heads, head_dim=args.head_dim, z_dim=args.z_dim,
        context_dim=args.context_dim, writer_rank=args.writer_rank,
        writer_hidden=args.writer_hidden, state_summary_dim=state_dim,
        state_pool_size=args.state_pool_size,
    ).to(device)
    decoder = ArcGridDecoder(
        z_dim=args.z_dim, query_dim=int(sample.query.shape[-1]), state_dim=state_dim,
        model_dim=args.decoder_dim,
    ).to(device)
    return reasoner, decoder


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--dev_count", type=int, default=40)
    parser.add_argument("--overfit_tasks", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--gate_exact", type=float, default=0.95)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=40)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--state_pool_size", type=int, default=4)
    parser.add_argument("--z_dim", type=int, default=32)
    parser.add_argument("--context_dim", type=int, default=128)
    parser.add_argument("--writer_rank", type=int, default=32)
    parser.add_argument("--writer_hidden", type=int, default=128)
    parser.add_argument("--decoder_dim", type=int, default=256)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    all_paths = sorted((args.feature_root / "training").glob("*.pt"))
    if not all_paths:
        raise ValueError(f"no feature shards under {args.feature_root / 'training'}")
    all_ids = sorted({_task_id(path) for path in all_paths})
    if args.overfit_tasks:
        selected = set(all_ids[: args.overfit_tasks])
        train_ids, dev_ids = selected, selected
    else:
        train_ids, dev_ids = split_training_task_ids(
            all_ids, dev_count=args.dev_count, seed=args.seed
        )
    train_paths = feature_paths(args.feature_root, train_ids)
    dev_paths = feature_paths(args.feature_root, dev_ids, original_only=True)
    sample = load_feature_record(train_paths[0])
    reasoner, decoder = build_models(sample, args, device)
    parameters = [*reasoner.parameters(), *decoder.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    start_step = 0
    epoch = 0
    best_metric = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        state = restore_checkpoint(
            checkpoint, reasoner=reasoner, decoder=decoder, optimizer=optimizer,
            scheduler=scheduler, scaler=scaler,
        )
        start_step, epoch, best_metric = state["step"], state["epoch"], state["best_metric"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "history.jsonl"
    optimizer.zero_grad(set_to_none=True)
    order_rng = random.Random(args.seed + epoch)
    order = train_paths.copy()
    order_rng.shuffle(order)
    position = 0
    reasoner.train()
    decoder.train()
    final_step = start_step
    for step in range(start_step + 1, args.max_steps + 1):
        final_step = step
        if position >= len(order):
            epoch += 1
            order_rng.shuffle(order)
            position = 0
        record = load_feature_record(order[position])
        position += 1
        evidence, query, base, targets = _to_device(record, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            trace = reasoner(evidence, query, steps=max(args.depths), base_state_features=base)
            loss, train_metrics = recurrent_arc_objective(
                decoder, trace, query, base, targets, depths=args.depths
            )
            scaled_loss = loss / args.grad_accum
        if not bool(torch.isfinite(loss)):
            emergency = make_checkpoint(
                reasoner=reasoner, decoder=decoder, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler, step=step, epoch=epoch, best_metric=best_metric, config=vars(args),
            )
            save_checkpoint(args.output_dir / f"emergency_nonfinite_step_{step:08d}.pt", emergency)
            raise FloatingPointError(f"non-finite ARC loss at step {step}")
        scaler.scale(scaled_loss).backward()
        if step % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        else:
            grad_norm = torch.tensor(float("nan"))
        event = {"step": step, "epoch": epoch, **train_metrics, "lr": optimizer.param_groups[0]["lr"]}
        if step == 1 or step % 10 == 0:
            print(json.dumps(event), flush=True)
            with history_path.open("a") as handle:
                handle.write(json.dumps(event) + "\n")
        if step % args.eval_every == 0 or step == args.max_steps:
            metrics = evaluate_paths(reasoner, decoder, dev_paths, device, tuple(args.depths))
            score = float(metrics[max(args.depths)]["pair_exact"])
            print(json.dumps({"step": step, "validation": metrics}), flush=True)
            reasoner.train(); decoder.train()
            if score >= best_metric:
                best_metric = score
                save_checkpoint(
                    args.output_dir / "best.pt",
                    make_checkpoint(reasoner=reasoner, decoder=decoder, optimizer=optimizer,
                                    scheduler=scheduler, scaler=scaler, step=step, epoch=epoch,
                                    best_metric=best_metric, config=vars(args)),
                )
            if args.overfit_tasks and score >= args.gate_exact:
                (args.output_dir / "gate_passed.json").write_text(
                    json.dumps({"step": step, "metrics": metrics}, indent=2) + "\n"
                )
                print(f"overfit gate passed at step {step}: exact={score:.3f}", flush=True)
                break
        if step % args.save_every == 0:
            save_checkpoint(
                args.output_dir / f"step_{step:08d}.pt",
                make_checkpoint(reasoner=reasoner, decoder=decoder, optimizer=optimizer,
                                scheduler=scheduler, scaler=scaler, step=step, epoch=epoch,
                                best_metric=best_metric, config=vars(args)),
            )
    save_checkpoint(
        args.output_dir / "last.pt",
        make_checkpoint(reasoner=reasoner, decoder=decoder, optimizer=optimizer,
                        scheduler=scheduler, scaler=scaler, step=final_step, epoch=epoch,
                        best_metric=best_metric, config=vars(args)),
    )
    peak_gib = (
        torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    )
    summary = {
        "step": final_step, "epoch": epoch, "best_pair_exact": best_metric,
        "peak_gpu_memory_gib": peak_gib, "train_records": len(train_paths),
        "dev_records": len(dev_paths), "train_tasks": len(train_ids), "dev_tasks": len(dev_ids),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    print(json.dumps({"complete": summary}), flush=True)


if __name__ == "__main__":
    main()
