"""Train the original LACES S2 with causal latent-block GRPO on GSM8K or MATH."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time

import torch

from .block_policy import block_grpo_update, potential_rewards, returns_to_go
from .block_runtime import generate_blocks, score_answer_from_cache, score_continuation_after_blocks
from .math_verify import extract_final_answer, verify_answer
from .native import load_native
from .policy import PolicyConfig, ddim_sample, rollout
from .prepare_math import read_evaluation_bundle, read_training_bundle

SCHEMA = "laces_math_block_grpo_v1"
PROMPT_VERSION = "math-block-reasoning-v1"
SELECTION_POLICY = "development checkpoints use fixed max_blocks only; block budgets are reported, not test-selected"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--output", default="results/math_block_grpo")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--rwkv-path")
    p.add_argument("--expected-step", type=int, default=30000)
    p.add_argument("--expected-writer", choices=["dynlowrank", "fixed"], default="dynlowrank")
    p.add_argument("--mode", choices=["preflight", "train", "eval"], default="preflight")
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--acknowledge-test", action="store_true")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--diffusion-steps", type=int, default=32)
    p.add_argument("--eta", type=float, default=.3)
    p.add_argument("--min-std", type=float, default=.02)
    p.add_argument("--cfg-scale", type=float, default=2.)
    p.add_argument("--blend", type=float, default=.7)
    p.add_argument("--lr", type=float, default=3e-7)
    p.add_argument("--weight-decay", type=float, default=0.)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--inner-epochs", type=int, default=1)
    p.add_argument("--advantage-mode", choices=["center", "normalized"], default="center")
    p.add_argument("--kl-coef", type=float, default=.05)
    p.add_argument("--clip-range", type=float, default=.2)
    p.add_argument("--max-grad-norm", type=float, default=1.)
    p.add_argument("--max-log-ratio", type=float, default=20.)
    p.add_argument("--max-blocks", type=int, default=16)
    p.add_argument("--tokens-per-block", type=int, default=32)
    p.add_argument("--exact-weight", type=float, default=1.)
    p.add_argument("--format-weight", type=float, default=.1)
    p.add_argument("--block-budgets", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--eval-samples", type=int, default=1)
    p.add_argument("--train-limit", type=int)
    p.add_argument("--dev-limit", type=int)
    p.add_argument("--test-limit", type=int)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume")
    p.add_argument("--s2-checkpoint")
    return p.parse_args(argv)


def format_problem(row: dict) -> str:
    return (
        "Solve the problem step by step. Use the internal plan across multiple reasoning blocks. "
        "End with one explicit line of the form 'Final answer: <answer>'.\n\n"
        f"Problem: {row['problem']}\nSolution:"
    )


def _answer_suffix(row: dict) -> str:
    return f"\nFinal answer: {row['answer']}"


def _seed(base: int, text: str) -> int:
    h = hashlib.sha256(f"{base}:{text}".encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**31 - 1)


def evaluation_seed(base: int, problem_hash: str, _budget: int, sample_idx: int) -> int:
    """Matched seed shared by arms and depth budgets for one latent trajectory."""
    return _seed(base, f"{problem_hash}:sample={sample_idx}")


def _generator(device, seed: int):
    try:
        return torch.Generator(device=device).manual_seed(seed)
    except Exception:
        return torch.Generator().manual_seed(seed)


def _json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def _contract(args, native, manifest) -> dict:
    return {
        "schema": SCHEMA,
        "prompt_version": PROMPT_VERSION,
        "parent": native.audit,
        "data_schema": manifest["schema"],
        "data_task": manifest["task"],
        "data_revision": manifest["source_revision"],
        "data_files": manifest["files"],
        "policy": {
            "diffusion_steps": args.diffusion_steps,
            "eta": args.eta,
            "min_std": args.min_std,
            "cfg_scale": args.cfg_scale,
            "blend": args.blend,
            "group_size": args.group_size,
            "inner_epochs": args.inner_epochs,
            "advantage_mode": args.advantage_mode,
            "kl_coef": args.kl_coef,
            "clip_range": args.clip_range,
            "max_grad_norm": args.max_grad_norm,
            "max_log_ratio": args.max_log_ratio,
            "max_blocks": args.max_blocks,
            "tokens_per_block": args.tokens_per_block,
            "exact_weight": args.exact_weight,
            "format_weight": args.format_weight,
        },
        "seed": args.seed,
    }


def _save(path: Path, native, optimizer, step: int, best_dev: float, contract: dict, rng: random.Random) -> None:
    payload = {
        "schema": SCHEMA,
        "s2": {k: v.detach().cpu() for k, v in native.s2.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "best_dev": best_dev,
        "contract": contract,
        "python_rng": rng.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _load(path: str | Path, native, *, contract: dict | None = None):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA:
        raise ValueError("Checkpoint schema mismatch")
    if contract is not None and payload.get("contract") != contract:
        raise ValueError("resume contract mismatch")
    native.s2.load_state_dict(payload["s2"], strict=True)
    return payload


def _candidate(native, prefix, cond, trajectory, row, args):
    z = trajectory.final.to(native.device)
    answer_ids = native._ids(_answer_suffix(row))
    baseline = float(native.score_tokens(prefix, answer_ids, z, raw=True))

    def boundary_potential(cache, pending, _tokens, _block_index):
        return float(score_answer_from_cache(native, cache, pending, answer_ids))

    generation = generate_blocks(native, prefix, z, tokens_per_block=args.tokens_per_block,
                                 max_blocks=args.max_blocks, eos_id=None, raw=False,
                                 potential_fn=boundary_potential)
    active_count = sum(bool(x) for x in generation.active_block_mask)
    block_potentials = list(generation.potentials)
    if len(block_potentials) != active_count:
        block_potentials = [float(score_continuation_after_blocks(
            native, prefix, z, generation.token_ids, generation.block_end_offsets, answer_ids, upto_block=h))
            for h in range(active_count)]
    potentials = [baseline, *block_potentials]
    while len(potentials) < args.max_blocks + 1:
        potentials.append(potentials[-1])
    exact = float(verify_answer(generation.text, row["answer"], row["task"]))
    format_ok = float(extract_final_answer(generation.text) is not None)
    return generation, potentials, exact, format_ok


def _group_credit(native, prefix, cond, group, row, args):
    candidates = [_candidate(native, prefix, cond, trajectory, row, args) for trajectory in group]
    phi = torch.tensor([x[1] for x in candidates], dtype=torch.float32)
    exact = torch.tensor([x[2] for x in candidates], dtype=torch.float32)
    format_ok = torch.tensor([x[3] for x in candidates], dtype=torch.float32)
    active = torch.tensor([x[0].active_block_mask for x in candidates], dtype=torch.bool)
    rewards = potential_rewards(phi, exact, format_ok, active,
                                exact_weight=args.exact_weight, format_weight=args.format_weight)
    returns = returns_to_go(rewards, active)
    total = rewards.sum(1)
    return candidates, rewards, returns, active, total


def _sample_plan(model, cond, native, cfg, seed: int):
    return ddim_sample(model, cond, native.horizon, cfg, _generator(native.device, seed)).to(native.device)


def _evaluate_arm(native, rows, model, args, cfg, *, raw: bool, arm: str):
    budgets = sorted(set(b for b in args.block_budgets if 1 <= b <= args.max_blocks))
    if args.max_blocks not in budgets:
        budgets.append(args.max_blocks)
    stats = {b: dict(n=0, correct=0, formatted=0, tokens=0) for b in budgets}
    original = {k: v.detach().clone() for k, v in native.s2.state_dict().items()}
    if not raw:
        native.s2.load_state_dict(model.state_dict(), strict=True)
    try:
        for row in rows:
            prompt = format_problem(row)
            prefix, cond = native.encode_prompt(prompt, seed=_seed(args.seed, row["problem_hash"] + ":evalcond"))
            for sample_idx in range(args.eval_samples):
                seed = evaluation_seed(args.seed, row["problem_hash"], args.max_blocks, sample_idx)
                z = (torch.zeros(1, native.horizon, native.audit["latent_dim"], device=native.device)
                     if raw else _sample_plan(model, cond, native, cfg, seed))
                for b in budgets:
                    gen = generate_blocks(native, prefix, z, tokens_per_block=args.tokens_per_block,
                                          max_blocks=b, eos_id=None, raw=raw)
                    st = stats[b]
                    st["n"] += 1
                    st["correct"] += int(verify_answer(gen.text, row["answer"], row["task"]))
                    st["formatted"] += int(extract_final_answer(gen.text) is not None)
                    st["tokens"] += len(gen.token_ids)
    finally:
        native.s2.load_state_dict(original, strict=True)
    return {
        str(b): {
            "n": st["n"],
            "accuracy": st["correct"] / max(1, st["n"]),
            "format_rate": st["formatted"] / max(1, st["n"]),
            "avg_generated_tokens": st["tokens"] / max(1, st["n"]),
        }
        for b, st in stats.items()
    }


def evaluate(native, rows, args, cfg, parent):
    current = copy.deepcopy(native.s2).requires_grad_(False)
    parent = copy.deepcopy(parent).requires_grad_(False)
    return {
        "metrics": {
            "raw_rwkv": _evaluate_arm(native, rows, parent, args, cfg, raw=True, arm="raw"),
            "parent_laces": _evaluate_arm(native, rows, parent, args, cfg, raw=False, arm="parent"),
            "current": _evaluate_arm(native, rows, current, args, cfg, raw=False, arm="current"),
        },
        "selection_policy": SELECTION_POLICY,
        "eval_samples": args.eval_samples,
    }


def _frozen_gradient_guard(native):
    model = getattr(native, "model", None)
    named = getattr(model, "named_parameters", None)
    if named is None:
        return
    bad = [name for name, p in named() if not name.startswith("trajectory_dit.") and getattr(p, "grad", None) is not None]
    if bad:
        raise RuntimeError(f"Frozen parent module received gradients: {bad[:3]}")


def run(args, *, runtime=None):
    if args.group_size < 2 or args.steps < 1 or args.eval_samples < 1:
        raise ValueError("group_size>=2, steps>=1 and eval_samples>=1 required")
    if args.mode == "eval":
        rows, manifest = read_evaluation_bundle(args.data, args.split, acknowledge_test=args.acknowledge_test)
        limit = args.test_limit if args.split == "test" else args.dev_limit
        if limit:
            rows = rows[:limit]
        train = dev = None
    else:
        train, dev, manifest = read_training_bundle(args.data)
        if args.train_limit:
            train = train[:args.train_limit]
        if args.dev_limit:
            dev = dev[:args.dev_limit]
    out = Path(args.output)
    if args.mode == "train" and not args.resume and any((out / x).exists() for x in ("latest.pt", "metrics.jsonl", "summary.json")):
        raise ValueError("Output already contains a run; use --resume or a new directory")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    owned = runtime is None
    native = runtime or load_native(args.ckpt_dir, args.device, rwkv_path=args.rwkv_path,
                                    expected_step=args.expected_step, expected_writer=args.expected_writer,
                                    blend=args.blend)
    if args.max_blocks > native.horizon or args.tokens_per_block > native.chunk:
        raise ValueError("Requested block budget exceeds native 16x32 LACES trajectory")
    native.enable_s2_training()
    cfg = PolicyConfig(args.diffusion_steps, args.eta, args.min_std, args.cfg_scale)
    parent = copy.deepcopy(native.s2).requires_grad_(False)
    parent.load_state_dict(native.parent_s2_state, strict=True)
    contract = _contract(args, native, manifest)
    out.mkdir(parents=True, exist_ok=True)
    _json(out / "run_config.json", {"args": vars(args), "parent": native.audit, "data": manifest,
                                     "contract": contract, "selection_policy": SELECTION_POLICY})
    loaded = None
    if args.resume:
        loaded = _load(args.resume, native, contract=contract)
    if args.s2_checkpoint:
        loaded = _load(args.s2_checkpoint, native, contract=None)
    if args.mode == "eval":
        result = evaluate(native, rows, args, cfg, parent)
        result.update(split=args.split, data_task=manifest["task"], selection_policy=SELECTION_POLICY)
        _json(out / f"evaluation_{args.split}.json", result)
        if owned:
            native.close()
        return result

    cache = {}
    def prepared(row):
        key = row["problem_hash"]
        if key not in cache:
            cache[key] = native.encode_prompt(format_problem(row), seed=_seed(args.seed, key))
        prefix, cond = cache[key]
        return prefix.to(native.device), cond.to(native.device)

    if args.mode == "preflight":
        row = train[0]
        prefix, cond = prepared(row)
        group = [rollout(native.s2, cond, native.horizon, cfg, _generator(native.device, args.seed + i))
                 for i in range(args.group_size)]
        candidates, rewards, returns, active, total = _group_credit(native, prefix, cond, group, row, args)
        probe = copy.deepcopy(native.s2)
        reference = copy.deepcopy(parent)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr)
        report = block_grpo_update(probe, reference, optimizer, cond, group, returns, active, cfg,
                                   advantage_mode=args.advantage_mode, clip_range=args.clip_range,
                                   kl_coef=args.kl_coef, inner_epochs=args.inner_epochs,
                                   max_grad_norm=args.max_grad_norm, max_log_ratio=args.max_log_ratio)
        active_blocks = int(active.any(0).sum())
        result = {
            "parent": native.audit,
            "data_task": manifest["task"],
            "active_blocks": active_blocks,
            "reward_std": float(total.std(unbiased=False)),
            "block_gradient_norm": report["grad_norm"],
            "reference_kl": report["reference_kl"],
            "advantage_std": report["advantage_std"],
            "ready_for_pilot": active_blocks == args.max_blocks and float(total.std(unbiased=False)) > 1e-7 and report["grad_norm"] > 0,
            "note": "Wiring/credit preflight only; not an accuracy result.",
        }
        _json(out / "preflight.json", result)
        if owned:
            native.close()
        return result

    optimizer = torch.optim.AdamW(native.s2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    step = 0; best = -1.
    if loaded is not None and args.resume:
        optimizer.load_state_dict(loaded["optimizer"])
        step = int(loaded["global_step"]); best = float(loaded["best_dev"])
        rng.setstate(loaded["python_rng"]); torch.set_rng_state(loaded["torch_rng"])
        if torch.cuda.is_available() and loaded.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(loaded["cuda_rng"])
    history = out / "metrics.jsonl"
    while step < args.steps:
        row = train[rng.randrange(len(train))]
        prefix, cond = prepared(row)
        started = time.perf_counter()
        group = [rollout(native.s2, cond, native.horizon, cfg) for _ in range(args.group_size)]
        candidates, rewards, returns, active, total = _group_credit(native, prefix, cond, group, row, args)
        metrics = block_grpo_update(native.s2, parent, optimizer, cond, group, returns, active, cfg,
                                    advantage_mode=args.advantage_mode, clip_range=args.clip_range,
                                    kl_coef=args.kl_coef, inner_epochs=args.inner_epochs,
                                    max_grad_norm=args.max_grad_norm, max_log_ratio=args.max_log_ratio)
        _frozen_gradient_guard(native)
        step += 1
        metrics.update(step=step, seconds=time.perf_counter() - started,
                       reward_mean=float(total.mean()), reward_std=float(total.std(unbiased=False)),
                       group_accuracy=sum(x[2] for x in candidates) / len(candidates),
                       format_rate=sum(x[3] for x in candidates) / len(candidates),
                       mean_active_blocks=float(active.sum(1).float().mean()), problem_hash=row["problem_hash"],
                       task=row["task"])
        with history.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, allow_nan=False) + "\n")
        print(json.dumps(metrics), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            _save(out / "latest.pt", native, optimizer, step, best, contract, rng)
        if step % args.eval_every == 0 or step == args.steps:
            report = evaluate(native, dev, args, cfg, parent)
            _json(out / f"dev_step_{step:06d}.json", report)
            primary = report["metrics"]["current"][str(args.max_blocks)]["accuracy"]
            if primary > best:
                best = primary
                _save(out / "best_dev.pt", native, optimizer, step, best, contract, rng)
    _save(out / "latest.pt", native, optimizer, step, best, contract, rng)
    summary = {"global_step": step, "best_dev_accuracy": best, "data_task": manifest["task"],
               "checkpoint": str(out / "latest.pt"), "selection_policy": SELECTION_POLICY,
               "verification": "training completed; sealed test not automatically evaluated"}
    _json(out / "summary.json", summary)
    if owned:
        native.close()
    return summary


def main(argv=None):
    result = run(parse_args(argv))
    print(json.dumps({k: v for k, v in result.items() if k != "parent"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
