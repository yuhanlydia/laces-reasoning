#!/usr/bin/env python3
"""E10: variable-budget recurrent latent reasoning with cumulative state writes.

The frozen RWKV supplies token-level fact and question features.  A shared recurrent
reasoner re-queries the facts at every step while keeping its workspace in R^32:

    q_r = Q([z_r, h_q])
    c_r = CrossAttn(q_r, H_facts, H_facts)
    z_{r+1} = GRU(c_r, z_r)
    C_r = R_dynamic(z_r)

``C_r`` is the cumulative recurrent-state correction after reasoning step ``r``.  At
inference we inject only ``C_R``; we never sum one full correction per step.  Training
unrolls to ``--train_max_steps`` and supervises every available budget.  Once the oracle
hop depth is reached, later steps are trained to preserve the solved latent/state.

Inference supports external budgets (1/2/4/8 by default), an oracle-hop diagnostic, and
validation-calibrated convergence stopping.  None of these modes updates parameters.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from models.recurrent_latent_reasoner import (  # noqa: E402
    RecurrentReasoner,
    cumulative_state_targets,
    normalize_depths,
    select_converged_depth,
    select_default_budget,
    state_at_depth,
    state_relative_mse,
    variable_depth_state_loss,
)
from models.state_hijacking_dit import _cache_layer_state, _reset_cache_layer_seen_tokens  # noqa: E402
from scripts.eval.diag_hard_problems_v2 import (  # noqa: E402
    gen_2agent,
    gen_3agent,
    gen_4agent,
    gen_conflict,
)
from scripts.eval.exp_multiagent_star_fusion import capture_final_state, raw_text  # noqa: E402
from scripts.eval.relay_utils import load_relay_model  # noqa: E402
from scripts.eval.run_cola_dlm_tasks_prefix_suffix_trajectory_cfg import sample_next_token  # noqa: E402

CKPT = str(
    REPO
    / "outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000"
)


def _split_query_ids(query_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if query_ids.ndim != 2 or query_ids.shape[0] != 1:
        raise ValueError(f"expected one tokenized query [1,T], got {tuple(query_ids.shape)}")
    if query_ids.shape[1] < 2:
        raise ValueError("query must contain at least two tokens for aligned state injection")
    return query_ids[:, :-1], query_ids[:, -1:]


@torch.no_grad()
def token_hidden(model, ids: torch.Tensor) -> torch.Tensor:
    mask = torch.ones_like(ids, dtype=torch.bool)
    output = model.rwkv_model(
        input_ids=ids,
        attention_mask=mask,
        output_hidden_states=True,
        use_cache=True,
        return_dict=True,
    )
    return output.hidden_states[-1][0].float()


def _to_cpu_half(states):
    return [state.detach().cpu().half() for state in states]


def _to_device_float(states, device):
    return [state.to(device=device, dtype=torch.float32) for state in states]


@torch.no_grad()
def generate_from_cumulative_state(
    model,
    tokenizer,
    query_prefix_ids: torch.Tensor,
    query_last_id: torch.Tensor,
    cumulative_state,
    args,
) -> list[int]:
    """Inject at the position immediately before the final query token.

    The previous E10 path consumed the whole query, modified its cache, and then consumed
    the whole query again.  Here the prefix is consumed once, the state is changed, and
    the final query token is consumed exactly once to obtain the first answer logits.
    """
    output = model.rwkv_model(
        input_ids=query_prefix_ids,
        attention_mask=torch.ones_like(query_prefix_ids, dtype=torch.bool),
        use_cache=True,
        return_dict=True,
    )
    past_kv = output.past_key_values
    for layer_index, correction in enumerate(cumulative_state):
        state = _cache_layer_state(past_kv, layer_index)
        current = state.get("recurrent_state")
        correction = correction.to(device=query_prefix_ids.device, dtype=torch.float32)
        state["recurrent_state"] = (
            current.float() + correction if isinstance(current, torch.Tensor) else correction
        )
        for sub_key in ("conv_state", "ffn_state"):
            cached = state.get(sub_key)
            if isinstance(cached, torch.Tensor):
                state[sub_key] = torch.zeros_like(cached)
        _reset_cache_layer_seen_tokens(past_kv, layer_index)
    if hasattr(past_kv, "_seen_tokens"):
        past_kv._seen_tokens = 0

    output = model.rwkv_model(
        input_ids=query_last_id,
        past_key_values=past_kv,
        use_cache=True,
        return_dict=True,
    )
    past_kv = output.past_key_values
    logits = output.logits[0, -1]
    all_ids = list(query_prefix_ids[0].tolist()) + list(query_last_id[0].tolist())
    new_ids: list[int] = []
    eos_id = getattr(tokenizer, "eos_token_id", None)

    for _ in range(args.max_new_tokens):
        next_id = sample_next_token(logits, all_ids, args)
        if eos_id is not None and next_id == eos_id:
            break
        new_ids.append(next_id)
        all_ids.append(next_id)
        output = model.rwkv_model(
            input_ids=torch.tensor([[next_id]], device=query_prefix_ids.device),
            past_key_values=past_kv,
            use_cache=True,
            return_dict=True,
        )
        past_kv = output.past_key_values
        logits = output.logits[0, -1]
    return new_ids


def _task_generators():
    return {
        "2agent": gen_2agent,
        "3agent": gen_3agent,
        "4agent": gen_4agent,
        "conflict": gen_conflict,
    }


def _build_dataset(model, tokenizer, tasks, device):
    def encode(text: str) -> torch.Tensor:
        return tokenizer(text, return_tensors="pt").input_ids.to(device)

    records = []
    for item in tasks:
        facts = list(item["agents"])
        query_ids = encode(item["q"])
        query_prefix_ids, query_last_id = _split_query_ids(query_ids)
        fact_ids = encode(" ".join(facts))

        H_facts = token_hidden(model, fact_ids).cpu().half()
        H_query = token_hidden(model, query_ids).cpu().half()
        context_state = capture_final_state(model, query_prefix_ids)

        oracle_states = []
        fact_prefix = ""
        for fact in facts:
            fact_prefix = f"{fact_prefix} {fact}".strip()
            oracle_ids = encode(f"{fact_prefix} {item['q']}")
            oracle_prefix_ids, oracle_last_id = _split_query_ids(oracle_ids)
            if int(oracle_last_id.item()) != int(query_last_id.item()):
                raise ValueError(
                    "the final query token changed after fact concatenation; cannot align state targets"
                )
            oracle_states.append(capture_final_state(model, oracle_prefix_ids))

        C_star = cumulative_state_targets(context_state, oracle_states)
        records.append(
            {
                "H_facts": H_facts,
                "H_query": H_query,
                "C_star": [_to_cpu_half(target) for target in C_star],
                "hop_count": len(facts),
                "query_prefix_ids": query_prefix_ids,
                "query_last_id": query_last_id,
                "gold": item["gold"].lower(),
            }
        )
    return records


def _optimizer_step(reasoner, optimizer, max_grad_norm: float) -> float:
    grad_norm = torch.nn.utils.clip_grad_norm_(reasoner.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(grad_norm.detach().item())


def _train_reasoner(reasoner, train_data, args, device):
    optimizer = torch.optim.AdamW(
        reasoner.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)
    rng = random.Random(args.seed)

    for epoch in range(args.epochs):
        reasoner.train()
        order = list(range(len(train_data)))
        rng.shuffle(order)
        epoch_loss = 0.0
        epoch_metrics = {
            "state_loss": 0.0,
            "cosine_loss": 0.0,
            "latent_stability": 0.0,
            "state_stability": 0.0,
        }
        grad_norm = 0.0

        for position, record_index in enumerate(order, start=1):
            record = train_data[record_index]
            trace = reasoner(
                record["H_facts"].to(device=device, dtype=torch.float32),
                record["H_query"].to(device=device, dtype=torch.float32),
                steps=args.train_max_steps,
            )
            targets = [
                _to_device_float(target, device)
                for target in record["C_star"]
            ]
            loss, metrics = variable_depth_state_loss(
                trace,
                targets,
                hop_count=record["hop_count"],
                lambda_cos=args.lambda_cos,
                lambda_latent_stability=args.lambda_latent_stability,
                lambda_state_stability=args.lambda_state_stability,
            )
            (loss / args.grad_accum).backward()
            epoch_loss += float(loss.detach().item())
            for key in epoch_metrics:
                epoch_metrics[key] += float(metrics[key])

            should_step = position % args.grad_accum == 0 or position == len(order)
            if should_step:
                grad_norm = _optimizer_step(
                    reasoner, optimizer, max_grad_norm=args.max_grad_norm
                )

        if (epoch + 1) % max(1, args.epochs // 10) == 0 or epoch == 0:
            count = max(1, len(train_data))
            metric_text = " ".join(
                f"{key}={value / count:.4f}" for key, value in epoch_metrics.items()
            )
            print(
                f"[train] epoch {epoch + 1}/{args.epochs} "
                f"loss={epoch_loss / count:.4f} {metric_text} grad={grad_norm:.3f}",
                flush=True,
            )


def _evaluation_depths(args, hop_count: int) -> list[int]:
    if args.R_mode == "fixed":
        return [args.R]
    if args.R_mode == "auto":
        # Oracle diagnostic only: real tasks do not expose their gold hop count.
        return [min(hop_count, args.eval_max_steps)]
    return normalize_depths(args.eval_depths, max_steps=args.eval_max_steps)


@torch.no_grad()
def _evaluate(reasoner, model, tokenizer, test_tasks, test_data, args, device):
    requested_depths = normalize_depths(args.eval_depths, max_steps=args.eval_max_steps)
    if not requested_depths:
        raise ValueError("--eval_depths must include at least one value within eval_max_steps")

    fixed_or_sweep_max = max(requested_depths + [min(args.R, args.eval_max_steps)])
    max_trace_steps = args.eval_max_steps if args.early_stop else fixed_or_sweep_max
    if args.R_mode == "auto":
        max_trace_steps = max(max_trace_steps, max(r["hop_count"] for r in test_data))
    max_trace_steps = min(max_trace_steps, args.eval_max_steps)

    correct: dict[str, int] = {"text_concat": 0, "inject_gold": 0}
    mse_sum: dict[str, float] = {}
    selected_depths: list[int] = []
    sample_args = SimpleNamespace(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        device=device,
    )

    for item_index, (item, record) in enumerate(zip(test_tasks, test_data), start=1):
        gold = record["gold"]
        text_concat = tokenizer.decode(
            raw_text(model, tokenizer, " ".join(item["agents"]), item["q"], sample_args),
            skip_special_tokens=True,
        ).strip().lower()
        correct["text_concat"] += int(gold in text_concat)

        full_target = _to_device_float(record["C_star"][-1], device)
        inject_gold_ids = generate_from_cumulative_state(
            model,
            tokenizer,
            record["query_prefix_ids"],
            record["query_last_id"],
            full_target,
            sample_args,
        )
        inject_gold_text = tokenizer.decode(
            inject_gold_ids, skip_special_tokens=True
        ).strip().lower()
        correct["inject_gold"] += int(gold in inject_gold_text)

        trace = reasoner(
            record["H_facts"].to(device=device, dtype=torch.float32),
            record["H_query"].to(device=device, dtype=torch.float32),
            steps=max_trace_steps,
        )
        depths = _evaluation_depths(args, record["hop_count"])
        depths = normalize_depths(depths, max_steps=max_trace_steps)

        for depth in depths:
            if args.R_mode == "auto":
                key = "reasoner_oracle_hop"
            else:
                key = f"reasoner_R{depth}"
            correct.setdefault(key, 0)
            mse_sum.setdefault(key, 0.0)
            predicted_state = state_at_depth(trace, depth)
            output_ids = generate_from_cumulative_state(
                model,
                tokenizer,
                record["query_prefix_ids"],
                record["query_last_id"],
                predicted_state,
                sample_args,
            )
            output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip().lower()
            correct[key] += int(gold in output_text)
            mse_sum[key] += float(
                (state_relative_mse(predicted_state, full_target) / len(full_target)).item()
            )

        if args.early_stop:
            stop_depth = select_converged_depth(
                trace.latents,
                trace.cumulative_states,
                min_steps=args.early_stop_min_steps,
                max_steps=max_trace_steps,
                latent_tolerance=args.latent_tolerance,
                state_tolerance=args.state_tolerance,
                patience=args.early_stop_patience,
            )
            selected_depths.append(stop_depth)
            key = "reasoner_early_stop"
            correct.setdefault(key, 0)
            mse_sum.setdefault(key, 0.0)
            predicted_state = state_at_depth(trace, stop_depth)
            output_ids = generate_from_cumulative_state(
                model,
                tokenizer,
                record["query_prefix_ids"],
                record["query_last_id"],
                predicted_state,
                sample_args,
            )
            output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip().lower()
            correct[key] += int(gold in output_text)
            mse_sum[key] += float(
                (state_relative_mse(predicted_state, full_target) / len(full_target)).item()
            )

        progress = " ".join(f"{key}={value}/{item_index}" for key, value in correct.items())
        if args.early_stop and selected_depths:
            progress += f" mean_stop={sum(selected_depths) / len(selected_depths):.2f}"
        print(f"[eval {item_index}/{len(test_data)}] {progress}", flush=True)

    count = len(test_data)
    accuracy = {key: round(value / count, 4) for key, value in correct.items()}
    depth_accuracy = {
        int(key.removeprefix("reasoner_R")): score
        for key, score in accuracy.items()
        if key.startswith("reasoner_R") and key.removeprefix("reasoner_R").isdigit()
    }
    recommended_default = (
        select_default_budget(depth_accuracy, tolerance=args.budget_tolerance)
        if depth_accuracy
        else None
    )
    return {
        "accuracy": accuracy,
        "relative_mse": {
            key: round(value / count, 6) for key, value in mse_sum.items()
        },
        "recommended_default_R": recommended_default,
        "budget_tolerance": args.budget_tolerance,
        "early_stop_mean_depth": (
            round(sum(selected_depths) / len(selected_depths), 4)
            if selected_depths
            else None
        ),
        "early_stop_depths": selected_depths,
    }


def run(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    print(f"[init] loading {args.ckpt_dir}", flush=True)
    model, _rwkv, tokenizer, _checkpoint, _config = load_relay_model(
        args.ckpt_dir, device
    )
    model.eval()
    model._prefix_suffix_trajectory_s2 = True

    generators = _task_generators()
    unknown = sorted(set(args.tasks) - set(generators))
    if unknown:
        raise ValueError(f"unknown tasks: {unknown}; choose from {sorted(generators)}")

    train_tasks = []
    test_tasks = []
    for task_name in args.tasks:
        train_tasks.extend(generators[task_name](args.n_train, args.seed))
        test_tasks.extend(generators[task_name](args.n_test, args.seed + 999))

    print(f"[data] building {len(train_tasks)} train records", flush=True)
    train_data = _build_dataset(model, tokenizer, train_tasks, device)
    print(f"[data] building {len(test_tasks)} test records", flush=True)
    test_data = _build_dataset(model, tokenizer, test_tasks, device)

    reasoner = RecurrentReasoner(
        hidden_dim=int(model.hidden_size),
        num_layers=int(model.num_layers),
        num_heads=int(model.num_heads),
        head_dim=int(model.head_dim),
        z_dim=args.z_dim,
        context_dim=args.context_dim,
        writer_rank=args.r_s,
        writer_hidden=args.writer_hidden,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in reasoner.parameters())
    print(
        f"[model] params={parameter_count / 1e6:.2f}M z={args.z_dim} "
        f"writer_rank={args.r_s} train_max_steps={args.train_max_steps}",
        flush=True,
    )

    _train_reasoner(reasoner, train_data, args, device)
    reasoner.eval()
    evaluation = _evaluate(
        reasoner, model, tokenizer, test_tasks, test_data, args, device
    )

    summary = {
        "ckpt_dir": args.ckpt_dir,
        "n_train": len(train_tasks),
        "n_test": len(test_tasks),
        "tasks": args.tasks,
        "train_max_steps": args.train_max_steps,
        "R_mode": args.R_mode,
        "fixed_R": args.R,
        "eval_depths": normalize_depths(
            args.eval_depths, max_steps=args.eval_max_steps
        ),
        "eval_max_steps": args.eval_max_steps,
        "early_stop": args.early_stop,
        "z_dim": args.z_dim,
        "writer_rank": args.r_s,
        "writer_hidden": args.writer_hidden,
        "parameter_count": parameter_count,
        "epochs": args.epochs,
        "lr": args.lr,
        **evaluation,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    torch.save(
        {
            "reasoner": reasoner.state_dict(),
            "model_config": {
                "hidden_dim": int(model.hidden_size),
                "num_layers": int(model.num_layers),
                "num_heads": int(model.num_heads),
                "head_dim": int(model.head_dim),
                "z_dim": args.z_dim,
                "context_dim": args.context_dim,
                "writer_rank": args.r_s,
                "writer_hidden": args.writer_hidden,
            },
            "training_args": vars(args),
        },
        str(output_path.with_suffix(".pt")),
    )

    print("\n================ E10 VARIABLE-DEPTH REASONING ================", flush=True)
    for key, value in evaluation["accuracy"].items():
        print(f"  {key:24s} acc = {value * 100:5.1f}%", flush=True)
    if evaluation["recommended_default_R"] is not None:
        print(
            f"  {'recommended_default_R':24s} = {evaluation['recommended_default_R']}",
            flush=True,
        )
    if evaluation["early_stop_mean_depth"] is not None:
        print(
            f"  {'early_stop_mean_depth':24s} = {evaluation['early_stop_mean_depth']:.2f}",
            flush=True,
        )
    print(f"written: {output_path}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train and evaluate variable-budget recurrent latent reasoning."
    )
    parser.add_argument("--ckpt_dir", default=CKPT)
    parser.add_argument(
        "--tasks", nargs="+", default=["2agent", "3agent", "4agent"]
    )
    parser.add_argument("--n_train", type=int, default=60)
    parser.add_argument("--n_test", type=int, default=25)
    parser.add_argument("--device", default="cuda:0")

    parser.add_argument("--train_max_steps", type=int, default=8)
    parser.add_argument("--eval_max_steps", type=int, default=8)
    parser.add_argument("--eval_depths", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--budget_tolerance", type=float, default=0.01)
    parser.add_argument(
        "--R_mode",
        default="sweep",
        choices=["sweep", "fixed", "auto"],
        help="sweep reports external budgets; fixed uses --R; auto is oracle-hop diagnostic",
    )
    parser.add_argument("--R", type=int, default=4)
    # Backward-compatible flag accepted by old launch commands. The new objective is
    # always cumulative variable-depth supervision.
    parser.add_argument("--mode", default="variable")

    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--early_stop_min_steps", type=int, default=2)
    parser.add_argument("--early_stop_patience", type=int, default=2)
    parser.add_argument("--latent_tolerance", type=float, default=1e-3)
    parser.add_argument("--state_tolerance", type=float, default=1e-3)

    parser.add_argument("--z_dim", type=int, default=32)
    parser.add_argument("--context_dim", type=int, default=128)
    parser.add_argument("--r_s", type=int, default=32)
    parser.add_argument("--writer_hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lambda_cos", type=float, default=0.5)
    parser.add_argument("--lambda_latent_stability", type=float, default=0.05)
    parser.add_argument("--lambda_state_stability", type=float, default=0.05)

    parser.add_argument("--max_new_tokens", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", default="results/capacity_audit/e10_variable_depth.json"
    )
    args = parser.parse_args(argv)

    if args.train_max_steps <= 0 or args.eval_max_steps <= 0:
        parser.error("train_max_steps and eval_max_steps must be positive")
    if args.R <= 0 or args.R > args.eval_max_steps:
        parser.error("R must be within 1..eval_max_steps")
    if args.budget_tolerance < 0:
        parser.error("budget_tolerance must be non-negative")
    if args.grad_accum <= 0:
        parser.error("grad_accum must be positive")
    if not normalize_depths(args.eval_depths, max_steps=args.eval_max_steps):
        parser.error("eval_depths must contain a value within 1..eval_max_steps")
    return args


if __name__ == "__main__":
    run(parse_args())
