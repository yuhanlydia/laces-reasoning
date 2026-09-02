#!/usr/bin/env python3
"""Main entry point for AAAI-27 multi-agent experiments.

Runs 3 tasks × 4 methods = 12 experiments.

Tasks:
1. GSM8K (200 samples): Sequential Latent Reasoning
2. HiddenBench (65 tasks): Distributed Evidence Fusion
3. StepGame (400 samples): Online Collaborative Reasoning

Methods:
1. LatentWeave (RWKV-7 2.9B): plan + recurrent state
2. Raw Qwen (Qwen3-4B): single model / full context
3. TextMAS-Qwen (Qwen3-4B): text sequential passing
4. LatentMAS (Qwen3-4B): latent thoughts + cumulative KV

Usage:
    python -m scripts.eval.aaai27.run_all --tasks gsm8k --methods latentweave raw_qwen
    python -m scripts.eval.aaai27.run_all --all
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import List

import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from scripts.eval.aaai27.common import (
    TaskSpec, TaskResult, MethodResult, save_results, load_results, print_summary,
)
from scripts.eval.aaai27.tasks.gsm8k import load_gsm8k_tasks
from scripts.eval.aaai27.tasks.hiddenbench import load_hiddenbench_tasks
from scripts.eval.aaai27.tasks.stepgame import load_stepgame_tasks


def load_tasks(task_names: List[str], n_samples: int = None) -> List[TaskSpec]:
    """Load tasks by name."""
    tasks = []

    if "gsm8k" in task_names:
        n = n_samples or 200
        gsm8k_tasks = load_gsm8k_tasks(n_samples=n)
        tasks.extend(gsm8k_tasks)
        print(f"Loaded {len(gsm8k_tasks)} GSM8K tasks")

    if "hiddenbench" in task_names:
        hb_tasks = load_hiddenbench_tasks(n_agents=4)
        tasks.extend(hb_tasks)
        print(f"Loaded {len(hb_tasks)} HiddenBench tasks")

    if "stepgame" in task_names:
        n_per_hop = (n_samples or 400) // 4
        sg_tasks = load_stepgame_tasks(n_per_hop=n_per_hop, hop_counts=[2, 4, 6, 8])
        tasks.extend(sg_tasks)
        print(f"Loaded {len(sg_tasks)} StepGame tasks")

    return tasks


def load_methods(method_names: List[str], device: str = "cuda:0") -> dict:
    """Load methods by name."""
    methods = {}

    # 基础方法（4个）
    if "raw_qwen" in method_names:
        from scripts.eval.aaai27.methods.raw_qwen import RawQwenMethod
        methods["raw_qwen"] = RawQwenMethod(device=device)
        print("Loaded Raw Qwen method")

    if "textmas_qwen" in method_names:
        from scripts.eval.aaai27.methods.textmas_qwen import TextMASQwenMethod
        methods["textmas_qwen"] = TextMASQwenMethod(device=device)
        print("Loaded TextMAS-Qwen method")

    if "latentmas" in method_names:
        from scripts.eval.aaai27.methods.latentmas import LatentMASMethod
        methods["latentmas"] = LatentMASMethod(device=device)
        print("Loaded LatentMAS method")

    if "latentweave" in method_names:
        from scripts.eval.aaai27.methods.latentweave import LatentWeaveMethod
        methods["latentweave"] = LatentWeaveMethod(device=device)
        print("Loaded LatentWeave method")

    # RWKV 消融方法（9个）
    if "raw_rwkv" in method_names:
        from scripts.eval.aaai27.methods.raw_rwkv import RawRWKVMethod
        methods["raw_rwkv"] = RawRWKVMethod(device=device)
        print("Loaded Raw RWKV method")

    if "state_only" in method_names:
        from scripts.eval.aaai27.methods.state_only_rwkv import StateOnlyRWKVMethod
        methods["state_only"] = StateOnlyRWKVMethod(device=device)
        print("Loaded State-only RWKV method")

    if "anchored_relay" in method_names:
        from scripts.eval.aaai27.methods.anchored_relay_rwkv import AnchoredRelayRWKVMethod
        methods["anchored_relay"] = AnchoredRelayRWKVMethod(device=device)
        print("Loaded Anchored Relay RWKV method")

    if "pca_k1" in method_names:
        from scripts.eval.aaai27.methods.pca_rwkv import PCARWKVMethod
        methods["pca_k1"] = PCARWKVMethod(device=device, pca_k=1)
        print("Loaded PCA K=1 RWKV method")

    if "pca_k2" in method_names:
        from scripts.eval.aaai27.methods.pca_rwkv import PCARWKVMethod
        methods["pca_k2"] = PCARWKVMethod(device=device, pca_k=2)
        print("Loaded PCA K=2 RWKV method")

    if "pca_k3" in method_names:
        from scripts.eval.aaai27.methods.pca_rwkv import PCARWKVMethod
        methods["pca_k3"] = PCARWKVMethod(device=device, pca_k=3)
        print("Loaded PCA K=3 RWKV method")

    if "anchored_pca_relay_state" in method_names:
        from scripts.eval.aaai27.methods.anchored_pca_relay_state import AnchoredPCARelayStateMethod
        methods["anchored_pca_relay_state"] = AnchoredPCARelayStateMethod(device=device)
        print("Loaded Anchored PCA Relay + State method")

    if "direct_relay" in method_names:
        from scripts.eval.aaai27.methods.direct_relay_rwkv import DirectRelayRWKVMethod
        methods["direct_relay"] = DirectRelayRWKVMethod(device=device)
        print("Loaded Direct Relay RWKV method")

    if "renormalized_relay" in method_names:
        from scripts.eval.aaai27.methods.renormalized_relay_rwkv import RenormalizedRelayRWKVMethod
        methods["renormalized_relay"] = RenormalizedRelayRWKVMethod(device=device)
        print("Loaded Renormalized Relay RWKV method")

    if "anchored_relay_state" in method_names:
        from scripts.eval.aaai27.methods.anchored_relay_state_rwkv import AnchoredRelayStateRWKVMethod
        methods["anchored_relay_state"] = AnchoredRelayStateRWKVMethod(device=device)
        print("Loaded Anchored Relay + State RWKV method")

    # HiddenBench 特定方法（5个）
    if "best_local_agent" in method_names:
        from scripts.eval.aaai27.methods.best_local_agent import BestLocalAgentMethod
        methods["best_local_agent"] = BestLocalAgentMethod(device=device)
        print("Loaded Best Local Agent method")

    if "full_information_raw" in method_names:
        from scripts.eval.aaai27.methods.full_information_raw import FullInformationRawMethod
        methods["full_information_raw"] = FullInformationRawMethod(device=device)
        print("Loaded Full Information Raw method")

    if "direct_latent_avg" in method_names:
        from scripts.eval.aaai27.methods.direct_latent_avg import DirectLatentAverageMethod
        methods["direct_latent_avg"] = DirectLatentAverageMethod(device=device)
        print("Loaded Direct Latent Average method")

    if "residual_fusion_k0" in method_names:
        from scripts.eval.aaai27.methods.residual_fusion_k0 import ResidualFusionK0Method
        methods["residual_fusion_k0"] = ResidualFusionK0Method(device=device)
        print("Loaded Residual Fusion K=0 method")

    if "shuffled_plan" in method_names:
        from scripts.eval.aaai27.methods.shuffled_plan import ShuffledPlanMethod
        methods["shuffled_plan"] = ShuffledPlanMethod(device=device)
        print("Loaded Shuffled Plan method")

    return methods


def load_existing_results(output_dir: str) -> dict:
    """Load existing results from checkpoint files."""
    import glob as globmod
    existing = {}
    files = sorted(globmod.glob(str(Path(output_dir) / "results_*.json")))
    for f in files:
        if "final" in f:
            continue
        try:
            data = load_results(f)
            for tr in data:
                if tr.results:
                    existing[tr.task_id] = tr
        except Exception:
            continue
    return existing


def run_experiments(
    tasks: List[TaskSpec],
    methods: dict,
    output_dir: str = "outputs_eval/aaai27",
    resume: bool = False,
) -> List[TaskResult]:
    """Run all experiments."""
    results = []
    skip_count = 0

    if resume:
        existing = load_existing_results(output_dir)
        print(f"Resuming: found {len(existing)} completed tasks in {output_dir}")
    else:
        existing = {}

    for i, task in enumerate(tasks):
        if resume and task.task_id in existing:
            results.append(existing[task.task_id])
            skip_count += 1
            continue

        if skip_count > 0:
            print(f"\n[Skipped {skip_count} completed tasks, resuming from task {i+1}]")
            skip_count = 0

        print(f"\n[{i+1}/{len(tasks)}] Task: {task.task_id} ({task.task_name})")

        task_result = TaskResult(
            task_id=task.task_id,
            task_name=task.task_name,
            gold_answer=task.gold_answer,
        )

        for method_name, method in methods.items():
            try:
                result = method.run(task)
                task_result.results[method_name] = result
                status = "✓" if result.is_correct else "✗"
                print(f"  {method_name}: {status} ({result.time_ms:.0f}ms)")
            except Exception as e:
                print(f"  {method_name}: ERROR - {e}")

        results.append(task_result)

        # Save intermediate results every 10 tasks
        if (i + 1) % 10 == 0:
            output_path = Path(output_dir) / f"results_{i+1}.json"
            save_results(results, str(output_path))

    # Save final results
    output_path = Path(output_dir) / "results_final.json"
    save_results(results, str(output_path))

    return results


def main():
    parser = argparse.ArgumentParser(description="AAAI-27 Multi-Agent Experiments")
    parser.add_argument("--tasks", nargs="+", default=["gsm8k", "hiddenbench", "stepgame"],
                        help="Tasks to run: gsm8k, hiddenbench, stepgame")
    parser.add_argument("--methods", nargs="+", default=["latentweave", "raw_qwen", "textmas_qwen", "latentmas"],
                        help="Methods to run: latentweave, raw_qwen, textmas_qwen, latentmas")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Number of samples per task (default: task-specific)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use")
    parser.add_argument("--output_dir", type=str, default="outputs_eval/aaai27",
                        help="Output directory")
    parser.add_argument("--all", action="store_true",
                        help="Run all tasks and methods")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoint results")

    args = parser.parse_args()

    if args.all:
        args.tasks = ["gsm8k", "hiddenbench", "stepgame"]
        args.methods = ["latentweave", "raw_qwen", "textmas_qwen", "latentmas"]

    print("=" * 80)
    print("AAAI-27 Multi-Agent Experiments")
    print("=" * 80)
    print(f"Tasks: {args.tasks}")
    print(f"Methods: {args.methods}")
    print(f"Device: {args.device}")
    print("=" * 80)

    # Load tasks
    tasks = load_tasks(args.tasks, args.n_samples)
    print(f"\nTotal tasks: {len(tasks)}")

    # Load methods
    methods = load_methods(args.methods, args.device)
    print(f"Total methods: {len(methods)}")

    # Run experiments
    results = run_experiments(tasks, methods, args.output_dir, resume=args.resume)

    # Print summary
    print_summary(results, args.methods)

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
