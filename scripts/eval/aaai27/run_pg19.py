#!/usr/bin/env python3
"""Run PG-19 long-form generation experiments."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List

import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from scripts.eval.aaai27.common import (
    TaskSpec, TaskResult, save_results, print_summary,
)
from scripts.eval.aaai27.tasks.pg19 import load_pg19_tasks


def load_tasks(
    n_books: int = 20,
    min_tokens: int = 8192,
    prefix_tokens: int = 512,
    generation_lengths: List[int] = None,
) -> List[TaskSpec]:
    """Load PG-19 tasks."""
    return load_pg19_tasks(
        n_books=n_books,
        min_tokens=min_tokens,
        prefix_tokens=prefix_tokens,
        generation_lengths=generation_lengths,
    )


def load_methods(method_names: List[str], device: str = "cuda:0") -> dict:
    """Load methods by name."""
    methods = {}

    if "raw_rwkv" in method_names:
        from scripts.eval.aaai27.methods.raw_rwkv_long import RawRWKVLongMethod
        methods["raw_rwkv"] = RawRWKVLongMethod(device=device)
        print("Loaded Raw RWKV method")

    if "single_plan_rwkv" in method_names:
        from scripts.eval.aaai27.methods.single_plan_rwkv_long import SinglePlanRWKVLongMethod
        methods["single_plan_rwkv"] = SinglePlanRWKVLongMethod(device=device)
        print("Loaded Single Plan RWKV method")

    if "latentweave_anchored" in method_names:
        from scripts.eval.aaai27.methods.latentweave_anchored_long import LatentWeaveAnchoredLongMethod
        methods["latentweave_anchored"] = LatentWeaveAnchoredLongMethod(device=device)
        print("Loaded LatentWeave Anchored method")

    if "raw_qwen" in method_names:
        from scripts.eval.aaai27.methods.raw_qwen_long import RawQwenLongMethod
        methods["raw_qwen"] = RawQwenLongMethod(device=device)
        print("Loaded Raw Qwen method")

    if "latentmas_qwen" in method_names:
        from scripts.eval.aaai27.methods.latentmas_qwen_long import LatentMASQwenLongMethod
        methods["latentmas_qwen"] = LatentMASQwenLongMethod(device=device)
        print("Loaded LatentMAS Qwen method")

    return methods


def run_experiments(
    tasks: List[TaskSpec],
    methods: dict,
    output_dir: str = "outputs_eval/aaai27/pg19",
) -> List[TaskResult]:
    """Run all experiments."""
    results = []

    for i, task in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] Task: {task.task_id}")

        task_result = TaskResult(
            task_id=task.task_id,
            task_name=task.task_name,
            gold_answer=task.gold_answer[:100] + "...",  # Truncate for display
        )

        for method_name, method in methods.items():
            try:
                result = method.run(task)
                task_result.results[method_name] = result

                # Print key metrics
                metrics = result.metadata
                print(f"  {method_name}:")
                print(f"    Time: {result.time_ms:.0f}ms")
                print(f"    Repetition: {metrics.get('repetition_4gram', 0):.3f}")
                print(f"    Entity consistency: {metrics.get('entity_consistency', 0):.3f}")
                print(f"    Topic retention: {metrics.get('topic_retention', 0):.3f}")
                print(f"    Avg coherence: {metrics.get('avg_coherence', 0):.3f}")

            except Exception as e:
                print(f"  {method_name}: ERROR - {e}")
                import traceback
                traceback.print_exc()

        results.append(task_result)

        # Save intermediate results every 5 tasks
        if (i + 1) % 5 == 0:
            output_path = Path(output_dir) / f"results_{i+1}.json"
            save_results(results, str(output_path))

    # Save final results
    output_path = Path(output_dir) / "results_final.json"
    save_results(results, str(output_path))

    return results


def main():
    parser = argparse.ArgumentParser(description="PG-19 Long-Form Generation Experiments")
    parser.add_argument("--n_books", type=int, default=20,
                        help="Number of books to use")
    parser.add_argument("--min_tokens", type=int, default=8192,
                        help="Minimum book length in tokens")
    parser.add_argument("--prefix_tokens", type=int, default=512,
                        help="Number of prefix tokens")
    parser.add_argument("--generation_lengths", nargs="+", type=int, default=[2048, 4096, 8192],
                        help="Generation lengths to test")
    parser.add_argument("--methods", nargs="+",
                        default=["raw_rwkv", "single_plan_rwkv", "latentweave_anchored", "raw_qwen", "latentmas_qwen"],
                        help="Methods to run")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use")
    parser.add_argument("--output_dir", type=str, default="outputs_eval/aaai27/pg19",
                        help="Output directory")

    args = parser.parse_args()

    print("=" * 80)
    print("PG-19 Long-Form Generation Experiments")
    print("=" * 80)
    print(f"Books: {args.n_books}")
    print(f"Min tokens: {args.min_tokens}")
    print(f"Prefix tokens: {args.prefix_tokens}")
    print(f"Generation lengths: {args.generation_lengths}")
    print(f"Methods: {args.methods}")
    print(f"Device: {args.device}")
    print("=" * 80)

    # Load tasks
    tasks = load_tasks(
        n_books=args.n_books,
        min_tokens=args.min_tokens,
        prefix_tokens=args.prefix_tokens,
        generation_lengths=args.generation_lengths,
    )
    print(f"\nTotal tasks: {len(tasks)}")

    # Load methods
    methods = load_methods(args.methods, args.device)
    print(f"Total methods: {len(methods)}")

    # Run experiments
    results = run_experiments(tasks, methods, args.output_dir)

    # Print summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)

    for method_name in args.methods:
        method_results = [r for r in results if method_name in r.results]
        if not method_results:
            continue

        # Aggregate metrics
        avg_time = sum(r.results[method_name].time_ms for r in method_results) / len(method_results)
        avg_repetition = sum(r.results[method_name].metadata.get('repetition_4gram', 0) for r in method_results) / len(method_results)
        avg_entity = sum(r.results[method_name].metadata.get('entity_consistency', 0) for r in method_results) / len(method_results)
        avg_topic = sum(r.results[method_name].metadata.get('topic_retention', 0) for r in method_results) / len(method_results)
        avg_coherence = sum(r.results[method_name].metadata.get('avg_coherence', 0) for r in method_results) / len(method_results)

        print(f"\n{method_name}:")
        print(f"  Avg time: {avg_time:.0f}ms")
        print(f"  Avg repetition (4-gram): {avg_repetition:.3f}")
        print(f"  Avg entity consistency: {avg_entity:.3f}")
        print(f"  Avg topic retention: {avg_topic:.3f}")
        print(f"  Avg coherence: {avg_coherence:.3f}")

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
