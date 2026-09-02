#!/usr/bin/env python3
"""Common interface for AAAI-27 multi-agent experiments.

Defines:
- TaskSpec: unified task specification
- MethodResult: unified result format
- Evaluation helpers: accuracy, timing, payload measurement
"""
from __future__ import annotations
import time
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path

import torch


@dataclass
class AgentInput:
    """Input for a single agent."""
    role: str  # e.g., "Planner", "Critic", "Agent_1"
    text: str  # The text this agent sees (private evidence + shared context)
    query: str = ""  # The question (visible to all agents in some tasks)


@dataclass
class TaskSpec:
    """Unified task specification for all 3 tasks."""
    task_id: str
    task_name: str  # "gsm8k", "hiddenbench", "stepgame"
    agents: List[AgentInput]  # List of agent inputs (ordered for chain)
    query: str  # Final question
    gold_answer: str  # Ground truth answer
    metadata: Dict[str, Any] = field(default_factory=dict)  # Task-specific info


@dataclass
class MethodResult:
    """Unified result format for all methods."""
    method_name: str  # "latentweave", "raw_qwen", "textmas_qwen", "latentmas"
    answer: str
    is_correct: bool
    time_ms: float
    payload_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Result for a single task across all methods."""
    task_id: str
    task_name: str
    gold_answer: str
    results: Dict[str, MethodResult] = field(default_factory=dict)


def measure_time(fn: Callable) -> tuple[Any, float]:
    """Measure execution time in milliseconds."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return result, (t1 - t0) * 1000


def compute_accuracy(results: List[TaskResult], method_name: str) -> float:
    """Compute accuracy for a method across tasks."""
    correct = sum(1 for r in results if method_name in r.results and r.results[method_name].is_correct)
    total = sum(1 for r in results if method_name in r.results)
    return correct / total if total > 0 else 0.0


def compute_avg_time(results: List[TaskResult], method_name: str) -> float:
    """Compute average time for a method across tasks."""
    times = [r.results[method_name].time_ms for r in results if method_name in r.results]
    return sum(times) / len(times) if times else 0.0


def compute_avg_payload(results: List[TaskResult], method_name: str) -> int:
    """Compute average payload for a method across tasks."""
    payloads = [r.results[method_name].payload_bytes for r in results if method_name in r.results]
    return sum(payloads) // len(payloads) if payloads else 0


def save_results(results: List[TaskResult], output_path: str):
    """Save results to JSON."""
    data = []
    for r in results:
        item = {
            "task_id": r.task_id,
            "task_name": r.task_name,
            "gold_answer": r.gold_answer,
            "results": {}
        }
        for method_name, mr in r.results.items():
            item["results"][method_name] = {
                "answer": mr.answer,
                "is_correct": mr.is_correct,
                "time_ms": mr.time_ms,
                "payload_bytes": mr.payload_bytes,
                "metadata": mr.metadata,
            }
        data.append(item)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(results)} results to {output_path}")


def load_results(path: str) -> List[TaskResult]:
    """Load results from JSON."""
    with open(path) as f:
        data = json.load(f)

    results = []
    for item in data:
        tr = TaskResult(
            task_id=item["task_id"],
            task_name=item["task_name"],
            gold_answer=item["gold_answer"],
        )
        for method_name, mr_data in item["results"].items():
            tr.results[method_name] = MethodResult(
                method_name=method_name,
                answer=mr_data["answer"],
                is_correct=mr_data["is_correct"],
                time_ms=mr_data["time_ms"],
                payload_bytes=mr_data.get("payload_bytes", 0),
                metadata=mr_data.get("metadata", {}),
            )
        results.append(tr)
    return results


def print_summary(results: List[TaskResult], methods: List[str]):
    """Print summary table."""
    print("\n" + "=" * 80)
    print(f"{'Method':<20} {'Accuracy':>10} {'Avg Time (ms)':>15} {'Avg Payload (MB)':>20}")
    print("-" * 80)
    for method in methods:
        acc = compute_accuracy(results, method)
        avg_time = compute_avg_time(results, method)
        avg_payload = compute_avg_payload(results, method) / (1024 * 1024)
        print(f"{method:<20} {acc:>9.1%} {avg_time:>14.0f} {avg_payload:>19.2f}")
    print("=" * 80)
