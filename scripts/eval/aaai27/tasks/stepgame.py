#!/usr/bin/env python3
"""Task 3: StepGame - Online Collaborative Reasoning

Multi-hop spatial reasoning benchmark.
Each hop's spatial relation is assigned to one agent.
Agents process in order of the reasoning chain.

Sample 100 examples per hop count (2, 4, 6, 8 hops) = 400 total.
"""
from __future__ import annotations
import re
from typing import List, Dict
from datasets import load_dataset

from scripts.eval.aaai27.common import TaskSpec, AgentInput


def load_stepgame_tasks(
    n_per_hop: int = 100,
    hop_counts: List[int] = None,
    seed: int = 42,
) -> List[TaskSpec]:
    """Load StepGame test samples as TaskSpec objects."""
    if hop_counts is None:
        hop_counts = [2, 4, 6, 8]

    ds = load_dataset("michaelszx/StepGame", split="test")

    # Filter by hop count and sample
    tasks_by_hop: Dict[int, List] = {k: [] for k in hop_counts}
    for example in ds:
        k = example.get("k_hop", 0)
        if k in hop_counts and len(tasks_by_hop[k]) < n_per_hop:
            tasks_by_hop[k].append(example)

    tasks = []
    task_idx = 0

    for k in hop_counts:
        for example in tasks_by_hop[k]:
            story_lines = example["story"]  # List of spatial relations
            question = example["question"]
            gold_label = example["label"]
            k_hop = example["k_hop"]

            # Each spatial relation is assigned to one agent
            # story_lines format: ["X is to the left of Y", "Y is in front of Z", ...]
            agents = []
            for i, relation in enumerate(story_lines[:k_hop]):
                agent_text = f"Spatial Relation {i+1}: {relation}"
                agents.append(AgentInput(
                    role=f"Agent_{i+1}",
                    text=agent_text,
                    query=question,
                ))

            task = TaskSpec(
                task_id=f"stepgame_{task_idx}",
                task_name="stepgame",
                agents=agents,
                query=question,
                gold_answer=gold_label,
                metadata={
                    "story": story_lines,
                    "k_hop": k_hop,
                },
            )
            tasks.append(task)
            task_idx += 1

    return tasks


# Valid spatial relation labels in StepGame
STEPGAME_LABELS = [
    "left", "right", "front", "back",
    "left and front", "left and back",
    "right and front", "right and back",
    "same", "different",
]


def check_stepgame_answer(predicted: str, gold: str) -> bool:
    """Check if predicted answer matches gold (spatial relation label)."""
    pred_lower = predicted.lower().strip()
    gold_lower = gold.lower().strip()

    # Direct match
    if gold_lower in pred_lower:
        return True

    # Check if any valid label is in predicted, and it matches gold
    for label in STEPGAME_LABELS:
        if label in pred_lower:
            return label == gold_lower

    return False
