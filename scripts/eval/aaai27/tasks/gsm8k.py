#!/usr/bin/env python3
"""Task 1: GSM8K - Sequential Latent Reasoning

200 test samples from GSM8K.
4 agents in chain: Planner → Critic → Refiner → Solver

Each agent sees the question + previous agent's output (for text methods)
or receives latent state (for latent methods).
"""
from __future__ import annotations
import re
from typing import List
from datasets import load_dataset

from scripts.eval.aaai27.common import TaskSpec, AgentInput


def load_gsm8k_tasks(n_samples: int = 200, seed: int = 42) -> List[TaskSpec]:
    """Load GSM8K test samples as TaskSpec objects."""
    ds = load_dataset("gsm8k", "main", split=f"test[:{n_samples}]")

    tasks = []
    for i, example in enumerate(ds):
        question = example["question"]
        answer_text = example["answer"]

        # Extract numerical answer from GSM8K format: "#### 42" or "The answer is 42"
        gold = extract_gsm8k_answer(answer_text)

        # Create 4-agent chain task
        # Agent 1 (Planner): sees question, forms initial plan
        # Agent 2 (Critic): checks plan
        # Agent 3 (Refiner): updates plan
        # Agent 4 (Solver): outputs final answer

        agents = [
            AgentInput(
                role="Planner",
                text=f"Question: {question}\n\nFormulate a step-by-step plan to solve this math problem.",
                query=question,
            ),
            AgentInput(
                role="Critic",
                text=f"Question: {question}\n\nReview the plan and check for errors.",
                query=question,
            ),
            AgentInput(
                role="Refiner",
                text=f"Question: {question}\n\nRefine the plan based on the critique.",
                query=question,
            ),
            AgentInput(
                role="Solver",
                text=f"Question: {question}\n\nSolve the problem and provide the final numerical answer.",
                query=question,
            ),
        ]

        task = TaskSpec(
            task_id=f"gsm8k_{i}",
            task_name="gsm8k",
            agents=agents,
            query=question,
            gold_answer=gold,
            metadata={"original_answer": answer_text},
        )
        tasks.append(task)

    return tasks


def extract_gsm8k_answer(answer_text: str) -> str:
    """Extract the final numerical answer from GSM8K format."""
    # GSM8K format: "... The answer is \\boxed{42}" or "... #### 42"
    match = re.search(r"####\s*(\d+)", answer_text)
    if match:
        return match.group(1)

    match = re.search(r"\\boxed\{(\d+)\}", answer_text)
    if match:
        return match.group(1)

    # Fallback: last number in the text
    numbers = re.findall(r"\d+", answer_text)
    return numbers[-1] if numbers else ""


def check_gsm8k_answer(predicted: str, gold: str) -> bool:
    """Check if predicted answer matches gold (numerical comparison)."""
    # Extract numbers from predicted answer
    pred_numbers = re.findall(r"\d+", predicted)
    if not pred_numbers:
        return False

    # Check if gold number appears in predicted
    return gold in pred_numbers or any(n == gold for n in pred_numbers)
