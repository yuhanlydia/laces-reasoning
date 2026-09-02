#!/usr/bin/env python3
"""Task 2: HiddenBench - Distributed Evidence Fusion

65 Hidden-Profile tasks.
4 agents, each receives:
- shared information (common to all)
- private evidence (unique to this agent)

The key challenge: no single agent can answer from its own evidence;
the group must FUSE distributed private information.
"""
from __future__ import annotations
import json
from typing import List
from pathlib import Path

from scripts.eval.aaai27.common import TaskSpec, AgentInput


def load_hiddenbench_tasks(
    data_path: str = None,
    n_agents: int = 4,
) -> List[TaskSpec]:
    """Load HiddenBench tasks as TaskSpec objects."""
    if data_path is None:
        # Default path
        repo_root = Path(__file__).resolve().parents[4]
        data_path = repo_root / "data" / "hiddenbench" / "benchmark.json"

    with open(data_path) as f:
        raw_tasks = json.load(f)

    tasks = []
    for i, raw in enumerate(raw_tasks):
        task_id = raw.get("id", i)
        task_name = raw.get("name", f"task_{i}")
        description = raw.get("description", "")
        shared_info = raw.get("shared_information", [])
        hidden_info = raw.get("hidden_information", [])
        possible_answers = raw.get("possible_answers", [])
        correct_answer = raw.get("correct_answer", "")

        # Distribute hidden information across agents
        # Each agent gets shared info + one piece of hidden info
        agents = []
        for j in range(n_agents):
            # Get this agent's private evidence
            if j < len(hidden_info):
                private_evidence = hidden_info[j]
            else:
                # If fewer hidden pieces than agents, cycle through
                private_evidence = hidden_info[j % len(hidden_info)] if hidden_info else ""

            # Agent text: shared context + private evidence
            agent_text = f"{description}\n\n"
            if shared_info:
                agent_text += "Shared Information:\n"
                for s in shared_info:
                    agent_text += f"- {s}\n"
                agent_text += "\n"

            agent_text += f"Your Private Evidence:\n{private_evidence}"

            # Build query with options
            options_str = " / ".join(possible_answers)
            query = f"Based on all information, which option is correct? Options: {options_str}."

            agents.append(AgentInput(
                role=f"Agent_{j+1}",
                text=agent_text,
                query=query,
            ))

        task = TaskSpec(
            task_id=f"hiddenbench_{task_id}",
            task_name="hiddenbench",
            agents=agents,
            query=agents[0].query if agents else "",
            gold_answer=correct_answer,
            metadata={
                "description": description,
                "shared_info": shared_info,
                "hidden_info": hidden_info,
                "possible_answers": possible_answers,
            },
        )
        tasks.append(task)

    return tasks


def check_hiddenbench_answer(predicted: str, gold: str, options: List[str]) -> bool:
    """Check if predicted answer matches gold (MCQ matching)."""
    pred_lower = predicted.lower().strip()
    gold_lower = gold.lower().strip()

    # Direct match
    if gold_lower in pred_lower:
        # Make sure no other option matches earlier
        for opt in options:
            opt_lower = opt.lower().strip()
            if opt_lower != gold_lower and opt_lower in pred_lower:
                if pred_lower.find(opt_lower) < pred_lower.find(gold_lower):
                    return False
        return True

    # Check if option index is mentioned (e.g., "A", "1")
    if gold in options:
        idx = options.index(gold)
        idx_str = chr(65 + idx)  # A, B, C, D
        if idx_str in predicted.upper():
            return True

    return False
