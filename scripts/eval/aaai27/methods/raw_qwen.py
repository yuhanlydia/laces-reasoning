#!/usr/bin/env python3
"""Method: Raw Qwen3-4B (full context baseline)

Single model sees ALL agent texts concatenated.
This is the ceiling for text-based methods.
"""
from __future__ import annotations
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

from scripts.eval.aaai27.common import TaskSpec, MethodResult, measure_time
from scripts.eval.aaai27.tasks.gsm8k import check_gsm8k_answer
from scripts.eval.aaai27.tasks.hiddenbench import check_hiddenbench_answer
from scripts.eval.aaai27.tasks.stepgame import check_stepgame_answer


class RawQwenMethod:
    """Raw Qwen3-4B with full context."""

    def __init__(self, model_path: str = None, device: str = "cuda:0"):
        if model_path is None:
            model_path = "/inspire/hdd/global_user/zhangjiaquan-253108540222/models/Qwen2.5-3B"
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device).eval()

    def run(self, task: TaskSpec) -> MethodResult:
        """Run Raw Qwen on a task."""
        # Concatenate all agent texts
        full_context = "\n\n".join([
            f"[{agent.role}]:\n{agent.text}"
            for agent in task.agents
        ])

        # Add query
        prompt = f"{full_context}\n\n{task.query}\nAnswer:"

        def generate():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                )
            # Decode only new tokens
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        answer, time_ms = measure_time(generate)

        # Check correctness based on task type
        is_correct = self._check_answer(task, answer)

        # Payload: full context in bytes
        payload_bytes = len(prompt.encode("utf-8"))

        return MethodResult(
            method_name="raw_qwen",
            answer=answer.strip(),
            is_correct=is_correct,
            time_ms=time_ms,
            payload_bytes=payload_bytes,
        )

    def _check_answer(self, task: TaskSpec, predicted: str) -> bool:
        """Check answer based on task type."""
        if task.task_name == "gsm8k":
            return check_gsm8k_answer(predicted, task.gold_answer)
        elif task.task_name == "hiddenbench":
            options = task.metadata.get("possible_answers", [])
            return check_hiddenbench_answer(predicted, task.gold_answer, options)
        elif task.task_name == "stepgame":
            return check_stepgame_answer(predicted, task.gold_answer)
        return False
