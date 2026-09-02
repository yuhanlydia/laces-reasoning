#!/usr/bin/env python3
"""Method: TextMAS-Qwen (text sequential passing with Qwen3-4B)

Agents process sequentially, each seeing:
- Their own private text
- All previous agents' generated text outputs

This simulates cumulative text-based communication.
"""
from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

from scripts.eval.aaai27.common import TaskSpec, MethodResult, measure_time
from scripts.eval.aaai27.tasks.gsm8k import check_gsm8k_answer
from scripts.eval.aaai27.tasks.hiddenbench import check_hiddenbench_answer
from scripts.eval.aaai27.tasks.stepgame import check_stepgame_answer


class TextMASQwenMethod:
    """TextMAS with Qwen3-4B: sequential text passing."""

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
        """Run TextMAS-Qwen on a task."""
        previous_outputs = []
        total_payload = 0

        def process_agents():
            nonlocal previous_outputs, total_payload

            for i, agent in enumerate(task.agents):
                # Build prompt: agent's text + all previous outputs
                prompt_parts = [agent.text]

                if previous_outputs:
                    prompt_parts.append("\n\nPrevious agents' reasoning:")
                    for j, output in enumerate(previous_outputs):
                        prompt_parts.append(f"[Agent {j+1}]: {output}")

                if i == len(task.agents) - 1:
                    # Last agent also sees the query
                    prompt_parts.append(f"\n\n{task.query}")

                prompt = "\n".join(prompt_parts) + "\n\nYour response:"

                # Generate
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=128,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                    )
                new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
                output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                previous_outputs.append(output_text.strip())

                # Accumulate payload
                total_payload += len(prompt.encode("utf-8"))

            return previous_outputs[-1] if previous_outputs else ""

        answer, time_ms = measure_time(process_agents)

        # Check correctness
        is_correct = self._check_answer(task, answer)

        return MethodResult(
            method_name="textmas_qwen",
            answer=answer.strip(),
            is_correct=is_correct,
            time_ms=time_ms,
            payload_bytes=total_payload,
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
