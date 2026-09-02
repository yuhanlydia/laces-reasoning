#!/usr/bin/env python3
"""Method: LatentMAS (latent thoughts + cumulative KV cache)

Based on LatentMAS paper (ICML 2026 Spotlight).
Each agent:
1. Processes their text
2. Generates "latent thinking tokens" (not decoded to text)
3. Passes cumulative KV cache to next agent

Final agent generates the answer from accumulated KV cache.
"""
from __future__ import annotations
import sys
import torch
from pathlib import Path
from types import SimpleNamespace
from typing import List

from scripts.eval.aaai27.common import TaskSpec, MethodResult, measure_time
from scripts.eval.aaai27.tasks.gsm8k import check_gsm8k_answer
from scripts.eval.aaai27.tasks.hiddenbench import check_hiddenbench_answer
from scripts.eval.aaai27.tasks.stepgame import check_stepgame_answer


class LatentMASMethod:
    """LatentMAS with Qwen3-4B: latent thoughts + cumulative KV."""

    def __init__(self, model_path: str = "Qwen/Qwen3-4B", device: str = "cuda:0"):
        self.device = torch.device(device)

        # Import LatentMAS
        repo_root = Path(__file__).resolve().parents[4]
        sys.path.insert(0, str(repo_root / "baselines" / "Cola_DLM"))
        sys.path.insert(0, str(repo_root / "LatentMAS"))

        from LatentMAS.models import ModelWrapper, _past_length

        args = SimpleNamespace(
            latent_space_realign=False,
            device=self.device,
            device2=self.device,
            use_vllm=False,
            use_second_HF_model=False,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            enable_prefix_caching=False,
            method="latent_mas",
            task="custom",
            max_new_tokens=256,
            think=False,
            latent_only=False,
            sequential_info_only=False,
        )

        self.model = ModelWrapper(model_path, self.device, use_vllm=False, args=args)
        self.tokenizer = self.model.tokenizer
        self._past_length = _past_length

    def run(self, task: TaskSpec, latent_steps: int = 10) -> MethodResult:
        """Run LatentMAS on a task."""
        def process_agents():
            past_kv = None
            total_kv_tokens = 0

            # Each agent processes their text with latent thinking
            for i, agent in enumerate(task.agents):
                prompt = f"You are {agent.role}. You know: {agent.text}\nThink about what this means."

                encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)

                # Generate latent steps (thinking tokens)
                past_kv = self.model.generate_latent_batch(
                    input_ids,
                    attention_mask=attention_mask,
                    latent_steps=latent_steps,
                    past_key_values=past_kv,
                )

                total_kv_tokens = self._past_length(past_kv)

            # Final agent generates answer
            judger_prompt = task.query
            encoded = self.tokenizer(judger_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            gen, _ = self.model.generate_text_batch(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.9,
                past_key_values=past_kv,
            )

            return gen[0], total_kv_tokens

        (answer, kv_tokens), time_ms = measure_time(process_agents)

        # Check correctness
        is_correct = self._check_answer(task, answer)

        # Payload: KV cache size (approximate)
        # Each token in KV cache: 2 (K+V) × layers × d_model × 2 bytes (fp16)
        # For Qwen3-4B: 36 layers, d_model=2560
        payload_bytes = kv_tokens * 2 * 36 * 2560 * 2

        return MethodResult(
            method_name="latentmas",
            answer=answer.strip(),
            is_correct=is_correct,
            time_ms=time_ms,
            payload_bytes=payload_bytes,
            metadata={"kv_tokens": kv_tokens},
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
