"""LatentMAS Qwen long-form generation method."""
from __future__ import annotations
import sys
import torch
from pathlib import Path
from typing import Dict, Any
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from scripts.eval.aaai27.common import TaskSpec, MethodResult, measure_time
from scripts.eval.aaai27.tasks.pg19 import evaluate_pg19_generation


class LatentMASQwenLongMethod:
    """LatentMAS Qwen: latent thoughts + cumulative KV cache."""

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-4B",
        device: str = "cuda:0",
        latent_steps: int = 10,
    ):
        self.device = torch.device(device)

        # Import LatentMAS
        sys.path.insert(0, str(REPO / "baselines" / "Cola_DLM"))
        sys.path.insert(0, str(REPO / "LatentMAS"))

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
        self.latent_steps = latent_steps

    def run(self, task: TaskSpec) -> MethodResult:
        prefix = task.agents[0].text
        max_length = task.metadata["generation_length"]

        def generate():
            # Process prefix with latent thinking
            prompt = f"You are a writer. Continue the following text:\n\n{prefix}\n\nContinue:"

            encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            # Generate latent steps
            past_kv = self.model.generate_latent_batch(
                input_ids,
                attention_mask=attention_mask,
                latent_steps=self.latent_steps,
                past_key_values=None,
            )

            # Generate text
            gen, _ = self.model.generate_text_batch(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_length,
                temperature=0.7,
                top_p=0.9,
                past_key_values=past_kv,
            )

            return gen[0]

        generated, time_ms = measure_time(generate)

        # Evaluate
        metrics = evaluate_pg19_generation(
            generated=generated,
            reference=task.gold_answer,
            prefix=prefix,
        )

        # Payload: KV cache size (approximate)
        # Qwen3-4B: 36 layers, d_model=2560
        kv_tokens = max_length + len(prefix.split())
        payload_bytes = kv_tokens * 2 * 36 * 2560 * 2  # K+V, layers, d_model, fp16

        return MethodResult(
            method_name="latentmas_qwen_long",
            answer=generated,
            is_correct=True,
            time_ms=time_ms,
            payload_bytes=payload_bytes,
            metadata={
                **metrics,
                "latent_steps": self.latent_steps,
                "kv_tokens": kv_tokens,
            },
        )
