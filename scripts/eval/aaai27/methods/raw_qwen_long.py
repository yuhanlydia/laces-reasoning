"""Raw Qwen long-form generation method."""
from __future__ import annotations
import sys
import torch
from pathlib import Path
from typing import Dict, Any

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from scripts.eval.aaai27.common import TaskSpec, MethodResult, measure_time
from scripts.eval.aaai27.tasks.pg19 import evaluate_pg19_generation


class RawQwenLongMethod:
    """Raw Qwen: direct autoregressive generation."""

    def __init__(
        self,
        model_path: str = None,
        device: str = "cuda:0",
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer
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
        prefix = task.agents[0].text
        max_length = task.metadata["generation_length"]

        def generate():
            # Tokenize prefix
            inputs = self.tokenizer(
                prefix, return_tensors="pt", add_special_tokens=False
            ).to(self.device)

            # Generate
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                )

            # Decode
            generated = self.tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            return generated

        generated, time_ms = measure_time(generate)

        # Evaluate
        metrics = evaluate_pg19_generation(
            generated=generated,
            reference=task.gold_answer,
            prefix=prefix,
        )

        # Payload: just the prefix (no communication)
        payload_bytes = len(prefix.encode('utf-8'))

        return MethodResult(
            method_name="raw_qwen_long",
            answer=generated,
            is_correct=True,
            time_ms=time_ms,
            payload_bytes=payload_bytes,
            metadata=metrics,
        )
