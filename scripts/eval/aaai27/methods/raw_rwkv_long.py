"""Raw RWKV long-form generation method."""
from __future__ import annotations
import sys
import torch
from pathlib import Path
from typing import Dict, Any

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from scripts.eval.aaai27.common import TaskSpec, MethodResult, measure_time
from scripts.eval.aaai27.tasks.pg19 import evaluate_pg19_generation


class RawRWKVLongMethod:
    """Raw RWKV: direct autoregressive generation."""

    def __init__(
        self,
        ckpt_dir: str = "outputs_relay/C-LDLM-4096-coadapt-8h200-b5-rnn-20260708/step_00022500",
        device: str = "cuda:0",
    ):
        from scripts.eval.relay_utils import load_relay_model
        self.device = torch.device(device)
        self.model, self.rwkv, self.tokenizer, self.ckpt, self.cfg = load_relay_model(
            ckpt_dir, device=str(self.device)
        )
        self.model.eval()

    def run(self, task: TaskSpec) -> MethodResult:
        prefix = task.agents[0].text
        max_length = task.metadata["generation_length"]

        def generate():
            # Tokenize prefix
            input_ids = self.tokenizer(
                prefix, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.device)

            # Generate
            with torch.no_grad():
                output = self.model.rwkv_model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                )

            # Decode
            generated = self.tokenizer.decode(
                output[0][input_ids.shape[1]:],
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
            method_name="raw_rwkv_long",
            answer=generated,
            is_correct=True,  # Not applicable for generation
            time_ms=time_ms,
            payload_bytes=payload_bytes,
            metadata=metrics,
        )
