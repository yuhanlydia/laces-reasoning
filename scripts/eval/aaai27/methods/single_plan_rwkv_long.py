"""RWKV + single latent plan method."""
from __future__ import annotations
import sys
import torch
from pathlib import Path
from typing import Dict, Any

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from scripts.eval.aaai27.common import TaskSpec, MethodResult, measure_time
from scripts.eval.aaai27.tasks.pg19 import evaluate_pg19_generation


class SinglePlanRWKVLongMethod:
    """RWKV + single latent plan: encode prefix once, use throughout."""

    def __init__(
        self,
        ckpt_dir: str = "outputs_relay/C-LDLM-4096-coadapt-8h200-b5-rnn-20260708/step_00022500",
        device: str = "cuda:0",
    ):
        from scripts.eval.relay_utils import load_relay_model
        from scripts.eval.sample_prefix_suffix_trajectory_cfg import (
            encode_prefix, sample_trajectory_cfg,
        )
        self.device = torch.device(device)
        self.model, self.rwkv, self.tokenizer, self.ckpt, self.cfg = load_relay_model(
            ckpt_dir, device=str(self.device)
        )
        self.model.eval()
        self.model._prefix_suffix_trajectory_s2 = True
        self.dtype = next(self.model.alpha_heads.parameters()).dtype
        self.H = int(self.model.trajectory_horizon)
        self.encode_prefix = encode_prefix
        self.sample_trajectory_cfg = sample_trajectory_cfg

    def run(self, task: TaskSpec) -> MethodResult:
        prefix = task.agents[0].text
        max_length = task.metadata["generation_length"]

        def generate():
            # Encode prefix to latent
            prefix_ids = self.tokenizer(
                prefix, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.device)

            with torch.no_grad():
                z = self.encode_prefix(self.model, prefix_ids, torch.ones_like(prefix_ids))[0].to(self.dtype)

                # Sample trajectory once
                Z = self.sample_trajectory_cfg(
                    self.model, z.unsqueeze(0), steps=100, cfg_scale=3.0,
                    device=self.device, dtype=self.dtype,
                )

                # Get plan states
                plan_states = [self.model.predict_states(Z[:, h]) for h in range(self.H)]

                # Inject first plan state
                dummy = self.model.rwkv_model(
                    input_ids=prefix_ids[:, :1],
                    attention_mask=torch.ones_like(prefix_ids[:, :1]).bool(),
                    use_cache=True, return_dict=True,
                )
                past = self.model.inject_into_cache(dummy.past_key_values, plan_states[0])

                # Prefill rest of prefix
                out = self.model.rwkv_model(
                    input_ids=prefix_ids,
                    attention_mask=torch.ones_like(prefix_ids).bool(),
                    past_key_values=past,
                    use_cache=True, return_dict=True,
                )
                past = out.past_key_values

                logits = out.logits[0, -1]
                generated_tokens = []
                all_ids = list(prefix_ids[0].tolist())
                eos_id = getattr(self.tokenizer, "eos_token_id", None)

                for _ in range(max_length):
                    probs = torch.softmax(logits / 0.7, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                    if eos_id is not None and next_token == eos_id:
                        break
                    generated_tokens.append(next_token)
                    all_ids.append(next_token)
                    cur_ids = torch.tensor([all_ids[-1:]], device=self.device)
                    out2 = self.model.rwkv_model(
                        input_ids=cur_ids,
                        past_key_values=past,
                        use_cache=True, return_dict=True,
                    )
                    past = out2.past_key_values
                    logits = out2.logits[0, -1]

            generated = self.tokenizer.decode(
                generated_tokens,
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

        # Payload: latent z (32 dimensions × 4 bytes)
        payload_bytes = 32 * 4

        return MethodResult(
            method_name="single_plan_rwkv_long",
            answer=generated,
            is_correct=True,
            time_ms=time_ms,
            payload_bytes=payload_bytes,
            metadata=metrics,
        )
