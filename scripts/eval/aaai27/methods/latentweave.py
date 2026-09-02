#!/usr/bin/env python3
"""Method: LatentWeave (plan + recurrent state injection)

Our core method using frozen RWKV-7 2.9B.
Each agent:
1. Encodes their text into latent plan z ∈ R^32
2. Sequentially passes recurrent state S (fixed size, ~20MB)
3. Final agent decodes answer from injected state

Key advantage: communication cost is FIXED (state size),
not growing with context length or number of agents.
"""
from __future__ import annotations
import sys
import copy
import torch
from pathlib import Path
from typing import List

from scripts.eval.aaai27.common import TaskSpec, MethodResult, measure_time
from scripts.eval.aaai27.tasks.gsm8k import check_gsm8k_answer
from scripts.eval.aaai27.tasks.hiddenbench import check_hiddenbench_answer
from scripts.eval.aaai27.tasks.stepgame import check_stepgame_answer


REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))


class LatentWeaveMethod:
    """LatentWeave with frozen RWKV-7 2.9B: plan + state injection."""

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

        # State size: 32 layers × state_size × 4 bytes (fp32)
        # For RWKV-7 2.9B: approximately 20MB
        self.state_bytes = self._estimate_state_bytes()

    def _estimate_state_bytes(self) -> int:
        """Estimate recurrent state size in bytes."""
        # Run a dummy forward pass to get state shape
        dummy_ids = torch.tensor([[1, 2, 3]], device=self.device)
        out = self.model.rwkv_model(
            input_ids=dummy_ids,
            attention_mask=torch.ones_like(dummy_ids).bool(),
            use_cache=True,
            return_dict=True,
        )
        past = out.past_key_values
        total = 0
        for layer in past.layers:
            state = layer.state.get("recurrent_state") if layer.state is not None else None
            if state is not None and isinstance(state, torch.Tensor):
                total += state.numel() * 4  # fp32
        return total

    def run(self, task: TaskSpec, use_plan: bool = True) -> MethodResult:
        """Run LatentWeave on a task."""
        def process_agents():
            # Step 1: Encode all agent texts into latent plans
            all_z = []
            for agent in task.agents:
                agent_ids = self.tokenizer(
                    agent.text, return_tensors="pt", add_special_tokens=False
                ).input_ids.to(self.device)
                z = self.encode_prefix(
                    self.model, agent_ids, torch.ones_like(agent_ids)
                )[0].to(self.dtype)
                all_z.append(z[0].float())

            # Step 2: Compute fused plan (residual fusion)
            z_mean = torch.stack(all_z, dim=0).mean(dim=0, keepdim=True).to(self.dtype)
            w = 1.0 / len(all_z)
            resid = [z - z_mean for z in all_z]
            cond = z_mean + sum((w * r for r in resid), torch.zeros_like(z_mean))

            # Step 3: Sample trajectory from fused plan
            if use_plan:
                Z = self.sample_trajectory_cfg(
                    self.model, cond, steps=100, cfg_scale=3.0,
                    device=self.device, dtype=self.dtype
                )
                plan_states = [self.model.predict_states(Z[:, h]) for h in range(self.H)]
            else:
                plan_states = None

            # Step 4: Sequential state carryover with plan injection
            past = None
            for i, agent in enumerate(task.agents):
                agent_ids = self.tokenizer(
                    agent.text, return_tensors="pt", add_special_tokens=False
                ).input_ids.to(self.device)

                # Inject plan state at first agent
                if i == 0 and plan_states is not None:
                    dummy = self.model.rwkv_model(
                        input_ids=agent_ids[:, :1],
                        attention_mask=torch.ones_like(agent_ids[:, :1]).bool(),
                        use_cache=True,
                        return_dict=True,
                    )
                    past = self.model.inject_into_cache(
                        dummy.past_key_values, plan_states[0]
                    )

                # Prefill agent text
                out = self.model.rwkv_model(
                    input_ids=agent_ids,
                    attention_mask=torch.ones_like(agent_ids).bool(),
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                past = out.past_key_values

            # Step 5: Final agent generates answer
            q_ids = self.tokenizer(
                f"Question: {task.query}\nAnswer:",
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids.to(self.device)

            out = self.model.rwkv_model(
                input_ids=q_ids,
                attention_mask=torch.ones_like(q_ids).bool(),
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = out.past_key_values

            # Decode answer
            logits = out.logits[0, -1]
            new_ids = []
            all_ids = list(q_ids[0].tolist())
            eos_id = getattr(self.tokenizer, "eos_token_id", None)

            for _ in range(128):
                # Sample next token
                probs = torch.softmax(logits / 0.7, dim=-1)
                nid = torch.multinomial(probs, 1).item()
                if eos_id is not None and nid == eos_id:
                    break
                new_ids.append(nid)
                all_ids.append(nid)

                out = self.model.rwkv_model(
                    input_ids=torch.tensor([[nid]], device=self.device),
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                past = out.past_key_values
                logits = out.logits[0, -1]

            return self.tokenizer.decode(new_ids, skip_special_tokens=True)

        answer, time_ms = measure_time(process_agents)

        # Check correctness
        is_correct = self._check_answer(task, answer)

        # Payload: fixed state size (doesn't grow with agents)
        payload_bytes = self.state_bytes

        return MethodResult(
            method_name="latentweave",
            answer=answer.strip(),
            is_correct=is_correct,
            time_ms=time_ms,
            payload_bytes=payload_bytes,
            metadata={"state_bytes": self.state_bytes},
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
