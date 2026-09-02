from __future__ import annotations
import sys
import copy
import torch
from pathlib import Path
from typing import List

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from scripts.eval.aaai27.common import TaskSpec, MethodResult, measure_time
from scripts.eval.aaai27.tasks.gsm8k import check_gsm8k_answer
from scripts.eval.aaai27.tasks.hiddenbench import check_hiddenbench_answer
from scripts.eval.aaai27.tasks.stepgame import check_stepgame_answer


class StateOnlyRWKVMethod:
    """State-only: sequential prefill → extract recurrent state → inject + decode question."""

    def __init__(
        self,
        ckpt_dir: str = "outputs_relay/C-LDLM-4096-coadapt-8h200-b5-rnn-20260708/step_00022500",
        device: str = "cuda:0",
    ):
        from scripts.eval.relay_utils import load_relay_model
        from scripts.eval.exp_multiagent_star_fusion import capture_final_state, decode_chunkwise
        self.device = torch.device(device)
        self.model, self.rwkv, self.tokenizer, self.ckpt, self.cfg = load_relay_model(
            ckpt_dir, device=str(self.device)
        )
        self.model.eval()
        self.model._prefix_suffix_trajectory_s2 = True
        self.dtype = next(self.model.alpha_heads.parameters()).dtype
        self.H = int(self.model.trajectory_horizon)
        self.capture_final_state = capture_final_state
        self.decode_chunkwise = decode_chunkwise
        self.state_bytes = self._estimate_state_bytes()

    def _estimate_state_bytes(self) -> int:
        dummy_ids = torch.tensor([[1, 2, 3]], device=self.device)
        out = self.model.rwkv_model(
            input_ids=dummy_ids, attention_mask=torch.ones_like(dummy_ids).bool(),
            use_cache=True, return_dict=True,
        )
        total = 0
        for layer in out.past_key_values.layers:
            state = layer.state.get("recurrent_state") if layer.state is not None else None
            if state is not None and isinstance(state, torch.Tensor):
                total += state.numel() * 4
        return total

    def run(self, task: TaskSpec) -> MethodResult:
        def process():
            past = None
            for agent in task.agents:
                ids = self.tokenizer(agent.text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
                out = self.model.rwkv_model(
                    input_ids=ids, attention_mask=torch.ones_like(ids).bool(),
                    past_key_values=past, use_cache=True, return_dict=True,
                )
                past = out.past_key_values

            states = []
            for layer in past.layers:
                s = layer.state.get("recurrent_state") if layer.state is not None else None
                states.append(s.float().clone() if isinstance(s, torch.Tensor) else None)

            q_ids = self.tokenizer(f"Question: {task.query}\nAnswer:", return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)

            from types import SimpleNamespace
            dec_args = SimpleNamespace(
                device=str(self.device), max_new_tokens=128,
                temperature=0.7, top_k=10, top_p=0.75, repetition_penalty=1.3,
            )
            new_ids = self.decode_chunkwise(
                self.model, self.tokenizer, q_ids,
                lambda h: states, self.H, dec_args,
            )
            return self.tokenizer.decode(new_ids, skip_special_tokens=True)

        answer, time_ms = measure_time(process)
        is_correct = self._check_answer(task, answer)

        return MethodResult(
            method_name="state_only", answer=answer.strip(),
            is_correct=is_correct, time_ms=time_ms, payload_bytes=self.state_bytes,
            metadata={"state_bytes": self.state_bytes},
        )

    def _check_answer(self, task: TaskSpec, predicted: str) -> bool:
        if task.task_name == "gsm8k":
            return check_gsm8k_answer(predicted, task.gold_answer)
        elif task.task_name == "hiddenbench":
            return check_hiddenbench_answer(predicted, task.gold_answer, task.metadata.get("possible_answers", []))
        elif task.task_name == "stepgame":
            return check_stepgame_answer(predicted, task.gold_answer)
        return False
