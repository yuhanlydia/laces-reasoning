from __future__ import annotations
import sys
import torch
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from scripts.eval.aaai27.common import TaskSpec, MethodResult, measure_time
from scripts.eval.aaai27.tasks.gsm8k import check_gsm8k_answer
from scripts.eval.aaai27.tasks.hiddenbench import check_hiddenbench_answer
from scripts.eval.aaai27.tasks.stepgame import check_stepgame_answer


class DirectRelayRWKVMethod:
    """Direct relay: z_{t+1} = sample(z_t), no anchor, no renorm. Expected to diverge."""

    def __init__(
        self,
        ckpt_dir: str = "outputs_relay/C-LDLM-4096-coadapt-8h200-b5-rnn-20260708/step_00022500",
        device: str = "cuda:0",
    ):
        from scripts.eval.relay_utils import load_relay_model
        from scripts.eval.sample_prefix_suffix_trajectory_cfg import encode_prefix, sample_trajectory_cfg
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
        all_z = []

        all_z = []
        def process():
            for agent in task.agents:
                ids = self.tokenizer(agent.text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
                z = self.encode_prefix(self.model, ids, torch.ones_like(ids))[0].to(self.dtype)
                all_z.append(z[0].float())

            z_cur = all_z[0]
            for i in range(1, len(all_z)):
                Z = self.sample_trajectory_cfg(
                    self.model, z_cur.unsqueeze(0).to(self.dtype), steps=100, cfg_scale=3.0,
                    device=self.device, dtype=self.dtype,
                )
                z_cur = Z[0].float().mean(dim=0)

            Z = self.sample_trajectory_cfg(
                self.model, z_cur.unsqueeze(0).to(self.dtype), steps=100, cfg_scale=3.0,
                device=self.device, dtype=self.dtype,
            )
            plan_states = [self.model.predict_states(Z[:, h]) for h in range(self.H)]

            q_ids = self.tokenizer(f"Question: {task.query}\nAnswer:", return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            dummy = self.model.rwkv_model(
                input_ids=q_ids[:, :1], attention_mask=torch.ones_like(q_ids[:, :1]).bool(),
                use_cache=True, return_dict=True,
            )
            past = self.model.inject_into_cache(dummy.past_key_values, plan_states[0])

            out = self.model.rwkv_model(
                input_ids=q_ids, attention_mask=torch.ones_like(q_ids).bool(),
                past_key_values=past, use_cache=True, return_dict=True,
            )
            past = out.past_key_values
            logits = out.logits[0, -1]

            new_ids = []
            all_ids = list(q_ids[0].tolist())
            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            for _ in range(128):
                probs = torch.softmax(logits / 0.7, dim=-1)
                nid = torch.multinomial(probs, 1).item()
                if eos_id is not None and nid == eos_id:
                    break
                new_ids.append(nid)
                all_ids.append(nid)
                out = self.model.rwkv_model(
                    input_ids=torch.tensor([[nid]], device=self.device),
                    past_key_values=past, use_cache=True, return_dict=True,
                )
                past = out.past_key_values
                logits = out.logits[0, -1]

            return self.tokenizer.decode(new_ids, skip_special_tokens=True)

        answer, time_ms = measure_time(process)
        is_correct = self._check_answer(task, answer)
        z_bytes = sum(z.numel() * 4 for z in all_z)

        return MethodResult(
            method_name="direct_relay", answer=answer.strip(),
            is_correct=is_correct, time_ms=time_ms, payload_bytes=z_bytes,
            metadata={"anchor": 0.0, "renorm": 0.0, "n_agents": len(task.agents)},
        )

    def _check_answer(self, task: TaskSpec, predicted: str) -> bool:
        if task.task_name == "gsm8k":
            return check_gsm8k_answer(predicted, task.gold_answer)
        elif task.task_name == "hiddenbench":
            return check_hiddenbench_answer(predicted, task.gold_answer, task.metadata.get("possible_answers", []))
        elif task.task_name == "stepgame":
            return check_stepgame_answer(predicted, task.gold_answer)
        return False
