from __future__ import annotations
import sys
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


class BestLocalAgentMethod:
    """Best local agent: each agent runs independently, pick the best answer (oracle selection)."""

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

    def _run_single_agent(self, agent_text: str, query: str) -> str:
        prompt = f"{agent_text}\n\n{query}\nAnswer:"
        ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)

        out = self.model.rwkv_model(
            input_ids=ids, attention_mask=torch.ones_like(ids).bool(),
            use_cache=True, return_dict=True,
        )
        past = out.past_key_values
        logits = out.logits[0, -1]

        new_ids = []
        all_ids = list(ids[0].tolist())
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

    def run(self, task: TaskSpec) -> MethodResult:
        def process():
            answers = []
            for agent in task.agents:
                ans = self._run_single_agent(agent.text, task.query)
                answers.append(ans)
            return answers

        answers, time_ms = measure_time(process)

        is_correct = any(self._check_answer(task, ans) for ans in answers)
        payload_bytes = sum(len(a.text.encode("utf-8")) for a in task.agents)

        return MethodResult(
            method_name="best_local_agent", answer=answers[0].strip() if answers else "",
            is_correct=is_correct, time_ms=time_ms, payload_bytes=payload_bytes,
            metadata={"n_agents": len(task.agents), "all_answers": answers},
        )

    def _check_answer(self, task: TaskSpec, predicted: str) -> bool:
        if task.task_name == "gsm8k":
            return check_gsm8k_answer(predicted, task.gold_answer)
        elif task.task_name == "hiddenbench":
            return check_hiddenbench_answer(predicted, task.gold_answer, task.metadata.get("possible_answers", []))
        elif task.task_name == "stepgame":
            return check_stepgame_answer(predicted, task.gold_answer)
        return False
