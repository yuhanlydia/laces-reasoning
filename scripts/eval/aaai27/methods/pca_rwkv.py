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


class PCARWKVMethod:
    """PCA fusion: encode agents → SVD → remove top-K PCs → fuse residuals → resample → decode."""

    def __init__(
        self,
        ckpt_dir: str = "outputs_relay/C-LDLM-4096-coadapt-8h200-b5-rnn-20260708/step_00022500",
        device: str = "cuda:0",
        pca_k: int = 1,
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
        self.pca_k = pca_k
        self.encode_prefix = encode_prefix
        self.sample_trajectory_cfg = sample_trajectory_cfg

    def _pca_remove_shared(self, z_all: torch.Tensor, K: int):
        z_mean = z_all.mean(dim=0, keepdim=True)
        z_centered = z_all - z_mean
        if K == 0 or z_centered.shape[0] <= K:
            return z_centered, z_mean
        U, S, Vh = torch.linalg.svd(z_centered, full_matrices=False)
        shared_dirs = Vh[:K]
        proj_scores = z_centered @ shared_dirs.T
        shared_part = proj_scores @ shared_dirs
        z_residuals = z_centered - shared_part
        return z_residuals, z_mean

    def run(self, task: TaskSpec) -> MethodResult:
        all_z = []
        def process():
            nonlocal all_z
            for agent in task.agents:
                ids = self.tokenizer(agent.text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
                z = self.encode_prefix(self.model, ids, torch.ones_like(ids))[0].to(self.dtype)
                all_z.append(z[0].float())

            z_stack = torch.stack(all_z, dim=0).to(self.device)
            z_residuals, z_mean = self._pca_remove_shared(z_stack, self.pca_k)

            fused_resid = z_residuals.mean(dim=0, keepdim=True)
            cond_pca = (z_mean + fused_resid).to(self.dtype)

            Z = self.sample_trajectory_cfg(
                self.model, cond_pca, steps=100, cfg_scale=3.0,
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
            method_name=f"pca_k{self.pca_k}", answer=answer.strip(),
            is_correct=is_correct, time_ms=time_ms, payload_bytes=z_bytes,
            metadata={"pca_k": self.pca_k, "n_agents": len(task.agents)},
        )

    def _check_answer(self, task: TaskSpec, predicted: str) -> bool:
        if task.task_name == "gsm8k":
            return check_gsm8k_answer(predicted, task.gold_answer)
        elif task.task_name == "hiddenbench":
            return check_hiddenbench_answer(predicted, task.gold_answer, task.metadata.get("possible_answers", []))
        elif task.task_name == "stepgame":
            return check_stepgame_answer(predicted, task.gold_answer)
        return False
