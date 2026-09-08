from __future__ import annotations

import copy
from collections.abc import Iterator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Literal, Protocol, cast, final

try:
    from typing import override
except ImportError:  # Python 3.10/3.11; project metadata supports >=3.10.
    from typing_extensions import override

import torch
import torch.nn as nn
import torch.nn.functional as F


CommitStrategy = Literal["all", "linear", "threshold"]


class RWKVOutput(Protocol):
    @property
    def logits(self) -> torch.Tensor: ...

    @property
    def past_key_values(self) -> object | None: ...


class RWKVLike(Protocol):
    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]: ...

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool = True,
        return_dict: bool = True,
    ) -> RWKVOutput: ...


@dataclass
class StatePrefillOutput:
    loss: torch.Tensor
    denoise_loss: torch.Tensor
    state_loss: torch.Tensor
    logit_loss: torch.Tensor
    ar_loss: torch.Tensor
    logits: torch.Tensor
    targets: torch.Tensor
    loss_mask: torch.Tensor
    corrupted: torch.Tensor
    n_loss_tokens: torch.Tensor


@dataclass
class DenoiseBlockOutput:
    tokens: torch.Tensor
    steps_used: int
    final_confidence: torch.Tensor


def clone_rwkv_state(state: object | None) -> object | None:
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        return state.clone()
    if isinstance(state, dict):
        state_dict = cast(dict[object, object], state)
        return {key: clone_rwkv_state(value) for key, value in state_dict.items()}
    if isinstance(state, list):
        state_list = cast(list[object], state)
        return [clone_rwkv_state(value) for value in state_list]
    if isinstance(state, tuple):
        state_tuple = cast(tuple[object, ...], state)
        return tuple(clone_rwkv_state(value) for value in state_tuple)
    state_vars_obj = getattr(state, "__dict__", None)
    if isinstance(state_vars_obj, dict):
        state_vars = cast(dict[str, object], state_vars_obj)
        cloned = copy.copy(state)
        cloned_vars_obj = getattr(cloned, "__dict__", None)
        if isinstance(cloned_vars_obj, dict):
            cloned_vars = cast(dict[str, object], cloned_vars_obj)
            for key, value in state_vars.items():
                cloned_vars[key] = clone_rwkv_state(value)
            return cloned
    return state


def _state_tensor_pairs(student: object | None, teacher: object | None) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    if isinstance(student, torch.Tensor) and isinstance(teacher, torch.Tensor):
        if student.shape == teacher.shape:
            yield student, teacher
        return
    if isinstance(student, dict) and isinstance(teacher, dict):
        student_dict = cast(dict[object, object], student)
        teacher_dict = cast(dict[object, object], teacher)
        for key, student_value in student_dict.items():
            if key in teacher_dict:
                yield from _state_tensor_pairs(student_value, teacher_dict[key])
        return
    if isinstance(student, list) and isinstance(teacher, list):
        student_list = cast(list[object], student)
        teacher_list = cast(list[object], teacher)
        for student_value, teacher_value in zip(student_list, teacher_list, strict=False):
            yield from _state_tensor_pairs(student_value, teacher_value)
        return
    if isinstance(student, tuple) and isinstance(teacher, tuple):
        student_tuple = cast(tuple[object, ...], student)
        teacher_tuple = cast(tuple[object, ...], teacher)
        for student_value, teacher_value in zip(student_tuple, teacher_tuple, strict=False):
            yield from _state_tensor_pairs(student_value, teacher_value)
        return
    student_obj = cast(object, student)
    teacher_obj = cast(object, teacher)
    student_vars_obj = getattr(student_obj, "__dict__", None)
    teacher_vars_obj = getattr(teacher_obj, "__dict__", None)
    if isinstance(student_vars_obj, dict) and isinstance(teacher_vars_obj, dict):
        student_vars = cast(dict[str, object], student_vars_obj)
        teacher_vars = cast(dict[str, object], teacher_vars_obj)
        for key, student_value in student_vars.items():
            if key in teacher_vars:
                yield from _state_tensor_pairs(student_value, teacher_vars[key])


def state_mse_loss(student: object | None, teacher: object | None, reference: torch.Tensor) -> torch.Tensor:
    total = reference.new_zeros((), dtype=torch.float32)
    count = reference.new_zeros((), dtype=torch.float32)
    for student_tensor, teacher_tensor in _state_tensor_pairs(student, teacher):
        total = total + F.mse_loss(student_tensor.float(), teacher_tensor.detach().float(), reduction="sum")
        count = count + student_tensor.new_tensor(student_tensor.numel(), dtype=torch.float32)
    return total / count.clamp_min(1.0)


def valid_token_mask(
    tokens: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    pad_id: int | None = None,
) -> torch.Tensor:
    valid = torch.ones_like(tokens, dtype=torch.bool)
    if attention_mask is not None:
        valid &= attention_mask.to(device=tokens.device).bool()
    if pad_id is not None:
        valid &= tokens.ne(int(pad_id))
    return valid


def sample_block_mask(
    tokens: torch.Tensor,
    valid_mask: torch.Tensor,
    min_mask_ratio: float = 0.0,
    max_mask_ratio: float = 1.0,
    full_mask_prob: float = 0.10,
    force_mask_eos: bool = True,
    eos_id: int = 0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if tokens.shape != valid_mask.shape:
        raise ValueError("tokens and valid_mask must have the same shape")
    if not (0.0 <= min_mask_ratio <= max_mask_ratio <= 1.0):
        raise ValueError("mask ratios must satisfy 0 <= min <= max <= 1")
    if not (0.0 <= full_mask_prob <= 1.0):
        raise ValueError("full_mask_prob must be in [0, 1]")

    batch, _ = tokens.shape
    device = tokens.device
    ratio = torch.empty(batch, 1, device=device).uniform_(
        float(min_mask_ratio), float(max_mask_ratio), generator=generator
    )
    if full_mask_prob > 0.0:
        full = torch.rand(batch, 1, device=device, generator=generator) < full_mask_prob
        ratio = torch.where(full, torch.ones_like(ratio), ratio)
    mask = torch.rand(tokens.shape, device=device, generator=generator) < ratio
    mask &= valid_mask
    if force_mask_eos:
        mask |= tokens.eq(int(eos_id)) & valid_mask
    return mask


def corrupt_with_mask(tokens: torch.Tensor, mask: torch.Tensor, mask_id: int) -> torch.Tensor:
    if tokens.shape != mask.shape:
        raise ValueError("tokens and mask must have the same shape")
    mask_fill = torch.full_like(tokens, int(mask_id))
    return torch.where(mask, mask_fill, tokens)


def _filter_logits(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    filtered = logits
    if top_k > 0:
        k = min(int(top_k), filtered.shape[-1])
        values, _ = torch.topk(filtered, k=k, dim=-1)
        filtered = filtered.masked_fill(filtered < values[..., -1:], float("-inf"))
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = sorted_logits.softmax(dim=-1)
        remove_sorted = sorted_probs.cumsum(dim=-1) > float(top_p)
        remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
        remove_sorted[..., 0] = False
        remove_mask = torch.zeros_like(filtered, dtype=torch.bool)
        _ = remove_mask.scatter_(-1, sorted_idx, remove_sorted)
        filtered = filtered.masked_fill(remove_mask, float("-inf"))
    return filtered


@final
class StatePrefillBlockDiffusion(nn.Module):
    rwkv_model: RWKVLike
    mask_id: int
    block_size: int
    pad_id: int | None
    eos_id: int
    min_mask_ratio: float
    max_mask_ratio: float
    full_mask_prob: float
    lambda_state: float
    lambda_logit: float
    lambda_ar: float
    state_mask_ratio_weight: float

    def __init__(
        self,
        rwkv_model: RWKVLike,
        mask_id: int,
        block_size: int = 32,
        pad_id: int | None = None,
        eos_id: int = 0,
        min_mask_ratio: float = 0.0,
        max_mask_ratio: float = 1.0,
        full_mask_prob: float = 0.10,
        lambda_state: float = 0.0,
        lambda_logit: float = 0.0,
        lambda_ar: float = 0.0,
        state_mask_ratio_weight: float = 0.0,
        freeze_rwkv: bool = False,
    ):
        nn.Module.__init__(self)  # pyright: ignore[reportUnknownMemberType]
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.rwkv_model = rwkv_model
        self.mask_id = int(mask_id)
        self.block_size = int(block_size)
        self.pad_id = None if pad_id is None else int(pad_id)
        self.eos_id = int(eos_id)
        self.min_mask_ratio = float(min_mask_ratio)
        self.max_mask_ratio = float(max_mask_ratio)
        self.full_mask_prob = float(full_mask_prob)
        self.lambda_state = float(lambda_state)
        self.lambda_logit = float(lambda_logit)
        self.lambda_ar = float(lambda_ar)
        self.state_mask_ratio_weight = float(state_mask_ratio_weight)

        for param in self.rwkv_model.parameters():
            param.requires_grad = not freeze_rwkv

    def scan(
        self,
        tokens: torch.Tensor,
        state: object | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> object | None:
        out = self.rwkv_model(
            input_ids=tokens,
            attention_mask=attention_mask.bool() if attention_mask is not None else None,
            past_key_values=state,
            use_cache=True,
            return_dict=True,
        )
        return out.past_key_values

    def scan_logits(
        self,
        tokens: torch.Tensor,
        state: object | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, object | None]:
        out = self.rwkv_model(
            input_ids=tokens,
            attention_mask=attention_mask.bool() if attention_mask is not None else None,
            past_key_values=state,
            use_cache=True,
            return_dict=True,
        )
        return out.logits, out.past_key_values

    def _iter_blocks(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor | None]]:
        length = tokens.shape[1]
        for start in range(0, length, self.block_size):
            end = min(start + self.block_size, length)
            block = tokens[:, start:end]
            block_mask = attention_mask[:, start:end] if attention_mask is not None else None
            yield block, block_mask

    @override
    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> StatePrefillOutput:
        if tokens.dim() != 2:
            raise ValueError("tokens must have shape [batch, length]")
        if attention_mask is not None and attention_mask.shape != tokens.shape:
            raise ValueError("attention_mask must match tokens shape")

        state = None
        logits_by_block: list[torch.Tensor] = []
        targets_by_block: list[torch.Tensor] = []
        masks_by_block: list[torch.Tensor] = []
        corrupted_by_block: list[torch.Tensor] = []
        total_loss = tokens.new_zeros((), dtype=torch.float32)
        total_count = tokens.new_zeros((), dtype=torch.float32)
        total_state_loss = tokens.new_zeros((), dtype=torch.float32)
        total_logit_loss = tokens.new_zeros((), dtype=torch.float32)
        total_ar_loss = tokens.new_zeros((), dtype=torch.float32)
        state_blocks = tokens.new_zeros((), dtype=torch.float32)
        logit_blocks = tokens.new_zeros((), dtype=torch.float32)
        ar_count = tokens.new_zeros((), dtype=torch.float32)

        for clean_block, block_attention in self._iter_blocks(tokens, attention_mask):
            eligible = valid_token_mask(clean_block, block_attention, self.pad_id)
            loss_mask = sample_block_mask(
                clean_block,
                eligible,
                min_mask_ratio=self.min_mask_ratio,
                max_mask_ratio=self.max_mask_ratio,
                full_mask_prob=self.full_mask_prob,
                force_mask_eos=True,
                eos_id=self.eos_id,
                generator=generator,
            )
            corrupted = corrupt_with_mask(clean_block, loss_mask, self.mask_id)

            history_state = clone_rwkv_state(state)
            prefill_state = self.scan(
                corrupted,
                state=clone_rwkv_state(history_state),
                attention_mask=block_attention,
            )
            query_logits, _ = self.scan_logits(
                corrupted,
                state=clone_rwkv_state(prefill_state),
                attention_mask=block_attention,
            )

            clean_logits_from_history: torch.Tensor | None = None
            if self.lambda_ar > 0.0:
                clean_logits_from_history, clean_state = self.scan_logits(
                    clean_block,
                    state=clone_rwkv_state(history_state),
                    attention_mask=block_attention,
                )
            else:
                clean_state = self.scan(
                    clean_block,
                    state=clone_rwkv_state(history_state),
                    attention_mask=block_attention,
                )

            if self.lambda_state > 0.0:
                mask_ratio = loss_mask.sum().to(torch.float32) / eligible.sum().to(torch.float32).clamp_min(1.0)
                state_weight = 1.0 + self.state_mask_ratio_weight * mask_ratio
                total_state_loss = total_state_loss + state_weight * state_mse_loss(prefill_state, clean_state, clean_block)
                state_blocks = state_blocks + 1.0

            if self.lambda_logit > 0.0 and loss_mask.any():
                clean_query_logits, _ = self.scan_logits(
                    clean_block,
                    state=clone_rwkv_state(clean_state),
                    attention_mask=block_attention,
                )
                student_log_probs = query_logits.float().log_softmax(dim=-1)
                teacher_probs = clean_query_logits.detach().float().softmax(dim=-1)
                flat_student = student_log_probs.reshape(-1, student_log_probs.shape[-1])
                flat_teacher = teacher_probs.reshape(-1, teacher_probs.shape[-1])
                flat_mask = loss_mask.reshape(-1)
                token_kl = F.kl_div(flat_student[flat_mask], flat_teacher[flat_mask], reduction="sum")
                total_logit_loss = total_logit_loss + token_kl / flat_mask.sum().to(token_kl.dtype).clamp_min(1.0)
                logit_blocks = logit_blocks + 1.0

            if clean_logits_from_history is not None and clean_block.shape[1] > 1:
                ar_mask = valid_token_mask(clean_block, block_attention, self.pad_id)[:, 1:]
                if ar_mask.any():
                    ar_logits = clean_logits_from_history[:, :-1].float().reshape(-1, clean_logits_from_history.shape[-1])
                    ar_targets = clean_block[:, 1:].reshape(-1)
                    flat_ar_mask = ar_mask.reshape(-1)
                    ar_token_loss = F.cross_entropy(
                        ar_logits[flat_ar_mask],
                        ar_targets[flat_ar_mask],
                        reduction="sum",
                    )
                    total_ar_loss = total_ar_loss + ar_token_loss
                    ar_count = ar_count + flat_ar_mask.sum().to(ar_count.dtype)

            if loss_mask.any():
                flat_logits = query_logits.float().reshape(-1, query_logits.shape[-1])
                flat_targets = clean_block.reshape(-1)
                flat_mask = loss_mask.reshape(-1)
                token_loss = F.cross_entropy(
                    flat_logits[flat_mask],
                    flat_targets[flat_mask],
                    reduction="sum",
                )
                total_loss = total_loss + token_loss
                total_count = total_count + flat_mask.sum().to(total_count.dtype)

            state = clean_state

            logits_by_block.append(query_logits)
            targets_by_block.append(clean_block)
            masks_by_block.append(loss_mask)
            corrupted_by_block.append(corrupted)

        denoise_loss = total_loss / total_count.clamp_min(1.0)
        state_loss = total_state_loss / state_blocks.clamp_min(1.0)
        logit_loss = total_logit_loss / logit_blocks.clamp_min(1.0)
        ar_loss = total_ar_loss / ar_count.clamp_min(1.0)
        loss = (
            denoise_loss
            + denoise_loss.new_tensor(self.lambda_state) * state_loss
            + denoise_loss.new_tensor(self.lambda_logit) * logit_loss
            + denoise_loss.new_tensor(self.lambda_ar) * ar_loss
        )
        return StatePrefillOutput(
            loss=loss,
            denoise_loss=denoise_loss,
            state_loss=state_loss,
            logit_loss=logit_loss,
            ar_loss=ar_loss,
            logits=torch.cat(logits_by_block, dim=1),
            targets=torch.cat(targets_by_block, dim=1),
            loss_mask=torch.cat(masks_by_block, dim=1),
            corrupted=torch.cat(corrupted_by_block, dim=1),
            n_loss_tokens=total_count,
        )

    @torch.no_grad()  # pyright: ignore[reportUntypedFunctionDecorator]
    def denoise_block(
        self,
        state: object | None,
        batch_size: int,
        steps: int = 8,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        strategy: CommitStrategy | str = "all",
        conf_threshold: float = 0.95,
        min_per_step: int = 0,
        self_correction: bool = False,
        remask_threshold: float = 0.25,
        device: torch.device | None = None,
    ) -> DenoiseBlockOutput:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if steps <= 0:
            raise ValueError("steps must be positive")
        device = device or next(self.rwkv_model.parameters()).device
        cur = torch.full((batch_size, self.block_size), self.mask_id, dtype=torch.long, device=device)
        still_masked = torch.ones_like(cur, dtype=torch.bool)
        final_confidence = torch.zeros_like(cur, dtype=torch.float32)
        if min_per_step <= 0:
            min_per_step = max(1, self.block_size // max(steps, 1))

        for step in range(1, steps + 1):
            if not still_masked.any():
                return DenoiseBlockOutput(cur, step - 1, final_confidence)

            prefill_state = self.scan(cur, state=clone_rwkv_state(state))
            logits, _ = self.scan_logits(cur, state=clone_rwkv_state(prefill_state))
            logits = logits.float()
            logits[..., self.mask_id] = float("-inf")
            if self.pad_id is not None:
                logits[..., self.pad_id] = float("-inf")

            raw_probs = logits.softmax(dim=-1)
            shaped_logits = logits if temperature == 1.0 else logits / max(float(temperature), 1e-6)
            shaped_logits = _filter_logits(shaped_logits, top_k=top_k, top_p=top_p)
            shaped_probs = shaped_logits.softmax(dim=-1)
            if temperature > 0.0:
                pred = torch.multinomial(shaped_probs.reshape(-1, shaped_probs.shape[-1]), 1).view_as(cur)
            else:
                pred = shaped_probs.argmax(dim=-1)
            confidence = raw_probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1)
            reconsider = torch.zeros_like(still_masked)
            if self_correction and step < steps:
                current_confidence = raw_probs.gather(-1, cur.unsqueeze(-1)).squeeze(-1)
                reconsider = (~still_masked) & (current_confidence < float(remask_threshold))
            update_mask = still_masked | reconsider
            confidence = confidence.masked_fill(~update_mask, float("-inf"))

            if step == steps:
                commit = update_mask
            elif strategy == "all":
                commit = update_mask
            elif strategy == "linear":
                target_clean = min(int(self.block_size * step // steps), self.block_size)
                current_clean = (~update_mask).sum(dim=1)
                commit = torch.zeros_like(still_masked)
                for row in range(batch_size):
                    n_commit = max(0, target_clean - int(current_clean[row].item()))
                    if n_commit:
                        _, idx = torch.topk(confidence[row], k=n_commit)
                        commit[row, idx] = True
            elif strategy == "threshold":
                commit = (confidence > float(conf_threshold)) & update_mask
                for row in range(batch_size):
                    if int(commit[row].sum().item()) < min_per_step:
                        k = min(min_per_step, int(update_mask[row].sum().item()))
                        if k > 0:
                            _, idx = torch.topk(confidence[row], k=k)
                            commit[row, idx] = True
            else:
                raise ValueError(f"unknown commit strategy: {strategy}")

            cur = torch.where(commit, pred, cur)
            reopened = reconsider & ~commit
            cur = torch.where(reopened, torch.full_like(cur, self.mask_id), cur)
            final_confidence = torch.where(commit, confidence.clamp_min(0.0), final_confidence)
            final_confidence = torch.where(reopened, torch.zeros_like(final_confidence), final_confidence)
            still_masked = (still_masked | reconsider) & ~commit

        return DenoiseBlockOutput(cur, steps, final_confidence)

    @torch.no_grad()  # pyright: ignore[reportUntypedFunctionDecorator]
    def generate_ids(
        self,
        prompt_ids: torch.Tensor,
        gen_len: int,
        steps: int = 8,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        strategy: CommitStrategy | str = "all",
        self_correction: bool = False,
        remask_threshold: float = 0.25,
    ) -> torch.Tensor:
        if prompt_ids.dim() != 2:
            raise ValueError("prompt_ids must have shape [batch, prompt_len]")
        if gen_len <= 0:
            return prompt_ids.new_empty((prompt_ids.shape[0], 0))

        batch_size = prompt_ids.shape[0]
        state = None
        if prompt_ids.shape[1] > 0:
            state = self.scan(prompt_ids)

        blocks: list[torch.Tensor] = []
        remaining = int(gen_len)
        while remaining > 0:
            out = self.denoise_block(
                state,
                batch_size=batch_size,
                steps=steps,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                strategy=strategy,
                self_correction=self_correction,
                remask_threshold=remask_threshold,
                device=prompt_ids.device,
            )
            block = out.tokens[:, : min(self.block_size, remaining)]
            blocks.append(block)
            state = self.scan(block, state=state)
            remaining -= block.shape[1]
        return torch.cat(blocks, dim=1)


def make_output(logits: torch.Tensor, past_key_values: object) -> SimpleNamespace:
    return SimpleNamespace(logits=logits, past_key_values=past_key_values)
