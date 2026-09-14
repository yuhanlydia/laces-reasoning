"""Causal execution helpers for the native LACES latent trajectory.

The existing ``NativeLACES`` interface stays unchanged. These helpers operate on that
runtime and preserve its S1 scale/blend semantics while making block boundaries explicit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import copy

import torch


@dataclass
class BlockGeneration:
    token_ids: list[int]
    text: str
    block_end_offsets: list[int]
    active_block_mask: list[bool]
    potentials: list[float]


def _state_dtype(native):
    if native.model.s1_writer_type == "dynlowrank":
        return next(native.model.s1_trunk.parameters()).dtype
    return next(native.model.alpha_heads.parameters()).dtype


def _write_block(native, cache, z, block: int):
    if native.blend == 0:
        return cache
    states = native.model.predict_states(z[:, block].to(device=native.device, dtype=_state_dtype(native)))
    return (native.model.blend_into_cache(cache, states, native.blend) if native.blend < 1
            else native.model.inject_into_cache(cache, states))


@torch.no_grad()
def score_answer_from_cache(native, cache, pending, answer_ids) -> torch.Tensor:
    """Score a gold continuation from a boundary cache copy without replaying prior blocks."""
    answer_ids = answer_ids.to(native.device)
    if answer_ids.ndim != 2 or answer_ids.shape[0] != 1 or answer_ids.shape[1] < 1:
        raise ValueError("answer_ids must be one nonempty sequence")
    out = native.model.rwkv_model(input_ids=pending.to(native.device), past_key_values=cache,
                                  use_cache=True, return_dict=True)
    cache, logits = out.past_key_values, out.logits[0, -1].float()
    scores = []
    for i, token in enumerate(answer_ids[0]):
        scores.append(torch.log_softmax(logits, -1)[token])
        if i + 1 < answer_ids.shape[1]:
            out = native.model.rwkv_model(input_ids=token.view(1, 1), past_key_values=cache,
                                          use_cache=True, return_dict=True)
            cache, logits = out.past_key_values, out.logits[0, -1].float()
    return torch.stack(scores).sum()


@torch.no_grad()
def generate_blocks(native, prefix, z, *, tokens_per_block=None, max_blocks=None, eos_id=None,
                    raw=False, potential_fn: Callable[[object, torch.Tensor, list[int], int], float] | None = None) -> BlockGeneration:
    """Execute ``z_1..z_H`` causally, writing each block before its boundary anchor."""
    custom = getattr(native, "generate_blocks", None)
    if callable(custom):
        return custom(prefix, z, tokens_per_block=tokens_per_block, max_blocks=max_blocks,
                      eos_id=eos_id, raw=raw, potential_fn=potential_fn)
    prefix = prefix.to(native.device)
    z = z.to(native.device)
    tokens_per_block = int(native.chunk if tokens_per_block is None else tokens_per_block)
    max_blocks = int(native.horizon if max_blocks is None else max_blocks)
    if prefix.ndim != 2 or prefix.shape[0] != 1 or prefix.shape[1] < 2:
        raise ValueError("Expected a single prefix with at least two tokens")
    latent_dim = getattr(native.model, "latent_dim", None)
    if latent_dim is None:
        latent_dim = native.audit["latent_dim"]
    latent_dim = int(latent_dim)
    if z.shape != (1, native.horizon, latent_dim):
        raise ValueError("Wrong pretrained trajectory shape")
    if not 1 <= tokens_per_block <= native.chunk:
        raise ValueError("tokens_per_block must be in [1, native chunk size]")
    if not 1 <= max_blocks <= native.horizon:
        raise ValueError("max_blocks must be in [1, native horizon]")
    eos_id = getattr(native.tokenizer, "eos_token_id", None) if eos_id is None else eos_id
    out = native.model.rwkv_model(input_ids=prefix[:, :-1], attention_mask=torch.ones_like(prefix[:, :-1]).bool(),
                                  use_cache=True, return_dict=True)
    cache, pending = out.past_key_values, prefix[:, -1:]
    ids: list[int] = []
    ends: list[int] = []
    potentials: list[float] = []
    active = [False] * max_blocks
    stopped = False
    for h in range(max_blocks):
        if stopped:
            break
        active[h] = True
        if not raw:
            cache = _write_block(native, cache, z, h)
        for _ in range(tokens_per_block):
            out = native.model.rwkv_model(input_ids=pending, past_key_values=cache, use_cache=True, return_dict=True)
            cache = out.past_key_values
            token = int(out.logits[0, -1].float().argmax())
            ids.append(token)
            pending = torch.tensor([[token]], device=native.device, dtype=torch.long)
            if eos_id is not None and token == int(eos_id):
                stopped = True
                break
        ends.append(len(ids))
        if potential_fn is not None:
            try:
                score_cache = copy.deepcopy(cache)
            except Exception as exc:
                raise RuntimeError("Could not clone RWKV cache for boundary potential scoring") from exc
            value = float(potential_fn(score_cache, pending.detach().clone(), list(ids), h))
            if not torch.isfinite(torch.tensor(value)):
                raise FloatingPointError("Nonfinite block potential")
            potentials.append(value)
    return BlockGeneration(ids, native.tokenizer.decode(ids, skip_special_tokens=False), ends, active, potentials)


@torch.no_grad()
def score_continuation_after_blocks(native, prefix, z, generated_ids, block_end_offsets, answer_ids,
                                    *, upto_block: int, raw=False) -> torch.Tensor:
    """Score an answer continuation after replaying completed blocks ``<= upto_block``."""
    custom = getattr(native, "score_continuation_after_blocks", None)
    if callable(custom):
        return custom(prefix, z, generated_ids, block_end_offsets, answer_ids,
                      upto_block=upto_block, raw=raw)
    prefix = prefix.to(native.device)
    z = z.to(native.device)
    answer_ids = answer_ids.to(native.device)
    if not 0 <= upto_block < len(block_end_offsets) or upto_block >= native.horizon:
        raise ValueError("upto_block outside completed block range")
    if answer_ids.ndim != 2 or answer_ids.shape[0] != 1 or answer_ids.shape[1] < 1:
        raise ValueError("answer_ids must be one nonempty sequence")
    if any(b <= 0 for b in block_end_offsets) or any(a >= b for a, b in zip(block_end_offsets, block_end_offsets[1:])):
        raise ValueError("block_end_offsets must be strictly increasing positive offsets")
    if block_end_offsets[-1] > len(generated_ids):
        raise ValueError("block_end_offsets exceed generated token history")
    out = native.model.rwkv_model(input_ids=prefix[:, :-1], attention_mask=torch.ones_like(prefix[:, :-1]).bool(),
                                  use_cache=True, return_dict=True)
    cache, pending = out.past_key_values, prefix[:, -1:]
    start = 0
    for h in range(upto_block + 1):
        if not raw:
            cache = _write_block(native, cache, z, h)
        end = block_end_offsets[h]
        for token in generated_ids[start:end]:
            out = native.model.rwkv_model(input_ids=pending, past_key_values=cache, use_cache=True, return_dict=True)
            cache = out.past_key_values
            pending = torch.tensor([[int(token)]], device=native.device, dtype=torch.long)
        start = end
    out = native.model.rwkv_model(input_ids=pending, past_key_values=cache, use_cache=True, return_dict=True)
    cache, logits = out.past_key_values, out.logits[0, -1].float()
    scores = []
    for i, token in enumerate(answer_ids[0]):
        scores.append(torch.log_softmax(logits, -1)[token])
        if i + 1 < answer_ids.shape[1]:
            out = native.model.rwkv_model(input_ids=token.view(1, 1), past_key_values=cache,
                                          use_cache=True, return_dict=True)
            cache, logits = out.past_key_values, out.logits[0, -1].float()
    return torch.stack(scores).sum()
