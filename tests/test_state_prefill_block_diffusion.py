from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from typing import final

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.state_prefill_block_diffusion import (
    StatePrefillBlockDiffusion,
    corrupt_with_mask,
    sample_block_mask,
    valid_token_mask,
)


@final
@dataclass
class FakeCache:
    hidden: torch.Tensor


@final
@dataclass
class FakeConfig:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    head_dim: int


@final
@dataclass
class FakeOutput:
    logits: torch.Tensor
    past_key_values: FakeCache


@final
class FakeRWKV:
    config: FakeConfig
    emb: nn.Embedding
    in_proj: nn.Linear
    head: nn.Linear

    def __init__(self, vocab_size: int = 16, hidden_size: int = 12):
        self.config = FakeConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=2,
            head_dim=4,
        )
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.in_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        _ = recurse
        yield from self.emb.parameters()
        yield from self.in_proj.parameters()
        yield from self.head.parameters()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool = True,
        return_dict: bool = True,
    ) -> FakeOutput:
        _ = use_cache
        _ = return_dict
        batch, length = input_ids.shape
        if past_key_values is None:
            hidden = torch.zeros(batch, self.config.hidden_size, device=input_ids.device)
        else:
            if not isinstance(past_key_values, FakeCache):
                raise TypeError("past_key_values must be FakeCache")
            hidden = past_key_values.hidden
        logits: list[torch.Tensor] = []
        emb = self.emb.forward(input_ids)
        for idx in range(length):
            x = emb[:, idx]
            hidden = torch.tanh(self.in_proj.forward(torch.cat([x, hidden], dim=-1)))
            if attention_mask is not None:
                keep = attention_mask[:, idx].to(hidden.dtype).unsqueeze(-1)
                hidden = hidden * keep + past_key_values.hidden * (1 - keep) if past_key_values is not None else hidden * keep
            logits.append(self.head.forward(hidden))
        return FakeOutput(torch.stack(logits, dim=1), FakeCache(hidden))

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool = True,
        return_dict: bool = True,
    ) -> FakeOutput:
        return self.forward(input_ids, attention_mask, past_key_values, use_cache, return_dict)


def test_mask_helpers_exclude_pad_and_corrupt_selected_positions():
    tokens = torch.tensor([[1, 2, 0, 15]])
    attention = torch.tensor([[1, 1, 1, 0]])
    valid = valid_token_mask(tokens, attention_mask=attention, pad_id=15)
    assert torch.equal(valid, torch.tensor([[True, True, True, False]]))

    gen = torch.Generator().manual_seed(0)
    mask = sample_block_mask(
        tokens,
        valid,
        min_mask_ratio=1.0,
        max_mask_ratio=1.0,
        full_mask_prob=0.0,
        eos_id=0,
        generator=gen,
    )
    assert torch.equal(mask, torch.tensor([[True, True, True, False]]))
    corrupted = corrupt_with_mask(tokens, mask, mask_id=14)
    assert torch.equal(corrupted, torch.tensor([[14, 14, 14, 15]]))


def test_state_prefill_forward_uses_masked_loss_only():
    wrapper = StatePrefillBlockDiffusion(
        FakeRWKV(),
        mask_id=14,
        pad_id=15,
        block_size=4,
        min_mask_ratio=1.0,
        max_mask_ratio=1.0,
        full_mask_prob=0.0,
    )
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 0, 15], [7, 8, 9, 10, 11, 12, 13, 15]])
    attention = tokens.ne(15)

    out = wrapper.forward(tokens, attention_mask=attention, generator=torch.Generator().manual_seed(2))

    assert out.logits.shape == (2, 8, 16)
    assert out.targets.shape == tokens.shape
    assert out.corrupted.shape == tokens.shape
    assert out.n_loss_tokens.item() == attention.sum().item()
    assert torch.isfinite(out.loss)
    assert out.loss.detach().item() > 0


def test_denoise_and_generate_ids_shapes():
    wrapper = StatePrefillBlockDiffusion(
        FakeRWKV(),
        mask_id=14,
        pad_id=15,
        block_size=4,
        min_mask_ratio=1.0,
        max_mask_ratio=1.0,
    )
    prompt = torch.tensor([[1, 2]])
    generated = wrapper.generate_ids(prompt, gen_len=6, steps=2, temperature=0.0, strategy="linear")

    assert generated.shape == (1, 6)
    assert not generated.eq(14).any()
    assert not generated.eq(15).any()
