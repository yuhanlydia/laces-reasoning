"""Recurrent reasoning through an OWT-pretrained LACES state writer."""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from models.state_hijacking_dit import _cache_layer_state


_WRITER_PREFIXES = ("s1_trunk.", "s1_u_head.", "s1_v_head.", "state_scale")


def native_reasoning_query_ids(tokenizer, *, min_tokens: int = 64) -> tuple[list[int], str]:
    """Build a meaningful suffix long enough to keep FLA state gradients enabled."""
    if min_tokens <= 0:
        raise ValueError("min_tokens must be positive")
    instruction = (
        "Review the question and every option above. Work through the relevant facts "
        "internally, compare all choices carefully, and reject choices that conflict "
        "with the evidence. Return only the capital letter of the best option."
    )
    filler = " Check the conclusion once more before selecting the final option."
    while True:
        text = instruction + "\nAnswer:"
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) >= min_tokens:
            return list(ids), text
        instruction += filler


def select_pretrained_writer_parameters(relay: nn.Module) -> dict[str, nn.Parameter]:
    """Freeze RELAY, then expose the exact dynlowrank writer loaded from its checkpoint."""
    for parameter in relay.parameters():
        parameter.requires_grad_(False)
    selected = {
        name: parameter
        for name, parameter in relay.named_parameters()
        if name == "state_scale" or name.startswith(_WRITER_PREFIXES[:-1])
    }
    if not selected or not all(any(name.startswith(prefix) for name in selected) for prefix in _WRITER_PREFIXES):
        raise ValueError("checkpoint does not contain a complete dynlowrank writer")
    for parameter in selected.values():
        parameter.requires_grad_(True)
    return selected


class OWTLatentTransition(nn.Module):
    """Shared recurrent computation initialized from the OWT encoder latent."""

    def __init__(
        self, *, hidden_dim: int, z_dim: int = 32, context_dim: int = 128,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        if context_dim % attention_heads:
            raise ValueError("context_dim must be divisible by attention_heads")
        self.hidden_projection = nn.Linear(hidden_dim, context_dim)
        self.query_projection = nn.Linear(z_dim, context_dim)
        self.attention = nn.MultiheadAttention(
            context_dim, attention_heads, batch_first=True
        )
        self.step_input = nn.Sequential(
            nn.Linear(context_dim + z_dim, context_dim), nn.GELU()
        )
        self.step_cell = nn.GRUCell(context_dim, z_dim)
        self.norm = nn.LayerNorm(z_dim)

    def forward(
        self, hidden_tokens: torch.Tensor, z0: torch.Tensor, *, steps: int
    ) -> list[torch.Tensor]:
        if hidden_tokens.ndim != 3 or z0.ndim != 2:
            raise ValueError("expected hidden tokens [B,T,C] and z0 [B,D]")
        if steps < 0:
            raise ValueError("steps must be non-negative")
        compute_dtype = self.hidden_projection.weight.dtype
        memory = self.hidden_projection(hidden_tokens.to(compute_dtype))
        z = z0.to(compute_dtype)
        trace = [z]
        for _ in range(steps):
            query = self.query_projection(z).unsqueeze(1)
            context, _ = self.attention(query, memory, memory, need_weights=False)
            cell_input = self.step_input(torch.cat([context[:, 0], z], dim=-1))
            z = self.norm(self.step_cell(cell_input, z))
            trace.append(z)
        return trace


def add_residual_states(
    evidence_states: Sequence[torch.Tensor], residual_states: Sequence[torch.Tensor]
) -> list[torch.Tensor]:
    """Apply writer outputs with the residual semantics used during reasoning."""
    if not evidence_states or len(evidence_states) != len(residual_states):
        raise ValueError("evidence and residual state lists must have equal non-zero length")
    written = []
    for layer, (evidence, residual) in enumerate(zip(evidence_states, residual_states)):
        if evidence.shape != residual.shape:
            raise ValueError(
                f"state shape mismatch at layer {layer}: {evidence.shape} != {residual.shape}"
            )
        written.append(evidence.detach().to(residual.dtype) + residual)
    return written


def inject_residual_into_cache(
    cache, residual_states: Sequence[torch.Tensor], *, scale: torch.Tensor | float = 1.0
):
    """Add differentiable writer residuals to an evidence-bearing RWKV cache."""
    for layer, residual in enumerate(residual_states):
        state = _cache_layer_state(cache, layer)
        evidence = state.get("recurrent_state")
        if not isinstance(evidence, torch.Tensor):
            raise ValueError(f"missing evidence recurrent state at layer {layer}")
        state["recurrent_state"] = evidence.detach() + residual.to(evidence.dtype) * scale
    return cache


def choice_token_logits(
    vocabulary_logits: torch.Tensor,
    label_token_ids: torch.Tensor,
    choice_counts: torch.Tensor,
) -> torch.Tensor:
    """Select A--J token logits and mask choices absent from each example."""
    if vocabulary_logits.ndim != 2:
        raise ValueError("vocabulary_logits must have shape [B,V]")
    selected = vocabulary_logits.index_select(-1, label_token_ids.to(vocabulary_logits.device))
    positions = torch.arange(selected.shape[-1], device=selected.device).unsqueeze(0)
    return selected.masked_fill(positions >= choice_counts.to(selected.device).unsqueeze(1), float("-inf"))
