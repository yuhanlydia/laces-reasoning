"""Variable-depth recurrent latent reasoning and dynamic state writing.

The reasoner keeps computation in a compact latent state while re-querying token-level
facts at every recurrent step.  A shared layer/head-conditioned hypernetwork maps each
latent state to a *cumulative* RWKV recurrent-state correction.  Using cumulative writes
keeps the magnitude of an intervention independent of the chosen reasoning budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

StateList = list[torch.Tensor]


def recurrent_state_summary(
    states: Sequence[torch.Tensor], *, pool_size: int | None = None
) -> torch.Tensor:
    """Compact differentiable readout used to feed written state back into z.

    With ``pool_size=None`` the function returns four scalar statistics per layer for
    backwards-compatible diagnostics.  The recurrent reasoner uses a small spatial
    grid per head instead, which preserves associative structure rather than only the
    magnitude of the memory.
    """
    if not states:
        raise ValueError("states must contain at least one layer")
    summaries = []
    batch_size = None
    device = states[0].device
    dtype = states[0].dtype if states[0].is_floating_point() else torch.float32
    for layer_index, state in enumerate(states):
        if state is None:
            if batch_size is None:
                raise ValueError("the first recurrent state cannot be None")
            summaries.append(torch.zeros(batch_size, 4, device=device, dtype=dtype))
            continue
        if not isinstance(state, torch.Tensor) or state.ndim < 2:
            raise ValueError(f"state at layer {layer_index} must be a batched tensor")
        if batch_size is None:
            batch_size = int(state.shape[0])
        elif int(state.shape[0]) != batch_size:
            raise ValueError("all recurrent states must have the same batch size")
        if pool_size is not None:
            if pool_size <= 0 or state.ndim != 4 or state.shape[-1] != state.shape[-2]:
                raise ValueError(
                    "pooled recurrent states must have shape [B,H,D,D] and positive pool_size"
                )
            pooled = F.adaptive_avg_pool2d(
                state.float().reshape(-1, 1, state.shape[-2], state.shape[-1]),
                (pool_size, pool_size),
            )
            summaries.append(pooled.reshape(state.shape[0], -1))
        else:
            flat = state.float().reshape(state.shape[0], -1)
            summaries.append(torch.stack((
                flat.mean(dim=-1),
                flat.std(dim=-1, unbiased=False),
                flat.abs().mean(dim=-1),
                flat.square().mean(dim=-1).sqrt(),
            ), dim=-1))
    if batch_size is None:
        raise ValueError("states must contain at least one tensor")
    return torch.cat(summaries, dim=-1).to(dtype=dtype)


@dataclass
class ReasoningTrace:
    """Outputs at every available inference-time reasoning budget."""

    latents: list[torch.Tensor]
    contexts: list[torch.Tensor]
    cumulative_states: list[StateList]
    attention_weights: list[torch.Tensor]


class SharedDynamicStateWriter(nn.Module):
    """Generate per-layer, per-head low-rank state directions with shared parameters.

    The old implementation used one giant output projection and shared ``V`` across all
    heads in a layer.  Here a compact hypernetwork is shared across layer/head pairs,
    while both U and V are conditioned on the latent, layer id, and head id.
    """

    def __init__(
        self,
        *,
        z_dim: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        rank: int,
        hidden_dim: int = 128,
        layer_embedding_dim: int = 16,
        head_embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        if min(z_dim, num_layers, num_heads, head_dim, rank, hidden_dim) <= 0:
            raise ValueError("all writer dimensions must be positive")

        self.z_dim = int(z_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.rank = int(rank)

        self.layer_embedding = nn.Embedding(self.num_layers, layer_embedding_dim)
        self.head_embedding = nn.Embedding(self.num_heads, head_embedding_dim)
        input_dim = self.z_dim + layer_embedding_dim + head_embedding_dim
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        # One shared head is evaluated for every (layer, head) pair. Both factors are
        # therefore per-head without introducing L*H independent projection matrices.
        self.uv_head = nn.Linear(hidden_dim, 2 * self.head_dim * self.rank)
        self.gain_head = nn.Linear(hidden_dim, 1)
        self.log_global_scale = nn.Parameter(torch.tensor(log(32.0)))

        nn.init.normal_(self.layer_embedding.weight, std=0.02)
        nn.init.normal_(self.head_embedding.weight, std=0.02)
        # Match the O(1e-2..1e-1) scale of observed RWKV state corrections at start-up.
        nn.init.constant_(self.gain_head.bias, -1.0)

        layer_ids = torch.arange(self.num_layers).view(self.num_layers, 1)
        head_ids = torch.arange(self.num_heads).view(1, self.num_heads)
        self.register_buffer(
            "_layer_ids",
            layer_ids.expand(self.num_layers, self.num_heads).reshape(-1),
            persistent=False,
        )
        self.register_buffer(
            "_head_ids",
            head_ids.expand(self.num_layers, self.num_heads).reshape(-1),
            persistent=False,
        )

    def forward(self, z: torch.Tensor) -> StateList:
        if z.ndim == 1:
            z = z.unsqueeze(0)
        if z.ndim != 2 or z.shape[-1] != self.z_dim:
            raise ValueError(f"expected z with shape [B,{self.z_dim}], got {tuple(z.shape)}")

        batch = z.shape[0]
        pairs = self.num_layers * self.num_heads
        layer_emb = self.layer_embedding(self._layer_ids).unsqueeze(0).expand(batch, -1, -1)
        head_emb = self.head_embedding(self._head_ids).unsqueeze(0).expand(batch, -1, -1)
        z_pairs = z.unsqueeze(1).expand(batch, pairs, self.z_dim)
        features = self.trunk(torch.cat([z_pairs, layer_emb, head_emb], dim=-1))

        uv = self.uv_head(features)
        u_flat, v_flat = uv.chunk(2, dim=-1)
        U = u_flat.reshape(
            batch, self.num_layers, self.num_heads, self.head_dim, self.rank
        )
        V = v_flat.reshape(
            batch, self.num_layers, self.num_heads, self.head_dim, self.rank
        )
        gain = F.softplus(self.gain_head(features)).reshape(
            batch, self.num_layers, self.num_heads, 1, 1
        )
        scale = self.log_global_scale.exp() / sqrt(float(self.rank))
        matrices = torch.matmul(U, V.transpose(-1, -2)) * gain * scale
        return [matrices[:, layer] for layer in range(self.num_layers)]


class RecurrentReasoner(nn.Module):
    """Query-conditioned recurrent computation in a compact latent space.

    Every step forms a new attention query from the current latent and the question
    representation, re-reads the fact tokens, and reads back a compact summary of the
    state written at the previous step. Parameters are shared across all steps, so a
    single checkpoint can run with different inference-time budgets.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        *,
        z_dim: int = 32,
        context_dim: int = 128,
        writer_rank: int = 32,
        writer_hidden: int = 128,
        attention_heads: int = 4,
        state_summary_dim: int | None = None,
        state_pool_size: int = 4,
    ) -> None:
        super().__init__()
        if context_dim % attention_heads != 0:
            raise ValueError("context_dim must be divisible by attention_heads")
        self.hidden_dim = int(hidden_dim)
        self.z_dim = int(z_dim)
        self.context_dim = int(context_dim)
        self.state_pool_size = int(state_pool_size)
        self.state_summary_dim = int(
            state_summary_dim or (num_layers * num_heads * self.state_pool_size**2)
        )

        self.fact_projection = nn.Linear(hidden_dim, context_dim)
        self.query_projection = nn.Linear(hidden_dim, context_dim)
        self.initial_attention_query = nn.Linear(context_dim, context_dim)
        self.state_feedback_projection = nn.Linear(self.state_summary_dim, context_dim)
        self.step_attention_query = nn.Linear(z_dim + 2 * context_dim, context_dim)
        self.cross_attention = nn.MultiheadAttention(
            context_dim, num_heads=attention_heads, batch_first=True
        )
        self.z0_head = nn.Sequential(
            nn.Linear(2 * context_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, z_dim),
        )
        self.step_input = nn.Sequential(
            nn.Linear(3 * context_dim, context_dim),
            nn.GELU(),
        )
        self.step_cell = nn.GRUCell(context_dim, z_dim)
        self.latent_norm = nn.LayerNorm(z_dim)
        self.writer = SharedDynamicStateWriter(
            z_dim=z_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            rank=writer_rank,
            hidden_dim=writer_hidden,
        )

    @staticmethod
    def _batched(tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim == 2:
            return tokens.unsqueeze(0)
        if tokens.ndim != 3:
            raise ValueError(f"expected token features [T,C] or [B,T,C], got {tokens.shape}")
        return tokens

    def forward(
        self,
        fact_tokens: torch.Tensor,
        query_tokens: torch.Tensor,
        *,
        steps: int,
        base_states: Sequence[torch.Tensor] | None = None,
        base_state_features: torch.Tensor | None = None,
    ) -> ReasoningTrace:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        facts = self._batched(fact_tokens)
        query = self._batched(query_tokens)
        if facts.shape[0] != query.shape[0]:
            raise ValueError("fact and query batches must match")

        fact_features = self.fact_projection(facts)
        query_features = self.query_projection(query)
        query_summary = query_features.mean(dim=1)

        if base_states is not None and base_state_features is not None:
            raise ValueError("pass base_states or base_state_features, not both")
        if base_state_features is not None:
            base_summary = base_state_features.to(device=facts.device, dtype=facts.dtype)
            if base_summary.ndim == 1:
                base_summary = base_summary.unsqueeze(0)
            if base_summary.shape != (facts.shape[0], self.state_summary_dim):
                raise ValueError(
                    f"expected base_state_features [{facts.shape[0]},{self.state_summary_dim}], "
                    f"got {tuple(base_summary.shape)}"
                )
            state_feedback = self.state_feedback_projection(base_summary)
        elif base_states is None:
            base_summary = None
            state_feedback = torch.zeros(
                facts.shape[0], self.context_dim,
                device=facts.device, dtype=facts.dtype,
            )
        else:
            base_states = [state.detach().to(device=facts.device) for state in base_states]
            base_summary = recurrent_state_summary(
                base_states, pool_size=self.state_pool_size
            ).to(dtype=facts.dtype)
            if base_summary.shape[-1] != self.state_summary_dim:
                raise ValueError(
                    f"expected state summary width {self.state_summary_dim}, "
                    f"got {base_summary.shape[-1]}"
                )
            state_feedback = self.state_feedback_projection(base_summary)
        query_summary = query_summary + state_feedback

        initial_query = self.initial_attention_query(query_summary).unsqueeze(1)
        initial_context, _ = self.cross_attention(
            initial_query, fact_features, fact_features, need_weights=False
        )
        z = self.z0_head(torch.cat([query_summary, initial_context[:, 0]], dim=-1))
        z = self.latent_norm(z)

        latents = [z]
        contexts: list[torch.Tensor] = []
        attention_weights: list[torch.Tensor] = []
        cumulative_states = [self.writer(z)]
        if base_states is not None or base_summary is not None:
            correction_summary = recurrent_state_summary(
                cumulative_states[0], pool_size=self.state_pool_size
            ).to(dtype=facts.dtype)
            current_summary = (
                base_summary + correction_summary
                if base_summary is not None
                else correction_summary
            )
            state_feedback = self.state_feedback_projection(
                current_summary
            )

        for _ in range(steps):
            attention_query = self.step_attention_query(
                torch.cat([z, query_summary, state_feedback], dim=-1)
            ).unsqueeze(1)
            context, weights = self.cross_attention(
                attention_query,
                fact_features,
                fact_features,
                need_weights=True,
                average_attn_weights=False,
            )
            context = context[:, 0]
            cell_input = self.step_input(
                torch.cat([context, query_summary, state_feedback], dim=-1)
            )
            z = self.latent_norm(self.step_cell(cell_input, z))

            contexts.append(context)
            attention_weights.append(weights)
            latents.append(z)
            cumulative_states.append(self.writer(z))
            if base_states is not None or base_summary is not None:
                correction_summary = recurrent_state_summary(
                    cumulative_states[-1], pool_size=self.state_pool_size
                ).to(dtype=facts.dtype)
                current_summary = (
                    base_summary + correction_summary
                    if base_summary is not None
                    else correction_summary
                )
                state_feedback = self.state_feedback_projection(
                    current_summary
                )

        return ReasoningTrace(
            latents=latents,
            contexts=contexts,
            cumulative_states=cumulative_states,
            attention_weights=attention_weights,
        )


def _validate_state_lists(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]) -> None:
    if len(left) != len(right) or not left:
        raise ValueError("state lists must have the same non-zero number of layers")
    for index, (a, b) in enumerate(zip(left, right)):
        if a.shape != b.shape:
            raise ValueError(f"state shape mismatch at layer {index}: {a.shape} != {b.shape}")


def cumulative_state_targets(
    context_state: Sequence[torch.Tensor],
    oracle_states: Sequence[Sequence[torch.Tensor]],
) -> list[StateList]:
    """Return C_r* = S_r* - S_context for every oracle reasoning depth."""
    targets: list[StateList] = []
    for oracle in oracle_states:
        _validate_state_lists(context_state, oracle)
        targets.append([gold - base for gold, base in zip(oracle, context_state)])
    return targets


def state_relative_mse(
    predicted: Sequence[torch.Tensor], target: Sequence[torch.Tensor], eps: float = 1e-8
) -> torch.Tensor:
    _validate_state_lists(predicted, target)
    losses = [
        (pred.float() - gold.float()).pow(2).sum()
        / gold.float().pow(2).sum().clamp(min=eps)
        for pred, gold in zip(predicted, target)
    ]
    return torch.stack(losses).sum()


def state_direction_cos(
    predicted: Sequence[torch.Tensor], target: Sequence[torch.Tensor], eps: float = 1e-8
) -> torch.Tensor:
    """Global state cosine that preserves the autograd graph."""
    _validate_state_lists(predicted, target)
    dot = sum((pred.float() * gold.float()).sum() for pred, gold in zip(predicted, target))
    pred_sq = sum(pred.float().pow(2).sum() for pred in predicted)
    gold_sq = sum(gold.float().pow(2).sum() for gold in target)
    return dot / (pred_sq.sqrt() * gold_sq.sqrt()).clamp(min=eps)


def _relative_tensor_change(current: torch.Tensor, previous: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (current.float() - previous.float()).norm() / previous.float().norm().clamp(min=eps)


def state_relative_change(
    current: Sequence[torch.Tensor], previous: Sequence[torch.Tensor], eps: float = 1e-8
) -> torch.Tensor:
    _validate_state_lists(current, previous)
    numerator = sum((cur.float() - prev.float()).pow(2).sum() for cur, prev in zip(current, previous))
    denominator = sum(prev.float().pow(2).sum() for prev in previous)
    return numerator.sqrt() / denominator.sqrt().clamp(min=eps)


def variable_depth_state_loss(
    trace: ReasoningTrace,
    cumulative_targets: Sequence[Sequence[torch.Tensor]],
    *,
    hop_count: int,
    lambda_cos: float = 0.5,
    lambda_latent_stability: float = 0.05,
    lambda_state_stability: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Deep supervision for all budgets plus post-solution fixed-point training.

    Step ``r`` is matched to the oracle cumulative state after ``min(r, hop_count)``
    facts.  Extra recurrent steps therefore learn to preserve the solved state instead of
    repeatedly adding another correction.
    """
    steps = len(trace.latents) - 1
    if len(trace.cumulative_states) != len(trace.latents):
        raise ValueError("trace state and latent lengths must match")
    if hop_count <= 0 or len(cumulative_targets) < hop_count:
        raise ValueError("cumulative targets must cover every oracle hop")
    if steps <= 0:
        raise ValueError("at least one recurrent step is required for training")

    layer_count = len(cumulative_targets[0])
    objective = trace.latents[0].new_zeros(())
    post_solution_steps = 0
    state_terms: list[torch.Tensor] = []
    cosine_terms: list[torch.Tensor] = []
    latent_stability_terms: list[torch.Tensor] = []
    state_stability_terms: list[torch.Tensor] = []

    for step in range(1, steps + 1):
        target = cumulative_targets[min(step, hop_count) - 1]
        state_term = state_relative_mse(trace.cumulative_states[step], target) / layer_count
        cosine_term = 1.0 - state_direction_cos(trace.cumulative_states[step], target)
        objective = objective + state_term + float(lambda_cos) * cosine_term
        state_terms.append(state_term)
        cosine_terms.append(cosine_term)

        if step > hop_count:
            post_solution_steps += 1
            latent_stability = F.mse_loss(trace.latents[step], trace.latents[step - 1])
            state_stability = (
                state_relative_mse(
                    trace.cumulative_states[step], trace.cumulative_states[step - 1]
                )
                / layer_count
            )
            objective = objective + float(lambda_latent_stability) * latent_stability
            objective = objective + float(lambda_state_stability) * state_stability
            latent_stability_terms.append(latent_stability)
            state_stability_terms.append(state_stability)

    objective = objective / steps

    def _mean_value(values: Sequence[torch.Tensor]) -> float:
        if not values:
            return 0.0
        return float(torch.stack([value.detach() for value in values]).mean().item())

    metrics: dict[str, float | int] = {
        "supervised_steps": steps,
        "post_solution_steps": post_solution_steps,
        "state_loss": _mean_value(state_terms),
        "cosine_loss": _mean_value(cosine_terms),
        "latent_stability": _mean_value(latent_stability_terms),
        "state_stability": _mean_value(state_stability_terms),
    }
    return objective, metrics


def state_at_depth(trace: ReasoningTrace, depth: int) -> StateList:
    if depth < 0 or depth >= len(trace.cumulative_states):
        raise ValueError(
            f"depth {depth} unavailable; trace contains 0..{len(trace.cumulative_states) - 1}"
        )
    return trace.cumulative_states[depth]


def normalize_depths(depths: Sequence[int], *, max_steps: int) -> list[int]:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    return sorted({int(depth) for depth in depths if 0 < int(depth) <= max_steps})



def select_default_budget(accuracy_by_depth: dict[int, float], *, tolerance: float = 0.01) -> int:
    """Choose the cheapest budget within ``tolerance`` of the best validation score."""
    if not accuracy_by_depth:
        raise ValueError("accuracy_by_depth must not be empty")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    cleaned = {int(depth): float(score) for depth, score in accuracy_by_depth.items() if int(depth) > 0}
    if not cleaned:
        raise ValueError("accuracy_by_depth must contain a positive depth")
    best = max(cleaned.values())
    threshold = best - tolerance
    return min(depth for depth, score in cleaned.items() if score >= threshold)

def select_converged_depth(
    latents: Sequence[torch.Tensor],
    cumulative_states: Sequence[Sequence[torch.Tensor]],
    *,
    min_steps: int,
    max_steps: int,
    latent_tolerance: float,
    state_tolerance: float,
    patience: int = 2,
) -> int:
    """Select a forward-only convergence stopping point.

    This is a validation-calibrated rule, not a learned halting policy.
    """
    available = min(len(latents), len(cumulative_states)) - 1
    limit = min(int(max_steps), available)
    if limit <= 0:
        raise ValueError("at least one reasoning transition is required")
    min_steps = max(1, int(min_steps))
    patience = max(1, int(patience))
    stable_count = 0

    for step in range(1, limit + 1):
        latent_change = float(_relative_tensor_change(latents[step], latents[step - 1]).item())
        state_change = float(
            state_relative_change(cumulative_states[step], cumulative_states[step - 1]).item()
        )
        stable = latent_change <= latent_tolerance and state_change <= state_tolerance
        stable_count = stable_count + 1 if stable else 0
        if step >= min_steps and stable_count >= patience:
            return step
    return limit


__all__ = [
    "ReasoningTrace",
    "RecurrentReasoner",
    "SharedDynamicStateWriter",
    "cumulative_state_targets",
    "normalize_depths",
    "select_converged_depth",
    "select_default_budget",
    "state_at_depth",
    "state_direction_cos",
    "state_relative_change",
    "state_relative_mse",
    "variable_depth_state_loss",
]
