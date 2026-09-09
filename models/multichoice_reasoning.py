"""Compact frozen-backbone features and recurrent residual scoring for MCQ tasks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.recurrent_latent_reasoner import ReasoningTrace


SCHEMA_VERSION = 1


def retarget_cosine_schedule(scheduler, *, total_steps: int, grad_accum: int) -> None:
    if total_steps <= 0 or grad_accum <= 0:
        raise ValueError("total_steps and grad_accum must be positive")
    scheduler.T_max = max(1, int(total_steps) // int(grad_accum))


def format_multiple_choice_prompt(
    question: str, options: Sequence[str], *, category: str | None = None,
) -> tuple[str, str]:
    if not 2 <= len(options) <= 10:
        raise ValueError("multiple-choice examples require 2..10 options")
    lines = []
    if category:
        lines.append(f"Category: {category}")
    lines.extend((f"Question: {question}", "Options:"))
    lines.extend(f"({chr(65 + index)}) {option}" for index, option in enumerate(options))
    return "\n".join(lines) + "\n", "Answer:"


@dataclass
class MultipleChoiceFeatureRecord:
    example_id: str
    category: str
    evidence: torch.Tensor
    query: torch.Tensor
    base_state_features: torch.Tensor
    base_choice_logits: torch.Tensor
    num_choices: int
    label: int


def stratified_three_way_split(
    categories: Sequence[str], *, seed: int, train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> tuple[list[int], list[int], list[int]]:
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1:
        raise ValueError("invalid split fractions")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a test split")
    grouped: dict[str, list[int]] = {}
    for index, category in enumerate(categories):
        grouped.setdefault(str(category), []).append(index)
    rng = random.Random(seed)
    train, validation, test = [], [], []
    for category in sorted(grouped):
        indices = grouped[category][:]
        rng.shuffle(indices)
        train_end = int(len(indices) * train_fraction)
        validation_end = train_end + int(len(indices) * validation_fraction)
        train.extend(indices[:train_end])
        validation.extend(indices[train_end:validation_end])
        test.extend(indices[validation_end:])
    rng.shuffle(train); rng.shuffle(validation); rng.shuffle(test)
    return train, validation, test


def save_feature_record(path: str | Path, record: MultipleChoiceFeatureRecord) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": SCHEMA_VERSION, **record.__dict__}
    for key, value in list(payload.items()):
        if isinstance(value, torch.Tensor):
            payload[key] = value.detach().cpu()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_feature_record(path: str | Path) -> MultipleChoiceFeatureRecord:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.pop("schema_version", None) != SCHEMA_VERSION:
        raise ValueError("unsupported multiple-choice feature schema")
    return MultipleChoiceFeatureRecord(**payload)


class MultipleChoiceResidualHead(nn.Module):
    """Add a writer-conditioned residual to frozen-backbone choice scores."""

    def __init__(self, *, state_dim: int, hidden_dim: int = 256, max_choices: int = 10):
        super().__init__()
        if min(state_dim, hidden_dim, max_choices) <= 0:
            raise ValueError("head dimensions must be positive")
        self.max_choices = int(max_choices)
        self.network = nn.Sequential(
            nn.LayerNorm(state_dim), nn.Linear(state_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, self.max_choices),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self, correction_features: torch.Tensor, base_choice_logits: torch.Tensor,
        choice_counts: torch.Tensor,
    ) -> torch.Tensor:
        if correction_features.ndim != 2 or base_choice_logits.ndim != 2:
            raise ValueError("correction and base logits must be batched matrices")
        width = int(base_choice_logits.shape[-1])
        if width > self.max_choices:
            raise ValueError("base logits exceed max_choices")
        logits = base_choice_logits + self.network(correction_features)[:, :width]
        positions = torch.arange(width, device=logits.device).unsqueeze(0)
        return logits.masked_fill(positions >= choice_counts.unsqueeze(1), float("-inf"))


def recurrent_multichoice_objective(
    head: MultipleChoiceResidualHead,
    trace: ReasoningTrace,
    *,
    base_choice_logits: torch.Tensor,
    choice_counts: torch.Tensor,
    labels: torch.Tensor,
    depths: Sequence[int],
    stability_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    selected = sorted({int(depth) for depth in depths if 0 < int(depth) < len(trace.latents)})
    if not selected:
        raise ValueError("no requested depth is available")
    losses = []
    accuracies = []
    for depth in selected:
        logits = head(trace.pooled_corrections[depth], base_choice_logits, choice_counts)
        losses.append(F.cross_entropy(logits, labels))
        accuracies.append((logits.argmax(-1) == labels).float().mean())
    stability = trace.latents[0].new_zeros(())
    if len(selected) > 1:
        stability = F.mse_loss(trace.latents[selected[-1]], trace.latents[selected[-2]])
    loss = torch.stack(losses).mean() + float(stability_weight) * stability
    return loss, {
        "loss": float(loss.detach()),
        "accuracy": float(torch.stack(accuracies).mean().detach()),
        "stability": float(stability.detach()),
        "supervised_depths": len(selected),
    }
