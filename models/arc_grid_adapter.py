"""Differentiable ARC grid decoding from recurrent latent reasoning traces."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.recurrent_latent_reasoner import ReasoningTrace, recurrent_state_summary

MAX_GRID_SIZE = 30
NUM_COLORS = 10
IGNORE_INDEX = -100


@dataclass
class ArcDecoderOutput:
    row_logits: torch.Tensor
    col_logits: torch.Tensor
    cell_logits: torch.Tensor


class ArcGridDecoder(nn.Module):
    """Decode shape and colors only through the dynamic writer's pooled correction."""

    def __init__(
        self,
        *,
        state_dim: int = 20480,
        model_dim: int = 256,
        max_grid_size: int = MAX_GRID_SIZE,
        num_colors: int = NUM_COLORS,
    ) -> None:
        super().__init__()
        if min(state_dim, model_dim, max_grid_size, num_colors) <= 0:
            raise ValueError("all decoder dimensions must be positive")
        self.max_grid_size = int(max_grid_size)
        self.num_colors = int(num_colors)
        self.state_projection = nn.Sequential(nn.LayerNorm(state_dim), nn.Linear(state_dim, model_dim))
        self.fusion = nn.Sequential(nn.GELU(), nn.LayerNorm(model_dim))
        self.row_head = nn.Linear(model_dim, self.max_grid_size)
        self.col_head = nn.Linear(model_dim, self.max_grid_size)
        self.row_embedding = nn.Embedding(self.max_grid_size, model_dim)
        self.col_embedding = nn.Embedding(self.max_grid_size, model_dim)
        self.cell_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, self.num_colors),
        )

    def forward(
        self,
        correction_features: torch.Tensor,
    ) -> ArcDecoderOutput:
        if correction_features.ndim != 2:
            raise ValueError("expected pooled writer correction [B,S]")
        context = self.fusion(self.state_projection(correction_features))
        rows = self.row_embedding.weight[None, :, None, :]
        cols = self.col_embedding.weight[None, None, :, :]
        cells = context[:, None, None, :] + rows + cols
        return ArcDecoderOutput(
            row_logits=self.row_head(context),
            col_logits=self.col_head(context),
            cell_logits=self.cell_head(cells),
        )


def pad_target_grids(
    targets: Sequence[torch.Tensor], *, max_grid_size: int = MAX_GRID_SIZE
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not targets:
        raise ValueError("targets must not be empty")
    device = targets[0].device
    padded = torch.full(
        (len(targets), max_grid_size, max_grid_size),
        IGNORE_INDEX, dtype=torch.long, device=device,
    )
    row_labels = torch.empty(len(targets), dtype=torch.long, device=device)
    col_labels = torch.empty(len(targets), dtype=torch.long, device=device)
    for index, target in enumerate(targets):
        if target.ndim != 2:
            raise ValueError("each ARC target must be a two-dimensional grid")
        rows, cols = map(int, target.shape)
        if not (1 <= rows <= max_grid_size and 1 <= cols <= max_grid_size):
            raise ValueError(f"ARC target dimensions must be in 1..{max_grid_size}")
        target = target.to(device=device, dtype=torch.long)
        if bool(((target < 0) | (target >= NUM_COLORS)).any()):
            raise ValueError("ARC target colors must be in 0..9")
        padded[index, :rows, :cols] = target
        row_labels[index] = rows - 1
        col_labels[index] = cols - 1
    return padded, row_labels, col_labels


def arc_grid_loss(
    output: ArcDecoderOutput,
    targets: Sequence[torch.Tensor],
    *,
    shape_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, float]]:
    padded, row_labels, col_labels = pad_target_grids(
        targets, max_grid_size=output.cell_logits.shape[1]
    )
    device = output.cell_logits.device
    padded = padded.to(device)
    row_labels = row_labels.to(device)
    col_labels = col_labels.to(device)
    cell_loss = F.cross_entropy(
        output.cell_logits.reshape(-1, output.cell_logits.shape[-1]), padded.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
    row_loss = F.cross_entropy(output.row_logits, row_labels)
    col_loss = F.cross_entropy(output.col_logits, col_labels)
    shape_loss = 0.5 * (row_loss + col_loss)
    loss = cell_loss + float(shape_weight) * shape_loss
    return loss, {
        "loss": float(loss.detach()),
        "cell_loss": float(cell_loss.detach()),
        "shape_loss": float(shape_loss.detach()),
    }


def decode_grid(output: ArcDecoderOutput) -> list[torch.Tensor]:
    rows = output.row_logits.argmax(dim=-1) + 1
    cols = output.col_logits.argmax(dim=-1) + 1
    colors = output.cell_logits.argmax(dim=-1)
    return [
        colors[index, : int(rows[index]), : int(cols[index])].detach().cpu()
        for index in range(colors.shape[0])
    ]


def recurrent_arc_objective(
    decoder: ArcGridDecoder,
    trace: ReasoningTrace,
    targets: Sequence[torch.Tensor],
    *,
    depths: Sequence[int] = (1, 2, 4, 8),
    shape_weight: float = 0.25,
    stability_weight: float = 0.05,
    state_magnitude_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Apply ARC supervision at each available recurrent inference budget."""
    selected = sorted({int(depth) for depth in depths if 0 < int(depth) < len(trace.latents)})
    if not selected:
        raise ValueError("no requested depth is available in the reasoning trace")
    losses = []
    cell_losses = []
    shape_losses = []
    state_penalties = []
    for depth in selected:
        correction = trace.pooled_corrections[depth]
        output = decoder(correction)
        grid_loss, grid_metrics = arc_grid_loss(output, targets, shape_weight=shape_weight)
        state_penalty = correction.float().square().mean()
        losses.append(grid_loss + float(state_magnitude_weight) * state_penalty)
        cell_losses.append(grid_metrics["cell_loss"])
        shape_losses.append(grid_metrics["shape_loss"])
        state_penalties.append(state_penalty)

    stability = trace.latents[0].new_zeros(())
    if len(selected) > 1:
        stability = F.mse_loss(trace.latents[selected[-1]], trace.latents[selected[-2]])
    objective = torch.stack(losses).mean() + float(stability_weight) * stability
    return objective, {
        "loss": float(objective.detach()),
        "cell_loss": sum(cell_losses) / len(cell_losses),
        "shape_loss": sum(shape_losses) / len(shape_losses),
        "state_magnitude": float(torch.stack(state_penalties).mean().detach()),
        "stability": float(stability.detach()),
        "supervised_depths": len(selected),
    }
