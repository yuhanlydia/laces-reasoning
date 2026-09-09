"""Exact ARC-AGI grid metrics with invalid-prediction containment."""
from __future__ import annotations

from typing import Mapping, Sequence

import torch


def valid_arc_grid(grid: object) -> bool:
    return (
        isinstance(grid, torch.Tensor)
        and grid.ndim == 2
        and 1 <= grid.shape[0] <= 30
        and 1 <= grid.shape[1] <= 30
        and not bool(((grid < 0) | (grid > 9)).any())
    )


def evaluate_arc_predictions(
    gold: Mapping[str, Sequence[torch.Tensor]],
    predictions: Mapping[str, Sequence[Sequence[torch.Tensor]]],
) -> dict[str, float | int]:
    """Score ordered candidate grids; candidate zero is pass@1."""
    pair_count = pair_exact = pass_two = shape_correct = 0
    task_exact = 0
    invalid = 0
    matching_shape_cells = matching_shape_correct = 0
    for task_id, gold_pairs in gold.items():
        predicted_pairs = predictions.get(task_id, ())
        all_first_exact = len(predicted_pairs) == len(gold_pairs)
        for index, target in enumerate(gold_pairs):
            pair_count += 1
            candidates = predicted_pairs[index] if index < len(predicted_pairs) else ()
            first = candidates[0] if candidates else None
            first_valid = valid_arc_grid(first)
            if not first_valid:
                invalid += 1
                all_first_exact = False
            else:
                same_shape = tuple(first.shape) == tuple(target.shape)
                shape_correct += int(same_shape)
                if same_shape:
                    matching_shape_cells += target.numel()
                    matching_shape_correct += int((first.cpu() == target.cpu()).sum())
                exact = same_shape and torch.equal(first.cpu(), target.cpu())
                pair_exact += int(exact)
                all_first_exact = all_first_exact and exact
            candidate_exact = False
            for candidate in list(candidates)[:2]:
                if not valid_arc_grid(candidate):
                    continue
                if tuple(candidate.shape) == tuple(target.shape) and torch.equal(
                    candidate.cpu(), target.cpu()
                ):
                    candidate_exact = True
                    break
            pass_two += int(candidate_exact)
        task_exact += int(all_first_exact)
    denominator = max(pair_count, 1)
    return {
        "task_count": len(gold),
        "pair_count": pair_count,
        "pair_exact": pair_exact / denominator,
        "task_exact": task_exact / max(len(gold), 1),
        "pass_at_1": pair_exact / denominator,
        "pass_at_2": pass_two / denominator,
        "shape_accuracy": shape_correct / denominator,
        "cell_accuracy_on_shape_matches": (
            matching_shape_correct / matching_shape_cells if matching_shape_cells else 0.0
        ),
        "invalid_predictions": invalid,
    }
