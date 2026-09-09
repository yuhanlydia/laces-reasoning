from __future__ import annotations

import random

import pytest
import torch

from models.arc_metrics import evaluate_arc_predictions
from scripts.eval.train_arc_recurrent_reasoner import (
    make_checkpoint,
    restore_checkpoint,
    split_training_task_ids,
)


def test_deterministic_360_40_split_and_eval_isolation():
    task_ids = [f"train-{index:03d}" for index in range(400)]
    first = split_training_task_ids(task_ids, dev_count=40, seed=17)
    second = split_training_task_ids(reversed(task_ids), dev_count=40, seed=17)
    assert first == second
    train_ids, dev_ids = first
    assert len(train_ids) == 360
    assert len(dev_ids) == 40
    assert train_ids.isdisjoint(dev_ids)
    with pytest.raises(ValueError, match="evaluation"):
        split_training_task_ids(task_ids, dev_count=40, seed=17, evaluation_ids={"train-001"})


def test_exact_pair_task_shape_cell_and_pass_at_two_metrics():
    gold = {
        "a": [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5]])],
        "b": [torch.tensor([[7, 8]])],
    }
    predictions = {
        "a": [
            [torch.tensor([[1, 2], [3, 4]])],
            [torch.tensor([[0]]), torch.tensor([[5]])],
        ],
        "b": [[torch.tensor([[7], [8]]), torch.tensor([[7, 8]])]],
    }
    metrics = evaluate_arc_predictions(gold, predictions)
    assert metrics["pair_count"] == 3
    assert metrics["pair_exact"] == pytest.approx(1 / 3)
    assert metrics["task_exact"] == 0
    assert metrics["pass_at_1"] == pytest.approx(1 / 3)
    assert metrics["pass_at_2"] == 1
    assert metrics["shape_accuracy"] == pytest.approx(2 / 3)
    assert metrics["cell_accuracy_on_shape_matches"] == pytest.approx(4 / 5)


def test_invalid_prediction_is_scored_incorrect_instead_of_crashing():
    gold = {"a": [torch.tensor([[1]])]}
    predictions = {"a": [[torch.tensor([[12]])]]}
    metrics = evaluate_arc_predictions(gold, predictions)
    assert metrics["invalid_predictions"] == 1
    assert metrics["pair_exact"] == 0


def test_checkpoint_restores_models_optimizer_scheduler_scaler_and_rng():
    torch.manual_seed(3)
    random.seed(3)
    model = torch.nn.Linear(2, 2)
    decoder = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW([*model.parameters(), *decoder.parameters()], lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    checkpoint = make_checkpoint(
        reasoner=model, decoder=decoder, optimizer=optimizer, scheduler=scheduler,
        scaler=scaler, step=7, epoch=2, best_metric=0.4, config={"x": 1},
    )
    expected_random = random.random()
    expected_torch = torch.rand(1)
    with torch.no_grad():
        model.weight.zero_()
    random.seed(99)
    torch.manual_seed(99)

    restored = restore_checkpoint(
        checkpoint, reasoner=model, decoder=decoder, optimizer=optimizer,
        scheduler=scheduler, scaler=scaler,
    )
    assert restored == {"step": 7, "epoch": 2, "best_metric": 0.4, "config": {"x": 1}}
    assert model.weight.abs().sum() > 0
    assert random.random() == expected_random
    assert torch.equal(torch.rand(1), expected_torch)
