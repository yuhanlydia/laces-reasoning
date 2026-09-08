"""Contracts for variable-depth recurrent latent reasoning."""

from __future__ import annotations

import torch

from models.recurrent_latent_reasoner import (
    RecurrentReasoner,
    cumulative_state_targets,
    normalize_depths,
    select_converged_depth,
    select_default_budget,
    state_direction_cos,
    state_relative_mse,
    state_at_depth,
    variable_depth_state_loss,
)


def _state(value: float, *, layers: int = 2, batch: int = 1, heads: int = 2, dim: int = 3):
    return [torch.full((batch, heads, dim, dim), value) for _ in range(layers)]


def test_reasoner_requeries_evidence_and_returns_every_budget_depth():
    torch.manual_seed(7)
    model = RecurrentReasoner(
        hidden_dim=12,
        num_layers=2,
        num_heads=2,
        head_dim=3,
        z_dim=4,
        context_dim=8,
        writer_rank=2,
        writer_hidden=16,
    )
    facts = torch.randn(1, 5, 12)
    query = torch.randn(1, 3, 12)

    trace = model(facts, query, steps=4)

    assert len(trace.latents) == 5
    assert len(trace.contexts) == 4
    assert len(trace.cumulative_states) == 5
    assert trace.latents[-1].shape == (1, 4)
    assert trace.cumulative_states[-1][0].shape == (1, 2, 3, 3)
    # Per-step evidence is queried from the current latent, not compressed once and reused.
    assert not torch.allclose(trace.contexts[0], trace.contexts[1])


def test_same_facts_with_different_queries_change_the_reasoning_trace():
    torch.manual_seed(11)
    model = RecurrentReasoner(
        hidden_dim=10,
        num_layers=1,
        num_heads=2,
        head_dim=2,
        z_dim=4,
        context_dim=8,
        writer_rank=2,
        writer_hidden=16,
    )
    facts = torch.randn(1, 6, 10)
    query_a = torch.zeros(1, 2, 10)
    query_b = torch.ones(1, 2, 10)

    trace_a = model(facts, query_a, steps=2)
    trace_b = model(facts, query_b, steps=2)

    assert not torch.allclose(trace_a.latents[0], trace_b.latents[0])
    assert not torch.allclose(trace_a.cumulative_states[2][0], trace_b.cumulative_states[2][0])


def test_dynamic_writer_uses_per_head_u_and_v_with_shared_parameters():
    model = RecurrentReasoner(
        hidden_dim=12,
        num_layers=4,
        num_heads=3,
        head_dim=5,
        z_dim=4,
        context_dim=8,
        writer_rank=2,
        writer_hidden=16,
    )
    writer = model.writer

    assert writer.uv_head.out_features == 2 * 5 * 2
    # A shared hypernetwork should not scale its parameters as one giant output head per layer/head.
    assert sum(p.numel() for p in writer.parameters()) < 20_000


def test_cumulative_targets_are_relative_to_one_context_not_incremental():
    context = _state(10.0)
    oracle_states = [_state(11.0), _state(13.0), _state(16.0)]

    targets = cumulative_state_targets(context, oracle_states)

    assert torch.allclose(targets[0][0], torch.ones_like(targets[0][0]))
    assert torch.allclose(targets[1][0], torch.full_like(targets[1][0], 3.0))
    assert torch.allclose(targets[2][0], torch.full_like(targets[2][0], 6.0))


def test_state_at_depth_does_not_sum_writes_across_reasoning_steps():
    class Trace:
        cumulative_states = [_state(0.0), _state(1.0), _state(2.0), _state(3.0)]

    selected = state_at_depth(Trace(), 3)

    assert torch.allclose(selected[0], torch.full_like(selected[0], 3.0))


def test_variable_depth_loss_handles_extra_steps_after_short_hop_chain():
    torch.manual_seed(5)
    model = RecurrentReasoner(
        hidden_dim=8,
        num_layers=2,
        num_heads=1,
        head_dim=2,
        z_dim=4,
        context_dim=8,
        writer_rank=2,
        writer_hidden=16,
    )
    trace = model(torch.randn(1, 4, 8), torch.randn(1, 2, 8), steps=4)
    targets = [_state(0.25, layers=2, heads=1, dim=2), _state(0.5, layers=2, heads=1, dim=2)]

    loss, metrics = variable_depth_state_loss(
        trace,
        targets,
        hop_count=2,
        lambda_cos=0.2,
        lambda_latent_stability=0.1,
        lambda_state_stability=0.1,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["supervised_steps"] == 4
    assert metrics["post_solution_steps"] == 2
    assert model.step_cell.weight_hh.grad is not None
    assert torch.isfinite(model.step_cell.weight_hh.grad).all()


def test_direction_cosine_remains_differentiable():
    pred = [torch.randn(1, 2, 2, 2, requires_grad=True)]
    target = [torch.randn(1, 2, 2, 2)]

    cosine = state_direction_cos(pred, target)
    (1.0 - cosine).backward()

    assert cosine.requires_grad
    assert pred[0].grad is not None
    assert torch.isfinite(pred[0].grad).all()


def test_relative_mse_is_zero_for_identical_state_lists():
    state = _state(2.0)
    assert state_relative_mse(state, state).item() == 0.0


def test_convergence_stopping_waits_for_patience_and_minimum_depth():
    latents = [
        torch.tensor([[0.0, 0.0]]),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[1.00001, 0.0]]),
        torch.tensor([[1.00002, 0.0]]),
        torch.tensor([[2.0, 0.0]]),
    ]
    states = [_state(0.0), _state(1.0), _state(1.00001), _state(1.00002), _state(2.0)]

    depth = select_converged_depth(
        latents,
        states,
        min_steps=2,
        max_steps=4,
        latent_tolerance=1e-3,
        state_tolerance=1e-3,
        patience=2,
    )

    assert depth == 3


def test_depth_normalization_keeps_valid_unique_ordered_budgets():
    assert normalize_depths([8, 1, 4, 4, -1, 16], max_steps=8) == [1, 4, 8]


def test_e10_script_wires_variable_depth_reasoner_and_cumulative_writes(repo_root):
    from pathlib import Path

    script = (Path(repo_root) / "scripts" / "eval" / "train_recurrent_reasoner.py").read_text()

    assert "from models.recurrent_latent_reasoner import" in script
    assert "variable_depth_state_loss" in script
    assert '"H_facts"' in script
    assert '"H_query"' in script
    assert "state_at_depth(trace" in script
    assert "sum_states(traj" not in script
    assert "recompute_logits_from_injected_cache" not in script
    assert "_cache_layer_state" in script


def test_e10_cli_exposes_train_and_inference_reasoning_budgets(repo_root):
    from pathlib import Path

    script = (Path(repo_root) / "scripts" / "eval" / "train_recurrent_reasoner.py").read_text()

    assert '"--train_max_steps"' in script
    assert '"--eval_depths"' in script
    assert '"--early_stop"' in script
    assert 'default="sweep"' in script


def test_oracle_hop_mode_uses_one_metric_key_across_mixed_hop_lengths(repo_root):
    from pathlib import Path

    script = (Path(repo_root) / "scripts" / "eval" / "train_recurrent_reasoner.py").read_text()
    assert 'key = "reasoner_oracle_hop"' in script


def test_default_budget_is_smallest_depth_within_tolerance_of_best_accuracy():
    accuracy = {1: 0.41, 2: 0.60, 4: 0.69, 8: 0.695}
    assert select_default_budget(accuracy, tolerance=0.01) == 4
