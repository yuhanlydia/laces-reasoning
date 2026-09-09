from __future__ import annotations

import pytest
import torch

from models.arc_grid_adapter import (
    ArcGridDecoder,
    arc_grid_loss,
    decode_grid,
    pad_target_grids,
    recurrent_arc_objective,
)
from models.recurrent_latent_reasoner import RecurrentReasoner


def _targets():
    return [torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([[7], [8], [9]])]


def test_decoder_shapes_and_target_mask():
    decoder = ArcGridDecoder(z_dim=8, query_dim=12, state_dim=16, model_dim=24)
    output = decoder(torch.randn(2, 8), torch.randn(2, 5, 12), torch.randn(2, 16))
    assert output.row_logits.shape == (2, 30)
    assert output.col_logits.shape == (2, 30)
    assert output.cell_logits.shape == (2, 30, 30, 10)

    padded, rows, cols = pad_target_grids(_targets())
    assert rows.tolist() == [1, 2]
    assert cols.tolist() == [2, 0]
    assert torch.equal(padded[0, :2, :3], _targets()[0])
    assert torch.all(padded[0, 2:] == -100)
    assert torch.all(padded[1, :, 1:] == -100)


def test_perfect_prediction_decodes_exactly_and_has_near_zero_loss():
    targets = _targets()
    padded, rows, cols = pad_target_grids(targets)
    row_logits = torch.full((2, 30), -40.0)
    col_logits = torch.full((2, 30), -40.0)
    cell_logits = torch.full((2, 30, 30, 10), -40.0)
    row_logits.scatter_(1, rows[:, None], 40.0)
    col_logits.scatter_(1, cols[:, None], 40.0)
    cell_logits.scatter_(-1, padded.clamp_min(0)[..., None], 40.0)

    from models.arc_grid_adapter import ArcDecoderOutput
    output = ArcDecoderOutput(row_logits, col_logits, cell_logits)
    loss, metrics = arc_grid_loss(output, targets)
    assert loss.item() < 1e-6
    assert metrics["cell_loss"] < 1e-6
    decoded = decode_grid(output)
    assert all(torch.equal(actual, expected) for actual, expected in zip(decoded, targets))


@pytest.mark.parametrize("shape", [(0, 1), (1, 0), (31, 1), (1, 31)])
def test_invalid_target_dimensions_rejected(shape):
    with pytest.raises(ValueError, match="1..30"):
        pad_target_grids([torch.zeros(shape, dtype=torch.long)])


def test_depth_two_objective_reaches_recurrent_cell_and_writer():
    torch.manual_seed(7)
    reasoner = RecurrentReasoner(
        hidden_dim=12, num_layers=2, num_heads=2, head_dim=4,
        z_dim=8, context_dim=16, writer_rank=2, writer_hidden=16,
        state_summary_dim=16, state_pool_size=2,
    )
    decoder = ArcGridDecoder(z_dim=8, query_dim=12, state_dim=16, model_dim=24)
    facts = torch.randn(1, 7, 12)
    query = torch.randn(1, 4, 12)
    base = torch.randn(1, 16)
    trace = reasoner(facts, query, steps=2, base_state_features=base)
    loss, metrics = recurrent_arc_objective(
        decoder, trace, query, base, [torch.tensor([[1, 2], [3, 4]])], depths=(1, 2)
    )
    loss.backward()

    assert metrics["supervised_depths"] == 2
    assert reasoner.step_cell.weight_hh.grad is not None
    assert torch.isfinite(reasoner.step_cell.weight_hh.grad).all()
    assert reasoner.step_cell.weight_hh.grad.abs().sum() > 0
    assert reasoner.writer.uv_head.weight.grad is not None
    assert torch.isfinite(reasoner.writer.uv_head.weight.grad).all()
    assert reasoner.writer.uv_head.weight.grad.abs().sum() > 0
