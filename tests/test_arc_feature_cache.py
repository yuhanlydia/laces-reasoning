from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from models.arc_feature_cache import (
    ArcFeatureRecord,
    deterministic_projection,
    load_feature_record,
    pooled_state_features,
    run_rwkv_chunks,
    save_feature_record,
)
from models.recurrent_latent_reasoner import RecurrentReasoner


def _states(value: float):
    return [torch.full((1, 2, 4, 4), value), torch.full((1, 2, 4, 4), value + 1)]


def test_projection_is_deterministic_and_orthonormal():
    first = deterministic_projection(12, 4, seed=19)
    second = deterministic_projection(12, 4, seed=19)
    assert torch.equal(first, second)
    assert torch.allclose(first.T @ first, torch.eye(4), atol=1e-5)


def test_pooled_state_composition_is_linear():
    base = _states(1.0)
    correction = _states(0.25)
    combined = [left + right for left, right in zip(base, correction)]
    assert torch.allclose(
        pooled_state_features(combined, pool_size=2),
        pooled_state_features(base, pool_size=2)
        + pooled_state_features(correction, pool_size=2),
    )


def test_feature_record_round_trip_and_schema_rejection(tmp_path):
    record = ArcFeatureRecord(
        task_id="abc",
        transform_id=3,
        query_index=0,
        evidence=torch.randn(5, 8),
        query=torch.randn(3, 8),
        base_state_features=torch.randn(1, 16),
        target_grid=torch.tensor([[1, 2], [3, 4]]),
    )
    path = tmp_path / "record.pt"
    save_feature_record(path, record)
    loaded = load_feature_record(path)
    assert loaded.task_id == record.task_id
    assert torch.equal(loaded.target_grid, record.target_grid)

    payload = torch.load(path, weights_only=False)
    payload["schema_version"] = 999
    torch.save(payload, path)
    with pytest.raises(ValueError, match="schema"):
        load_feature_record(path)


def test_reasoner_accepts_prepooled_base_state():
    torch.manual_seed(5)
    model = RecurrentReasoner(
        hidden_dim=8,
        num_layers=2,
        num_heads=2,
        head_dim=4,
        z_dim=4,
        context_dim=8,
        writer_rank=2,
        writer_hidden=16,
        state_pool_size=2,
    )
    facts = torch.randn(1, 4, 8)
    query = torch.randn(1, 2, 8)
    states = _states(0.5)
    pooled = pooled_state_features(states, pool_size=2)
    raw_trace = model(facts, query, steps=2, base_states=states)
    pooled_trace = model(facts, query, steps=2, base_state_features=pooled)
    assert torch.allclose(raw_trace.latents[-1], pooled_trace.latents[-1], atol=1e-6)


class _FakeRwkv:
    def __call__(
        self,
        *,
        input_ids,
        past_key_values=None,
        use_cache=True,
        return_dict=True,
        output_hidden_states=True,
        attention_mask=None,
    ):
        previous = 0 if past_key_values is None else past_key_values
        hidden = input_ids.float().unsqueeze(-1) + previous
        return SimpleNamespace(
            hidden_states=(hidden,),
            past_key_values=previous + input_ids.shape[1],
        )


def test_chunked_rwkv_keeps_all_tokens_and_cache():
    ids = torch.arange(11).view(1, -1)
    hidden, cache = run_rwkv_chunks(_FakeRwkv(), ids, chunk_size=4)
    assert hidden.shape == (1, 11, 1)
    assert cache == 11
    assert torch.equal(hidden[:, :4, 0], ids[:, :4].float())
    assert torch.equal(hidden[:, 4:8, 0], ids[:, 4:8].float() + 4)
