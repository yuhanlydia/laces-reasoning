from types import SimpleNamespace

import pytest
import torch

from scripts.eval.run_cola_dlm_tasks_prefix_suffix_trajectory_cfg import (
    generate_answer_trajectory,
    generate_answer_trajectory_batch,
)


class _BlendSensitiveRwkv:
    def __call__(self, input_ids, past_key_values, **_kwargs):
        batch_size, sequence_length = input_ids.shape
        token_id = 1 if past_key_values["blend"] < 1.0 else 2
        logits = torch.zeros(batch_size, sequence_length, 3)
        logits[:, :, token_id] = 10.0
        return SimpleNamespace(past_key_values=past_key_values, logits=logits)


class _IndependentModel:
    trajectory_chunk_size = 1

    def __init__(self, blend):
        self.config = {
            "trajectory_s1_mode": "independent",
            "trajectory_state_blend": blend,
        }
        self.trajectory_state_blend = blend
        self.rwkv_model = _BlendSensitiveRwkv()

    def predict_states(self, latent):
        return [latent]

    def blend_into_cache(self, cache, _states, blend):
        cache["blend"] = float(blend)
        return cache

    def inject_into_cache(self, cache, _states):
        cache["blend"] = 1.0
        return cache


class _Tokenizer:
    eos_token_id = None

    def decode(self, token_ids, **_kwargs):
        return " ".join("partial" if token_id == 1 else "full" for token_id in token_ids)


def _args():
    return SimpleNamespace(
        max_new_tokens=1,
        repetition_penalty=1.0,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
    )


@pytest.mark.parametrize("blend", [0.25, 0.5, 0.7, 0.75])
def test_independent_single_generation_uses_configured_partial_blend(blend):
    generated, _, _ = generate_answer_trajectory(
        _IndependentModel(blend),
        _Tokenizer(),
        torch.tensor([[0]]),
        {"blend": 1.0},
        torch.zeros(3),
        torch.zeros(1, 1, 1),
        _args(),
    )

    assert generated == "partial"


def test_independent_single_generation_keeps_full_replacement_at_blend_one():
    generated, _, _ = generate_answer_trajectory(
        _IndependentModel(1.0),
        _Tokenizer(),
        torch.tensor([[0]]),
        {"blend": 0.0},
        torch.zeros(3),
        torch.zeros(1, 1, 1),
        _args(),
    )

    assert generated == "full"


def test_independent_batch_generation_uses_configured_partial_blend():
    generated, _, _ = generate_answer_trajectory_batch(
        _IndependentModel(0.5),
        _Tokenizer(),
        torch.tensor([[0], [0]]),
        torch.ones(2, 1, dtype=torch.long),
        {"blend": 1.0},
        torch.zeros(2, 3),
        torch.zeros(2, 1, 1),
        _args(),
    )

    assert generated == ["partial", "partial"]
