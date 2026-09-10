import torch
import torch.nn as nn

from models.owt_writer_reasoning import (
    OWTLatentTransition,
    add_residual_states,
    choice_token_logits,
    inject_residual_into_cache,
    native_reasoning_query_ids,
    select_pretrained_writer_parameters,
)


def test_add_residual_states_preserves_evidence_and_keeps_writer_gradient():
    evidence = [torch.full((1, 2, 3, 3), 2.0)]
    residual = [torch.full((1, 2, 3, 3), 0.5, requires_grad=True)]

    written = add_residual_states(evidence, residual)
    written[0].sum().backward()

    assert torch.equal(written[0], torch.full((1, 2, 3, 3), 2.5))
    assert residual[0].grad is not None
    assert torch.equal(residual[0].grad, torch.ones_like(residual[0]))


def test_choice_token_logits_selects_checkpoint_token_ids_and_masks_options():
    vocabulary_logits = torch.arange(24.0).reshape(2, 12)
    label_token_ids = torch.tensor([3, 7, 9, 11])

    selected = choice_token_logits(
        vocabulary_logits, label_token_ids, torch.tensor([2, 4])
    )

    assert torch.equal(selected[0, :2], torch.tensor([3.0, 7.0]))
    assert torch.isneginf(selected[0, 2:]).all()
    assert torch.equal(selected[1], torch.tensor([15.0, 19.0, 21.0, 23.0]))


def test_select_pretrained_writer_parameters_reuses_only_relay_writer():
    relay = nn.Module()
    relay.s1_trunk = nn.Linear(3, 4)
    relay.s1_u_head = nn.Linear(4, 5)
    relay.s1_v_head = nn.Linear(4, 5)
    relay.state_scale = nn.Parameter(torch.ones(1))
    relay.encoder = nn.Linear(3, 3)
    original = relay.s1_u_head.weight.data_ptr()

    selected = select_pretrained_writer_parameters(relay)

    assert relay.s1_u_head.weight.data_ptr() == original
    assert set(selected) == {
        "s1_trunk.weight", "s1_trunk.bias", "s1_u_head.weight", "s1_u_head.bias",
        "s1_v_head.weight", "s1_v_head.bias", "state_scale",
    }
    assert not relay.encoder.weight.requires_grad
    assert all(parameter.requires_grad for parameter in selected.values())


def test_transition_returns_one_latent_per_requested_depth():
    transition = OWTLatentTransition(hidden_dim=8, z_dim=4, context_dim=8, attention_heads=2)
    hidden = torch.randn(2, 5, 8)
    z0 = torch.randn(2, 4)

    trace = transition(hidden, z0, steps=3)

    assert len(trace) == 4
    assert all(z.shape == (2, 4) for z in trace)
    trace[-1].sum().backward()
    assert transition.step_cell.weight_hh.grad is not None


def test_inject_residual_into_cache_adds_instead_of_replacing_state():
    cache = type("Cache", (), {})()
    cache.states = [{"recurrent_state": torch.full((1, 2, 3, 3), 4.0)}]
    residual = torch.ones(1, 2, 3, 3, requires_grad=True)

    returned = inject_residual_into_cache(cache, [residual], scale=torch.tensor(0.25))

    assert returned is cache
    assert torch.equal(cache.states[0]["recurrent_state"], torch.full((1, 2, 3, 3), 4.25))
    cache.states[0]["recurrent_state"].sum().backward()
    assert torch.equal(residual.grad, torch.full_like(residual, 0.25))


def test_native_reasoning_query_is_long_enough_for_differentiable_chunk_backend():
    class CharacterTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return type("Tokens", (), {"input_ids": list(range(len(text)))})()

    ids, text = native_reasoning_query_ids(CharacterTokenizer(), min_tokens=64)

    assert len(ids) >= 64
    assert text.endswith("Answer:")
    assert "(A)" not in text
