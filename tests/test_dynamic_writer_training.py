"""Trainability contracts for LACES S1 writer variants."""

import pytest
import torch
from torch import nn

import train_state_hijacking_dit as training
from models import state_hijacking_dit as state_model


class TinyRelay(nn.Module):
    """Small parameter graph with the same trainability boundaries as RELAY."""

    def __init__(self, writer_type: str):
        super().__init__()
        self.s1_writer_type = writer_type
        self.rwkv_model = nn.Linear(2, 2)
        self.encoder_trunk = nn.Linear(2, 2)
        self.mu_head = nn.Linear(2, 2)
        self.logvar_head = nn.Linear(2, 2)
        self.alpha_heads = nn.ModuleList([nn.Linear(2, 2)])
        self.alpha_trunk = nn.Linear(2, 2)
        self.state_basis = nn.Parameter(torch.zeros(1, 2, 1, 2, 2))
        self.state_scale = nn.Parameter(torch.zeros(1))
        self.latent_dit = nn.Linear(2, 2)
        self.trajectory_dit = nn.Linear(2, 2)
        self.trajectory_state_decoder = nn.Linear(2, 2)
        self.blend_gate_logit = None
        self.use_learnable_blend = False

        if writer_type == "dynlowrank":
            self.s1_trunk = nn.Linear(2, 2)
            self.s1_u_head = nn.Linear(2, 2)
            self.s1_v_head = nn.Linear(2, 2)
        elif writer_type == "mixture":
            self.s1_banks = nn.Parameter(torch.zeros(2, 1, 2, 1, 2, 2))
            self.s1_gate = nn.Linear(2, 2)
            self.s1_alpha_head = nn.Linear(2, 4)


def _configure(model, **kwargs):
    """Call the production selector; no-op fallback makes pre-feature RED behavioral."""
    selector = getattr(training, "_configure_trainable_parameters", lambda *_args, **_kwargs: None)
    for parameter in model.parameters():
        parameter.requires_grad = False
    selector(model, **kwargs)


def _all_trainable(module):
    return all(parameter.requires_grad for parameter in module.parameters())


def _all_frozen(module):
    return all(not parameter.requires_grad for parameter in module.parameters())


def test_stage1_dynlowrank_trains_dynamic_writer_instead_of_fixed_basis():
    model = TinyRelay("dynlowrank")

    _configure(
        model,
        train_stage=1,
        trajectory_mode=True,
        trajectory_s1_mode="independent",
        s2_unfreeze_s1=False,
        freeze_s2=False,
        sft_response_only=False,
    )

    assert _all_trainable(model.s1_trunk)
    assert _all_trainable(model.s1_u_head)
    assert _all_trainable(model.s1_v_head)
    assert model.state_scale.requires_grad
    assert _all_frozen(model.alpha_heads)
    assert not model.state_basis.requires_grad
    assert _all_frozen(model.rwkv_model)


def test_stage2_coadapt_keeps_trained_s0_frozen():
    model = TinyRelay("dynlowrank")

    _configure(
        model,
        train_stage=2,
        trajectory_mode=True,
        trajectory_s1_mode="independent",
        s2_unfreeze_s1=True,
        freeze_s2=False,
        sft_response_only=False,
    )

    assert _all_trainable(model.s1_trunk)
    assert _all_trainable(model.s1_u_head)
    assert _all_trainable(model.s1_v_head)
    assert _all_frozen(model.encoder_trunk)
    assert _all_frozen(model.mu_head)
    assert _all_frozen(model.logvar_head)
    assert _all_trainable(model.trajectory_dit)
    assert model.state_scale.requires_grad
    assert _all_frozen(model.alpha_heads)
    assert not model.state_basis.requires_grad
    assert _all_frozen(model.rwkv_model)


def test_stage2_can_explicitly_unfreeze_s0_encoder():
    model = TinyRelay("dynlowrank")

    _configure(
        model,
        train_stage=2,
        trajectory_mode=True,
        trajectory_s1_mode="independent",
        s2_unfreeze_s1=True,
        s2_unfreeze_s0=True,
        freeze_s2=False,
        sft_response_only=False,
    )

    assert _all_trainable(model.encoder_trunk)
    assert _all_trainable(model.mu_head)
    assert _all_trainable(model.logvar_head)


def test_stage2_coadapt_trains_mixture_writer_bank_and_heads():
    model = TinyRelay("mixture")

    _configure(
        model,
        train_stage=2,
        trajectory_mode=True,
        trajectory_s1_mode="independent",
        s2_unfreeze_s1=True,
        freeze_s2=False,
        sft_response_only=False,
    )

    assert model.s1_banks.requires_grad
    assert _all_trainable(model.s1_gate)
    assert _all_trainable(model.s1_alpha_head)
    assert model.state_scale.requires_grad
    assert not model.state_basis.requires_grad


def test_stage1_fixed_writer_keeps_original_basis_trainability():
    model = TinyRelay("fixed")

    _configure(
        model,
        train_stage=1,
        trajectory_mode=True,
        trajectory_s1_mode="independent",
        s2_unfreeze_s1=False,
        freeze_s2=False,
        sft_response_only=False,
    )

    assert _all_trainable(model.alpha_heads)
    assert _all_trainable(model.alpha_trunk)
    assert model.state_basis.requires_grad
    assert model.state_scale.requires_grad
    assert _all_frozen(model.rwkv_model)


def test_module_grad_stats_distinguishes_missing_and_nonzero_gradients():
    module = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 1))
    module[0].weight.grad = torch.ones_like(module[0].weight)
    module[0].bias.grad = torch.zeros_like(module[0].bias)

    stats = training._module_grad_stats(module)

    assert stats["trainable_tensors"] == 4
    assert stats["grad_tensors"] == 2
    assert stats["nonzero_tensors"] == 1
    assert stats["norm"] == pytest.approx(6 ** 0.5)


def test_chunk_rwkv7_wrapper_omits_new_kwargs_for_fla_03(monkeypatch):
    seen = {}

    def old_chunk_rwkv7(**kwargs):
        seen.update(kwargs)
        return kwargs["v"], None

    monkeypatch.setattr(state_model, "_chunk_rwkv7", old_chunk_rwkv7)
    monkeypatch.setattr(
        state_model,
        "_CHUNK_RWKV7_PARAMETERS",
        {"r", "w", "k", "v", "a", "b", "scale", "initial_state", "output_final_state"},
    )
    x = torch.zeros(1, 2, 1, 4)

    y, final_state = state_model._call_chunk_rwkv7(x, x, x, x, x, x)

    assert y is x
    assert final_state is None
    assert "safe_gate" not in seen
    assert "chunk_size" not in seen


def test_cache_layer_state_supports_fla_03_states_and_fla_04_layers():
    old_cache = type("OldCache", (), {"states": [{"recurrent_state": None}]})()
    new_layer = type("NewLayer", (), {"state": {"recurrent_state": None}})()
    new_cache = type("NewCache", (), {"layers": [new_layer]})()

    assert state_model._cache_layer_state(old_cache, 0) is old_cache.states[0]
    assert state_model._cache_layer_state(new_cache, 0) is new_layer.state
