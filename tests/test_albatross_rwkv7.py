from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import fields, is_dataclass
from typing import Protocol, cast

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


DIM = 16
HEAD_DIM = 8
BATCH = 2
SEQ = 4


class AlbatrossLayer(Protocol):
    def __call__(self, x: torch.Tensor, *, state: object | None = None) -> tuple[torch.Tensor, object]: ...

    def eval(self) -> AlbatrossLayer: ...

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, nn.Parameter]]: ...

    def zero_grad(self, set_to_none: bool = False) -> None: ...


class TMixFactory(Protocol):
    def __call__(self, *, dim: int, head_dim: int, layer_id: int, n_layer: int) -> AlbatrossLayer: ...


class CMixFactory(Protocol):
    def __call__(self, *, dim: int, layer_id: int, n_layer: int) -> AlbatrossLayer: ...


class BlockFactory(Protocol):
    def __call__(
        self,
        *,
        dim: int,
        head_dim: int,
        layer_id: int,
        n_layer: int,
        mlp_ratio: float,
    ) -> AlbatrossLayer: ...


class RecurrentFallback(Protocol):
    def __call__(
        self,
        r: torch.Tensor,
        w: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


def _future_api() -> tuple[TMixFactory, CMixFactory, BlockFactory, RecurrentFallback]:
    module = importlib.import_module("models.albatross_rwkv7")
    return (
        cast(TMixFactory, getattr(module, "AlbatrossTMix")),
        cast(CMixFactory, getattr(module, "AlbatrossCMix")),
        cast(BlockFactory, getattr(module, "AlbatrossRWKV7Block")),
        cast(RecurrentFallback, getattr(module, "albatross_recurrent_fallback")),
    )


def _inputs(batch: int = BATCH, seq: int = SEQ, dim: int = DIM) -> torch.Tensor:
    values = torch.linspace(-0.75, 0.85, steps=batch * seq * dim, dtype=torch.float32)
    return values.reshape(batch, seq, dim)


def _tmix() -> AlbatrossLayer:
    tmix_factory, _, _, _ = _future_api()
    return tmix_factory(dim=DIM, head_dim=HEAD_DIM, layer_id=0, n_layer=2)


def _cmix() -> AlbatrossLayer:
    _, cmix_factory, _, _ = _future_api()
    return cmix_factory(dim=DIM, layer_id=0, n_layer=2)


def _block() -> AlbatrossLayer:
    _, _, block_factory, _ = _future_api()
    return block_factory(dim=DIM, head_dim=HEAD_DIM, layer_id=0, n_layer=2, mlp_ratio=2.0)


def _state_leaves(state: object | None) -> tuple[torch.Tensor, ...]:
    if state is None:
        return ()
    if torch.is_tensor(state):
        return (state,)
    if is_dataclass(state):
        dataclass_leaves: list[torch.Tensor] = []
        for field in fields(state):
            dataclass_leaves.extend(_state_leaves(cast(object, getattr(state, field.name))))
        return tuple(dataclass_leaves)
    if isinstance(state, Mapping):
        mapping_state = cast(Mapping[object, object], state)
        mapping_leaves: list[torch.Tensor] = []
        for key in sorted(mapping_state.keys(), key=str):
            mapping_leaves.extend(_state_leaves(mapping_state[key]))
        return tuple(mapping_leaves)
    if isinstance(state, Sequence) and not isinstance(state, (str, bytes, bytearray)):
        sequence_leaves: list[torch.Tensor] = []
        for item in state:
            sequence_leaves.extend(_state_leaves(item))
        return tuple(sequence_leaves)
    if hasattr(state, "__dict__"):
        object_leaves: list[torch.Tensor] = []
        attrs = cast(Mapping[str, object], vars(state))
        for key, value in sorted(attrs.items()):
            if not key.startswith("_"):
                object_leaves.extend(_state_leaves(value))
        return tuple(object_leaves)
    raise TypeError(f"state must be None, tensor, dataclass, mapping, or sequence; got {type(state)!r}")


def _assert_state_close(left: object | None, right: object | None, *, atol: float = 1e-5, rtol: float = 1e-5) -> None:
    left_leaves = _state_leaves(left)
    right_leaves = _state_leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for left_tensor, right_tensor in zip(left_leaves, right_leaves):
        assert left_tensor.shape == right_tensor.shape
        assert left_tensor.dtype == right_tensor.dtype
        assert torch.allclose(left_tensor, right_tensor, atol=atol, rtol=rtol)


def _assert_state_batch_item_close(left: object | None, right: object | None, batch_idx: int) -> None:
    left_leaves = _state_leaves(left)
    right_leaves = _state_leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for left_tensor, right_tensor in zip(left_leaves, right_leaves):
        assert left_tensor.shape == right_tensor.shape
        assert left_tensor.shape[0] > batch_idx
        assert torch.allclose(left_tensor[batch_idx], right_tensor[batch_idx], atol=1e-5, rtol=1e-5)


def _assert_module_contract_output(y: torch.Tensor, state: object | None, x: torch.Tensor) -> None:
    assert y.shape == x.shape
    assert y.dtype == torch.float32
    assert y.device.type == "cpu"
    assert torch.isfinite(y).all()
    leaves = _state_leaves(state)
    assert leaves, "state carry must be explicit and observable"
    for tensor in leaves:
        assert tensor.shape[0] == x.shape[0]
        assert tensor.dtype == torch.float32
        assert tensor.device.type == "cpu"
        assert torch.isfinite(tensor).all()


def test_tmix_and_block_sequence_outputs_are_deterministic_fp32_cpu():
    x = _inputs()
    for module in (_tmix(), _block()):
        _ = module.eval()
        with torch.no_grad():
            y1, state1 = module(x)
            y2, state2 = module(x)

        _assert_module_contract_output(y1, state1, x)
        assert torch.allclose(y1, y2, atol=0.0, rtol=0.0)
        _assert_state_close(state1, state2, atol=0.0, rtol=0.0)


def test_tmix_sequence_and_token_state_carry_are_equivalent():
    tmix = _tmix().eval()
    x = _inputs()

    with torch.no_grad():
        y_sequence, state_sequence = tmix(x)
        state: object | None = None
        token_outputs: list[torch.Tensor] = []
        for index in range(x.shape[1]):
            y_token, state = tmix(x[:, index : index + 1], state=state)
            token_outputs.append(y_token)

    assert torch.allclose(y_sequence, torch.cat(token_outputs, dim=1), atol=1e-5, rtol=1e-5)
    _assert_state_close(state_sequence, state)


def test_block_state_carry_is_equivalent_for_sequence_and_tokens():
    block = _block().eval()
    x = _inputs()

    with torch.no_grad():
        y_sequence, state_sequence = block(x)
        state: object | None = None
        token_outputs: list[torch.Tensor] = []
        for index in range(x.shape[1]):
            y_token, state = block(x[:, index : index + 1], state=state)
            token_outputs.append(y_token)

    assert torch.allclose(y_sequence, torch.cat(token_outputs, dim=1), atol=1e-5, rtol=1e-5)
    _assert_state_close(state_sequence, state)


def test_tmix_batch_boundary_isolation_for_outputs_and_state():
    tmix = _tmix().eval()
    x = _inputs()
    changed = x.clone()
    changed[0] = changed[0].flip(0) + 0.5

    with torch.no_grad():
        y, state = tmix(x)
        changed_y, changed_state = tmix(changed)

    assert torch.allclose(y[1], changed_y[1], atol=1e-5, rtol=1e-5)
    _assert_state_batch_item_close(state, changed_state, batch_idx=1)


def test_cmix_token_shift_uses_previous_token_not_only_current_token():
    cmix = _cmix().eval()
    current = torch.linspace(-0.2, 0.4, steps=DIM, dtype=torch.float32)
    previous_a = torch.linspace(-1.0, 1.0, steps=DIM, dtype=torch.float32)
    previous_b = torch.linspace(1.0, -1.0, steps=DIM, dtype=torch.float32)
    seq_a = torch.stack([previous_a, current]).reshape(1, 2, DIM)
    seq_b = torch.stack([previous_b, current]).reshape(1, 2, DIM)

    with torch.no_grad():
        y_a, state_a = cmix(seq_a)
        y_b, state_b = cmix(seq_b)

    _assert_module_contract_output(y_a, state_a, seq_a)
    _assert_module_contract_output(y_b, state_b, seq_b)
    assert not torch.allclose(y_a[:, 1], y_b[:, 1], atol=1e-6, rtol=1e-6)


def test_pure_torch_recurrent_fallback_shape_state_and_token_equivalence():
    _, _, _, albatross_recurrent_fallback = _future_api()
    batch, seq, heads, head_dim = 2, 3, 2, 4
    shape = (batch, seq, heads, head_dim)
    values = torch.linspace(-0.4, 0.6, steps=batch * seq * heads * head_dim, dtype=torch.float32).reshape(shape)
    r = values * 0.1
    w = -values.abs() * 0.5
    k = values.flip(1) * 0.2
    v = values.roll(shifts=1, dims=1) * 0.2
    a = values.flip(-1) * 0.1
    b = values.roll(shifts=1, dims=-1) * 0.1

    y_sequence, state_sequence = albatross_recurrent_fallback(r, w, k, v, a, b, state=None)
    state: torch.Tensor | None = None
    pieces: list[torch.Tensor] = []
    for index in range(seq):
        y_token, state = albatross_recurrent_fallback(
            r[:, index : index + 1],
            w[:, index : index + 1],
            k[:, index : index + 1],
            v[:, index : index + 1],
            a[:, index : index + 1],
            b[:, index : index + 1],
            state=state,
        )
        pieces.append(y_token)

    assert y_sequence.shape == shape
    assert y_sequence.dtype == torch.float32
    assert torch.isfinite(y_sequence).all()
    assert torch.allclose(y_sequence, torch.cat(pieces, dim=1), atol=1e-5, rtol=1e-5)
    assert state_sequence.shape == (batch, heads, head_dim, head_dim)
    assert state_sequence.dtype == torch.float32
    assert torch.isfinite(state_sequence).all()
    assert state is not None
    assert torch.allclose(state_sequence, state, atol=1e-5, rtol=1e-5)


def test_synthetic_backward_gives_finite_gradients_for_all_trainable_parameters():
    x = _inputs(batch=2, seq=3).requires_grad_(True)
    for name, module in (("tmix", _tmix()), ("cmix", _cmix()), ("block", _block())):
        y, state = module(x)
        loss = y.square().mean()
        for tensor in _state_leaves(state):
            loss = loss + tensor.float().square().mean() * 1e-4
        torch.autograd.backward(loss, retain_graph=True)

        params = [(param_name, param) for param_name, param in module.named_parameters() if param.requires_grad]
        assert params, f"{name} must expose trainable parameters"
        missing: list[str] = []
        bad: list[str] = []
        for param_name, param in params:
            if param.grad is None:
                missing.append(param_name)
                continue
            if not torch.isfinite(param.grad).all():
                bad.append(param_name)
        assert not bad, f"{name} non-finite gradients for {bad}"
        assert not missing, f"{name} missing gradients for {missing}"
        module.zero_grad(set_to_none=True)
