#!/usr/bin/env python3
# pyright: reportAny=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportDeprecated=false, reportExplicitAny=false, reportImplicitStringConcatenation=false, reportMissingParameterType=false, reportOperatorIssue=false, reportOptionalCall=false, reportOptionalMemberAccess=false, reportPrivateUsage=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnannotatedClassAttribute=false, reportUnnecessaryIsInstance=false, reportUnusedCallResult=false, reportUntypedFunctionDecorator=false
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from models.state_hijacking_dit import StateInjectionDiTRELAY  # noqa: E402


DEFAULT_CONFIG = "rwkv_relay_2.9B_state_hijack_dit_vae32_traj32x16"
DEFAULT_SINGLEZ = "outputs_relay/test-v6-2.9B-s2-prefix-suffix-cfg/step_00150000/model.pt"
DEFAULT_OUT = "outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s1-merged/step_00000000/model.pt"
DEFAULT_DIAG = "/tmp/diffrwkv_traj_diag_cleanz/merge_singlez_bridge_v2.json"
DEFAULT_VALIDATION_TEXT = (
    "The history of artificial intelligence began with attempts to describe reasoning as a formal "
    "process. Early researchers built symbolic systems, search programs, and game-playing machines."
)


class DummyRWKV(torch.nn.Module):
    def __init__(self, *, num_layers: int = 32, hidden_size: int = 2560, head_dim: int = 64):
        super().__init__()
        self.config = SimpleNamespace(
            num_hidden_layers=num_layers,
            hidden_size=hidden_size,
            head_dim=head_dim,
        )


def _repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _shape(t: torch.Tensor) -> list[int]:
    return list(t.shape)


def _load_trainable(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("trainable_state", ckpt.get("state_dict", ckpt))
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint state is not a dict: {path}")
    return state, ckpt


def _inspect_keys(state: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    prefixes = (
        "encoder_trunk.",
        "mu_head.",
        "logvar_head.",
        "alpha_heads.",
        "alpha_predictors.",
        "alpha_trunk.",
    )
    exact = {"state_basis", "state_scale"}
    out: dict[str, dict[str, Any]] = {}
    for key in sorted(state):
        if key in exact or key.startswith(prefixes):
            value = state[key]
            out[key] = {
                "shape": _shape(value) if isinstance(value, torch.Tensor) else None,
                "dtype": str(value.dtype) if isinstance(value, torch.Tensor) else str(type(value)),
            }
    return out


def _load_config(config_name: str, n_basis: int, chunk_size: int = 32, horizon: int = 16):
    cfg_path = REPO_ROOT / "configs" / f"{config_name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.update(cfg, "model.n_basis", int(n_basis), merge=False)
    OmegaConf.update(cfg, "model.trajectory_s1_mode", "independent", merge=False)
    OmegaConf.update(cfg, "model.trajectory_chunk_size", int(chunk_size), merge=False)
    OmegaConf.update(cfg, "model.trajectory_horizon", int(horizon), merge=False)
    OmegaConf.update(cfg, "model.trajectory_mode", True, merge=False)
    return cfg


def _read_backbone_dims(cfg) -> dict[str, int]:
    local = cfg.model.get("rwkv_local_path", None)
    if local:
        cfg_path = _repo_path(local) / "config.json"
        if cfg_path.exists():
            import json

            bc = json.loads(cfg_path.read_text())
            return {
                "num_layers": int(bc["num_hidden_layers"]),
                "hidden_size": int(bc["hidden_size"]),
                "head_dim": int(bc.get("head_dim", 64)),
            }
    return {"num_layers": 32, "hidden_size": 2560, "head_dim": 64}


def _build_model(cfg) -> StateInjectionDiTRELAY:
    rwkv = DummyRWKV(**_read_backbone_dims(cfg))
    model = StateInjectionDiTRELAY(
        config=cfg.model,
        rwkv_model=rwkv,
        vocab_size=65536,
        latent_dim=int(cfg.model.latent_dim),
        n_basis=int(cfg.model.n_basis),
        dit_hidden=int(cfg.model.get("dit_hidden", 256)),
        dit_depth=int(cfg.model.get("dit_depth", 4)),
        dit_num_heads=int(cfg.model.get("dit_num_heads", 4)),
        dit_num_tokens=int(cfg.model.get("dit_num_tokens", 4)),
        encoder_type=str(cfg.model.get("encoder_type", "mlp")),
        alpha_type=str(cfg.model.get("alpha_type", "linear")),
        alpha_hidden=int(cfg.model.get("alpha_hidden", 256)),
        latent_stats_path=cfg.model.get("latent_stats_path", None),
    )
    if model.trajectory_state_decoder is not None:
        raise RuntimeError("expected independent trajectory_s1_mode to leave trajectory_state_decoder=None")
    return model


def _candidate_source_keys(dst_key: str) -> list[str]:
    if dst_key.startswith("alpha_heads."):
        legacy = "alpha_predictors." + dst_key[len("alpha_heads."):]
        return [dst_key, legacy]
    return [dst_key]


def _bridge_target_keys(model_state: dict[str, torch.Tensor]) -> list[str]:
    prefixes = ("encoder_trunk.", "mu_head.", "logvar_head.", "alpha_heads.", "alpha_trunk.")
    keys = [k for k in model_state if k.startswith(prefixes)]
    keys.extend(["state_basis", "state_scale"])
    return sorted(k for k in keys if k in model_state)


def _copy_checked(
    dst: dict[str, torch.Tensor],
    source: dict[str, torch.Tensor],
    keys: Iterable[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    copied: list[dict[str, Any]] = []
    missing: list[str] = []
    for dst_key in keys:
        src_key = None
        src_tensor = None
        for candidate in _candidate_source_keys(dst_key):
            if candidate in source:
                src_key = candidate
                src_tensor = source[candidate]
                break
        if src_tensor is None or src_key is None:
            missing.append(dst_key)
            continue
        if tuple(dst[dst_key].shape) != tuple(src_tensor.shape):
            raise RuntimeError(
                f"shape mismatch for {dst_key}: model={tuple(dst[dst_key].shape)} "
                f"source {src_key}={tuple(src_tensor.shape)}"
            )
        dst[dst_key].copy_(src_tensor.to(dtype=dst[dst_key].dtype))
        copied.append(
            {
                "dst_key": dst_key,
                "src_key": src_key,
                "src_checkpoint": "singlez_source",
                "shape": _shape(dst[dst_key]),
                "model_dtype": str(dst[dst_key].dtype),
                "source_dtype": str(src_tensor.dtype),
            }
        )
    if missing:
        raise RuntimeError("missing source keys for transplant: " + ", ".join(missing[:20]))
    return copied, missing


def _source_n_basis(state: dict[str, torch.Tensor]) -> int:
    basis = state.get("state_basis")
    if not isinstance(basis, torch.Tensor) or basis.ndim < 2:
        raise RuntimeError("source checkpoint missing state_basis tensor with n_basis dimension")
    return int(basis.shape[1])


def _load_validation_model(cfg: Any, state: dict[str, torch.Tensor], rwkv: torch.nn.Module, vocab_size: int, device: str):
    model = StateInjectionDiTRELAY(
        config=cfg.model,
        rwkv_model=rwkv,
        vocab_size=vocab_size,
        latent_dim=int(cfg.model.latent_dim),
        n_basis=int(cfg.model.get("n_basis", 16)),
        dit_hidden=int(cfg.model.get("dit_hidden", 256)),
        dit_depth=int(cfg.model.get("dit_depth", 4)),
        dit_num_heads=int(cfg.model.get("dit_num_heads", 4)),
        dit_num_tokens=int(cfg.model.get("dit_num_tokens", 4)),
        encoder_type=str(cfg.model.get("encoder_type", "mlp")),
        alpha_type=str(cfg.model.get("alpha_type", "linear")),
        alpha_hidden=int(cfg.model.get("alpha_hidden", 256)),
        latent_stats_path=cfg.model.get("latent_stats_path", None),
    ).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    critical_missing = [k for k in missing if k in _bridge_target_keys(model.state_dict())]
    if critical_missing:
        raise RuntimeError("validation model missing transplanted keys: " + ", ".join(critical_missing[:20]))
    model.eval()
    return model, list(missing), list(unexpected)


def _flatten_tensors(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.detach().float().reshape(-1).cpu() for t in tensors])


@torch.no_grad()
def _encode_single_chunk(model: StateInjectionDiTRELAY, input_ids: torch.Tensor, attention_mask: torch.Tensor, seed: int):
    out = model.rwkv_model(
        input_ids=input_ids,
        attention_mask=attention_mask.bool(),
        output_hidden_states=True,
        use_cache=True,
        return_dict=True,
    )
    pooled = model._pool_hidden(out.hidden_states[-1], attention_mask)
    if model.encoder_type == "variational":
        h = model.encoder_trunk(pooled.to(next(model.encoder_trunk.parameters()).dtype))
        mu = model.mu_head(h)
        logvar = model.logvar_head(h).clamp(min=-10.0, max=10.0)
    else:
        mu = None
        logvar = None
    torch.manual_seed(seed)
    if input_ids.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    z, _kl = model._encode_pooled(pooled)
    return z, pooled, mu, logvar


@torch.no_grad()
def _encode_trajectory_first_chunk(model: StateInjectionDiTRELAY, input_ids: torch.Tensor, attention_mask: torch.Tensor, seed: int):
    if input_ids.shape[1] != int(model.trajectory_chunk_size):
        raise RuntimeError(
            f"validation chunk length {input_ids.shape[1]} != trajectory_chunk_size {model.trajectory_chunk_size}"
        )
    chunks, chunk_mask, _h_eff, _c = model._trajectory_view(input_ids, attention_mask)
    flat_tokens = chunks.reshape(1, int(model.trajectory_chunk_size))
    flat_mask = chunk_mask.reshape(1, int(model.trajectory_chunk_size)) if chunk_mask is not None else None
    out = model.rwkv_model(
        input_ids=flat_tokens,
        attention_mask=flat_mask.bool() if flat_mask is not None else None,
        output_hidden_states=True,
        use_cache=True,
        return_dict=True,
    )
    pooled = model._pool_hidden(out.hidden_states[-1], flat_mask)
    if model.encoder_type == "variational":
        h = model.encoder_trunk(pooled.to(next(model.encoder_trunk.parameters()).dtype))
        mu = model.mu_head(h)
        logvar = model.logvar_head(h).clamp(min=-10.0, max=10.0)
    else:
        mu = None
        logvar = None
    torch.manual_seed(seed)
    if input_ids.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    z_flat, _kl = model._encode_pooled(pooled)
    return z_flat, pooled, mu, logvar


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.detach().float().reshape(1, -1), b.detach().float().reshape(1, -1), dim=-1).item())


@torch.no_grad()
def _validate_reference(
    source_ckpt: dict[str, Any],
    source_state: dict[str, torch.Tensor],
    merged_cfg: Any,
    merged_state: dict[str, torch.Tensor],
    validation_text: str,
    device: str,
    seed: int,
) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source_cfg = OmegaConf.create(source_ckpt["config"])
    rwkv_path = source_cfg.model.rwkv_local_path
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    rwkv = AutoModelForCausalLM.from_pretrained(
        rwkv_path, trust_remote_code=True, torch_dtype=dtype, local_files_only=True
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(rwkv_path, trust_remote_code=True, local_files_only=True)
    source_model, source_missing, source_unexpected = _load_validation_model(
        source_cfg, source_state, rwkv, len(tokenizer), device
    )
    merged_model, merged_missing, merged_unexpected = _load_validation_model(
        merged_cfg, merged_state, rwkv, len(tokenizer), device
    )
    chunk_size = int(merged_model.trajectory_chunk_size)
    ids = tokenizer(validation_text, add_special_tokens=False).input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    ids = [int(x) for x in ids]
    if len(ids) < chunk_size:
        eos = getattr(tokenizer, "eos_token_id", 0) or 0
        ids = ids + [int(eos)] * (chunk_size - len(ids))
    ids = ids[:chunk_size]
    input_ids = torch.tensor([ids], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    z_single, pooled_single, mu_single, logvar_single = _encode_single_chunk(
        source_model, input_ids, attention_mask, seed
    )
    z_merged, pooled_merged, mu_merged, logvar_merged = _encode_trajectory_first_chunk(
        merged_model, input_ids, attention_mask, seed
    )
    states_single = source_model.predict_states(z_single)
    states_merged = merged_model.predict_states(z_merged)
    state_cosines = [
        _cosine(a, b) for a, b in zip(states_single, states_merged)
    ]
    states_flat_cosine = _cosine(_flatten_tensors(states_single), _flatten_tensors(states_merged))
    validation = {
        "device": device,
        "validation_text": validation_text,
        "chunk_size": chunk_size,
        "token_count": len(ids),
        "source_model_missing_count": len(source_missing),
        "source_model_unexpected_count": len(source_unexpected),
        "merged_model_missing_count": len(merged_missing),
        "merged_model_unexpected_count": len(merged_unexpected),
        "z_single_shape": _shape(z_single),
        "z_merged_shape": _shape(z_merged),
        "pooled_cosine": _cosine(pooled_single, pooled_merged),
        "z_h_cosine": _cosine(z_single, z_merged),
        "z_h_max_abs_diff": float((z_single.detach().float() - z_merged.detach().float()).abs().max().item()),
        "predict_states_flat_cosine": states_flat_cosine,
        "predict_states_min_layer_cosine": min(state_cosines),
        "predict_states_max_abs_diff": float(max((a.detach().float() - b.detach().float()).abs().max().item() for a, b in zip(states_single, states_merged))),
        "state_layer_cosines": state_cosines,
    }
    if mu_single is not None and mu_merged is not None and logvar_single is not None and logvar_merged is not None:
        validation.update({
            "mu_cosine": _cosine(mu_single, mu_merged),
            "mu_max_abs_diff": float((mu_single.detach().float() - mu_merged.detach().float()).abs().max().item()),
            "logvar_cosine": _cosine(logvar_single, logvar_merged),
            "logvar_max_abs_diff": float((logvar_single.detach().float() - logvar_merged.detach().float()).abs().max().item()),
        })
    validation["passed"] = bool(
        validation["z_h_cosine"] > 0.99
        and validation["predict_states_flat_cosine"] > 0.99
        and validation["predict_states_min_layer_cosine"] > 0.99
    )
    if not validation["passed"]:
        if validation.get("pooled_cosine", 0.0) <= 0.99 or validation.get("mu_cosine", 0.0) <= 0.99 or validation["z_h_cosine"] <= 0.99:
            diverged_at = "encode_path"
        else:
            diverged_at = "alpha_mapping_or_state_basis"
        validation["diverged_at"] = diverged_at
        raise RuntimeError("merged-vs-single-z validation failed: " + json.dumps(validation, indent=2))
    return validation


@torch.no_grad()
def _sanity(model: StateInjectionDiTRELAY) -> dict[str, Any]:
    model.eval()
    z = torch.randn(2, model.latent_dim)
    Z = torch.randn(2, model.trajectory_horizon, model.latent_dim)
    states = model.predict_states(z)
    traj_states = model.predict_trajectory_states(Z)
    flat_states = model.predict_states(Z.reshape(-1, model.latent_dim))
    rerouted = [s.reshape(2, model.trajectory_horizon, model.num_heads, model.head_dim, model.head_dim) for s in flat_states]
    max_diffs = [(a - b).abs().max().item() for a, b in zip(traj_states, rerouted)]
    return {
        "predict_states_layers": len(states),
        "predict_states_first_shape": _shape(states[0]),
        "predict_trajectory_states_layers": len(traj_states),
        "predict_trajectory_states_first_shape": _shape(traj_states[0]),
        "trajectory_state_decoder_is_none": model.trajectory_state_decoder is None,
        "independent_route_max_abs_diff": max(max_diffs),
        "independent_route_allclose": all(d == 0.0 for d in max_diffs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default=DEFAULT_CONFIG)
    parser.add_argument("--singlez-ckpt", default=DEFAULT_SINGLEZ)
    parser.add_argument("--output", default=DEFAULT_OUT)
    parser.add_argument("--diag-json", default=DEFAULT_DIAG)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--validation-text", default=DEFAULT_VALIDATION_TEXT)
    parser.add_argument("--validation-seed", type=int, default=1234)
    parser.add_argument("--skip-reference-validation", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=32, help="trajectory_chunk_size (512=32, 4096=64)")
    parser.add_argument("--horizon", type=int, default=16, help="trajectory_horizon (512=16, 4096=64)")
    args = parser.parse_args()

    primary_state, primary_ckpt = _load_trainable(_repo_path(args.singlez_ckpt))
    n_basis = _source_n_basis(primary_state)
    cfg = _load_config(args.config_name, n_basis, chunk_size=args.chunk_size, horizon=args.horizon)

    model = _build_model(cfg)
    model_state = model.state_dict()
    target_keys = _bridge_target_keys(model_state)
    copied, _ = _copy_checked(model_state, primary_state, target_keys)
    model.load_state_dict(model_state, strict=True)

    sanity = _sanity(model)
    trainable_state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if not name.startswith("rwkv_model.")
    }
    validation = None
    if not args.skip_reference_validation:
        validation = _validate_reference(
            primary_ckpt,
            primary_state,
            cfg,
            trainable_state,
            str(args.validation_text),
            str(args.device),
            int(args.validation_seed),
        )

    out_path = _repo_path(args.output)
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite existing checkpoint: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=False)

    payload = {
        "trainable_state": trainable_state,
        "state_basis": model.state_basis.detach().cpu(),
        "state_scale": model.state_scale.detach().cpu(),
        "step": 0,
        "config": OmegaConf.to_yaml(cfg),
        "merge_info": {
            "source_singlez": str(_repo_path(args.singlez_ckpt)),
            "architecture": f"traj32x16 basis{n_basis} independent S1 bridge from validated single-z checkpoint",
            "reference_validation": validation,
        },
    }
    torch.save(payload, out_path)

    diag = {
        "config_name": args.config_name,
        "overrides": {
            "model.n_basis": int(cfg.model.n_basis),
            "model.trajectory_s1_mode": str(cfg.model.trajectory_s1_mode),
            "model.trajectory_chunk_size": int(cfg.model.trajectory_chunk_size),
            "model.trajectory_horizon": int(cfg.model.trajectory_horizon),
        },
        "source_singlez": {
            "path": str(_repo_path(args.singlez_ckpt)),
            "step": primary_ckpt.get("step"),
            "n_basis": n_basis,
            "keys": _inspect_keys(primary_state),
        },
        "copied": copied,
        "copied_count": len(copied),
        "output": str(out_path),
        "sanity": sanity,
        "reference_validation": validation,
    }
    diag_path = Path(args.diag_json)
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    diag_path.write_text(json.dumps(diag, indent=2), encoding="utf-8")

    print(json.dumps({"output": str(out_path), "copied_count": len(copied), "n_basis": n_basis, "sanity": sanity, "reference_validation": validation}, indent=2))


if __name__ == "__main__":
    main()
