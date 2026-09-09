"""Compact, versioned ARC features extracted from a frozen recurrent RWKV."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from models.recurrent_latent_reasoner import recurrent_state_summary

SCHEMA_VERSION = 1


@dataclass
class ArcFeatureRecord:
    task_id: str
    transform_id: int
    query_index: int
    evidence: torch.Tensor
    query: torch.Tensor
    base_state_features: torch.Tensor
    target_grid: torch.Tensor


def deterministic_projection(input_dim: int, output_dim: int, *, seed: int) -> torch.Tensor:
    if input_dim < output_dim or output_dim <= 0:
        raise ValueError("projection requires input_dim >= output_dim > 0")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn(input_dim, output_dim, generator=generator, dtype=torch.float32)
    projection, diagonal = torch.linalg.qr(matrix, mode="reduced")
    signs = torch.where(diagonal.diag() < 0, -1.0, 1.0)
    return projection * signs.unsqueeze(0)


def pooled_state_features(states, *, pool_size: int = 4) -> torch.Tensor:
    return recurrent_state_summary(states, pool_size=pool_size).float()


def run_rwkv_chunks(model, input_ids: torch.Tensor, *, chunk_size: int, past_key_values=None):
    if input_ids.ndim != 2 or chunk_size <= 0:
        raise ValueError("input_ids must be [B,T] and chunk_size must be positive")
    cache = past_key_values
    hidden_chunks = []
    for start in range(0, input_ids.shape[1], chunk_size):
        token_chunk = input_ids[:, start : start + chunk_size]
        output = model(
            input_ids=token_chunk,
            attention_mask=torch.ones_like(token_chunk, dtype=torch.bool),
            past_key_values=cache,
            output_hidden_states=True,
            use_cache=True,
            return_dict=True,
        )
        hidden_chunks.append(output.hidden_states[-1])
        cache = output.past_key_values
    return torch.cat(hidden_chunks, dim=1), cache


def save_feature_record(path: str | Path, record: ArcFeatureRecord) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "task_id": record.task_id,
        "transform_id": int(record.transform_id),
        "query_index": int(record.query_index),
        "evidence": record.evidence.detach().cpu(),
        "query": record.query.detach().cpu(),
        "base_state_features": record.base_state_features.detach().cpu(),
        "target_grid": record.target_grid.detach().cpu().long(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_feature_record(path: str | Path) -> ArcFeatureRecord:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported ARC feature schema {payload.get('schema_version')}; "
            f"expected {SCHEMA_VERSION}"
        )
    required = {
        "task_id", "transform_id", "query_index", "evidence", "query",
        "base_state_features", "target_grid",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"ARC feature record missing keys: {sorted(missing)}")
    return ArcFeatureRecord(**{key: payload[key] for key in required})
