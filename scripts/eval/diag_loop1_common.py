# pyright: reportAny=false, reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnusedCallResult=false, reportAttributeAccessIssue=false
"""Shared helpers for Loop-1 no-retrain diagnostics.

All diagnostics load a trajectory single-z-bridge + birwkv condboundary checkpoint
and compare the CLEAN encoder latent Z (z from real text) against the S2-SAMPLED
latent Z (z from the diffusion sampler conditioned on a prefix). The central
question: is the sampled-vs-clean downstream gap (clean ~55 -> sampled ~46) caused
by the sampled state landing OFF the reachable RWKV state manifold?

These helpers only READ existing model APIs; they never train or mutate weights.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.relay_utils import load_relay_model  # noqa: E402
from scripts.eval.sample_prefix_suffix_trajectory_cfg import (  # noqa: E402
    encode_prefix,
    sample_trajectory_cfg,
)
from scripts.eval.diag_clean_z_trajectory import (  # noqa: E402
    _set_pad_token,
    _tokenize_without_specials,
    encode_clean_trajectory_from_ids,
)

# 16 diagnostic prompts (shared across scripts). Each is a (prefix, continuation)
# pair carved from a single coherent passage so that z_clean(continuation) and
# z_sampled(prefix) describe the SAME text region.
PASSAGES = [
    "The history of artificial intelligence is long and rich. Researchers began "
    "exploring machine reasoning in the 1950s, and over the following decades the "
    "field moved from symbolic logic systems toward statistical learning and, more "
    "recently, large neural networks trained on vast text corpora.",
    "Photosynthesis is the process by which green plants convert sunlight into "
    "chemical energy. Chlorophyll in the leaves absorbs light, water is split to "
    "release oxygen, and carbon dioxide is fixed into sugars that the plant uses to "
    "grow and store energy for later use.",
    "The stock market fell sharply today after the central bank signaled that "
    "interest rates would remain elevated for longer than investors had expected. "
    "Technology shares led the decline, while defensive sectors such as utilities "
    "and consumer staples held up comparatively well.",
    "Ancient Rome was founded, according to legend, by the twins Romulus and Remus. "
    "Over centuries it grew from a small settlement on the Tiber into a vast empire "
    "that controlled the Mediterranean world, spreading its language, law, and "
    "engineering across three continents.",
    "Machine learning models require large amounts of high quality data to reach "
    "good performance. The data must be cleaned, labeled where necessary, and split "
    "into training and evaluation sets so that the model can be tuned without "
    "overfitting to the examples it has already seen.",
    "Climate change is one of the most pressing challenges of our era. Rising "
    "concentrations of greenhouse gases trap heat in the atmosphere, driving higher "
    "average temperatures, shifting rainfall patterns, and more frequent extreme "
    "weather events across many regions of the world.",
    "The human immune system defends the body against infection through a layered "
    "set of mechanisms. Physical barriers block most pathogens, innate cells respond "
    "quickly to intruders, and adaptive lymphocytes learn to recognize specific "
    "threats and remember them for the future.",
    "During the Renaissance, artists began to explore perspective, anatomy, and the "
    "play of light with new precision. Painters such as Leonardo and Raphael combined "
    "careful observation of nature with mathematical technique to produce works of "
    "unprecedented realism and depth.",
    "To solve a quadratic equation, first move all terms to one side so the "
    "expression equals zero. Then either factor the quadratic, complete the square, "
    "or apply the quadratic formula, checking each candidate solution back in the "
    "original equation to confirm it is valid.",
    "The novel opens with a quiet description of a coastal town at dawn. Fishing "
    "boats drift out past the harbor wall, gulls wheel overhead, and the narrator "
    "introduces the family whose fortunes the story will follow through several "
    "turbulent decades of change.",
    "Basketball is a sport played by two teams of five players each. The objective "
    "is to score by shooting the ball through the opponent's hoop, while defending "
    "your own basket. Games are fast paced, requiring a blend of individual skill and "
    "coordinated team strategy.",
    "The chemical formula for water shows that each molecule contains two hydrogen "
    "atoms bonded to a single oxygen atom. This simple structure gives water its "
    "polarity, which in turn explains many of its remarkable properties as a solvent "
    "and as the basis of life.",
    "The Supreme Court ruled today that the disputed statute exceeded the powers "
    "granted to the legislature. Writing for the majority, the chief justice argued "
    "that the law intruded on rights the constitution reserves to individuals, and "
    "remanded the case to the lower court.",
    "In a shocking turn of events, scientists discovered a previously unknown species "
    "of deep sea creature living near hydrothermal vents. The organism thrives in "
    "total darkness and extreme pressure, drawing energy from chemicals rather than "
    "sunlight, and challenges old assumptions about life.",
    "Once upon a time in a distant mountain kingdom, a young shepherd found a strange "
    "glowing stone in a cave. The discovery would change the course of his life, "
    "drawing him into an adventure that tested his courage and revealed secrets the "
    "kingdom had guarded for generations.",
    "The recipe calls for two cups of flour, a teaspoon of salt, and a packet of "
    "yeast dissolved in warm water. After kneading the dough until it is smooth and "
    "elastic, let it rise in a warm place until doubled in size before shaping it and "
    "baking it in a hot oven.",
]


def build_model(ckpt_dir: str, device: str):
    """Load the trajectory model + tokenizer and return (model, tokenizer, dtype, pad_id)."""
    model, _rwkv, tokenizer, _ckpt, _cfg = load_relay_model(ckpt_dir, device)
    model.eval()
    pad_id = _set_pad_token(tokenizer)
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    return model, tokenizer, dtype, pad_id


def split_prefix_suffix(tokenizer: Any, passage: str, model: Any, pad_id: int):
    """Tokenize a passage and split into (prefix_ids, suffix_ids).

    Suffix has length horizon*chunk_size (the trajectory window). Prefix is the
    text preceding it (at least a few tokens). Returns padded suffix ids of exactly
    `total` length plus the real suffix length.
    """
    horizon = int(model.trajectory_horizon)
    chunk_size = int(model.trajectory_chunk_size)
    total = horizon * chunk_size
    ids = _tokenize_without_specials(tokenizer, passage)
    if len(ids) < total + 8:
        # repeat the passage until we have enough tokens for a clean prefix+suffix
        reps = (total + 8) // max(1, len(ids)) + 1
        ids = (ids * reps)
    # prefix = first ~1/3, suffix = the trajectory window right after it
    prefix_len = max(8, min(len(ids) - total, total // 2))
    prefix_ids = ids[:prefix_len]
    suffix_ids = ids[prefix_len:prefix_len + total]
    actual_suffix_len = len(suffix_ids)
    if actual_suffix_len < total:
        suffix_ids = suffix_ids + [pad_id] * (total - actual_suffix_len)
    return prefix_ids, suffix_ids, actual_suffix_len


@torch.no_grad()
def get_z_clean(model: Any, suffix_ids: list[int], device: str) -> torch.Tensor:
    """Clean encoder latent Z[1,H,D] from real suffix tokens."""
    return encode_clean_trajectory_from_ids(model, suffix_ids, device)


@torch.no_grad()
def get_z_sampled(
    model: Any,
    prefix_ids: list[int],
    device: str,
    dtype: torch.dtype,
    steps: int,
    cfg_scale: float,
) -> torch.Tensor:
    """S2-sampled latent Z[1,H,D] conditioned on the prefix (fresh Gaussian init)."""
    input_ids = torch.tensor([prefix_ids], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    z_prefix, _cache, _logits = encode_prefix(model, input_ids, attention_mask)
    z_sampled = sample_trajectory_cfg(model, z_prefix, steps, cfg_scale, device, dtype)
    return z_sampled


@torch.no_grad()
def per_layer_states_flat(model: Any, z: torch.Tensor) -> list[torch.Tensor]:
    """predict_trajectory_states(z[1,H,D]) -> list per layer of [H, heads, hd, hd]
    flattened to float32 [H, -1] on the same device."""
    states = model.predict_trajectory_states(z)  # list, each [B,H,heads,hd,hd]
    out = []
    for s in states:
        s0 = s[0].float()  # [H, heads, hd, hd]
        out.append(s0.reshape(s0.shape[0], -1))  # [H, feat]
    return out
