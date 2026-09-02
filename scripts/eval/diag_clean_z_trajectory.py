#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.eval.relay_utils import load_relay_model
from models.state_hijacking_dit import cosine_alpha_bar
from scripts.eval.sample_prefix_suffix_cfg import encode_prefix as encode_single_prefix
from scripts.eval.sample_prefix_suffix_trajectory_cfg import encode_prefix, sample_trajectory_cfg
from scripts.eval.run_cola_dlm_tasks_prefix_suffix_cfg import build_prompt, get_ground_truth, iter_jsonl
from scripts.eval.run_cola_dlm_tasks_prefix_suffix_trajectory_cfg import sample_next_token, recompute_logits_from_injected_cache


OPEN_PROMPT_PAIRS = [
    ("The history of artificial intelligence", "began with attempts to describe reasoning as a formal process. Early researchers built symbolic systems, search programs, and game-playing machines, then later shifted toward statistical learning as larger datasets became available. The field changed again when neural networks became practical at scale, connecting perception, language, planning, and tool use in a single research agenda."),
    ("In a distant future", "human settlements orbit quiet moons and trade stories through slow radio links. Engineers maintain gardens under glass while historians preserve fragments of old languages. The most valuable machines are not weapons or engines, but patient archives that remember how each colony solved hunger, illness, and isolation."),
    ("The economy of the region", "depends on ports, farms, small manufacturers, and a growing network of service companies. When transport costs rise, local producers cooperate more closely and diversify their supply chains. Public investment in schools, clinics, and reliable power often matters more than a single headline industry."),
    ("A useful explanation of climate change", "starts with the balance between incoming sunlight and outgoing heat. Greenhouse gases absorb part of the heat that Earth would otherwise radiate into space. As their concentration increases, temperatures shift, weather patterns become less stable, and communities must adapt infrastructure, agriculture, and water management."),
    ("The young scientist opened the notebook", "and found a careful record of failed experiments. Each page contained a question, a measurement, and a note about what to try next. The failures were not embarrassing; they were the map that showed which assumptions had to be replaced before the discovery could be repeated."),
    ("During the long voyage", "the crew learned to treat routine as a form of safety. Instruments were checked twice, meals were shared at regular hours, and small repairs were logged before they became emergencies. The journey felt less lonely when everyone understood how their work protected the others."),
    ("The city council debated", "whether to restore the old railway station or replace it with a modern terminal. Residents argued that the building carried memories of migration and reunion. Planners replied that accessibility and capacity mattered too. The final proposal kept the facade, expanded the platforms, and added public space."),
    ("When children learn to read", "they combine memory, sound, attention, and curiosity. Good teachers do not merely drill symbols; they connect words to stories, questions, and familiar experiences. Practice matters, but confidence matters as well, because a hesitant reader often needs encouragement before speed."),
    ("The mountain village", "survived winter by planning together. Families stored grain, repaired roofs before the first snow, and shared tools that no single household could afford. Outsiders sometimes saw only poverty, but the villagers understood the wealth contained in trust and practical knowledge."),
    ("Modern medicine has changed", "because diagnosis, prevention, and treatment now depend on many kinds of evidence. Laboratory tests, imaging, clinical judgment, and patient history each reveal a different part of the problem. The best care combines technical precision with clear conversation and respect."),
    ("The old library", "stood at the center of town with its stone steps worn smooth by generations of readers. Inside, the shelves held maps, letters, manuals, poems, and newspapers. People came for information, but they often stayed because the quiet room made difficult thoughts easier to arrange."),
    ("A good software system", "is shaped by constraints that are easy to forget: failures, updates, confusing inputs, and users with urgent goals. Clean interfaces help teams change one part without breaking another. Tests are not decorations; they are agreements about behavior that should remain true."),
    ("The river after the storm", "carried branches, soil, and pieces of broken fence toward the lowlands. By morning the water had begun to fall, leaving marks on the bridge supports. Neighbors walked the banks together, checking pumps, clearing drains, and planning stronger barriers for the next season."),
    ("The teacher asked the class", "to explain not only the answer but the path that led to it. Some students drew diagrams, others wrote equations, and a few described examples from daily life. The discussion showed that understanding can appear in several forms before it becomes formal language."),
    ("In the history of trade", "routes often mattered as much as goods. Roads, ports, canals, and later railways changed what people could buy and which towns could grow. Trade also carried techniques, beliefs, diseases, and laws, so its influence was cultural as well as economic."),
    ("The small robot entered the greenhouse", "and measured moisture, leaf color, temperature, and light. It did not replace the gardener; it gave the gardener earlier warnings. Together they found patterns that would have been invisible during a quick morning inspection."),
    ("A fair legal system", "requires rules that are public, procedures that are consistent, and officials who can be challenged when they misuse power. Fairness is not only the outcome of a single case. It is also the public belief that evidence matters more than status or influence."),
    ("The expedition reached the coast", "after weeks of crossing dry plains. The travelers expected silence, but the shore was full of movement: birds circling, waves striking black rocks, and fishing boats returning at dusk. Their maps ended there, while the questions they carried became larger."),
    ("The invention of printing", "changed scholarship by making texts cheaper, more consistent, and easier to compare. Readers could argue over the same page instead of relying on fragile copies. Over time, printing strengthened schools, religious debate, scientific exchange, and political pamphlets."),
    ("The farmer watched the clouds", "and decided to delay planting for two more days. Experience had taught her that a calendar was useful but never sufficient. Soil temperature, wind, insects, and the behavior of birds all contributed small pieces to a practical forecast."),
]

CKPT_512_TRAJ = "outputs_relay/owt512-traj32x16-2.9B-basis32-prefix-suffix-blend0p5-s2-rwkv-ddpm/step_00150000"
CKPT_4096_A = "outputs_relay/fineweb4096-traj64x64-2.9B-basis32-prefix-suffix-blend0p5-s2-rwkv-s1-birwkv-ddpm-h200x4-dai-from-s1step13k-20260629/step_00055000"
CKPT_4096_B = "outputs_relay/albatross-goose-2.9B-fineweb4096-launch-20260629-145500/step_00045000"
CKPT_SINGLE_Z = "outputs_relay/test-v6-2.9B-s2-prefix-suffix-cfg/step_00150000"


def parse_args():
    p = argparse.ArgumentParser(description="Run clean-Z trajectory diagnostic on Cola-DLM task JSONL files.")
    p.add_argument("--diag_mode", choices=("tasks", "single_clean", "longgen", "cfg_probe", "both", "convffn", "blend_gate", "sample_512_4096"), default="tasks")
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--task_data_dir", default="baseline/Cola-DLM/eval_output/tasks_default")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--tasks", default="mmlu,obqa,race")
    p.add_argument("--max_samples", type=int, default=100)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--longgen_prompts", type=int, default=20)
    p.add_argument("--trajectory_s1_mode", choices=("independent", "transformer", "rwkv", "birwkv"), default=None)
    p.add_argument("--trajectory_state_blend", type=float, required=True)
    p.add_argument("--inject_every", type=int, default=1)
    p.add_argument("--chunk0_only", action="store_true")
    p.add_argument("--convffn_variant", choices=("zero", "preserve", "blend"), default="zero")
    p.add_argument("--calibration_samples", type=int, default=50)
    p.add_argument("--json_output", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _set_pad_token(tokenizer: Any) -> int:
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is not None:
        return int(pad_id)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
        return int(eos_id)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is not None:
        tokenizer.pad_token = tokenizer.unk_token
        return int(unk_id)
    return 0


def _tokenize_without_specials(tokenizer: Any, text: str) -> list[int]:
    try:
        ids = tokenizer(text, add_special_tokens=False).input_ids
    except TypeError:
        ids = tokenizer(text).input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


@torch.no_grad()
def encode_clean_trajectory(model: Any, tokenizer: Any, continuation: str, device: str, pad_id: int):
    horizon = int(model.trajectory_horizon)
    chunk_size = int(model.trajectory_chunk_size)
    total = horizon * chunk_size
    ids = _tokenize_without_specials(tokenizer, continuation)
    if not ids:
        eos_id = getattr(tokenizer, "eos_token_id", None)
        ids = [int(eos_id if eos_id is not None else pad_id)]
    ids = ids[:total]
    actual_len = len(ids)
    ids = ids + [pad_id] * (total - actual_len)
    input_ids = torch.tensor([ids], device=device, dtype=torch.long)
    attention_mask = torch.zeros((1, total), device=device, dtype=torch.long)
    attention_mask[:, :actual_len] = 1
    z_clean, _kl, h_eff, c_eff = model._encode_trajectory_chunks(input_ids, attention_mask)
    expected = (1, horizon, int(model.latent_dim))
    if tuple(z_clean.shape) != expected:
        raise RuntimeError(f"clean z shape {tuple(z_clean.shape)} != expected {expected}; H_eff={h_eff}, C={c_eff}")
    return z_clean, actual_len


@torch.no_grad()
def encode_clean_trajectory_from_ids(model: Any, suffix_ids: list[int], device: str):
    input_ids = torch.tensor([suffix_ids], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    z_clean, _kl, _h_eff, _c_eff = model._encode_trajectory_chunks(input_ids, attention_mask)
    return z_clean


@torch.no_grad()
def encode_clean_single_z(model: Any, tokenizer: Any, continuation: str, device: str, pad_id: int):
    ids = _tokenize_without_specials(tokenizer, continuation)
    if not ids:
        eos_id = getattr(tokenizer, "eos_token_id", None)
        ids = [int(eos_id if eos_id is not None else pad_id)]
    input_ids = torch.tensor([ids], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model.rwkv_model(
            input_ids=input_ids,
            attention_mask=attention_mask.bool(),
            output_hidden_states=True,
            use_cache=True,
            return_dict=True,
        )
        pooled = model._pool_hidden(out.hidden_states[-1], attention_mask)
    z_clean, _kl = model._encode_pooled(pooled)
    expected = (1, int(model.latent_dim))
    if tuple(z_clean.shape) != expected:
        raise RuntimeError(f"single clean z shape {tuple(z_clean.shape)} != expected {expected}")
    return z_clean, len(ids)


@torch.no_grad()
def generate_answer_single_clean(model: Any, tokenizer: Any, input_ids, z_clean, args):
    attention_mask = torch.ones_like(input_ids)
    states = model.predict_states(z_clean)
    out = model.rwkv_model(input_ids=input_ids, attention_mask=attention_mask.bool(), use_cache=True, return_dict=True)
    cache = model.inject_into_cache(out.past_key_values, states)
    out = model.rwkv_model(input_ids=input_ids, past_key_values=cache, use_cache=True, return_dict=True)
    cache = out.past_key_values
    new_ids: list[int] = []
    all_ids = list(input_ids[0].tolist())
    logits = out.logits[0, -1]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    for _ in range(args.max_new_tokens):
        next_id = sample_next_token(logits, all_ids, args)
        if eos_id is not None and next_id == eos_id:
            break
        new_ids.append(next_id)
        all_ids.append(next_id)
        out = model.rwkv_model(
            input_ids=torch.tensor([[next_id]], device=input_ids.device),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = out.past_key_values
        logits = out.logits[0, -1]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip(), input_ids.shape[1], len(new_ids)


def _should_inject_chunk(h: int, inject_every: int, chunk0_only: bool) -> bool:
    if chunk0_only:
        return h == 0
    return h % max(1, int(inject_every)) == 0


def blend_recurrent_with_convffn_variant(model: Any, cache, predicted_states, blend: float, variant: str, chunk_idx: int):
    if chunk_idx == 0 or variant == "zero":
        return model.blend_into_cache(cache, predicted_states, blend)
    blend = float(blend)
    for layer_idx, st in enumerate(predicted_states):
        layer = cache.layers[layer_idx]
        if layer.state is None:
            layer.state = {"recurrent_state": None, "attn_state": None, "conv_state": None, "ffn_state": None}
        state = cast(dict[str, Any], layer.state)
        current = state.get("recurrent_state")
        planned = st.to(torch.float32)
        if isinstance(current, torch.Tensor) and 0.0 < blend < 1.0:
            state["recurrent_state"] = current.to(torch.float32) * (1.0 - blend) + planned * blend
        elif blend <= 0.0 and isinstance(current, torch.Tensor):
            state["recurrent_state"] = current.to(torch.float32)
        else:
            state["recurrent_state"] = planned
        if variant == "blend":
            for sub_key in ("conv_state", "ffn_state"):
                value = state.get(sub_key)
                if isinstance(value, torch.Tensor):
                    state[sub_key] = value.to(torch.float32) * (1.0 - blend)
    return cache


def layer_band_indices(num_layers: int, bands: int = 4) -> list[tuple[int, int]]:
    result = []
    for band in range(bands):
        start = (num_layers * band) // bands
        end = (num_layers * (band + 1)) // bands
        result.append((start, end))
    return result


def expand_band_blends(num_layers: int, band_blends: list[float]) -> list[float]:
    per_layer = [0.0 for _ in range(num_layers)]
    for band_value, (start, end) in zip(band_blends, layer_band_indices(num_layers, len(band_blends))):
        for layer_idx in range(start, end):
            per_layer[layer_idx] = float(band_value)
    return per_layer


def blend_into_cache_per_layer(model: Any, cache, predicted_states, per_layer_blends: list[float], norm_match: bool):
    for layer_idx, st in enumerate(predicted_states):
        layer = cache.layers[layer_idx]
        if layer.state is None:
            layer.state = {"recurrent_state": None, "attn_state": None, "conv_state": None, "ffn_state": None}
        state = cast(dict[str, Any], layer.state)
        current = state.get("recurrent_state")
        planned = st.to(torch.float32)
        if norm_match and isinstance(current, torch.Tensor):
            current_norm = current.detach().float().norm()
            planned_norm = planned.detach().float().norm().clamp(min=1e-6)
            planned = planned * (current_norm / planned_norm)
        blend = float(per_layer_blends[layer_idx])
        if isinstance(current, torch.Tensor) and 0.0 < blend < 1.0:
            state["recurrent_state"] = current.to(torch.float32) * (1.0 - blend) + planned * blend
        elif blend <= 0.0 and isinstance(current, torch.Tensor):
            state["recurrent_state"] = current.to(torch.float32)
        else:
            state["recurrent_state"] = planned
        for sub_key in ("conv_state", "ffn_state"):
            value = state.get(sub_key)
            if isinstance(value, torch.Tensor):
                state[sub_key] = torch.zeros_like(value)
        layer._seen_tokens = 0
    if hasattr(cache, "_seen_tokens"):
        cache._seen_tokens = 0
    return cache


def with_per_layer_blends(args, per_layer_blends: list[float] | None, norm_match: bool = False):
    local_args = argparse.Namespace(**vars(args))
    local_args.per_layer_blends = per_layer_blends
    local_args.norm_match = bool(norm_match)
    return local_args


@torch.no_grad()
def generate_answer_trajectory_frequency(model: Any, tokenizer: Any, input_ids, prefix_cache, prefix_logits, z_traj, args):
    chunk_size = int(model.trajectory_chunk_size)
    s1_mode = str(model.config.get("trajectory_s1_mode", "independent"))
    all_ids = list(input_ids[0].tolist())
    new_ids: list[int] = []
    past_kv = prefix_cache
    logits = prefix_logits
    eos_id = getattr(tokenizer, "eos_token_id", None)
    inject_every = max(1, int(args.inject_every))
    injected_chunks: list[int] = []

    if s1_mode in ("transformer", "rwkv", "birwkv"):
        layer_states = model.predict_trajectory_states(z_traj)
        blend = float(model.config.get("trajectory_state_blend", getattr(model, "trajectory_state_blend", 1.0)))
    else:
        layer_states = None
        blend = 1.0

    stop = False
    for h in range(z_traj.shape[1]):
        if stop or len(new_ids) >= args.max_new_tokens:
            break
        if _should_inject_chunk(h, inject_every, bool(args.chunk0_only)):
            injected_chunks.append(h)
            if layer_states is not None:
                states_h = [layer_state[:, h] for layer_state in layer_states]
                per_layer_blends = getattr(args, "per_layer_blends", None)
                if per_layer_blends is not None:
                    past_kv = blend_into_cache_per_layer(
                        model, past_kv, states_h, list(per_layer_blends), bool(getattr(args, "norm_match", False))
                    )
                else:
                    past_kv = blend_recurrent_with_convffn_variant(
                        model, past_kv, states_h, blend, str(getattr(args, "convffn_variant", "zero")), h
                    )
            else:
                states_h = model.predict_states(z_traj[:, h])
                past_kv = model.inject_into_cache(past_kv, states_h)
            # Recompute logits from the injected cache, so the first token sampled
            # for this chunk conditions on the injected state rather than the
            # pre-injection prefix logits (matches run_cola_dlm_tasks_*_trajectory_cfg).
            context_ids = torch.tensor([all_ids], device=input_ids.device, dtype=torch.long)
            context_mask = torch.ones_like(context_ids)
            past_kv, logits_batch = recompute_logits_from_injected_cache(model, context_ids, context_mask, past_kv)
            logits = logits_batch[0]
        for _ in range(chunk_size):
            if len(new_ids) >= args.max_new_tokens:
                stop = True
                break
            next_id = sample_next_token(logits, all_ids, args)
            if eos_id is not None and next_id == eos_id:
                stop = True
                break
            new_ids.append(next_id)
            all_ids.append(next_id)
            out = model.rwkv_model(
                input_ids=torch.tensor([[next_id]], device=input_ids.device),
                past_key_values=past_kv,
                use_cache=True,
                return_dict=True,
            )
            past_kv = out.past_key_values
            logits = out.logits[0, -1]
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return text, input_ids.shape[1], len(new_ids), injected_chunks


@torch.no_grad()
def generate_token_ids_trajectory_frequency(model: Any, tokenizer: Any, input_ids, prefix_cache, prefix_logits, z_traj, args, inject_every: int, chunk0_only: bool):
    local_args = argparse.Namespace(**vars(args))
    local_args.inject_every = inject_every
    local_args.chunk0_only = chunk0_only
    text, prompt_tokens, generated_tokens, injected_chunks = generate_answer_trajectory_frequency(
        model, tokenizer, input_ids, prefix_cache, prefix_logits, z_traj, local_args
    )
    ids = _tokenize_without_specials(tokenizer, text)
    return ids, text, prompt_tokens, generated_tokens, injected_chunks


@torch.no_grad()
def generate_token_ids_raw(model: Any, tokenizer: Any, input_ids, args):
    out = model.rwkv_model(input_ids=input_ids, use_cache=True, return_dict=True)
    cache = out.past_key_values
    all_ids = list(input_ids[0].tolist())
    new_ids: list[int] = []
    logits = out.logits[0, -1]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    for _ in range(args.max_new_tokens):
        next_id = sample_next_token(logits, all_ids, args)
        if eos_id is not None and next_id == eos_id:
            break
        new_ids.append(next_id)
        all_ids.append(next_id)
        out = model.rwkv_model(
            input_ids=torch.tensor([[next_id]], device=input_ids.device),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = out.past_key_values
        logits = out.logits[0, -1]
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return new_ids, text


@torch.no_grad()
def generate_hybrid_single_start(model: Any, single_model: Any, single_tokenizer: Any, tokenizer: Any, prompt: str, z_traj, args):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
    attention_mask = torch.ones_like(input_ids)
    _z_prefix, prefix_cache, prefix_logits = encode_prefix(model, input_ids, attention_mask)
    single_ids = single_tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
    single_mask = torch.ones_like(single_ids)
    z_single = encode_single_prefix(single_model, single_ids, single_mask)
    single_states = single_model.predict_states(z_single)
    layer_states = model.predict_trajectory_states(z_traj)
    chunk_size = int(model.trajectory_chunk_size)
    past_kv = prefix_cache
    logits = prefix_logits
    all_ids = list(input_ids[0].tolist())
    new_ids: list[int] = []
    eos_id = getattr(tokenizer, "eos_token_id", None)
    blend = float(model.config.get("trajectory_state_blend", getattr(model, "trajectory_state_blend", 0.4)))
    for h in range(z_traj.shape[1]):
        if len(new_ids) >= args.max_new_tokens:
            break
        if h == 0:
            past_kv = model.blend_into_cache(past_kv, single_states, blend)
        else:
            states_h = [layer_state[:, h] for layer_state in layer_states]
            past_kv = model.blend_into_cache(past_kv, states_h, blend)
        for _ in range(chunk_size):
            if len(new_ids) >= args.max_new_tokens:
                break
            next_id = sample_next_token(logits, all_ids, args)
            if eos_id is not None and next_id == eos_id:
                return new_ids, tokenizer.decode(new_ids, skip_special_tokens=True).strip(), float(z_single.detach().float().norm(dim=-1).mean().item())
            new_ids.append(next_id)
            all_ids.append(next_id)
            out = model.rwkv_model(
                input_ids=torch.tensor([[next_id]], device=input_ids.device),
                past_key_values=past_kv,
                use_cache=True,
                return_dict=True,
            )
            past_kv = out.past_key_values
            logits = out.logits[0, -1]
    return new_ids, tokenizer.decode(new_ids, skip_special_tokens=True).strip(), float(z_single.detach().float().norm(dim=-1).mean().item())


def configure_trajectory_model(model: Any, cfg: Any, s1_mode: str | None, blend: float):
    model._prefix_suffix_trajectory_s2 = True
    model._training_stage = 2
    model._cfg_drop_prob = float(cfg.training.get("cfg_drop_prob", 0.0))
    if s1_mode is not None:
        model.config.trajectory_s1_mode = s1_mode
        model.trajectory_s1_mode = s1_mode
    model.config.trajectory_state_blend = float(blend)
    model.trajectory_state_blend = float(blend)


def safe_load_checkpoint(ckpt_dir: str, device: str):
    if not (Path(ckpt_dir) / "model.pt").exists():
        return None, {"status": "skipped", "reason": f"missing model.pt under {ckpt_dir}"}
    try:
        return load_relay_model(ckpt_dir, device), {"status": "loaded", "ckpt_dir": ckpt_dir}
    except Exception as exc:
        return None, {"status": "skipped", "reason": repr(exc), "ckpt_dir": ckpt_dir}


def ngram_repetition(ids: list[int], n: int) -> float:
    total = max(0, len(ids) - n + 1)
    if total == 0:
        return 0.0
    grams = [tuple(ids[i : i + n]) for i in range(total)]
    return float((total - len(set(grams))) / total)


def distinct_n(ids: list[int], n: int) -> float:
    total = max(0, len(ids) - n + 1)
    if total == 0:
        return 0.0
    grams = [tuple(ids[i : i + n]) for i in range(total)]
    return float(len(set(grams)) / total)


def aggregate_generation_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    count = max(1, len(records))
    return {
        "repeat4": sum(float(r["repeat4"]) for r in records) / count,
        "repeat8": sum(float(r["repeat8"]) for r in records) / count,
        "distinct2": sum(float(r["distinct2"]) for r in records) / count,
        "mean_len": sum(float(r["length"]) for r in records) / count,
        "frac_full_len": sum(1.0 for r in records if int(r["length"]) >= 256) / count,
    }


def aggregate_sample_records(mode_name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = aggregate_generation_metrics(records)
    return {"mode": mode_name, **metrics}


@torch.no_grad()
def sample_trajectory_mode_records(model: Any, tokenizer: Any, prompts: list[str], args, max_new_tokens: int, mode_name: str) -> list[dict[str, Any]]:
    dtype = torch.bfloat16
    records: list[dict[str, Any]] = []
    local_args = argparse.Namespace(**vars(args))
    local_args.max_new_tokens = max_new_tokens
    local_args.temperature = 0.7
    local_args.top_k = 10
    local_args.top_p = 0.75
    local_args.repetition_penalty = 1.0
    local_args.inject_every = 1
    local_args.chunk0_only = False
    for prompt_i, prompt in enumerate(prompts):
        torch.manual_seed(int(args.seed) + prompt_i)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
        attention_mask = torch.ones_like(input_ids)
        z_prefix, prefix_cache, prefix_logits = encode_prefix(model, input_ids, attention_mask)
        z_traj = sample_trajectory_cfg(model, z_prefix, 100, 2.0, args.device, dtype)
        ids, text, _ptok, gtok, injected_chunks = generate_token_ids_trajectory_frequency(
            model, tokenizer, input_ids, prefix_cache, prefix_logits, z_traj, local_args, 1, False
        )
        records.append({
            "prompt": prompt,
            "length": int(gtok),
            "repeat4": ngram_repetition(ids, 4),
            "repeat8": ngram_repetition(ids, 8),
            "distinct2": distinct_n(ids, 2),
            "injected_chunks": injected_chunks,
            "text": text,
            "z_norm": float(z_traj.detach().float().norm(dim=-1).mean().item()),
        })
    return records


@torch.no_grad()
def sample_hybrid_records(model: Any, tokenizer: Any, single_model: Any, single_tokenizer: Any, prompts: list[str], args) -> list[dict[str, Any]]:
    dtype = torch.bfloat16
    records: list[dict[str, Any]] = []
    local_args = argparse.Namespace(**vars(args))
    local_args.max_new_tokens = 256
    local_args.temperature = 0.7
    local_args.top_k = 10
    local_args.top_p = 0.75
    local_args.repetition_penalty = 1.0
    for prompt_i, prompt in enumerate(prompts):
        torch.manual_seed(int(args.seed) + prompt_i)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
        attention_mask = torch.ones_like(input_ids)
        z_prefix, _cache, _logits = encode_prefix(model, input_ids, attention_mask)
        z_traj = sample_trajectory_cfg(model, z_prefix, 100, 2.0, args.device, dtype)
        ids, text, single_norm = generate_hybrid_single_start(model, single_model, single_tokenizer, tokenizer, prompt, z_traj, local_args)
        records.append({
            "prompt": prompt,
            "length": len(ids),
            "repeat4": ngram_repetition(ids, 4),
            "repeat8": ngram_repetition(ids, 8),
            "distinct2": distinct_n(ids, 2),
            "text": text,
            "z_traj_norm": float(z_traj.detach().float().norm(dim=-1).mean().item()),
            "z_single_prefix_norm": single_norm,
        })
    return records


def run_sample_512_vs_4096(args) -> dict[str, Any]:
    prompts = [pair[0] for pair in OPEN_PROMPT_PAIRS[:10]]
    modes: list[dict[str, Any]] = []
    load_notes: dict[str, Any] = {}

    loaded_512, note_512 = safe_load_checkpoint(CKPT_512_TRAJ, args.device)
    load_notes["trajectory_512"] = note_512
    if loaded_512 is not None:
        model_512, _rwkv_512, tok_512, _ckpt_512, cfg_512 = loaded_512
        model_512 = cast(Any, model_512)
        configure_trajectory_model(model_512, cfg_512, "rwkv", 0.4)
        records_512 = sample_trajectory_mode_records(model_512, tok_512, prompts, args, 256, "512 trajectory")
        modes.append({"mode": "512 trajectory", "checkpoint": CKPT_512_TRAJ, "records": records_512, "metrics": aggregate_sample_records("512 trajectory", records_512), "examples": records_512[:3]})

        loaded_single, note_single = safe_load_checkpoint(CKPT_SINGLE_Z, args.device)
        load_notes["single_z_for_hybrid"] = note_single
        if loaded_single is not None:
            single_model, _rwkv_s, single_tok, _ckpt_s, _cfg_s = loaded_single
            single_model = cast(Any, single_model)
            hybrid_records = sample_hybrid_records(model_512, tok_512, single_model, single_tok, prompts, args)
            modes.append({"mode": "512 hybrid single-z chunk0 + trajectory", "checkpoint": CKPT_512_TRAJ, "single_checkpoint": CKPT_SINGLE_Z, "records": hybrid_records, "metrics": aggregate_sample_records("512 hybrid single-z chunk0 + trajectory", hybrid_records), "examples": hybrid_records[:3]})
            del single_model, _rwkv_s, single_tok, loaded_single
            gc.collect()
            torch.cuda.empty_cache()
        del model_512, _rwkv_512, tok_512, loaded_512
        gc.collect()
        torch.cuda.empty_cache()

    for key, ckpt_dir, label, s1_mode in (
        ("trajectory_4096_A", CKPT_4096_A, "4096 trajectory A", "rwkv"),
        ("trajectory_4096_B", CKPT_4096_B, "4096 trajectory B albatross", "rwkv"),
    ):
        loaded, note = safe_load_checkpoint(ckpt_dir, args.device)
        load_notes[key] = note
        if loaded is None:
            continue
        model = _rwkv = tok = None
        try:
            model, _rwkv, tok, _ckpt, cfg = loaded
            model = cast(Any, model)
            configure_trajectory_model(model, cfg, s1_mode, 0.4)
            records = sample_trajectory_mode_records(model, tok, prompts, args, 512, label)
            modes.append({"mode": label, "checkpoint": ckpt_dir, "records": records, "metrics": aggregate_sample_records(label, records), "examples": records[:3]})
        except Exception as exc:
            load_notes[key] = {"status": "skipped_after_load", "reason": repr(exc), "ckpt_dir": ckpt_dir}
        finally:
            del model, _rwkv, tok, loaded
            gc.collect()
            torch.cuda.empty_cache()

    return {
        "settings": {"gpu": "CUDA_VISIBLE_DEVICES=2", "prompt_count": len(prompts), "steps": 100, "cfg_scale": 2.0, "temperature": 0.7, "top_k": 10, "top_p": 0.75, "repetition_penalty": 1.0, "blend": 0.4, "seed": int(args.seed)},
        "load_notes": load_notes,
        "modes": modes,
        "metrics_rows": [mode["metrics"] for mode in modes],
    }


def make_calibration_examples(tokenizer: Any, task_data_dir: str, max_examples: int) -> list[tuple[list[int], list[int]]]:
    examples: list[tuple[list[int], list[int]]] = []
    suffix_len = 96
    for task in ("race", "mmlu", "obqa"):
        path = Path(task_data_dir) / f"{task}.jsonl"
        if not path.exists():
            continue
        for _sample_i, item in iter_jsonl(path, max_examples * 4):
            prompt = build_prompt(task, item)
            ids = _tokenize_without_specials(tokenizer, prompt)
            if len(ids) < suffix_len + 32:
                continue
            prefix_ids = ids[:-suffix_len]
            suffix_ids = ids[-suffix_len:]
            examples.append((prefix_ids, suffix_ids))
            if len(examples) >= max_examples:
                return examples
    return examples


@torch.no_grad()
def teacher_forced_suffix_ce(model: Any, prefix_ids: list[int], suffix_ids: list[int], per_layer_blends: list[float], norm_match: bool, device: str) -> float:
    C = int(model.trajectory_chunk_size)
    usable = (len(suffix_ids) // C) * C
    suffix_ids = suffix_ids[:usable]
    if len(prefix_ids) == 0 or usable < C * 2:
        return float("nan")
    prefix = torch.tensor([prefix_ids], device=device, dtype=torch.long)
    suffix = torch.tensor([suffix_ids], device=device, dtype=torch.long)
    z_clean = encode_clean_trajectory_from_ids(model, suffix_ids, device)
    chunks, _chunk_mask, h_eff, _chunk_size = model._trajectory_view(suffix, torch.ones_like(suffix))
    layer_states = model.predict_trajectory_states(z_clean)
    with torch.no_grad():
        prefix_out = model.rwkv_model(input_ids=prefix, use_cache=True, return_dict=True)
        cache = prefix_out.past_key_values
    logits_by_chunk = []
    for h in range(h_eff):
        states_h = [layer_state[:, h] for layer_state in layer_states]
        cache = blend_into_cache_per_layer(model, cache, states_h, per_layer_blends, norm_match)
        out_h = model.rwkv_model(input_ids=chunks[:, h], past_key_values=cache, use_cache=True, return_dict=True)
        cache = out_h.past_key_values
        logits_by_chunk.append(out_h.logits)
    logits = torch.stack(logits_by_chunk, dim=1).reshape(1, usable, -1)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), suffix[:, 1:].reshape(-1))
    return float(loss.item())


def mean_calibration_ce(model: Any, examples: list[tuple[list[int], list[int]]], band_blends: list[float], norm_match: bool, device: str) -> float:
    per_layer = expand_band_blends(int(model.num_layers), band_blends)
    losses = []
    for prefix_ids, suffix_ids in examples:
        value = teacher_forced_suffix_ce(model, prefix_ids, suffix_ids, per_layer, norm_match, device)
        if value == value:
            losses.append(value)
    return sum(losses) / max(1, len(losses))


def coordinate_search_band_blends(model: Any, examples: list[tuple[list[int], list[int]]], device: str) -> dict[str, Any]:
    values = [0.0, 0.1, 0.25, 0.4, 0.55]
    band_blends = [0.4, 0.4, 0.4, 0.4]
    history = []
    for pass_idx in range(2):
        for band_idx in range(4):
            candidates = []
            for value in values:
                trial = list(band_blends)
                trial[band_idx] = value
                ce = mean_calibration_ce(model, examples, trial, False, device)
                candidates.append({"value": value, "ce": ce, "trial": trial})
            best = min(candidates, key=lambda x: float(x["ce"]))
            band_blends = list(best["trial"])
            history.append({"pass": pass_idx, "band": band_idx, "candidates": candidates, "chosen": best})
    per_layer = expand_band_blends(int(model.num_layers), band_blends)
    collapse_fraction = sum(1 for x in per_layer if x < 0.05) / max(1, len(per_layer))
    return {
        "band_blends": band_blends,
        "per_layer_blends": per_layer,
        "stats": {
            "mean": sum(per_layer) / max(1, len(per_layer)),
            "min": min(per_layer),
            "max": max(per_layer),
            "fraction_lt_0p05": collapse_fraction,
        },
        "history": history,
        "final_ce": mean_calibration_ce(model, examples, band_blends, False, device),
        "global0p4_ce": mean_calibration_ce(model, examples, [0.4, 0.4, 0.4, 0.4], False, device),
    }


def load_acc_calc_module():
    path = Path(__file__).resolve().parents[2] / "baseline" / "Cola-DLM" / "scripts" / "acc_calc.py"
    spec = importlib.util.spec_from_file_location("cola_acc_calc_diag", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_short_answer(acc_mod: Any, task: str, generated: str, ground_truth: str, choices: list[Any]) -> float:
    if task in ("mmlu", "obqa", "race", "siqa"):
        pred_choice = acc_mod.extract_mmlu_choice_letter(generated, choices)
        gt_choice = acc_mod.extract_gt_mmlu_choice_letter(ground_truth, choices)
        if pred_choice and gt_choice and pred_choice == gt_choice:
            return 1.0
        sim = acc_mod.calculate_similarity(generated, ground_truth)
        return 1.0 if sim >= 1.0 else float(sim)
    return float(acc_mod.calculate_similarity(generated, ground_truth))


@torch.no_grad()
def run_short_answer_mode(model: Any, tokenizer: Any, args, pad_id: int, mode: str, per_layer_blends: list[float] | None, norm_match: bool, output_root: Path) -> dict[str, float]:
    acc_mod = load_acc_calc_module()
    tasks = ["mmlu", "obqa", "race"]
    alias = mode.replace("+", "_plus_").replace(".", "p")
    mode_dir = output_root / f"tasks_blend_gate_{alias}"
    mode_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, float] = {}
    local_args = with_per_layer_blends(args, per_layer_blends, norm_match)
    local_args.temperature = 0.0
    local_args.max_new_tokens = 32
    local_args.inject_every = 1
    local_args.chunk0_only = False
    local_args.trajectory_state_blend = 0.4
    for task in tasks:
        input_path = Path(args.task_data_dir) / f"{task}.jsonl"
        correctish = 0.0
        total = 0
        out_path = mode_dir / f"{task}.jsonl"
        with out_path.open("w", encoding="utf-8") as out_f:
            for sample_i, item in iter_jsonl(input_path, 100):
                prompt = build_prompt(task, item)
                gt = get_ground_truth(item)
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
                if mode == "raw":
                    generated_ids, generated = generate_token_ids_raw(model, tokenizer, input_ids, local_args)
                    prompt_tokens = int(input_ids.shape[1])
                    generated_tokens = len(generated_ids)
                    injected_chunks: list[int] = []
                else:
                    attention_mask = torch.ones_like(input_ids)
                    z_clean, _clean_tokens = encode_clean_trajectory(model, tokenizer, gt, args.device, pad_id)
                    _z_prefix, prefix_cache, prefix_logits = encode_prefix(model, input_ids, attention_mask)
                    generated_ids, generated, prompt_tokens, generated_tokens, injected_chunks = generate_token_ids_trajectory_frequency(
                        model, tokenizer, input_ids, prefix_cache, prefix_logits, z_clean, local_args, 1, False
                    )
                choices = item.get("choices", [])
                score = score_short_answer(acc_mod, task, generated, gt, choices)
                correctish += 1.0 if score >= 1.0 else 0.0
                total += 1
                rec = dict(item)
                rec.update({
                    "id": item.get("id", sample_i),
                    "prompt": prompt,
                    "generate": generated,
                    "ground_truth": gt,
                    "choices": choices,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated_tokens,
                    "method": mode,
                    "injected_chunks": injected_chunks,
                    "similarity_score_diag": score,
                })
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        results[task.upper() if task != "obqa" else "OBQA"] = 100.0 * correctish / max(1, total)
    results["avg"] = (results["MMLU"] + results["OBQA"] + results["RACE"]) / 3.0
    return results


def run_longgen_blend_compare(model: Any, tokenizer: Any, args, pad_id: int, per_layer_blends: list[float]) -> dict[str, Any]:
    mode_records: dict[str, list[dict[str, Any]]] = {"global0.4": [], "per-band": [], "per-band+norm": []}
    for prompt_i, (prompt, clean_continuation) in enumerate(OPEN_PROMPT_PAIRS[:20]):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
        attention_mask = torch.ones_like(input_ids)
        z_clean, _clean_tokens = encode_clean_trajectory(model, tokenizer, clean_continuation, args.device, pad_id)
        for mode, blends, norm in (("global0.4", None, False), ("per-band", per_layer_blends, False), ("per-band+norm", per_layer_blends, True)):
            torch.manual_seed(args.seed + 1000 * prompt_i + len(mode_records[mode]))
            local_args = with_per_layer_blends(args, blends, norm)
            local_args.max_new_tokens = 256
            local_args.temperature = 0.7
            local_args.top_k = 10
            local_args.top_p = 0.75
            local_args.repetition_penalty = 1.0
            _z_prefix, prefix_cache, prefix_logits = encode_prefix(model, input_ids, attention_mask)
            generated_ids, text, _ptok, _gtok, injected_chunks = generate_token_ids_trajectory_frequency(
                model, tokenizer, input_ids, prefix_cache, prefix_logits, z_clean, local_args, 1, False
            )
            mode_records[mode].append({
                "prompt": prompt,
                "length": len(generated_ids),
                "repeat4": ngram_repetition(generated_ids, 4),
                "repeat8": ngram_repetition(generated_ids, 8),
                "distinct2": distinct_n(generated_ids, 2),
                "injected_chunks": injected_chunks,
                "text": text[:1000],
            })
    rows = []
    for mode, records in mode_records.items():
        metrics = aggregate_generation_metrics(records)
        rows.append({"mode": mode, "mean_len": metrics["mean_len"], "repeat4": metrics["repeat4"], "repeat8": metrics["repeat8"], "distinct2": metrics["distinct2"]})
    return {"rows": rows, "per_prompt": mode_records}


def run_blend_gate_diagnostic(model: Any, tokenizer: Any, args, pad_id: int) -> dict[str, Any]:
    examples = make_calibration_examples(tokenizer, args.task_data_dir, int(args.calibration_samples))
    search = coordinate_search_band_blends(model, examples, args.device)
    per_layer = list(search["per_layer_blends"])
    output_root = Path(args.output_dir) / "blend_gate_eval"
    accuracy_rows = []
    for mode, blends, norm in (
        ("raw", None, False),
        ("global0.4", None, False),
        ("per-band", per_layer, False),
        ("per-band+norm", per_layer, True),
    ):
        scores = run_short_answer_mode(model, tokenizer, args, pad_id, mode, blends, norm, output_root)
        accuracy_rows.append({"mode": mode, **scores})
    longgen = run_longgen_blend_compare(model, tokenizer, args, pad_id, per_layer)
    return {
        "settings": {
            "calibration_examples": len(examples),
            "candidate_values": [0.0, 0.1, 0.25, 0.4, 0.55],
            "bands": layer_band_indices(int(model.num_layers), 4),
            "short_eval_samples_per_task": 100,
            "longgen_prompts": 20,
        },
        "search": search,
        "accuracy_rows": accuracy_rows,
        "longgen": longgen,
    }


def run_single_clean_tasks(model: Any, tokenizer: Any, args, pad_id: int) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    summary: dict[str, Any] = {
        "ckpt_dir": args.ckpt_dir,
        "method": "single_clean_z",
        "latent_dim": int(model.latent_dim),
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": float(args.temperature),
        "tasks": {},
    }
    for task_i, task in enumerate(tasks):
        input_path = Path(args.task_data_dir) / f"{task}.jsonl"
        output_path = output_dir / f"{task}.jsonl"
        n = 0
        first_shape = None
        first_norm = None
        with output_path.open("w", encoding="utf-8") as out_f:
            for sample_i, item in iter_jsonl(input_path, int(args.max_samples)):
                prompt = build_prompt(task, item)
                gt = get_ground_truth(item)
                torch.manual_seed(args.seed + 100000 * task_i + sample_i)
                z_clean, clean_tokens = encode_clean_single_z(model, tokenizer, gt, args.device, pad_id)
                if first_shape is None:
                    first_shape = list(z_clean.shape)
                    first_norm = float(z_clean.detach().float().norm(dim=-1).mean().item())
                    print(f"[SANITY] {task} single_clean_z.shape={first_shape} norm={first_norm:.6f}", flush=True)
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
                generated, prompt_tokens, generated_tokens = generate_answer_single_clean(model, tokenizer, input_ids, z_clean, args)
                rec = dict(item)
                rec.update({
                    "id": item.get("id", sample_i),
                    "prompt": prompt,
                    "generate": generated,
                    "ground_truth": gt,
                    "choices": item.get("choices", []),
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated_tokens,
                    "method": "single_clean_z",
                    "clean_z_shape": list(z_clean.shape),
                    "clean_z_norm": float(z_clean.detach().float().norm(dim=-1).mean().item()),
                    "clean_text_tokens": clean_tokens,
                })
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                if n % 20 == 0:
                    print(f"[{task}] {n} samples", flush=True)
        summary["tasks"][task] = {"samples": n, "output": str(output_path), "first_clean_z_shape": first_shape, "first_clean_z_norm": first_norm}
        print(f"[DONE] {task}: {n} -> {output_path}", flush=True)
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[SUMMARY] {output_dir / 'run_summary.json'}", flush=True)


@torch.no_grad()
def sample_trajectory_cfg_with_probe(model: Any, cond, steps: int, cfg_scale: float, device: str, dtype: torch.dtype):
    horizon = int(model.trajectory_horizon)
    z = torch.randn(cond.shape[0], horizon, model.latent_dim, device=device, dtype=dtype)
    uncond = torch.zeros_like(cond)
    ratios: list[float] = []
    cosines: list[float] = []
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)
    for i in range(steps):
        t_cur, t_nxt = ts[i], ts[i + 1]
        ab_cur = cosine_alpha_bar(t_cur.unsqueeze(0)).to(dtype).clamp(min=1e-4)
        ab_nxt = cosine_alpha_bar(t_nxt.unsqueeze(0)).to(dtype).clamp(min=1e-4)
        t_batch = t_cur.expand(cond.shape[0])
        eps_cond = model.trajectory_dit(z, t_batch, cond=cond)
        eps_uncond = model.trajectory_dit(z, t_batch, cond=uncond)
        diff = (eps_cond.float() - eps_uncond.float()).reshape(cond.shape[0], -1)
        uncond_flat = eps_uncond.float().reshape(cond.shape[0], -1)
        cond_flat = eps_cond.float().reshape(cond.shape[0], -1)
        ratios.extend((diff.norm(dim=-1) / uncond_flat.norm(dim=-1).clamp(min=1e-6)).tolist())
        cosines.extend(F.cosine_similarity(cond_flat, uncond_flat, dim=-1).tolist())
        eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
        z0_pred = (z - (1 - ab_cur).sqrt() * eps) / ab_cur.sqrt()
        z = ab_nxt.sqrt() * z0_pred + (1 - ab_nxt).sqrt() * eps
    return z, ratios, cosines


def run_long_generation_diagnostic(model: Any, tokenizer: Any, args, pad_id: int) -> dict[str, Any]:
    modes = [
        ("raw", None, False),
        ("inject_every=1", 1, False),
        ("inject_every=4", 4, False),
        ("chunk0_only", 1, True),
    ]
    mode_records: dict[str, list[dict[str, Any]]] = {name: [] for name, _every, _chunk0 in modes}
    prompt_count = max(1, min(int(args.longgen_prompts), len(OPEN_PROMPT_PAIRS)))
    for prompt_i, (prompt, clean_continuation) in enumerate(OPEN_PROMPT_PAIRS[:prompt_count]):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
        attention_mask = torch.ones_like(input_ids)
        for mode_name, inject_every, chunk0_only in modes:
            torch.manual_seed(args.seed + 1000 * prompt_i + len(mode_records[mode_name]))
            if mode_name == "raw":
                generated_ids, text = generate_token_ids_raw(model, tokenizer, input_ids, args)
                injected_chunks: list[int] = []
            else:
                inject_every_value = 1 if inject_every is None else int(inject_every)
                z_clean, _clean_tokens = encode_clean_trajectory(model, tokenizer, clean_continuation, args.device, pad_id)
                _z_prefix, prefix_cache, prefix_logits = encode_prefix(model, input_ids, attention_mask)
                generated_ids, text, _ptok, _gtok, injected_chunks = generate_token_ids_trajectory_frequency(
                    model, tokenizer, input_ids, prefix_cache, prefix_logits, z_clean, args, inject_every_value, bool(chunk0_only)
                )
            mode_records[mode_name].append(
                {
                    "prompt": prompt,
                    "length": len(generated_ids),
                    "repeat4": ngram_repetition(generated_ids, 4),
                    "repeat8": ngram_repetition(generated_ids, 8),
                    "distinct2": distinct_n(generated_ids, 2),
                    "injected_chunks": injected_chunks,
                    "text": text[:1000],
                }
            )
    rows = []
    for mode_name, _inject_every, _chunk0_only in modes:
        metrics = aggregate_generation_metrics(mode_records[mode_name])
        rows.append({"mode": mode_name, **metrics})
    return {
        "settings": {
            "prompt_count": prompt_count,
            "max_new_tokens": int(args.max_new_tokens),
            "temperature": float(args.temperature),
            "top_k": int(args.top_k),
            "top_p": float(args.top_p),
            "repetition_penalty": float(args.repetition_penalty),
            "trajectory_state_blend": float(args.trajectory_state_blend),
        },
        "rows": rows,
        "per_prompt": mode_records,
    }


def run_cfg_residual_diagnostic(model: Any, tokenizer: Any, args) -> dict[str, Any]:
    dtype = torch.bfloat16
    cfg_scales = [1.0, 2.0, 3.0, 5.0]
    prompts = [pair[0] for pair in OPEN_PROMPT_PAIRS[:5]]
    rows = []
    per_cfg: dict[str, Any] = {}
    for cfg_scale in cfg_scales:
        all_ratios: list[float] = []
        all_cosines: list[float] = []
        z_norms: list[float] = []
        pairwise: list[float] = []
        per_prompt = []
        for prompt_i, prompt in enumerate(prompts):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
            attention_mask = torch.ones_like(input_ids)
            cond, _cache, _logits = encode_prefix(model, input_ids, attention_mask)
            samples = []
            prompt_ratios: list[float] = []
            prompt_cosines: list[float] = []
            for seed_offset in (0, 100000):
                torch.manual_seed(args.seed + seed_offset + prompt_i)
                z, ratios, cosines = sample_trajectory_cfg_with_probe(
                    model, cond, int(args.steps), float(cfg_scale), args.device, dtype
                )
                samples.append(z.detach().float().cpu())
                z_norms.append(float(z.detach().float().norm(dim=-1).mean().item()))
                all_ratios.extend(float(x) for x in ratios)
                all_cosines.extend(float(x) for x in cosines)
                prompt_ratios.extend(float(x) for x in ratios)
                prompt_cosines.extend(float(x) for x in cosines)
            pair_l2 = float((samples[0].reshape(-1) - samples[1].reshape(-1)).norm().item())
            pairwise.append(pair_l2)
            per_prompt.append(
                {
                    "prompt": prompt,
                    "mean_residual_ratio": sum(prompt_ratios) / max(1, len(prompt_ratios)),
                    "mean_cosine": sum(prompt_cosines) / max(1, len(prompt_cosines)),
                    "pairwise_l2": pair_l2,
                }
            )
        row = {
            "cfg_scale": cfg_scale,
            "mean_residual_ratio": sum(all_ratios) / max(1, len(all_ratios)),
            "mean_cosine": sum(all_cosines) / max(1, len(all_cosines)),
            "mean_Z_norm": sum(z_norms) / max(1, len(z_norms)),
            "mean_pairwise_dist": sum(pairwise) / max(1, len(pairwise)),
        }
        rows.append(row)
        per_cfg[str(cfg_scale)] = per_prompt
    return {
        "settings": {
            "prompt_count": 5,
            "steps": int(args.steps),
            "cfg_scales": cfg_scales,
            "seeds_per_prompt": 2,
        },
        "rows": rows,
        "per_cfg": per_cfg,
    }


def run_convffn_diagnostic(model: Any, tokenizer: Any, args, pad_id: int) -> dict[str, Any]:
    modes = [("raw", "zero"), ("zero", "zero"), ("preserve", "preserve"), ("blend", "blend")]
    mode_records: dict[str, list[dict[str, Any]]] = {name: [] for name, _variant in modes}
    for prompt_i, (prompt, clean_continuation) in enumerate(OPEN_PROMPT_PAIRS[:20]):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
        attention_mask = torch.ones_like(input_ids)
        for mode_name, variant in modes:
            torch.manual_seed(args.seed + 1000 * prompt_i + len(mode_records[mode_name]))
            if mode_name == "raw":
                generated_ids, text = generate_token_ids_raw(model, tokenizer, input_ids, args)
                injected_chunks: list[int] = []
            else:
                local_args = argparse.Namespace(**vars(args))
                local_args.inject_every = 1
                local_args.chunk0_only = False
                local_args.convffn_variant = variant
                z_clean, _clean_tokens = encode_clean_trajectory(model, tokenizer, clean_continuation, args.device, pad_id)
                _z_prefix, prefix_cache, prefix_logits = encode_prefix(model, input_ids, attention_mask)
                generated_ids, text, _ptok, _gtok, injected_chunks = generate_token_ids_trajectory_frequency(
                    model, tokenizer, input_ids, prefix_cache, prefix_logits, z_clean, local_args, 1, False
                )
            mode_records[mode_name].append(
                {
                    "prompt": prompt,
                    "length": len(generated_ids),
                    "repeat4": ngram_repetition(generated_ids, 4),
                    "repeat8": ngram_repetition(generated_ids, 8),
                    "distinct2": distinct_n(generated_ids, 2),
                    "injected_chunks": injected_chunks,
                    "text": text[:1000],
                }
            )
    rows = []
    for mode_name, _variant in modes:
        metrics = aggregate_generation_metrics(mode_records[mode_name])
        rows.append({"mode": mode_name, **metrics})
    return {
        "settings": {
            "prompt_count": 20,
            "max_new_tokens": int(args.max_new_tokens),
            "temperature": float(args.temperature),
            "top_k": int(args.top_k),
            "top_p": float(args.top_p),
            "repetition_penalty": float(args.repetition_penalty),
            "trajectory_state_blend": float(args.trajectory_state_blend),
            "inject_every": 1,
            "blend_variant_note": "blend scales existing conv_state/ffn_state by 1-blend because no planned conv/ffn state exists",
        },
        "rows": rows,
        "per_prompt": mode_records,
    }


def print_longgen_table(rows: list[dict[str, Any]]):
    print("\nDiagnostic A: long-generation repetition vs injection frequency")
    print("| mode | repeat4 | repeat8 | distinct2 | mean_len |")
    print("|---|---:|---:|---:|---:|")
    for row in rows:
        print(f"| {row['mode']} | {row['repeat4']:.4f} | {row['repeat8']:.4f} | {row['distinct2']:.4f} | {row['mean_len']:.2f} |")


def print_cfg_table(rows: list[dict[str, Any]]):
    print("\nDiagnostic B: CFG residual / conditioning saturation")
    print("| cfg | mean_residual_ratio | mean_cosine | mean_Z_norm | mean_pairwise_dist |")
    print("|---:|---:|---:|---:|---:|")
    for row in rows:
        print(f"| {row['cfg_scale']:.0f} | {row['mean_residual_ratio']:.6f} | {row['mean_cosine']:.6f} | {row['mean_Z_norm']:.4f} | {row['mean_pairwise_dist']:.4f} |")


def print_convffn_table(rows: list[dict[str, Any]]):
    print("\nExperiment #5: conv/ffn handling at chunk boundaries")
    print("| mode | repeat4 | repeat8 | distinct2 | mean_len | frac_full_len |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        print(f"| {row['mode']} | {row['repeat4']:.4f} | {row['repeat8']:.4f} | {row['distinct2']:.4f} | {row['mean_len']:.2f} | {row['frac_full_len']:.2f} |")


def print_blend_gate_tables(result: dict[str, Any]):
    stats = result["search"]["stats"]
    print("\nExperiment #6: optimized per-band blend")
    print(f"band_blends = {result['search']['band_blends']}")
    print(
        f"collapse_stats: mean={stats['mean']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}, fraction<0.05={stats['fraction_lt_0p05']:.4f}"
    )
    print("\nShort-answer accuracy")
    print("| mode | MMLU | OBQA | RACE | avg |")
    print("|---|---:|---:|---:|---:|")
    for row in result["accuracy_rows"]:
        print(f"| {row['mode']} | {row['MMLU']:.2f} | {row['OBQA']:.2f} | {row['RACE']:.2f} | {row['avg']:.2f} |")
    print("\nLong generation")
    print("| mode | mean_len | repeat4 |")
    print("|---|---:|---:|")
    for row in result["longgen"]["rows"]:
        print(f"| {row['mode']} | {row['mean_len']:.2f} | {row['repeat4']:.4f} |")


def print_sample_512_4096(result: dict[str, Any]):
    print("\nExperiment #8: 512 vs 4096 trajectory samples")
    print("| mode | repeat4 | repeat8 | distinct2 | mean_len | frac_full_len |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in result["metrics_rows"]:
        print(f"| {row['mode']} | {row['repeat4']:.4f} | {row['repeat8']:.4f} | {row['distinct2']:.4f} | {row['mean_len']:.2f} | {row['frac_full_len']:.2f} |")
    print("\nExample generations")
    for mode in result["modes"]:
        print(f"\n[{mode['mode']}]")
        for example in mode["examples"][:2]:
            text = str(example.get("text", "")).replace("\n", " ")[:300]
            print(f"- {example['prompt']}: {text}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.diag_mode == "sample_512_4096":
        result = {"sample_512_vs_4096": run_sample_512_vs_4096(args)}
        print_sample_512_4096(result["sample_512_vs_4096"])
        output_path = Path(args.json_output) if args.json_output else output_dir / "sample_512_vs_4096.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[SUMMARY] {output_path}", flush=True)
        return

    model, _rwkv, tokenizer, ckpt, cfg = load_relay_model(args.ckpt_dir, args.device)
    model_any = cast(Any, model)
    cfg_any = cast(Any, cfg)
    tokenizer_any = cast(Any, tokenizer)
    pad_id = _set_pad_token(tokenizer_any)

    model_any._prefix_suffix_trajectory_s2 = True
    model_any._training_stage = 2
    model_any._cfg_drop_prob = float(cfg_any.training.get("cfg_drop_prob", 0.0))
    if args.trajectory_s1_mode is not None:
        model_any.config.trajectory_s1_mode = args.trajectory_s1_mode
        model_any.trajectory_s1_mode = args.trajectory_s1_mode
    model_any.config.trajectory_state_blend = float(args.trajectory_state_blend)
    model_any.trajectory_state_blend = float(args.trajectory_state_blend)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    summary: dict[str, Any] = {
        "ckpt_dir": args.ckpt_dir,
        "checkpoint_step": ckpt.get("step", -1),
        "method": "clean_z",
        "trajectory_s1_mode": str(model_any.config.get("trajectory_s1_mode", "independent")),
        "trajectory_state_blend": float(args.trajectory_state_blend),
        "inject_every": int(args.inject_every),
        "chunk0_only": bool(args.chunk0_only),
        "trajectory_horizon": int(model_any.trajectory_horizon),
        "trajectory_chunk_size": int(model_any.trajectory_chunk_size),
        "latent_dim": int(model_any.latent_dim),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "tasks": {},
    }

    if args.diag_mode in ("longgen", "cfg_probe", "both", "convffn", "blend_gate", "sample_512_4096"):
        diagnostic_result: dict[str, Any] = {
            "ckpt_dir": args.ckpt_dir,
            "checkpoint_step": ckpt.get("step", -1),
            "trajectory_s1_mode": str(model_any.config.get("trajectory_s1_mode", "independent")),
            "trajectory_state_blend": float(args.trajectory_state_blend),
            "trajectory_horizon": int(model_any.trajectory_horizon),
            "trajectory_chunk_size": int(model_any.trajectory_chunk_size),
        }
        if args.diag_mode in ("longgen", "both"):
            diagnostic_result["long_generation"] = run_long_generation_diagnostic(model_any, tokenizer_any, args, pad_id)
            print_longgen_table(diagnostic_result["long_generation"]["rows"])
        if args.diag_mode in ("cfg_probe", "both"):
            diagnostic_result["cfg_residual"] = run_cfg_residual_diagnostic(model_any, tokenizer_any, args)
            print_cfg_table(diagnostic_result["cfg_residual"]["rows"])
        if args.diag_mode == "convffn":
            diagnostic_result["convffn_handling"] = run_convffn_diagnostic(model_any, tokenizer_any, args, pad_id)
            print_convffn_table(diagnostic_result["convffn_handling"]["rows"])
            diagnostic_result["hypothesis_status"] = {
                "convffn_zeroing_at_boundaries_harms_long_generation": "pending_interpretation_after_run"
            }
        if args.diag_mode == "blend_gate":
            diagnostic_result["blend_gate"] = run_blend_gate_diagnostic(model_any, tokenizer_any, args, pad_id)
            print_blend_gate_tables(diagnostic_result["blend_gate"])
        if args.diag_mode == "sample_512_4096":
            diagnostic_result["sample_512_vs_4096"] = run_sample_512_vs_4096(args)
            print_sample_512_4096(diagnostic_result["sample_512_vs_4096"])
        output_path = Path(args.json_output) if args.json_output else output_dir / "longgen_and_cfg.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(diagnostic_result, f, indent=2, ensure_ascii=False)
        print(f"[SUMMARY] {output_path}", flush=True)
        return

    if args.diag_mode == "single_clean":
        run_single_clean_tasks(model_any, tokenizer_any, args, pad_id)
        return

    for task_i, task in enumerate(tasks):
        input_path = Path(args.task_data_dir) / f"{task}.jsonl"
        if not input_path.exists():
            print(f"[SKIP] missing {input_path}", flush=True)
            continue
        output_path = output_dir / f"{task}.jsonl"
        n = 0
        first_shape = None
        first_norm = None
        with output_path.open("w", encoding="utf-8") as out_f:
            for sample_i, item in iter_jsonl(input_path, args.max_samples):
                prompt = build_prompt(task, item)
                gt = get_ground_truth(item)
                torch.manual_seed(args.seed + 100000 * task_i + sample_i)
                z_clean, clean_tokens = encode_clean_trajectory(model_any, tokenizer_any, gt, args.device, pad_id)
                if first_shape is None:
                    first_shape = list(z_clean.shape)
                    first_norm = float(z_clean.detach().float().norm(dim=-1).mean().item())
                    print(f"[SANITY] {task} clean_z.shape={first_shape} norm={first_norm:.6f}", flush=True)

                input_ids = tokenizer_any(prompt, return_tensors="pt").input_ids.to(args.device)
                attention_mask = torch.ones_like(input_ids)
                z_prefix, prefix_cache, prefix_logits = encode_prefix(model_any, input_ids, attention_mask)
                generated, prompt_tokens, generated_tokens, injected_chunks = generate_answer_trajectory_frequency(
                    model_any, tokenizer_any, input_ids, prefix_cache, prefix_logits, z_clean, args
                )
                rec = dict(item)
                rec.update(
                    {
                        "id": item.get("id", sample_i),
                        "prompt": prompt,
                        "generate": generated,
                        "ground_truth": gt,
                        "choices": item.get("choices", []),
                        "prompt_tokens": prompt_tokens,
                        "generated_tokens": generated_tokens,
                        "method": "clean_z",
                        "trajectory_state_blend": float(args.trajectory_state_blend),
                        "inject_every": int(args.inject_every),
                        "chunk0_only": bool(args.chunk0_only),
                        "injected_chunks": injected_chunks,
                        "clean_z_shape": list(z_clean.shape),
                        "clean_z_norm": float(z_clean.detach().float().norm(dim=-1).mean().item()),
                        "clean_text_tokens": clean_tokens,
                        "z_prefix_norm": float(z_prefix.detach().float().norm(dim=-1).mean().item()),
                    }
                )
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                if n % 20 == 0:
                    print(f"[{task}] {n} samples", flush=True)
        summary["tasks"][task] = {
            "samples": n,
            "output": str(output_path),
            "first_clean_z_shape": first_shape,
            "first_clean_z_norm": first_norm,
        }
        print(f"[DONE] {task}: {n} -> {output_path}", flush=True)

    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[SUMMARY] {output_dir / 'run_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
