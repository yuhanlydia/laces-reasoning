#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/tmp/gorilla/berkeley-function-call-leaderboard")

# Bypass newer-gorilla heavy API-SDK import chain (anthropic/mistralai/tree_sitter version drift).
# ast_checker only touches MODEL_CONFIG_MAPPING for a Python underscore_to_dot flag; stub it.
import types as _types
from collections import defaultdict as _defaultdict


class _StubCfg:
    underscore_to_dot = False


_stub = _types.ModuleType("bfcl_eval.constants.model_config")
_stub.MODEL_CONFIG_MAPPING = _defaultdict(lambda: _StubCfg())
sys.modules["bfcl_eval.constants.model_config"] = _stub

from relay_utils import load_relay_model
from bfcl_eval.constants.enums import Language
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

BFCL_DIR = "/tmp/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data"
SYS_TMPL = (
    "You are an expert in composing functions. You are given a question and a set of "
    "possible functions. Based on the question, you will need to make one function call "
    "to achieve the purpose.\n"
    "You MUST output ONLY a python-style function call on a single line, e.g. "
    "func_name(arg1=value1, arg2=value2). Do not output anything else.\n\n"
    "Available functions:\n{funcs}\n"
)


def build_prompt(question, functions):
    func_str = json.dumps(functions, ensure_ascii=False)
    user = question[0][0]["content"]
    return SYS_TMPL.format(funcs=func_str) + f"\nQuestion: {user}\nFunction call: "


def parse_call(text):
    # search func(args) anywhere; model output has emoji/junk prefixes before the call
    for line in text.split("\n"):
        m = re.search(r"([A-Za-z_][\w\.]*)\s*\(([^\n]*)\)", line)
        if m:
            break
    else:
        return None
    name, argstr = m.group(1), m.group(2)
    args = {}
    if argstr.strip():
        for part in re.split(r",(?![^\[\]\{\}\(\)]*[\]\}\)])", argstr):
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            k = k.strip()
            v = v.strip()
            try:
                v = json.loads(v.replace("'", '"'))
            except Exception:
                v = v.strip("'\"")
            args[k] = v
    return {name: args}


@torch.no_grad()
def build_cache(relay, rwkv, input_ids, mode, sample_steps, device):
    if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
        torch.compiler.cudagraph_mark_step_begin()
    out = rwkv(input_ids=input_ids[:, :1], use_cache=True, return_dict=True)
    cache = out.past_key_values
    if mode == "trajectory":
        dtype = next(relay.trajectory_dit.parameters()).dtype
        z = relay.trajectory_sample(1, num_steps=sample_steps, device=device, dtype=dtype)
        states = [ls[:, 0] for ls in relay.predict_trajectory_states(z)]
    else:
        dtype = next(relay.latent_dit.parameters()).dtype
        z = relay.ddpm_sample(1, num_steps=sample_steps, device=device, dtype=dtype)
        states = relay.predict_states(z)
    return relay.inject_into_cache(cache, states)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="raw", choices=("raw", "single", "trajectory"))
    p.add_argument("--ckpt_dir", default=None)
    p.add_argument("--raw_model", default="hf_release/base_models/RWKV7-Goose-World3-2.9B-HF")
    p.add_argument("--category", default="simple_python")
    p.add_argument("--max_samples", type=int, default=100)
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--sample_steps", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    device = args.device

    if args.mode == "raw":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        rwkv = AutoModelForCausalLM.from_pretrained(
            args.raw_model, trust_remote_code=True, torch_dtype=torch.bfloat16, local_files_only=True
        ).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.raw_model, trust_remote_code=True, local_files_only=True)
        relay = None
    else:
        relay, rwkv, tokenizer, _c, _cfg = load_relay_model(args.ckpt_dir, device)

    cat = args.category
    data_file = f"BFCL_v4_simple_python.json" if cat == "simple_python" else f"BFCL_v4_{cat}.json"
    test_cat = "simple" if cat == "simple_python" else cat
    data = [json.loads(l) for l in open(f"{BFCL_DIR}/{data_file}")][: args.max_samples]
    gt = {r["id"]: r for r in (json.loads(l) for l in open(f"{BFCL_DIR}/possible_answer/{data_file}"))}

    n_valid = 0
    n_parsed = 0
    records = []
    for i, row in enumerate(data):
        prompt = build_prompt(row["question"], row["function"])
        inp = tokenizer(prompt, return_tensors="pt").to(device)
        gkw = dict(max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1)
        if args.mode == "raw":
            out = rwkv.generate(**inp, **gkw)[0]
        else:
            cache = build_cache(relay, rwkv, inp.input_ids, args.mode, args.sample_steps, device)
            out = rwkv.generate(input_ids=inp.input_ids, past_key_values=cache, use_cache=True, **gkw)[0]
        gen = tokenizer.decode(out[inp.input_ids.shape[-1]:], skip_special_tokens=True)
        parsed = parse_call(gen)
        valid = False
        if parsed is not None:
            n_parsed += 1
            try:
                res = ast_checker(row["function"], [parsed], gt[row["id"]]["ground_truth"],
                                  Language.PYTHON, test_cat, "relay")
                valid = bool(res.get("valid", False))
            except Exception:
                valid = False
        if valid:
            n_valid += 1
        records.append({"id": row["id"], "parsed": parsed is not None, "valid": valid, "gen_head": gen[:80]})
        if (i + 1) % 20 == 0:
            print(f"{args.mode}: {i+1}/{len(data)} acc={n_valid/(i+1):.4f} parse_rate={n_parsed/(i+1):.4f}", flush=True)

    summary = {
        "mode": args.mode,
        "ckpt_dir": args.ckpt_dir,
        "category": cat,
        "n": len(data),
        "accuracy": round(n_valid / max(1, len(data)), 4),
        "parse_rate": round(n_parsed / max(1, len(data)), 4),
        "n_valid": n_valid,
        "records": records[:30],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.output, "w"), indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
