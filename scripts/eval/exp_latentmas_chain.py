#!/usr/bin/env python3
"""LatentMAS (Qwen3-4B) vs our LatentWeave on the distributed-fact chain task.

Fair comparison: each agent sees ONLY its own fact shard; information must
travel through the communication channel. Conditions:
  - latent_relay : LatentMAS-style KV-cache relay (generate_latent_batch)
  - text_relay   : agents pass text thoughts, judger sees all text
  - concat       : ceiling, all facts + question in one prompt
  - q_only       : floor, question only

We also report the KV-cache token count at judger time (the communicated
object size for latent_relay) to contrast with our fixed-dimensional state.
"""
from __future__ import annotations
import argparse, json, re, sys
from types import SimpleNamespace
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "LatentMAS")

from scripts.eval.diag_hard_problems_v2 import gen_2agent, gen_4agent
from LatentMAS.models import ModelWrapper, _past_length
from LatentMAS.utils import set_seed

SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
KV_BYTES_PER_TOKEN = 2 * 36 * 8 * 128 * 2  # K,V x layers x kv_heads x head_dim x bf16


def agent_fact_message(name: str, fact: str):
    user = (
        f"You are {name} in a team solving a question together. "
        f"You privately know the following fact:\n{fact}\n\n"
        f"Think about what this fact implies for the team, but do NOT answer any question yet."
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def judger_message(question: str):
    user = (
        f"Target Question: {question}\n\n"
        "You are provided with latent information for reference and a target question to solve. "
        "The latent information might contain irrelevant contents. Ignore it if it is not helpful.\n"
        "Answer the question directly and output the final answer inside \\boxed{YOUR_FINAL_ANSWER}. "
        "Do not use <think> tags."
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def concat_message(facts, question):
    joined = "\n".join(facts)
    user = (
        f"You are given the following facts from a team of agents:\n{joined}\n\n"
        f"Question: {question}\n"
        "Answer the question directly and output the final answer inside \\boxed{YOUR_FINAL_ANSWER}."
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def extract_boxed(text: str) -> str:
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m[-1].strip()
    return text.strip()


def match_gold(text: str, gold: str) -> bool:
    pred = extract_boxed(text).lower()
    return gold.lower() in pred or gold.lower() in text.lower()


@torch.no_grad()
def run_task_latent(model, task, latent_steps, judger_max_new_tokens, temperature, top_p):
    past_kv = None
    for i, fact in enumerate(task["agents"]):
        msgs = agent_fact_message(f"Agent {chr(65 + i)}", fact)
        _, ids, mask, _ = model.prepare_chat_batch([msgs], add_generation_prompt=True)
        past_kv = model.generate_latent_batch(
            ids, attention_mask=mask, latent_steps=latent_steps, past_key_values=past_kv
        )
    kv_tokens = _past_length(past_kv)
    msgs = judger_message(task["q"])
    _, ids, mask, _ = model.prepare_chat_batch([msgs], add_generation_prompt=True)
    gen, _ = model.generate_text_batch(
        ids, mask, max_new_tokens=judger_max_new_tokens,
        temperature=temperature, top_p=top_p, past_key_values=past_kv,
    )
    return gen[0], kv_tokens


@torch.no_grad()
def run_task_text(model, task, judger_max_new_tokens, temperature, top_p):
    thoughts = []
    for i, fact in enumerate(task["agents"]):
        prior = "\n".join(thoughts)
        user = (
            f"You are Agent {chr(65 + i)} in a team solving a question together. "
            f"You privately know the following fact:\n{fact}\n"
            + (f"\nPrevious agents' notes:\n{prior}\n" if prior else "")
            + "\nSummarize in one short sentence what your fact (and the notes, if any) implies. Do NOT answer any question yet."
        )
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        _, ids, mask, _ = model.prepare_chat_batch([msgs], add_generation_prompt=True)
        gen, _ = model.generate_text_batch(
            ids, mask, max_new_tokens=64, temperature=temperature, top_p=top_p
        )
        thoughts.append(f"Agent {chr(65 + i)}: {gen[0]}")
    user = (
        f"Target Question: {task['q']}\n\nTeam notes:\n" + "\n".join(thoughts)
        + "\n\nAnswer the question directly and output the final answer inside \\boxed{YOUR_FINAL_ANSWER}."
    )
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    _, ids, mask, _ = model.prepare_chat_batch([msgs], add_generation_prompt=True)
    gen, _ = model.generate_text_batch(
        ids, mask, max_new_tokens=judger_max_new_tokens, temperature=temperature, top_p=top_p
    )
    total_text_tokens = sum(len(model.tokenizer(t, add_special_tokens=False)["input_ids"]) for t in thoughts)
    return gen[0], total_text_tokens


@torch.no_grad()
def run_task_concat(model, task, judger_max_new_tokens, temperature, top_p):
    msgs = concat_message(task["agents"], task["q"])
    _, ids, mask, _ = model.prepare_chat_batch([msgs], add_generation_prompt=True)
    gen, _ = model.generate_text_batch(
        ids, mask, max_new_tokens=judger_max_new_tokens, temperature=temperature, top_p=top_p
    )
    return gen[0], int(mask.sum().item())


@torch.no_grad()
def run_task_qonly(model, task, judger_max_new_tokens, temperature, top_p):
    msgs = judger_message(task["q"])
    _, ids, mask, _ = model.prepare_chat_batch([msgs], add_generation_prompt=True)
    gen, _ = model.generate_text_batch(
        ids, mask, max_new_tokens=judger_max_new_tokens, temperature=temperature, top_p=top_p
    )
    return gen[0], 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, default=2, choices=[2, 4])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    ap.add_argument("--latent_steps", type=int, default=10)
    ap.add_argument("--judger_max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--out", type=str, required=True)
    cli = ap.parse_args()

    set_seed(cli.seed)
    device = torch.device(cli.device)
    wrapper_args = SimpleNamespace(latent_space_realign=False)
    model = ModelWrapper(cli.model_name, device, use_vllm=False, args=wrapper_args)

    tasks = gen_2agent(cli.n, cli.seed) if cli.agents == 2 else gen_4agent(cli.n, cli.seed)
    print(f"tasks={cli.agents}agent N={cli.n} seed={cli.seed} latent_steps={cli.latent_steps}", flush=True)

    res = {"latent_relay": [], "text_relay": [], "concat": [], "q_only": []}
    kv_toks = {"latent_relay": [], "text_relay": [], "concat": []}
    traces = []
    for ii, it in enumerate(tasks):
        gold = it["gold"].lower()
        out, kvt = run_task_latent(model, it, cli.latent_steps, cli.judger_max_new_tokens, cli.temperature, cli.top_p)
        res["latent_relay"].append(match_gold(out, gold)); kv_toks["latent_relay"].append(kvt)
        latent_out = out
        out, kvt = run_task_text(model, it, cli.judger_max_new_tokens, cli.temperature, cli.top_p)
        res["text_relay"].append(match_gold(out, gold)); kv_toks["text_relay"].append(kvt)
        out, kvt = run_task_concat(model, it, cli.judger_max_new_tokens, cli.temperature, cli.top_p)
        res["concat"].append(match_gold(out, gold)); kv_toks["concat"].append(kvt)
        out, _ = run_task_qonly(model, it, cli.judger_max_new_tokens, cli.temperature, cli.top_p)
        res["q_only"].append(match_gold(out, gold))
        traces.append({"gold": gold, "latent_pred": extract_boxed(latent_out), "latent_raw": latent_out[:300]})
        if (ii + 1) % 5 == 0:
            def a(k): return sum(res[k]) / len(res[k]) * 100
            print(f"[{ii+1}/{cli.n}] latent={a('latent_relay'):.0f}% text={a('text_relay'):.0f}% "
                  f"concat={a('concat'):.0f}% qonly={a('q_only'):.0f}% "
                  f"kv_tok={sum(kv_toks['latent_relay'])/len(kv_toks['latent_relay']):.0f}", flush=True)

    acc = {k: round(sum(v) / len(v) * 100, 1) for k, v in res.items()}
    mean_kv = {k: round(sum(v) / len(v), 1) for k, v in kv_toks.items()}
    print(json.dumps({"acc": acc, "mean_kv_tokens": mean_kv}, indent=2), flush=True)
    payload = {
        "agents": cli.agents, "n": cli.n, "seed": cli.seed, "model": cli.model_name,
        "latent_steps": cli.latent_steps, "acc": acc, "mean_kv_tokens": mean_kv,
        "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
        "mean_latent_relay_kv_bytes": mean_kv["latent_relay"] * KV_BYTES_PER_TOKEN,
        "raw": {k: [int(x) for x in v] for k, v in res.items()}, "traces": traces,
    }
    with open(cli.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {cli.out}", flush=True)


if __name__ == "__main__":
    main()
