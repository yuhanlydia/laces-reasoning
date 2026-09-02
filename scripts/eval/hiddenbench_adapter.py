#!/usr/bin/env python3
"""A01: HiddenBench (public, YuxuanLi1225/HiddenBench, 65 Hidden-Profile tasks) adapter.

Each task: a shared `description`, `shared_information` (all agents see), a list of
`hidden_information` pieces (one distributed per agent), MCQ `possible_answers`, and
`correct_answer`. The Hidden-Profile paradigm: no single agent can answer from its own
hidden piece; the group must FUSE the distributed private information.

We first run a capability GATE (Full-profile: one agent sees ALL hidden pieces) to check
the 2.9B decoder is not at floor on a task; only GATE-passing tasks measure fusion.

Conditions (subset of the paper's H0-H13, the training-free ones we can run zero-shot):
  H0  local_only        : agent sees shared + its own hidden piece only (lower bound)
  H2  full_oracle       : one agent sees shared + ALL hidden pieces (capability ceiling)
  H3  text_chain        : facts relayed as text, no budget (grows with #agents)
  H4  text_budget       : text truncated to a fixed char budget
  H7  state_carryover   : sequential recurrent-state carryover, decoder sees query only
  H9  parallel_avg      : mean of per-agent recurrent states (negative control operator)
  H10 shuffled_state    : carryover with one agent's hidden piece corrupted (negative control)

Two receiver conditions:
  context_aware   : final receiver sees the query + shared_information
  context_unaware : final receiver sees ONLY the query (all evidence must arrive via the
                    communicated channel) -- proves the state truly carries distributed facts

Metric: MCQ accuracy by generation-then-match against the correct answer string / option letter.
Also reports Integration Ratio = (A_hidden - A_local) / (A_full - A_local) per condition.

Output: results/hiddenbench/<tag>.json  (per-task records + aggregate).
Zero training; champion checkpoint. Set CUDA_VISIBLE_DEVICES to a free card.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from scripts.eval.relay_utils import load_relay_model  # noqa: E402
from scripts.eval.sample_prefix_suffix_trajectory_cfg import (  # noqa: E402
    encode_prefix, sample_trajectory_cfg,
)
from scripts.eval.run_cola_dlm_tasks_prefix_suffix_trajectory_cfg import (  # noqa: E402
    recompute_logits_from_injected_cache, sample_next_token,
)
from scripts.eval.exp_multiagent_star_fusion import (  # noqa: E402
    capture_final_state, fuse_states, decode_chunkwise, raw_text, seq_carryover_state,
)


def load_tasks(path):
    return json.load(open(path))


def build_answer_prompt(description, options, receiver_shared=None):
    """Query the receiver sees. In context_unaware mode receiver_shared is None."""
    opt_str = " / ".join(options)
    body = ""
    if receiver_shared:
        body += "Shared information:\n" + "\n".join(f"- {s}" for s in receiver_shared) + "\n"
    body += (f"\nQuestion: Based on all information, which option is correct? "
             f"Options: {opt_str}.\nAnswer:")
    return (description + "\n" + body) if receiver_shared is not None else (
        f"Question: which option is correct? Options: {opt_str}.\nAnswer:")


def agent_texts(task, m):
    """One private fact per agent (m agents). Pads/truncates hidden_information to m."""
    hid = list(task["hidden_information"])
    shared = list(task["shared_information"])
    if not hid:
        return []
    while len(hid) < m:
        hid.append(hid[len(hid) % len(hid)])
    hid = hid[:m]
    # each agent = shared context + its own private piece
    return [(" ".join(shared) + " " + h).strip() for h in hid]


def match_answer(gen_text, correct, options):
    """Generation-then-match: correct option string (case-insensitive substring),
    or its 1-based index letter/number if present."""
    g = gen_text.strip().lower()
    c = correct.strip().lower()
    if c and c in g:
        # ensure no OTHER option also matched earlier (avoid ambiguous)
        others = [o.strip().lower() for o in options if o.strip().lower() != c]
        first_c = g.find(c)
        for o in others:
            if o and o in g and g.find(o) < first_c:
                return False
        return True
    return False


@torch.no_grad()
def run(args):
    torch.manual_seed(args.seed)
    device = args.device
    model, _rwkv, tokenizer, _ckpt, _cfg = load_relay_model(args.ckpt_dir, device)
    model.eval()
    model._prefix_suffix_trajectory_s2 = True
    dtype = next(model.alpha_heads.parameters()).dtype
    H = int(model.trajectory_horizon)

    def enc(text):
        return tokenizer(text, return_tensors="pt").input_ids.to(device)

    tasks = load_tasks(args.data)
    if args.limit:
        tasks = tasks[: args.limit]

    conds = ["local_only", "full_oracle", "text_chain", "text_budget",
             "state_carryover", "parallel_avg", "shuffled_state"]
    receiver_modes = ["context_aware", "context_unaware"]

    agg = {rm: {c: 0 for c in conds} for rm in receiver_modes}
    ntask = {rm: 0 for rm in receiver_modes}
    gate_pass = 0
    items_out = []

    for ti, task in enumerate(tasks):
        options = task["possible_answers"]
        correct = task["correct_answer"]
        m = min(args.num_agents, max(1, len(task["hidden_information"])))
        facts = agent_texts(task, m)
        if not facts:
            continue
        shared = list(task["shared_information"])
        full_hidden = " ".join(task["hidden_information"])

        rec = {"id": task["id"], "name": task.get("name", ""), "m": m,
               "correct": correct, "gen": {}}

        for rm in receiver_modes:
            receiver_shared = shared if rm == "context_aware" else None
            q = build_answer_prompt(task["description"], options, receiver_shared)
            q_ids = enc(q)

            # capability probe (full oracle): one agent sees shared + ALL hidden
            full_ctx = " ".join(shared) + " " + full_hidden
            pred_full = raw_text(model, tokenizer, full_ctx, q, args)
            local_ctx = facts[0]
            pred_local = raw_text(model, tokenizer, local_ctx, q, args)
            chain_ctx = " ".join(facts)
            pred_chain = raw_text(model, tokenizer, chain_ctx, q, args)
            pred_budget = raw_text(model, tokenizer, chain_ctx[: args.budget_chars], q, args)

            # state channel
            mems = [capture_final_state(model, enc(f)) for f in facts]
            z_q = encode_prefix(model, q_ids, torch.ones_like(q_ids))[0].to(dtype)
            Zq = sample_trajectory_cfg(model, z_q, args.steps, args.cfg_scale, device, dtype)
            plan_states = [model.predict_states(Zq[:, h]) for h in range(H)]
            seq_state = seq_carryover_state(model, tokenizer, facts, device)
            # shuffled control: corrupt one agent's piece with a distractor from next task
            sh_facts = list(facts)
            donor = tasks[(ti + 1) % len(tasks)]
            donor_facts = agent_texts(donor, m)
            if donor_facts:
                sh_facts[-1] = donor_facts[-1]
            seq_state_shuf = seq_carryover_state(model, tokenizer, sh_facts, device)

            a = args.plan_weight
            w = 1.0 / m
            pred_carry = decode_chunkwise(model, tokenizer, q_ids,
                lambda h: fuse_states([(a, plan_states[h]), ((1 - a), seq_state)]), H, args)
            pred_avg = decode_chunkwise(model, tokenizer, q_ids,
                lambda h: fuse_states([(a, plan_states[h])] + [((1 - a) * w, mm) for mm in mems]), H, args)
            pred_shuf = decode_chunkwise(model, tokenizer, q_ids,
                lambda h: fuse_states([(a, plan_states[h]), ((1 - a), seq_state_shuf)]), H, args)

            preds = {
                "local_only": pred_local, "full_oracle": pred_full,
                "text_chain": pred_chain, "text_budget": pred_budget,
                "state_carryover": pred_carry, "parallel_avg": pred_avg,
                "shuffled_state": pred_shuf,
            }
            rec["gen"][rm] = {}
            for c in conds:
                txt = tokenizer.decode(preds[c], skip_special_tokens=True)
                hit = match_answer(txt, correct, options)
                agg[rm][c] += int(hit)
                rec["gen"][rm][c] = {"text": txt.strip()[:80], "hit": hit}
            ntask[rm] += 1

        items_out.append(rec)
        ca = rec["gen"]["context_aware"]
        print(f"[{ti+1}/{len(tasks)}] {task.get('name','')[:20]:20s} "
              f"full={'1' if ca['full_oracle']['hit'] else '.'} "
              f"loc={'1' if ca['local_only']['hit'] else '.'} "
              f"chain={'1' if ca['text_chain']['hit'] else '.'} "
              f"carry={'1' if ca['state_carryover']['hit'] else '.'} "
              f"avg={'1' if ca['parallel_avg']['hit'] else '.'} "
              f"shuf={'1' if ca['shuffled_state']['hit'] else '.'}", flush=True)

    # aggregate + integration ratio
    out = {"ckpt_dir": args.ckpt_dir, "n_tasks": len(items_out),
           "num_agents": args.num_agents, "steps": args.steps,
           "cfg_scale": args.cfg_scale, "budget_chars": args.budget_chars,
           "plan_weight": args.plan_weight, "accuracy": {}, "integration_ratio": {}}
    for rm in receiver_modes:
        n = max(1, ntask[rm])
        acc = {c: round(agg[rm][c] / n, 3) for c in conds}
        out["accuracy"][rm] = acc
        a_local, a_full = acc["local_only"], acc["full_oracle"]
        denom = (a_full - a_local)
        ir = {}
        for c in conds:
            ir[c] = round((acc[c] - a_local) / denom, 3) if abs(denom) > 1e-6 else None
        out["integration_ratio"][rm] = ir
    out["items"] = items_out

    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(f"\n=== HiddenBench ({len(items_out)} tasks, M={args.num_agents}) ===")
    for rm in receiver_modes:
        print(f"  [{rm}]")
        for c in conds:
            ir = out["integration_ratio"][rm][c]
            irs = f" IR={ir}" if ir is not None else ""
            print(f"    {out['accuracy'][rm][c]*100:5.0f}%  {c}{irs}")
    print(f"written: {args.output}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default=str(
        REPO / "outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000"))
    p.add_argument("--data", default=str(REPO / "data/hiddenbench/benchmark.json"))
    p.add_argument("--output", default=str(REPO / "results/hiddenbench/hiddenbench_M4.json"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_agents", type=int, default=4)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--cfg_scale", type=float, default=3.0)
    p.add_argument("--budget_chars", type=int, default=120)
    p.add_argument("--plan_weight", type=float, default=0.7)
    p.add_argument("--max_new_tokens", type=int, default=24)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.2)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
