#!/usr/bin/env python3
"""A05: Two-agent latent-plan fusion causal ablation at scale (>=200 tasks).

Extends the 16-task two-agent fusion diagnostic to a randomized generator so the
81%-vs-62% claim rests on a large sample with proper controls. Each task splits a
two-hop fact across two agents (A knows X->Y, B knows Y->Z, query asks X->Z), so no
single agent can answer -> any high score is genuine fusion.

Conditions (A0-A15 subset, the training-free operators on the champion):
  A0  agent1_only          : agent-1 memory only (floor)
  A1  agent2_only          : agent-2 memory only (floor)
  A2  text_concat          : both facts as text (oracle ceiling)
  A3  raw_avg_noresample   : linear latent avg, NO resample (failing baseline)
  A4  raw_cond_resample    : raw avg as condition + resample, plan-state only
  A6  resid_cond_resample  : residual-space fusion + resample, plan-state only
  A7  state_only           : both agents' memory states fused (no plan)
  A9  resid_plan_plus_state: residual plan + both memory states  [OURS, full dual]

Four KEY causal interventions (state x plan):
  K_cc  correct plan     + correct state    (= A9, full method)
  K_sc  SHUFFLED plan     + correct state    (does plan contribute coordination?)
  K_cs  correct plan     + SHUFFLED state   (does state carry the facts?)
  K_xc  COUNTERFACTUAL plan + correct state  (does plan causally steer memory use?)

Parameter sweeps (optional, --sweep): residual gain lam, diffusion steps, guidance.

Output: results/fusion_ablation/<tag>.json  Zero training; champion checkpoint.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from scripts.eval.sample_prefix_suffix_trajectory_cfg import (  # noqa: E402
    encode_prefix, sample_trajectory_cfg,
)
from scripts.eval.exp_multiagent_star_fusion import (  # noqa: E402
    capture_final_state, fuse_states, decode_chunkwise,
)
from scripts.eval.relay_utils import load_relay_model  # noqa: E402
from scripts.eval.run_cola_dlm_tasks_prefix_suffix_trajectory_cfg import (  # noqa: E402
    sample_next_token,
)

# --- randomized two-hop task generator ---
_LINK_ENTITIES = [
    "oak tree", "blue vial", "red door", "green jacket", "iron gate", "stone bridge",
    "silver key", "black notebook", "old lighthouse", "marble statue", "copper pipe",
    "glass tower", "wooden crate", "brass compass", "velvet curtain", "ceramic urn",
]
_PLACES = [
    "Wexford", "Ashby", "Thornfield", "Kestrel", "Draymoor", "Fenwick", "Halcyon",
    "Brimmel", "Corvane", "Ostend", "Marlow", "Renwick", "Selby", "Tanager",
    "Verdon", "Wychley",
]
_SUBJECTS = [
    "treasure", "antidote", "manuscript", "artifact", "prototype", "ledger",
    "relic", "serum", "blueprint", "medallion", "specimen", "recording",
    "cipher", "sculpture", "vaccine", "heirloom", "diary", "sample",
]


def gen_tasks(n, seed):
    rng = random.Random(seed)
    tasks = []
    for i in range(n):
        subj = _SUBJECTS[i % len(_SUBJECTS)]
        link = rng.choice(_LINK_ENTITIES)
        gold = rng.choice(_PLACES)
        a = f"Agent A knows: the {subj} is hidden by the {link}."
        b = f"Agent B knows: the {link} is located in the town of {gold}."
        q = f"Question: in which town is the {subj} hidden? Answer: the town of"
        tasks.append({"a": a, "b": b, "q": q, "gold": gold.lower()})
    return tasks


@torch.no_grad()
def run(args):
    torch.manual_seed(args.seed)
    device = args.device
    model, _rwkv, tokenizer, _ckpt, _cfg = load_relay_model(args.ckpt_dir, device)
    model.eval()
    model._prefix_suffix_trajectory_s2 = True
    dtype = next(model.alpha_heads.parameters()).dtype
    H = int(model.trajectory_horizon)

    def enc(t):
        return tokenizer(t, return_tensors="pt").input_ids.to(device)

    tasks = gen_tasks(args.num_tasks, args.seed)

    # shared mean over all agent latents (residual fusion reference)
    all_z = []
    for it in tasks:
        for side in ("a", "b"):
            ids = enc(it[side])
            all_z.append(encode_prefix(model, ids, torch.ones_like(ids))[0].to(dtype)[0])
    zbar = torch.stack(all_z, 0).mean(0, keepdim=True)
    print(f"[pass0] n={len(tasks)} shared-mean ||z||={float(zbar.norm()):.3f} "
          f"mean-resid ||.||={float(torch.stack(all_z).sub(zbar).norm(dim=-1).mean()):.4f}", flush=True)

    conds = ["agent1_only", "agent2_only", "text_concat", "raw_avg_noresample",
             "raw_cond_resample", "resid_cond_resample", "state_only",
             "resid_plan_plus_state", "K_sc_shuf_plan", "K_cs_shuf_state",
             "K_xc_cfact_plan"]
    correct = {c: 0 for c in conds}
    both_fail_win = 0  # genuine-fusion count for the full method
    items = []

    W = dict(w_plan=0.5, w_m1=0.25, w_m2=0.25)

    def raw_text(context, q):
        ids = enc(context + "\n" + q)
        out = model.rwkv_model(input_ids=ids, attention_mask=torch.ones_like(ids).bool(),
                               use_cache=True, return_dict=True)
        past = out.past_key_values
        logits = out.logits[0, -1]
        allids = list(ids[0].tolist()); new = []
        for _ in range(args.max_new_tokens):
            nid = sample_next_token(logits, allids, args)
            eos = getattr(tokenizer, "eos_token_id", None)
            if eos is not None and nid == eos:
                break
            new.append(nid); allids.append(nid)
            out = model.rwkv_model(input_ids=torch.tensor([[nid]], device=device),
                                   past_key_values=past, use_cache=True, return_dict=True)
            past = out.past_key_values; logits = out.logits[0, -1]
        return new

    for ii, it in enumerate(tasks):
        gold = it["gold"].lower()
        q_ids = enc(it["q"])
        a_ids, b_ids = enc(it["a"]), enc(it["b"])
        mem1 = capture_final_state(model, a_ids)
        mem2 = capture_final_state(model, b_ids)
        z_a1 = encode_prefix(model, a_ids, torch.ones_like(a_ids))[0].to(dtype)
        z_a2 = encode_prefix(model, b_ids, torch.ones_like(b_ids))[0].to(dtype)

        Z_raw = sample_trajectory_cfg(model, 0.5 * z_a1 + 0.5 * z_a2, args.steps, args.cfg_scale, device, dtype)
        plan_raw = [model.predict_states(Z_raw[:, h]) for h in range(H)]
        cond_resid = zbar + args.lam * (0.5 * (z_a1 - zbar) + 0.5 * (z_a2 - zbar))
        Z_resid = sample_trajectory_cfg(model, cond_resid, args.steps, args.cfg_scale, device, dtype)
        plan_resid = [model.predict_states(Z_resid[:, h]) for h in range(H)]
        Z_noresample = torch.stack([0.5 * z_a1 + 0.5 * z_a2 for _ in range(H)], 1)
        plan_noresample = [model.predict_states(Z_noresample[:, h]) for h in range(H)]

        # counterfactual plan: fuse a DIFFERENT task's agents (wrong coordination signal)
        other = tasks[(ii + 7) % len(tasks)]
        oa_ids, ob_ids = enc(other["a"]), enc(other["b"])
        zc1 = encode_prefix(model, oa_ids, torch.ones_like(oa_ids))[0].to(dtype)
        zc2 = encode_prefix(model, ob_ids, torch.ones_like(ob_ids))[0].to(dtype)
        cond_cf = zbar + args.lam * (0.5 * (zc1 - zbar) + 0.5 * (zc2 - zbar))
        Z_cf = sample_trajectory_cfg(model, cond_cf, args.steps, args.cfg_scale, device, dtype)
        plan_cf = [model.predict_states(Z_cf[:, h]) for h in range(H)]

        # shuffled state: replace agent-2 memory with a different task's agent-2
        wrong_mem2 = capture_final_state(model, enc(other["b"]))
        a = W["w_plan"]

        def dc(fn):
            return decode_chunkwise(model, tokenizer, q_ids, fn, H, args)

        preds = {}
        preds["agent1_only"] = dc(lambda h: mem1)
        preds["agent2_only"] = dc(lambda h: mem2)
        preds["text_concat"] = raw_text(it["a"] + " " + it["b"], it["q"])
        preds["raw_avg_noresample"] = dc(lambda h: plan_noresample[h])
        preds["raw_cond_resample"] = dc(lambda h: plan_raw[h])
        preds["resid_cond_resample"] = dc(lambda h: plan_resid[h])
        preds["state_only"] = dc(lambda h: fuse_states([(W["w_m1"], mem1), (W["w_m2"], mem2)]))
        preds["resid_plan_plus_state"] = dc(lambda h: fuse_states(
            [(a, plan_resid[h]), (W["w_m1"], mem1), (W["w_m2"], mem2)]))
        preds["K_sc_shuf_plan"] = dc(lambda h: fuse_states(
            [(a, plan_cf[h]), (W["w_m1"], mem1), (W["w_m2"], mem2)]))  # wrong plan, correct state
        preds["K_cs_shuf_state"] = dc(lambda h: fuse_states(
            [(a, plan_resid[h]), (W["w_m1"], mem1), (W["w_m2"], wrong_mem2)]))  # correct plan, wrong state
        preds["K_xc_cfact_plan"] = dc(lambda h: fuse_states(
            [(a, plan_cf[h]), (W["w_m1"], mem1), (W["w_m2"], mem2)]))  # counterfactual plan

        rec = {"q": it["q"], "gold": gold, "hit": {}}
        for c in conds:
            txt = tokenizer.decode(preds[c], skip_special_tokens=True).strip().lower()
            hit = gold in txt
            correct[c] += int(hit)
            rec["hit"][c] = hit
        if (rec["hit"]["resid_plan_plus_state"] and not rec["hit"]["agent1_only"]
                and not rec["hit"]["agent2_only"]):
            both_fail_win += 1
        items.append(rec)
        if (ii + 1) % 20 == 0 or ii < 3:
            print(f"[{ii+1}/{len(tasks)}] dual={'1' if rec['hit']['resid_plan_plus_state'] else '.'} "
                  f"state={'1' if rec['hit']['state_only'] else '.'} "
                  f"concat={'1' if rec['hit']['text_concat'] else '.'}", flush=True)

    n = len(tasks)
    out = {"ckpt_dir": args.ckpt_dir, "n": n, "lam": args.lam, "steps": args.steps,
           "cfg_scale": args.cfg_scale,
           "accuracy": {c: round(correct[c] / n, 3) for c in conds},
           "genuine_fusion_wins": both_fail_win, "items": items}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\n=== FUSION ABLATION (n={n}) ===")
    for c in conds:
        print(f"  {out['accuracy'][c]*100:5.0f}%  {c}")
    print(f"  genuine-fusion wins (dual right, both agents wrong): {both_fail_win}/{n}")
    print(f"written: {args.output}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_tasks", type=int, default=200)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--cfg_scale", type=float, default=3.0)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--max_new_tokens", type=int, default=24)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=str(REPO / "results/fusion_ablation/fusion_200.json"))
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
