#!/usr/bin/env python3
"""Multi-agent two-layer fusion on the CHAMPION (scratch joint co-adapt trajectory).

A problem needs TWO agents' knowledge. Agent-1 sees context_A, Agent-2 sees context_B.
The question q can only be answered by COMBINING both (neither context alone suffices).

Architecture under test (Method C, two-layer fusion):
  CONDITION LAYER (latent): fuse the two agents' thoughts as a diffusion CONDITION,
    then RE-SAMPLE through S2 (diffusion projects the fused condition back to the
    plan manifold -- this is why it does not collapse):
      z_a1 = encode(context_A);  z_a2 = encode(context_B)
      cond_fused = 0.5*z_a1 + 0.5*z_a2
      Z_plan = sample_trajectory_cfg(model, cond_fused)      # fresh trajectory sample
  RESULT LAYER (state): fuse the plan state with the two agents' memory states:
      state_plan   = S1(Z_plan[:,h])
      state_mem_i  = final recurrent state after running context_i
      state_init_h = w_plan*state_plan + w_m1*state_mem1 + w_m2*state_mem2

Controls:
  C  = two-layer  (cond fuse + resample) + (state fuse plan+mem1+mem2)   [ours]
  L  = latent-only: cond_fused -> resample -> decode from plan state ONLY (no mem fuse)
  S  = state-only: pure-plan from a SINGLE agent's cond + state fuse mem1+mem2
  M2 = latent-avg WITHOUT resample (the failing baseline): Z_hist=0.5*z_a1+0.5*z_a2 ->
       predict_states directly (no diffusion) -> decode
  A1 = agent-1 alone (context_A memory only)   -> should FAIL (missing B's fact)
  A2 = agent-2 alone (context_B memory only)   -> should FAIL (missing A's fact)

Zero training. Champion checkpoint. Answer-accuracy is the metric: q needs BOTH facts,
so only a method that truly fuses both agents can score high.
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

from scripts.eval.sample_prefix_suffix_trajectory_cfg import (  # noqa: E402
    encode_prefix, sample_trajectory_cfg,
)
from scripts.eval.run_cola_dlm_tasks_prefix_suffix_trajectory_cfg import (  # noqa: E402
    recompute_logits_from_injected_cache, sample_next_token,
)
from scripts.eval.relay_utils import load_relay_model  # noqa: E402


# Each item: context_A (agent-1 fact), context_B (agent-2 fact), q (needs BOTH), gold.
# The answer combines a fact from A with a fact from B.
ITEMS = [
    {"a": "Agent A knows: the treasure chest is buried under the old oak tree.",
     "b": "Agent B knows: the old oak tree stands in the northern courtyard.",
     "q": "Question: In which courtyard is the treasure buried? Answer:",
     "gold": "northern"},
    {"a": "Agent A knows: the antidote is stored in the blue vial.",
     "b": "Agent B knows: the blue vial is kept in the laboratory freezer.",
     "q": "Question: Where is the antidote stored? Answer:",
     "gold": "freezer"},
    {"a": "Agent A knows: the stolen painting is a portrait of a woman.",
     "b": "Agent B knows: the portrait of a woman was hidden in the attic.",
     "q": "Question: Where is the stolen painting hidden? Answer:",
     "gold": "attic"},
    {"a": "Agent A knows: the fastest route uses the mountain tunnel.",
     "b": "Agent B knows: the mountain tunnel leads directly to Denver.",
     "q": "Question: Which city does the fastest route reach? Answer:",
     "gold": "denver"},
    {"a": "Agent A knows: the rare orchid blooms only at midnight.",
     "b": "Agent B knows: the flower that blooms at midnight is poisonous.",
     "q": "Question: Is the rare orchid poisonous? Answer:",
     "gold": "poisonous"},
    {"a": "Agent A knows: the missing key opens the red door.",
     "b": "Agent B knows: behind the red door is the archive room.",
     "q": "Question: Which room does the missing key open? Answer:",
     "gold": "archive"},
    {"a": "Agent A knows: the champion athlete trained in Kenya.",
     "b": "Agent B knows: the athlete who trained in Kenya runs the marathon.",
     "q": "Question: Which event does the champion athlete run? Answer:",
     "gold": "marathon"},
    {"a": "Agent A knows: the secret recipe requires saffron.",
     "b": "Agent B knows: saffron is harvested in the region of Kashmir.",
     "q": "Question: From which region is the secret recipe's spice? Answer:",
     "gold": "kashmir"},
    {"a": "Agent A knows: the encrypted file is named 'phoenix'.",
     "b": "Agent B knows: the file named 'phoenix' contains the launch codes.",
     "q": "Question: What does the encrypted file contain? Answer:",
     "gold": "launch"},
    {"a": "Agent A knows: the injured hiker was carried on a stretcher.",
     "b": "Agent B knows: the person on the stretcher was airlifted to Boston.",
     "q": "Question: To which city was the injured hiker taken? Answer:",
     "gold": "boston"},
    {"a": "Agent A knows: the winning lottery ticket was bought on Tuesday.",
     "b": "Agent B knows: the ticket bought on Tuesday was sold in Chicago.",
     "q": "Question: In which city was the winning ticket sold? Answer:",
     "gold": "chicago"},
    {"a": "Agent A knows: the ancient scroll is written in Greek.",
     "b": "Agent B knows: the Greek scroll describes a lost city.",
     "q": "Question: What does the ancient scroll describe? Answer:",
     "gold": "city"},
    {"a": "Agent A knows: the prototype engine runs on hydrogen.",
     "b": "Agent B knows: the hydrogen engine powers the new aircraft.",
     "q": "Question: What does the prototype engine power? Answer:",
     "gold": "aircraft"},
    {"a": "Agent A knows: the suspect wore a green jacket.",
     "b": "Agent B knows: the person in the green jacket fled toward the harbor.",
     "q": "Question: Where did the suspect flee? Answer:",
     "gold": "harbor"},
    {"a": "Agent A knows: the vaccine must be kept at minus twenty degrees.",
     "b": "Agent B knows: the freezer at minus twenty degrees is in Building C.",
     "q": "Question: In which building is the vaccine kept? Answer:",
     "gold": "building"},
    {"a": "Agent A knows: the melody was composed for a violin.",
     "b": "Agent B knows: the violin piece premiered in Prague.",
     "q": "Question: In which city did the melody premiere? Answer:",
     "gold": "prague"},
]

WEIGHTS = {"w_plan": 0.5, "w_m1": 0.25, "w_m2": 0.25}  # result-layer state fusion weights


def _repeat_n(ids, n=4):
    if len(ids) < n:
        return 0.0
    g = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return 1.0 - len(set(g)) / max(1, len(g))


@torch.no_grad()
def capture_final_state(model, ids, am):
    out = model.rwkv_model(input_ids=ids, attention_mask=am.bool(), use_cache=True, return_dict=True)
    cache = out.past_key_values
    states = []
    for l in range(model.num_layers):
        st = cache.layers[l].state.get("recurrent_state") if cache.layers[l].state is not None else None
        states.append(st.float().clone() if isinstance(st, torch.Tensor) else None)
    return states


@torch.no_grad()
def decode_chunkwise(model, tokenizer, q_ids, per_chunk_state_fn, H, args):
    chunk_size = int(model.trajectory_chunk_size)
    # fresh prefix cache primed on q
    out = model.rwkv_model(input_ids=q_ids, attention_mask=torch.ones_like(q_ids).bool(),
                           use_cache=True, return_dict=True)
    past_kv = out.past_key_values
    logits = out.logits[0, -1]
    all_ids = list(q_ids[0].tolist())
    new_ids: list[int] = []
    eos_id = getattr(tokenizer, "eos_token_id", None)
    stop = False
    for h in range(H):
        if stop or len(new_ids) >= args.max_new_tokens:
            break
        states_h = per_chunk_state_fn(h)
        past_kv = model.inject_into_cache(past_kv, states_h)
        ctx = torch.tensor([all_ids], device=q_ids.device, dtype=torch.long)
        past_kv, lb = recompute_logits_from_injected_cache(model, ctx, torch.ones_like(ctx), past_kv)
        logits = lb[0]
        for _ in range(chunk_size):
            if len(new_ids) >= args.max_new_tokens:
                stop = True
                break
            nid = sample_next_token(logits, all_ids, args)
            if eos_id is not None and nid == eos_id:
                stop = True
                break
            new_ids.append(nid)
            all_ids.append(nid)
            out = model.rwkv_model(input_ids=torch.tensor([[nid]], device=q_ids.device),
                                   past_key_values=past_kv, use_cache=True, return_dict=True)
            past_kv = out.past_key_values
            logits = out.logits[0, -1]
    return new_ids


def _fuse_states(*state_lists_with_w):
    """Each arg is (weight, list-of-per-layer-states-or-None). Returns fused per-layer list."""
    L = None
    for _, sl in state_lists_with_w:
        if sl is not None:
            L = len(sl); break
    out = []
    for l in range(L):
        acc = None
        for w, sl in state_lists_with_w:
            if sl is None or sl[l] is None:
                continue
            term = w * sl[l].float()
            acc = term if acc is None else acc + term
        out.append(acc)
    return out


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
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        return ids, torch.ones_like(ids)

    W = WEIGHTS
    methods = ["C_twolayer", "C_resid", "C_resid_amp", "L_latent_resample_only",
               "S_state_only", "M2_latent_avg_noresample", "A1_agent1_only", "A2_agent2_only"]
    results = {"ckpt_dir": args.ckpt_dir, "H": H, "steps": args.steps,
               "cfg_scale": args.cfg_scale, "weights": W, "resid_amp": args.resid_amp,
               "items": [], "methods": methods}
    correct = {m: 0 for m in methods}

    all_z = []
    for it in ITEMS:
        for side in ("a", "b"):
            ids = tokenizer(it[side], return_tensors="pt").input_ids.to(device)
            all_z.append(encode_prefix(model, ids, torch.ones_like(ids))[0].to(dtype)[0])
    z_shared_mean = torch.stack(all_z, dim=0).mean(dim=0, keepdim=True)
    print(f"[pass0] shared-mean ||z||={float(z_shared_mean.norm()):.3f} "
          f"mean residual ||.||={float(torch.stack(all_z).sub(z_shared_mean).norm(dim=-1).mean()):.4f}", flush=True)

    for ii, it in enumerate(ITEMS):
        a_ids, a_am = enc(it["a"])
        b_ids, b_am = enc(it["b"])
        q_ids, _ = enc(it["q"])
        gold = it["gold"].lower()

        # agents' memory states (result layer)
        mem1 = capture_final_state(model, a_ids, a_am)
        mem2 = capture_final_state(model, b_ids, b_am)
        # agents' thought latents (condition layer)
        z_a1 = encode_prefix(model, a_ids, a_am)[0].to(dtype)
        z_a2 = encode_prefix(model, b_ids, b_am)[0].to(dtype)

        cond_fused = 0.5 * z_a1 + 0.5 * z_a2
        Z_plan = sample_trajectory_cfg(model, cond_fused, args.steps, args.cfg_scale, device, dtype)
        Z_plan_a1 = sample_trajectory_cfg(model, z_a1, args.steps, args.cfg_scale, device, dtype)
        plan_states = [model.predict_states(Z_plan[:, h]) for h in range(H)]
        plan_states_a1 = [model.predict_states(Z_plan_a1[:, h]) for h in range(H)]

        r_a1 = z_a1 - z_shared_mean
        r_a2 = z_a2 - z_shared_mean
        cond_resid = z_shared_mean + 0.5 * r_a1 + 0.5 * r_a2
        cond_resid_amp = z_shared_mean + args.resid_amp * (0.5 * r_a1 + 0.5 * r_a2)
        Z_plan_resid = sample_trajectory_cfg(model, cond_resid, args.steps, args.cfg_scale, device, dtype)
        Z_plan_resid_amp = sample_trajectory_cfg(model, cond_resid_amp, args.steps, args.cfg_scale, device, dtype)
        plan_states_resid = [model.predict_states(Z_plan_resid[:, h]) for h in range(H)]
        plan_states_resid_amp = [model.predict_states(Z_plan_resid_amp[:, h]) for h in range(H)]

        Z_hist = torch.stack([0.5 * z_a1 + 0.5 * z_a2 for _ in range(H)], dim=1)
        hist_states = [model.predict_states(Z_hist[:, h]) for h in range(H)]

        item_rec = {"q": it["q"], "gold": gold, "gen": {}}
        for m in methods:
            if m == "C_twolayer":
                fn = lambda h: _fuse_states((W["w_plan"], plan_states[h]),
                                            (W["w_m1"], mem1), (W["w_m2"], mem2))
            elif m == "C_resid":
                fn = lambda h: _fuse_states((W["w_plan"], plan_states_resid[h]),
                                            (W["w_m1"], mem1), (W["w_m2"], mem2))
            elif m == "C_resid_amp":
                fn = lambda h: _fuse_states((W["w_plan"], plan_states_resid_amp[h]),
                                            (W["w_m1"], mem1), (W["w_m2"], mem2))
            elif m == "L_latent_resample_only":
                fn = lambda h: plan_states[h]                     # fused cond + resample, plan state only
            elif m == "S_state_only":
                fn = lambda h: _fuse_states((W["w_plan"], plan_states_a1[h]),
                                            (W["w_m1"], mem1), (W["w_m2"], mem2))  # single-agent plan + mem fuse
            elif m == "M2_latent_avg_noresample":
                fn = lambda h: hist_states[h]                     # latent avg, NO resample
            elif m == "A1_agent1_only":
                fn = lambda h: [s for s in mem1]                  # agent-1 memory only
            elif m == "A2_agent2_only":
                fn = lambda h: [s for s in mem2]
            gen = decode_chunkwise(model, tokenizer, q_ids, fn, H, args)
            txt = tokenizer.decode(gen, skip_special_tokens=True).strip()
            hit = gold in txt.lower()
            correct[m] += int(hit)
            item_rec["gen"][m] = {"text": txt, "hit": hit, "repeat4": _repeat_n(gen, 4)}
        results["items"].append(item_rec)
        print(f"[item {ii+1}/{len(ITEMS)}] " +
              " ".join(f"{m.split('_')[0]}={'1' if item_rec['gen'][m]['hit'] else '.'}" for m in methods),
              flush=True)

    n = len(ITEMS)
    results["accuracy"] = {m: round(correct[m] / n, 3) for m in methods}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print("\n=== MULTI-AGENT TWO-LAYER FUSION (champion, n={}) ===".format(n))
    labels = {
        "C_twolayer": "C  two-layer (cond-fuse+resample + state-fuse mem)  [OURS]",
        "C_resid": "Cr two-layer, RESIDUAL-space cond fusion (subtract shared mean)",
        "C_resid_amp": "Cr+ residual fusion, amplified residual (x{})".format(args.resid_amp),
        "L_latent_resample_only": "L  cond-fuse+resample, plan-state only (no mem)",
        "S_state_only": "S  single-agent plan + state-fuse mem",
        "M2_latent_avg_noresample": "M2 latent-avg NO resample (failing baseline)",
        "A1_agent1_only": "A1 agent-1 memory only (should fail)",
        "A2_agent2_only": "A2 agent-2 memory only (should fail)",
    }
    for m in methods:
        print(f"  {results['accuracy'][m]*100:5.0f}%  {labels[m]}")
    print(f"\nwritten: {args.output}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000")
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--cfg_scale", type=float, default=3.0)
    p.add_argument("--max_new_tokens", type=int, default=48)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resid_amp", type=float, default=5.0)
    p.add_argument("--output", default="outputs_eval/diag_multiagent_twolayer_fusion.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
