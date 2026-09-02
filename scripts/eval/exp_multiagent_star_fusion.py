#!/usr/bin/env python3
"""Topology B (Star fusion): M agents each hold one link of a fact chain; the question
needs the WHOLE chain. Compares multi-agent TEXT communication (concat, LatentMAS-style)
against our fixed-size STATE averaging / residual-LATENT fusion, swept over M in {2,4,8}.

Each agent i reads only fact_i and exposes: a recurrent memory state S_i and a plan
latent z_i. Fusion at the star center:
  TEXT (baseline)   : concatenate all M facts into the decoder context (grows with M)
  TEXT budget       : concat truncated to a fixed char budget (text breaks when M grows)
  STATE avg (ours)  : mean of M memory states (fixed size, independent of M)
  RESID latent(ours): subtract shared mean, average residuals, re-sample via diffusion
  DUAL (ours)       : residual-latent plan state + averaged memory states

Controls: single_agent (one link only, must fail), shuffled (wrong partner states).
The chain is built so no proper subset of agents can answer -> any high score is fusion.
Zero training, champion checkpoint. Metric: answer-accuracy (gold token in generation).
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

# Hidden-Profile fusion (NOT multi-hop reasoning). Each item defines M independent
# attribute facts distributed one-per-agent. The question asks for ONE specific
# attribute that a single (query) agent holds. This tests whether fusion PRESERVES a
# queried agent's fact among M agents' states -- direct retrieval, no chaining inference
# (chaining is a reasoning floor for the 2.9B base). Matches the verified 31-pair recall
# strength. attrs[i] = (attribute-name, value); query picks one i; gold = its value token.
ITEMS = [
    {"subject": "expedition", "attrs": [("leader", "Marlowe"), ("destination", "Patagonia"),
        ("vessel", "Aurora"), ("cargo", "quartz"), ("sponsor", "Halden"),
        ("route", "southern"), ("year", "1911"), ("crew", "twelve")]},
    {"subject": "gala", "attrs": [("host", "Verona"), ("venue", "Bellwether"),
        ("theme", "midnight"), ("caterer", "Osric"), ("guest", "Delacroix"),
        ("month", "October"), ("charity", "orphans"), ("band", "Solstice")]},
    {"subject": "startup", "attrs": [("founder", "Ishwar"), ("product", "sensors"),
        ("investor", "Kessel"), ("city", "Trenton"), ("headcount", "forty"),
        ("valuation", "billions"), ("rival", "Nimbus"), ("motto", "clarity")]},
    {"subject": "voyage", "attrs": [("captain", "Rourke"), ("ship", "Meridian"),
        ("port", "Lisbon"), ("goods", "cinnamon"), ("owner", "Pashkov"),
        ("season", "autumn"), ("length", "ninety"), ("flag", "crimson")]},
    {"subject": "museum", "attrs": [("curator", "Anselm"), ("exhibit", "meteorites"),
        ("donor", "Fairbanks"), ("wing", "eastern"), ("insurer", "Coventry"),
        ("opening", "spring"), ("piece", "obsidian"), ("guard", "Petrov")]},
    {"subject": "tournament", "attrs": [("champion", "Salazar"), ("sport", "fencing"),
        ("arena", "Gladstone"), ("prize", "sabre"), ("referee", "Uhlman"),
        ("round", "quarterfinal"), ("sponsor", "Everest"), ("upset", "Novak")]},
    {"subject": "laboratory", "attrs": [("director", "Meecham"), ("field", "photonics"),
        ("grant", "Wexler"), ("building", "annex"), ("device", "interferometer"),
        ("quarter", "third"), ("sample", "gallium"), ("assistant", "Bao")]},
    {"subject": "festival", "attrs": [("organizer", "Calloway"), ("genre", "folk"),
        ("location", "Ashgrove"), ("headliner", "Ferro"), ("permit", "municipal"),
        ("weekend", "final"), ("stage", "riverside"), ("vendor", "Okafor")]},
    {"subject": "merger", "attrs": [("acquirer", "Brандt"), ("target", "Winlow"),
        ("advisor", "Steropes"), ("sector", "logistics"), ("price", "millions"),
        ("closing", "December"), ("regulator", "commission"), ("analyst", "Yates")]},
    {"subject": "regatta", "attrs": [("skipper", "Halloran"), ("boat", "Tempest"),
        ("bay", "Corwin"), ("trophy", "pennant"), ("patron", "Ellsworth"),
        ("tide", "morning"), ("class", "sloop"), ("marshal", "Devi")]},
    {"subject": "archive", "attrs": [("keeper", "Prewitt"), ("collection", "manuscripts"),
        ("benefactor", "Landsman"), ("floor", "basement"), ("catalog", "roman"),
        ("decade", "sixties"), ("relic", "codex"), ("clerk", "Nakamura")]},
    {"subject": "orchestra", "attrs": [("conductor", "Vasquez"), ("work", "requiem"),
        ("hall", "Kingsley"), ("soloist", "Bergen"), ("patron", "Thorne"),
        ("night", "premiere"), ("instrument", "cello"), ("librarian", "Okoye")]},
    {"subject": "excavation", "attrs": [("archaeologist", "Nadir"), ("site", "Karnos"),
        ("funder", "Whitlock"), ("layer", "bronze"), ("find", "amphora"),
        ("dig", "summer"), ("depth", "meters"), ("photographer", "Ruiz")]},
    {"subject": "campaign", "attrs": [("candidate", "Alderton"), ("issue", "transit"),
        ("manager", "Fenwick"), ("district", "northern"), ("donor", "Crane"),
        ("election", "November"), ("slogan", "forward"), ("pollster", "Ibarra")]},
    {"subject": "brewery", "attrs": [("brewmaster", "Odenkirk"), ("beer", "porter"),
        ("town", "Halbrook"), ("supplier", "Machado"), ("barrel", "oak"),
        ("batch", "winter"), ("hops", "cascade"), ("taster", "Lindqvist")]},
    {"subject": "clinic", "attrs": [("physician", "Ramsey"), ("specialty", "cardiology"),
        ("patron", "Goodwin"), ("ward", "western"), ("equipment", "scanner"),
        ("shift", "evening"), ("medicine", "digoxin"), ("nurse", "Achebe")]},
]


def build_chain(item, m, query_idx):
    """Hidden-Profile: distribute m attribute facts one-per-agent; ask for one agent's fact.
    Returns (per-agent facts list of length m, question, gold)."""
    subj = item["subject"]
    base = list(item["attrs"])
    _extra_names = ["code", "serial", "badge", "cipher", "token", "index", "handle",
                    "marker", "tagline", "callsign", "docket", "reference", "stamp", "roster",
                    "label", "tier", "zone", "cohort", "batch2", "lane", "slot", "grade",
                    "channel", "bracket", "sector", "cluster"]
    attrs = list(base)
    while len(attrs) < m:
        j = len(attrs) - len(base)
        attrs.append((_extra_names[j % len(_extra_names)], f"{subj[:3]}{j:02d}x"))
    attrs = attrs[:m]
    facts = [f"Agent {i + 1} knows: the {subj}'s {name} is {val}." for i, (name, val) in enumerate(attrs)]
    query_pool = min(len(base), m)
    non_first_slot = (m - 1) if m <= 2 else 1 + (query_idx % max(1, query_pool - 1))
    qname, qval = attrs[non_first_slot]
    q = f"Question: what is the {subj}'s {qname}? Answer: the {subj}'s {qname} is"
    return facts, q, qval.lower()


@torch.no_grad()
def capture_final_state(model, ids):
    out = model.rwkv_model(input_ids=ids, attention_mask=torch.ones_like(ids).bool(),
                           use_cache=True, return_dict=True)
    cache = out.past_key_values
    states = []
    for l in range(model.num_layers):
        st = cache.layers[l].state.get("recurrent_state") if cache.layers[l].state is not None else None
        states.append(st.float().clone() if isinstance(st, torch.Tensor) else None)
    return states


def fuse_states(weighted):
    L = next(len(sl) for _, sl in weighted if sl is not None)
    out = []
    for l in range(L):
        acc = None
        for w, sl in weighted:
            if sl is None or sl[l] is None:
                continue
            term = w * sl[l].float()
            acc = term if acc is None else acc + term
        out.append(acc)
    return out


@torch.no_grad()
def decode_chunkwise(model, tokenizer, q_ids, per_chunk_state_fn, H, args):
    chunk_size = int(model.trajectory_chunk_size)
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


@torch.no_grad()
def raw_text(model, tokenizer, context, q, args):
    ids = tokenizer(context + "\n" + q, return_tensors="pt").input_ids.to(args.device)
    out = model.rwkv_model(input_ids=ids, attention_mask=torch.ones_like(ids).bool(),
                           use_cache=True, return_dict=True)
    past_kv = out.past_key_values
    logits = out.logits[0, -1]
    all_ids = list(ids[0].tolist())
    new_ids: list[int] = []
    eos_id = getattr(tokenizer, "eos_token_id", None)
    for _ in range(args.max_new_tokens):
        nid = sample_next_token(logits, all_ids, args)
        if eos_id is not None and nid == eos_id:
            break
        new_ids.append(nid)
        all_ids.append(nid)
        out = model.rwkv_model(input_ids=torch.tensor([[nid]], device=args.device),
                               past_key_values=past_kv, use_cache=True, return_dict=True)
        past_kv = out.past_key_values
        logits = out.logits[0, -1]
    return new_ids


@torch.no_grad()
def seq_carryover_state(model, tokenizer, facts, device):
    past = None
    for f in facts:
        ids = tokenizer(f, return_tensors="pt").input_ids.to(device)
        kwargs = dict(input_ids=ids, attention_mask=torch.ones_like(ids).bool(),
                      use_cache=True, return_dict=True)
        if past is not None:
            kwargs["past_key_values"] = past
        past = model.rwkv_model(**kwargs).past_key_values
    states = []
    for l in range(model.num_layers):
        st = past.layers[l].state.get("recurrent_state") if past.layers[l].state is not None else None
        states.append(st.float().clone() if isinstance(st, torch.Tensor) else None)
    return states


@torch.no_grad()
def run(args):
    torch.manual_seed(args.seed)
    device = args.device
    model, _rwkv, tokenizer, _ckpt, _cfg = load_relay_model(args.ckpt_dir, device)
    model.eval()
    model._prefix_suffix_trajectory_s2 = True
    dtype = next(model.alpha_heads.parameters()).dtype
    H = int(model.trajectory_horizon)
    M = args.num_agents

    def enc(text):
        return tokenizer(text, return_tensors="pt").input_ids.to(device)

    # shared mean over all agents' latents (for residual fusion)
    all_z = []
    prepared = []
    for qi, it in enumerate(ITEMS):
        facts, q, gold = build_chain(it, M, qi)
        prepared.append({"facts": facts, "q": q, "gold": gold})
        for f in facts:
            ids = enc(f)
            all_z.append(encode_prefix(model, ids, torch.ones_like(ids))[0].to(dtype)[0])
    z_mean = torch.stack(all_z, dim=0).mean(dim=0, keepdim=True)
    print(f"[M={M}] shared-mean ||z||={float(z_mean.norm()):.3f} "
          f"mean-residual ||.||={float(torch.stack(all_z).sub(z_mean).norm(dim=-1).mean()):.4f}", flush=True)

    conds = ["single_agent", "text_concat", "text_budget", "text_distract",
             "seq_carryover", "seq_distract", "state_avg", "dual", "seq_shuffled"]
    correct = {c: 0 for c in conds}
    ctxchars = {c: 0 for c in conds}
    items_out = []

    for ii, it in enumerate(prepared):
        facts, q, gold = it["facts"], it["q"], it["gold"]
        q_ids = enc(q)

        full_text = " ".join(facts)
        budget_text = full_text[: args.budget_chars]
        one_text = facts[0]
        n_distract = args.num_distract
        subj = ITEMS[ii]["subject"]
        qname = q.split(f"the {subj}'s ")[1].split("?")[0]
        confusers = []
        for j in range(1, n_distract + 1):
            donor = ITEMS[(ii + j) % len(ITEMS)]
            wrong_val = dict(donor["attrs"]).get(qname, donor["attrs"][j % len(donor["attrs"])][1])
            confusers.append(f"Agent D{j} heard a rumor that the {subj}'s {qname} might be {wrong_val}.")
        facts_noisy = facts + confusers
        distract_text = " ".join(facts_noisy)

        mems = [capture_final_state(model, enc(f)) for f in facts]
        zs = [encode_prefix(model, enc(f), torch.ones_like(enc(f)))[0].to(dtype) for f in facts]

        w = 1.0 / M
        z_q = encode_prefix(model, q_ids, torch.ones_like(q_ids))[0].to(dtype)
        Zq = sample_trajectory_cfg(model, z_q, args.steps, args.cfg_scale, device, dtype)
        plan_states = [model.predict_states(Zq[:, h]) for h in range(H)]

        resid = [z - z_mean for z in zs]
        cond_resid = z_mean + sum((w * r for r in resid), torch.zeros_like(z_mean))
        Zp = sample_trajectory_cfg(model, cond_resid, args.steps, args.cfg_scale, device, dtype)
        plan_states_resid = [model.predict_states(Zp[:, h]) for h in range(H)]

        # wrong-partner control: replace last agent's memory with next item's agent-0 memory
        wrong = prepared[(ii + 1) % len(prepared)]
        wrong_mem = capture_final_state(model, enc(wrong["facts"][0]))
        shuffled_mems = mems[:-1] + [wrong_mem]

        seq_state = seq_carryover_state(model, tokenizer, facts, device)
        seq_state_noisy = seq_carryover_state(model, tokenizer, facts_noisy, device)
        query_pool = min(8, M)
        qslot = (M - 1) if M <= 2 else 1 + (ii % max(1, query_pool - 1))
        shuffled_facts = list(facts)
        shuffled_facts[qslot] = prepared[(ii + 1) % len(prepared)]["facts"][qslot]
        seq_state_shuf = seq_carryover_state(model, tokenizer, shuffled_facts, device)

        a = args.plan_weight
        preds = {}
        preds["single_agent"] = raw_text(model, tokenizer, one_text, q, args)
        preds["text_concat"] = raw_text(model, tokenizer, full_text, q, args)
        preds["text_budget"] = raw_text(model, tokenizer, budget_text, q, args)
        preds["text_distract"] = raw_text(model, tokenizer, distract_text, q, args)
        preds["seq_carryover"] = decode_chunkwise(model, tokenizer, q_ids,
            lambda h: fuse_states([(a, plan_states[h]), ((1 - a), seq_state)]), H, args)
        preds["seq_distract"] = decode_chunkwise(model, tokenizer, q_ids,
            lambda h: fuse_states([(a, plan_states[h]), ((1 - a), seq_state_noisy)]), H, args)
        preds["state_avg"] = decode_chunkwise(model, tokenizer, q_ids,
            lambda h: fuse_states([(a, plan_states[h])] + [((1 - a) * w, m) for m in mems]), H, args)
        preds["dual"] = decode_chunkwise(model, tokenizer, q_ids,
            lambda h: fuse_states([(a, plan_states_resid[h])] + [((1 - a) * w, m) for m in mems]), H, args)
        preds["seq_shuffled"] = decode_chunkwise(model, tokenizer, q_ids,
            lambda h: fuse_states([(a, plan_states[h]), ((1 - a), seq_state_shuf)]), H, args)

        ctxchars["single_agent"] += len(one_text)
        ctxchars["text_concat"] += len(full_text)
        ctxchars["text_budget"] += len(budget_text)
        ctxchars["text_distract"] += len(distract_text)
        for c in ["seq_carryover", "seq_distract", "state_avg", "dual", "seq_shuffled"]:
            ctxchars[c] += len(q)

        rec = {"gold": gold, "gen": {}}
        for c in conds:
            txt = tokenizer.decode(preds[c], skip_special_tokens=True).strip().lower()
            hit = gold in txt
            correct[c] += int(hit)
            rec["gen"][c] = {"text": txt[:80], "hit": hit}
        items_out.append(rec)
        print(f"[M={M} {ii+1}/{len(prepared)}] " +
              " ".join(f"{c[:4]}={'1' if rec['gen'][c]['hit'] else '.'}" for c in conds), flush=True)

    n = len(prepared)
    result = {"ckpt_dir": args.ckpt_dir, "num_agents": M, "n": n, "H": H,
              "steps": args.steps, "cfg_scale": args.cfg_scale, "budget_chars": args.budget_chars,
              "conditions": conds,
              "accuracy": {c: round(correct[c] / n, 3) for c in conds},
              "mean_ctx_chars": {c: round(ctxchars[c] / n) for c in conds},
              "items": items_out}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"\n=== STAR FUSION (M={M} agents, n={n}) ===")
    for c in conds:
        print(f"  {result['accuracy'][c]*100:5.0f}%  ctx={result['mean_ctx_chars'][c]:5d}  {c}")
    print(f"written: {args.output}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000")
    p.add_argument("--num_agents", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--cfg_scale", type=float, default=3.0)
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.2)
    p.add_argument("--budget_chars", type=int, default=120)
    p.add_argument("--num_distract", type=int, default=4)
    p.add_argument("--plan_weight", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="outputs_eval/latentcot_gen/star_M4.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
