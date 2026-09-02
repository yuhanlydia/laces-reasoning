#!/usr/bin/env python3
"""A07: Sequential state-carryover forgetting, order sensitivity, and conflict semantics.

The multi-round result shows carryover holds at ceiling when querying a non-first agent's
single fact. This probes the boundaries the paper still needs:

  (1) RETENTION vs WRITE DISTANCE: with N facts threaded into one recurrent state, query a
      fact written at position p in {first, 25%, mid, 75%, last}. Accuracy vs (N - p) traces
      how a fixed-dimensional state forgets as more facts are written after the target.

  (2) ORDER SENSITIVITY: for each (N, target) run several random write orders; report mean,
      std, and best-worst gap across orders.

  (3) CONFLICT SEMANTICS: two facts assert different values for the SAME key at different
      write positions; measure whether the readout is last-write-wins, first-write, or mixed.

Output: results/order_retention/<tag>.json + fig_retention_heatmap.pdf
Zero training; champion checkpoint. Reuses the carryover threading of probe_multiround.
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

from scripts.eval.relay_utils import load_relay_model  # noqa: E402
from scripts.eval.run_cola_dlm_tasks_prefix_suffix_trajectory_cfg import sample_next_token  # noqa: E402
from scripts.eval.probe_multiround_state_carryover import (  # noqa: E402
    decode_with_state as _mr_decode_with_state,
)

# --- Original pool (multi-word values, action-style queries) ---
ENTITIES_OLD = [
    ("the red key", "opens the north gate"), ("the blue vial", "holds the antidote"),
    ("the oak chest", "contains the gold"), ("the stone bridge", "leads to the mill"),
    ("the silver ring", "belongs to the queen"), ("the black horse", "won the race"),
    ("the old map", "shows the harbor"), ("the green lamp", "marks the safe house"),
    ("the iron door", "hides the archive"), ("the white tower", "guards the coast"),
    ("the copper coin", "dates to the war"), ("the tall pine", "stands by the lake"),
    ("the glass jar", "stores the seeds"), ("the wooden flute", "plays the anthem"),
    ("the brass compass", "points to camp"), ("the leather bag", "carries the letters"),
    ("the round mirror", "reflects the moon"), ("the sharp blade", "cuts the rope"),
    ("the small boat", "crosses the river"), ("the golden crown", "sits in the vault"),
    ("the paper scroll", "names the heir"), ("the clay pot", "keeps the honey"),
    ("the steel chain", "locks the cellar"), ("the velvet cloak", "hides the scar"),
    ("the marble step", "counts to seven"), ("the amber bead", "warms in sunlight"),
    ("the linen sheet", "covers the loom"), ("the bronze bell", "rings at dawn"),
    ("the cedar box", "smells of pine"), ("the ivory key", "winds the clock"),
    ("the crimson flag", "flies at noon"), ("the frozen pond", "hides the ring"),
]

# --- M-agent-style pool (attribute-named facts, single-token values) ---
# Each entry is (subject, attribute_name, value). Facts become
# "the expedition's leader is Marlowe" and queries become
# "what is the expedition's leader?" — matching the format that
# scored 100% in M-agent star_fusion.
ATTR_ENTITIES = [
    ("the expedition", "leader", "Marlowe"), ("the expedition", "destination", "Patagonia"),
    ("the expedition", "vessel", "Aurora"), ("the expedition", "cargo", "quartz"),
    ("the gala", "host", "Verona"), ("the gala", "venue", "Bellwether"),
    ("the gala", "theme", "midnight"), ("the gala", "caterer", "Osric"),
    ("the merger", "acquirer", "Brantt"), ("the merger", "target", "Winlow"),
    ("the merger", "valuation", "Caldwell"), ("the merger", "regulator", "commission"),
    ("the regatta", "skipper", "Halloran"), ("the regatta", "boat", "Tempest"),
    ("the regatta", "port", "Seaview"), ("the regatta", "prize", "silver"),
    ("the archive", "director", "Prewitt"), ("the archive", "location", "Ashby"),
    ("the archive", "collection", "manuscripts"), ("the archive", "year", "1893"),
    ("the orchestra", "conductor", "Vasquez"), ("the orchestra", "work", "requiem"),
    ("the orchestra", "venue", "Odeon"), ("the orchestra", "night", "premiere"),
    ("the excavation", "leader", "Nadir"), ("the excavation", "site", "Tell-el"),
    ("the excavation", "period", "Bronze"), ("the excavation", "find", "amulet"),
    ("the campaign", "candidate", "Alderton"), ("the campaign", "party", "Reform"),
    ("the campaign", "district", "Northgate"), ("the campaign", "slogan", "forward"),
    ("the brewery", "master", "Odenkirk"), ("the brewery", "origin", "Flanders"),
    ("the brewery", "batch", "winter"), ("the brewery", "hops", "cascade"),
    ("the clinic", "physician", "Ramsey"), ("the clinic", "specialty", "cardiology"),
    ("the clinic", "patron", "Goodwin"), ("the clinic", "ward", "western"),
    ("the tribunal", "judge", "Whitmore"), ("the tribunal", "case", "inheritance"),
    ("the tribunal", "verdict", "guilty"), ("the tribunal", "defendant", "Kessler"),
    ("the sanctuary", "order", "Kallos"), ("the sanctuary", "location", "Cyclades"),
    ("the sanctuary", "deity", "Apollo"), ("the sanctuary", "ritual", "purification"),
    ("the observatory", "astronomer", "Arceneaux"), ("the observatory", "telescope", "refractor"),
    ("the observatory", "peak", "Mauna"), ("the observatory", "discovery", "exoplanet"),
    ("the consulate", "ambassador", "Marchetti"), ("the consulate", "nation", "Italy"),
    ("the consulate", "district", "Embassy"), ("the consulate", "protocol", "diplomatic"),
    ("the monastery", "abbot", "Theron"), ("the monastery", "rule", "Benedictine"),
    ("the monastery", "province", "Thrace"), ("the monastery", "founding", "970"),
    ("the fortress", "commander", "Draconis"), ("the fortress", "garrison", "twelve"),
    ("the fortress", "wall", "granite"), ("the fortress", "siege", "failed"),
    ("the lighthouse", "keeper", "Pharos"), ("the lighthouse", "coast", "Attica"),
    ("the lighthouse", "signal", "amber"), ("the lighthouse", "height", "forty"),
    ("the academy", "founder", "Callista"), ("the academy", "discipline", "logic"),
    ("the academy", "student", "fifty"), ("the academy", "term", "autumn"),
    ("the guildhall", "master", "Evert"), ("the guildhall", "trade", "silversmith"),
    ("the guildhall", "quarter", "Oldtown"), ("the guildhall", "charter", "1248"),
    ("the vault", "keeper", "Kael"), ("the vault", "content", "crown"),
    ("the vault", "level", "basement"), ("the vault", "lock", "cipher"),
    ("the harbor", "captain", "Marinos"), ("the harbor", "port", "Seaview"),
    ("the harbor", "fleet", "seven"), ("the harbor", "tide", "morning"),
    ("the cathedral", "bishop", "Benedict"), ("the cathedral", "style", "gothic"),
    ("the cathedral", "bell", "bronze"), ("the cathedral", "patron", "saint"),
]

# Legacy pools (kept for backward compat with --no_attr_format)
ENTITIES = ENTITIES_OLD

SINGLE_TOKEN_ENTITIES = [
    ("the merger", "Brantt"), ("the regatta", "Halloran"), ("the archive", "Prewitt"),
    ("the orchestra", "Vasquez"), ("the excavation", "Nadir"), ("the campaign", "Alderton"),
    ("the brewery", "Odenkirk"), ("the expedition", "Hargrove"), ("the tribunal", "Whitmore"),
    ("the sanctuary", "Kallos"), ("the observatory", "Arceneaux"), ("the consulate", "Marchetti"),
    ("the monastery", "Theron"), ("the marketplace", "Zephyros"), ("the fortress", "Draconis"),
    ("the lighthouse", "Pharos"), ("the academy", "Callista"), ("the guildhall", "Evert"),
    ("the vault", "Kael"), ("the harbor", "Marinos"), ("the cathedral", "Benedict"),
    ("the outpost", "Ridge"), ("the archive II", "Sable"), ("the temple", "Oracle"),
    ("the citadel", "Constantine"), ("the embassy", "Arden"), ("the laboratory", "Curie"),
    ("the museum", "Archivist"), ("the station", "Tracker"), ("the depot", "Logan"),
    ("the observatory II", "Kepler"), ("the prison", "Warden"),
]


class GenArgs:
    max_new_tokens = 12
    temperature = 0.0
    top_k = 50
    top_p = 0.9
    repetition_penalty = 1.1


def normalize(s):
    return "".join(c for c in s.lower() if c.isalnum() or c.isspace()).strip()


def em(pred, gold):
    p, g = normalize(pred), normalize(gold)
    return bool(g) and g in p


@torch.no_grad()
def thread_state(model, tokenizer, facts, device):
    """Natural carryover: process facts sequentially through the RWKV recurrent
    state, exactly like M-agent seq_carryover_state. Each fact is processed ONCE
    with the accumulated past_key_values — no inject-then-reprocess double-counting."""
    past = None
    for f in facts:
        fids = tokenizer(f + "\n", return_tensors="pt").input_ids.to(device)
        kw = dict(input_ids=fids, attention_mask=torch.ones_like(fids).bool(),
                  use_cache=True, return_dict=True)
        if past is not None:
            kw["past_key_values"] = past
        past = model.rwkv_model(**kw).past_key_values
    st = []
    for l in range(model.num_layers):
        rs = past.layers[l].state.get("recurrent_state") if past.layers[l].state is not None else None
        st.append(rs.float().clone() if isinstance(rs, torch.Tensor) else None)
    return st


@torch.no_grad()
def decode_with_state(model, tokenizer, q, st, device, gen):
    """Inject state into a fresh (empty) cache, then process the query ONCE.
    Avoids the double-processing bug where the query is first processed without
    state, then re-processed with injected state — which corrupts the recurrent
    state because the first pass's KV entries are retained."""
    # Start with a minimal 1-token forward to get an empty cache structure
    dummy = tokenizer(" ", return_tensors="pt").input_ids.to(device)
    out = model.rwkv_model(input_ids=dummy, attention_mask=torch.ones_like(dummy).bool(),
                           use_cache=True, return_dict=True)
    past = model.inject_into_cache(out.past_key_values, st)
    # Now process the actual query ONCE with the injected state
    q_ids = tokenizer(q, return_tensors="pt").input_ids.to(device)
    out = model.rwkv_model(input_ids=q_ids, past_key_values=past,
                           attention_mask=torch.ones_like(q_ids).bool(),
                           use_cache=True, return_dict=True)
    past = out.past_key_values
    logits = out.logits[0, -1]
    allids = list(q_ids[0].tolist()); new = []
    eos = getattr(tokenizer, "eos_token_id", None)
    for _ in range(gen.max_new_tokens):
        nid = sample_next_token(logits, allids, gen)
        if eos is not None and nid == eos:
            break
        new.append(nid); allids.append(nid)
        out = model.rwkv_model(input_ids=torch.tensor([[nid]], device=device),
                               past_key_values=past, use_cache=True, return_dict=True)
        past = out.past_key_values; logits = out.logits[0, -1]
    return tokenizer.decode(new, skip_special_tokens=True).strip()


@torch.no_grad()
def run(a):
    random.seed(a.seed); torch.manual_seed(a.seed)
    dev = a.device
    model, _rwkv, tokenizer, _ckpt, _cfg = load_relay_model(a.ckpt_dir, dev)
    model.eval()
    gen = GenArgs()
    use_attr = a.attr_format

    if use_attr:
        pool = ATTR_ENTITIES
    else:
        pool = SINGLE_TOKEN_ENTITIES if a.single_token_pool else ENTITIES

    Ns = [int(x) for x in a.rounds.split(",")]
    positions = ["first", "q25", "mid", "q75", "last"]

    def pos_index(N, tag):
        return {"first": 0, "q25": N // 4, "mid": N // 2,
                "q75": (3 * N) // 4, "last": N - 1}[tag]

    retention = {str(N): {} for N in Ns}
    order_stats = {str(N): {} for N in Ns}

    for N in Ns:
        for tag in positions:
            p = pos_index(N, tag)
            if p >= N:
                continue
            per_order_acc = []
            for o in range(a.orders):
                hits = 0
                for ex in range(a.n):
                    sampled = random.sample(pool, N)
                    order = list(range(N))
                    random.shuffle(order)
                    ordered = [sampled[i] for i in order]
                    tgt = sampled[p]
                    tpos = order.index(p)

                    if use_attr:
                        tgt_subj, tgt_attr, tgt_val = tgt
                        facts = [
                            f"Agent {i+1} knows: the {s}'s {attr} is {val}."
                            for i, (s, attr, val) in enumerate(ordered)
                        ]
                        q = f"Question: what is the {tgt_subj}'s {tgt_attr}? Answer: the {tgt_subj}'s {tgt_attr} is"
                    else:
                        tgt_ent, tgt_val = tgt
                        facts = [f"Fact {i+1}: {e} {v}." for i, (e, v) in enumerate(ordered)]
                        q = f"Question: What does {tgt_ent} do? Answer: {tgt_ent}"

                    st = thread_state(model, tokenizer, facts, dev)
                    pred = decode_with_state(model, tokenizer, q, st, dev, gen)
                    hits += em(pred, tgt_val)
                per_order_acc.append(hits / a.n)
            import statistics as st_
            mean = sum(per_order_acc) / len(per_order_acc)
            std = st_.pstdev(per_order_acc) if len(per_order_acc) > 1 else 0.0
            retention[str(N)][tag] = round(mean, 3)
            order_stats[str(N)][tag] = {"mean": round(mean, 3), "std": round(std, 3),
                                        "min": round(min(per_order_acc), 3),
                                        "max": round(max(per_order_acc), 3)}
            print(f"[N={N} pos={tag:5s} write_pos={p:2d}] acc={mean*100:.0f}% "
                  f"(std={std*100:.0f} min={min(per_order_acc)*100:.0f} max={max(per_order_acc)*100:.0f})",
                  flush=True)

    if not use_attr:
        conflict = {"last_write_wins": 0, "first_write_wins": 0, "neither": 0, "n": 0}
        for ex in range(a.n_conflict):
            ent = random.choice(pool)[0]
            v_early, v_late = random.sample([e[1] for e in pool], 2)
            filler = random.sample([e for e in pool if e[0] != ent], max(0, a.conflict_gap))
            facts = [f"Fact 1: {ent} {v_early}."]
            facts += [f"Fact: {e} {v}." for e, v in filler]
            facts += [f"Fact: {ent} {v_late}."]
            st = thread_state(model, tokenizer, facts, dev)
            q = f"Question: What does {ent} do? Answer: {ent}"
            pred = decode_with_state(model, tokenizer, q, st, dev, gen)
            conflict["n"] += 1
            if em(pred, v_late):
                conflict["last_write_wins"] += 1
            elif em(pred, v_early):
                conflict["first_write_wins"] += 1
            else:
                conflict["neither"] += 1
        print(f"[conflict] last_write={conflict['last_write_wins']}/{conflict['n']} "
              f"first_write={conflict['first_write_wins']}/{conflict['n']} "
              f"neither={conflict['neither']}/{conflict['n']}", flush=True)
    else:
        conflict = None

    out = {"ckpt_dir": a.ckpt_dir, "rounds": Ns, "positions": positions,
           "n_per_cell": a.n, "orders": a.orders, "attr_format": use_attr,
           "retention": retention, "order_stats": order_stats, "conflict": conflict}
    outp = Path(a.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(f"written: {a.output}", flush=True)
    _heatmap(retention, Ns, positions, outp.parent)


def _heatmap(retention, Ns, positions, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    grid = np.full((len(Ns), len(positions)), np.nan)
    for i, N in enumerate(Ns):
        for j, tag in enumerate(positions):
            v = retention[str(N)].get(tag)
            if v is not None:
                grid[i, j] = v
    plt.figure(figsize=(5, 3.6))
    im = plt.imshow(grid, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im, label="accuracy")
    plt.xticks(range(len(positions)), positions)
    plt.yticks(range(len(Ns)), [str(N) for N in Ns])
    plt.xlabel("query write position"); plt.ylabel("num facts N")
    plt.title("Carryover retention vs write position")
    for i in range(len(Ns)):
        for j in range(len(positions)):
            if not np.isnan(grid[i, j]):
                plt.text(j, i, f"{grid[i,j]*100:.0f}", ha="center", va="center",
                         color="w" if grid[i, j] < 0.6 else "k", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_retention_heatmap.pdf")
    plt.close()
    print(f"written: {out_dir / 'fig_retention_heatmap.pdf'}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000")
    p.add_argument("--device", default="cuda")
    p.add_argument("--rounds", default="4,8,16,32")
    p.add_argument("--n", type=int, default=15)
    p.add_argument("--orders", type=int, default=3)
    p.add_argument("--n_conflict", type=int, default=30)
    p.add_argument("--conflict_gap", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--single_token_pool", action="store_true")
    p.add_argument("--attr_format", action="store_true",
                   help="Use M-agent-style attribute-named format: facts are 'the expedition's leader is Marlowe', queries are 'what is the expedition's leader?'")
    p.add_argument("--output", default=str(REPO / "results/order_retention/carryover_forgetting.json"))
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
