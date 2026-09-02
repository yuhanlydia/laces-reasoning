#!/usr/bin/env python3
"""Generate multi-agent figures for the latentcom paper from the star/hierarchical eval
JSONs. Produces three PDFs: (1) accuracy vs M for seq-carryover vs parallel-average vs
text-budget; (2) communication cost (chars) vs M for text-concat vs ours; (3) budget
frontier at M=16. Reads outputs_eval/latentcot_gen/star/*.json, writes paper Figures/."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
STAR = REPO / "outputs_eval" / "latentcot_gen" / "star"
FIGDIR = REPO / "paper_AAAI27" / "latentcom" / "Figures"


def load(path):
    p = STAR / path
    return json.loads(p.read_text()) if p.exists() else None


def acc(d, k):
    return d["accuracy"][k] * 100 if d else None


def ctx(d, k):
    return d["mean_ctx_chars"][k] if d else None


def m_scaling_data():
    src = {2: "seq2_M2.json", 4: "seq2_M4.json", 8: "seq2_M8.json",
           16: "bigM_M16.json", 20: "fix_M20.json", 32: "fix_M32.json"}
    rows = {}
    for m, f in src.items():
        d = load(f)
        if d:
            rows[m] = d
    return rows


def fig_accuracy(rows):
    Ms = sorted(rows)
    seq = [acc(rows[m], "seq_carryover") for m in Ms]
    avg = [acc(rows[m], "state_avg") for m in Ms]
    tbud = [acc(rows[m], "text_budget") for m in Ms]
    tcon = [acc(rows[m], "text_concat") for m in Ms]
    plt.figure(figsize=(4.2, 3.0))
    plt.plot(Ms, seq, "o-", color="#1b6", label="Seq. carryover (ours)", lw=2)
    plt.plot(Ms, tcon, "s--", color="#888", label="Text concat (grows)")
    plt.plot(Ms, tbud, "^--", color="#e80", label="Text, fixed budget")
    plt.plot(Ms, avg, "x:", color="#c33", label="Parallel average")
    plt.xlabel("Number of agents $M$")
    plt.ylabel("Answer accuracy (%)")
    plt.ylim(-5, 105)
    plt.legend(fontsize=7, loc="center left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig_magents_acc.pdf")
    plt.close()


def fig_cost(rows):
    Ms = sorted(rows)
    tcon = [ctx(rows[m], "text_concat") for m in Ms]
    ours = [ctx(rows[m], "seq_carryover") for m in Ms]
    plt.figure(figsize=(4.2, 3.0))
    plt.plot(Ms, tcon, "s-", color="#888", label="Text concat")
    plt.plot(Ms, ours, "o-", color="#1b6", label="Seq. carryover (ours)", lw=2)
    plt.xlabel("Number of agents $M$")
    plt.ylabel("Communication context (chars)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig_magents_cost.pdf")
    plt.close()


def fig_budget():
    budgets = [120, 240, 480]
    files = {120: "M16_b120.json", 240: "budget/M16_b240.json", 480: "budget/M16_b480.json"}
    tb, ours = [], []
    for b in budgets:
        d = load(files[b]) if b != 120 else load("bigM_M16.json")
        tb.append(acc(d, "text_budget") if d else None)
        ours.append(acc(d, "seq_carryover") if d else None)
    plt.figure(figsize=(4.2, 3.0))
    plt.plot(budgets, tb, "^-", color="#e80", label="Text, fixed budget")
    plt.plot(budgets, ours, "o-", color="#1b6", label="Seq. carryover (ours, 73 chars)", lw=2)
    plt.axvline(73, color="#1b6", ls=":", alpha=0.6)
    plt.xlabel("Text byte budget (chars), $M{=}16$")
    plt.ylabel("Answer accuracy (%)")
    plt.ylim(-5, 105)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig_magents_budget.pdf")
    plt.close()


def _seed_acc(paths):
    import json
    vals = []
    for p in paths:
        try:
            recs = [json.loads(l) for l in open(p)]
        except FileNotFoundError:
            continue
        if len(recs) < 100:
            continue
        h = sum(1 for r in recs
                if str(r.get("ground_truth", "")).strip().lower()
                in str(r.get("generate", "")).strip().lower())
        vals.append(100 * h / len(recs))
    return vals


def fig_babilong():
    base = REPO / "outputs_eval" / "babilong5seed"
    import statistics
    conds = [("Raw", "raw_qa1", "qa1"), ("Single-$z$", "singlez_qa1", "qa1"),
             ("Ours", "champ_qa1", "qa1")]
    means, stds, labels = [], [], []
    for name, pat, task in conds:
        vals = _seed_acc([base / f"{pat}_s{s}" / f"babilong_{task}.jsonl" for s in range(1, 6)])
        if vals:
            means.append(statistics.mean(vals))
            stds.append(statistics.pstdev(vals))
            labels.append(name)
    plt.figure(figsize=(3.4, 3.0))
    colors = ["#888", "#68a", "#1b6"]
    plt.bar(range(len(means)), means, yerr=stds, capsize=4,
            color=colors[:len(means)])
    plt.xticks(range(len(labels)), labels)
    plt.ylabel("bAbILong qa1 accuracy (%)")
    plt.ylim(0, 55)
    for i, m in enumerate(means):
        plt.text(i, m + 1.5, f"{m:.0f}", ha="center", fontsize=9)
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig_babilong.pdf")
    plt.close()


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    rows = m_scaling_data()
    print("M-scaling rows:", sorted(rows))
    fig_accuracy(rows)
    fig_cost(rows)
    fig_budget()
    fig_babilong()
    print("wrote:", [p.name for p in FIGDIR.glob("fig_*.pdf")])


if __name__ == "__main__":
    main()
