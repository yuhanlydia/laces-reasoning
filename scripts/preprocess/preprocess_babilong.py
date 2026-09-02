# pyright: reportMissingImports=false
"""Prepare BABILong long-context QA as CoLA-compatible task jsonl for 4096 eval.

BABILong embeds fact/reasoning sentences inside long distractor text, then asks a
question whose short answer must be retrieved/inferred from the buried facts. This
directly tests whether the 4096 trajectory model's per-chunk latents preserve
long-context information better than a single global latent (single-z).

Each row -> {"id", "prompt", "generate":"", "ground_truth", "choices":[], "others":"", "few_shot_prefix":""}
prompt = context + "\n" + question + " Answer:"  (generate-then-match, free-form short answer)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def build_prompt(context: str, question: str) -> str:
    q = question.strip()
    return f"{context.strip()}\n\nQuestion: {q}\nAnswer:"


def main() -> None:
    ap = argparse.ArgumentParser(description="BABILong -> CoLA task jsonl for long-context eval.")
    ap.add_argument("--length", default="4k", help="BABILong context length config: 0k,1k,2k,4k,8k,16k,...")
    ap.add_argument("--tasks", default="qa1,qa2", help="BABILong tasks (qa1=single-fact retrieval, qa2=two-fact reasoning)")
    ap.add_argument("--max_samples", type=int, default=100, help="Samples per task")
    ap.add_argument("--out_dir", default="baseline/Cola-DLM/eval_output/tasks_babilong_4k", help="Output dir (tasks_ prefix for acc_calc)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    for task in tasks:
        ds = load_dataset("RMT-team/babilong", args.length, split=f"{task}[:{args.max_samples}]")
        rows = []
        for i, r in enumerate(ds):
            ctx = str(r["input"])
            q = str(r["question"])
            tgt = str(r["target"]).strip()
            rows.append({
                "id": i,
                "prompt": build_prompt(ctx, q),
                "generate": "",
                "ground_truth": tgt,
                "choices": [],
                "others": "",
                "few_shot_prefix": "",
            })
        path = out / f"babilong_{task}.jsonl"
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        avg_len = sum(len(r["prompt"]) for r in rows) / max(1, len(rows))
        print(f"[{task}] wrote {len(rows)} rows -> {path} (avg prompt {avg_len:.0f} chars)")

    print(f"done. output dir: {out}")


if __name__ == "__main__":
    main()
