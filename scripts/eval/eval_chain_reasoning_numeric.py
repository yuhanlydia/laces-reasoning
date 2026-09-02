"""Zero-training test: does an ITERATED single-z latent chain reason better on
tasks with a correct numeric answer (GSM8K / SVAMP)?

The single-z S2 already models p(z_next | z_cond). We chain it: z_1 = p(z|problem),
z_{k+1} = p(z | anchor*z_problem + (1-anchor)*z_k), decode the final z's state into
text, extract the numeric answer, and compare to ground truth. We sweep the number
of relay steps K to see whether iterating the latent (more "thinking") raises accuracy
vs a single shot (K=1). This does NOT slice reasoning steps and needs no training.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.sample_prefix_suffix_cfg import encode_prefix, sample_ddim_cfg, generate  # noqa: E402
from scripts.eval import diag_loop1_common as C  # noqa: E402


def _load_task(task, n):
    from datasets import load_dataset

    items = []
    if task == "gsm8k":
        d = load_dataset("gsm8k", "main", split="test")
        for r in list(d)[:n]:
            m = re.search(r"####\s*([-0-9.,]+)", r["answer"])
            items.append((r["question"], (m.group(1).replace(",", "") if m else None)))
    elif task == "svamp":
        for line in open("baseline/reasoning_eval/svamp.jsonl"):
            r = json.loads(line)
            q = (r.get("Body", "") + " " + r.get("Question", "")).strip()
            items.append((q, str(r.get("Answer")).replace(",", "")))
            if len(items) >= n:
                break
    return items


def _extract_num(text):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return nums[-1] if nums else None


def _match(pred, gt):
    if pred is None or gt is None:
        return False
    try:
        return abs(float(pred) - float(gt)) < 1e-3
    except ValueError:
        return pred.strip() == gt.strip()


@torch.no_grad()
def run(ckpt_dir, device, task, n, steps, cfg_scale, k_list, anchor, output):
    model, tokenizer, _dtype, pad_id = C.build_model(ckpt_dir, device)
    model.eval()
    model._prefix_suffix_s2 = True
    dtype = next(model.latent_dit.parameters()).dtype
    gen_args = SimpleNamespace(max_new_tokens=200, temperature=0.6, top_k=30,
                               top_p=0.9, repetition_penalty=1.2)

    items = _load_task(task, n)
    results = {k: {"correct": 0, "total": 0} for k in k_list}

    for qi, (question, gt) in enumerate(items):
        ids = tokenizer(question, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        am = torch.ones_like(ids)
        z_problem = encode_prefix(model, ids, am).to(dtype)
        for K in k_list:
            z = z_problem
            for _ in range(K):
                cond = anchor * z_problem + (1.0 - anchor) * z if anchor > 0 else z
                z = sample_ddim_cfg(model, cond, steps, cfg_scale, device, dtype)
            text, _ = generate(model, tokenizer, ids, am, z, gen_args)
            gen = text[len(tokenizer.decode(ids[0])):]
            ok = _match(_extract_num(gen), gt)
            results[K]["correct"] += int(ok)
            results[K]["total"] += 1
        if (qi + 1) % 20 == 0:
            line = " ".join(f"K{k}={results[k]['correct']}/{results[k]['total']}" for k in k_list)
            print(f"[{qi+1}/{len(items)}] {line}", flush=True)

    summary = {
        "ckpt_dir": ckpt_dir, "task": task, "n": len(items), "cfg_scale": cfg_scale,
        "anchor": anchor,
        "accuracy_by_relay_steps": {
            str(k): (results[k]["correct"] / max(1, results[k]["total"])) for k in k_list
        },
        "verdict_hint": "REASONING if accuracy rises with K (iterating latent helps); "
                        "FLAT/DOWN if K=1 is as good (no latent reasoning gain).",
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="outputs_relay/test-v6-2.9B-s2-prefix-suffix-cfg/step_00150000")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--task", default="gsm8k", choices=["gsm8k", "svamp"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cfg_scale", type=float, default=3.0)
    ap.add_argument("--k_list", default="1,2,4,6")
    ap.add_argument("--anchor", type=float, default=0.5)
    ap.add_argument("--output", default="outputs_eval/chain_reasoning_gsm8k.json")
    a = ap.parse_args()
    run(a.ckpt_dir, a.device, a.task, a.n, a.steps, a.cfg_scale,
        [int(x) for x in a.k_list.split(",")], a.anchor, a.output)


if __name__ == "__main__":
    main()
