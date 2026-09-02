#!/usr/bin/env python3
"""Grid search over key hyperparams for 4096 trajectory model eval.

Distributes combos across GPUs. Each GPU runs its assigned combos sequentially.
Usage: python grid_search_4096.py --gpu 0 --ckpt_dir PATH [--combos 0..N]
"""
from __future__ import annotations
import argparse, itertools, json, os, subprocess, sys, time
from pathlib import Path

REPO = "/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv"
CKPT = "outputs_relay/C-xchunk-4096-coadapt-8h200-b5-rnn-20260708/step_00022500"
LOGDIR = "/inspire/hdd/global_user/zhangjiaquan-253108540222/validation_logs_qz/grid_4096_xchunk_20260715"
TASKS = "mmlu,obqa,race"
MAX_SAMPLES = 100

GRID = {
    "cfg_scale": [1, 2, 3, 5],
    "steps": [50, 100],
    "temperature": [0.0, 0.5],
    "top_k": [10, 50],
    "top_p": [0.5, 0.75, 0.9],
}


def make_name(cfg_scale, steps, temperature, top_k, top_p):
    return f"cfg{cfg_scale}_s{steps}_t{temperature}_k{top_k}_p{top_p}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, required=True, help="CUDA device index")
    p.add_argument("--ckpt_dir", default=CKPT)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stride", type=int, default=4)
    args = p.parse_args()

    # Build all combos
    keys = list(GRID.keys())
    combos = list(itertools.product(*GRID.values()))
    
    # This GPU runs combos where index % stride == start
    my_combos = [(i, dict(zip(keys, combo))) for i, combo in enumerate(combos) if i % args.stride == args.start]

    logdir = Path(LOGDIR)
    os.makedirs(logdir, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"

    results = []
    t0 = time.time()

    for idx, (combo_i, params) in enumerate(my_combos):
        name = make_name(**params)
        out_dir = logdir / name
        tag = f"[GPU{args.gpu}][{idx+1}/{len(my_combos)}] combo={combo_i} {name}"
        print(f"{tag} START", flush=True)

        cmd = [
            sys.executable,
            f"{REPO}/scripts/eval/run_cola_dlm_tasks_prefix_suffix_trajectory_cfg.py",
            "--ckpt_dir", args.ckpt_dir,
            "--output_dir", str(out_dir),
            "--tasks", TASKS,
            "--max_samples", str(MAX_SAMPLES),
            "--steps", str(params["steps"]),
            "--cfg_scale", str(params["cfg_scale"]),
            "--temperature", str(params["temperature"]),
            "--top_k", str(params["top_k"]),
            "--top_p", str(params["top_p"]),
            "--max_new_tokens", "32",
            "--seed", "42",
        ]

        start = time.time()
        proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=7200)
        elapsed = time.time() - start

        if proc.returncode != 0:
            print(f"{tag} FAILED rc={proc.returncode}", flush=True)
            print(proc.stderr[-2000:], flush=True)
            results.append({"combo": combo_i, "name": name, "params": params, "status": "FAILED"})
            continue

        # Parse scores
        acc_cmd = [
            sys.executable,
            f"{REPO}/baseline/Cola-DLM/scripts/acc_calc.py",
            "--result_dir", str(out_dir),
        ]
        acc_proc = subprocess.run(acc_cmd, cwd=REPO, capture_output=True, text=True, timeout=60)
        scores = acc_proc.stdout.strip()

        result = {"combo": combo_i, "name": name, "params": params, "status": "OK", "elapsed_s": elapsed}
        # Try to parse scores
        for line in scores.split("\n"):
            line = line.strip()
            if "avg" in line.lower() or "mean" in line.lower():
                result["score_line"] = line

        print(f"{tag} DONE in {elapsed:.0f}s scores={scores[-200:]}", flush=True)
        results.append(result)

        # Save checkpoint of results after each combo
        with open(logdir / f"results_gpu{args.gpu}.json", "w") as f:
            json.dump(results, f, indent=2)

    total = time.time() - t0
    print(f"[GPU{args.gpu}] ALL DONE {len(my_combos)} combos in {total:.0f}s", flush=True)


if __name__ == "__main__":
    main()
