#!/usr/bin/env python3
"""HiddenBench with dual-channel + PCA fusion conditions.

Extends hiddenbench_adapter.py with:
  - memory_only      : pure sequential carryover, no plan channel
  - dual_resid       : agent-plan residual fusion + seq carryover state
  - dual_pca_k1/k2/k3: PCA-truncated agent-plan fusion + seq carryover state

The existing state_carryover condition uses the QUERY's plan (z_q).
The new dual_* conditions fuse AGENT latents into a plan condition.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from scripts.eval.relay_utils import load_relay_model
from scripts.eval.sample_prefix_suffix_trajectory_cfg import encode_prefix, sample_trajectory_cfg
from scripts.eval.run_cola_dlm_tasks_prefix_suffix_trajectory_cfg import recompute_logits_from_injected_cache, sample_next_token
from scripts.eval.exp_multiagent_star_fusion import capture_final_state, fuse_states, decode_chunkwise, raw_text, seq_carryover_state
from scripts.eval.fusion_ablation_pca import pca_remove_shared

def load_tasks(path):
    return json.load(open(path))

def build_answer_prompt(description, options, receiver_shared=None):
    opt_str = " / ".join(options)
    body = ""
    if receiver_shared:
        body += "Shared information:\n" + "\n".join(f"- {s}" for s in receiver_shared) + "\n"
    body += (f"\nQuestion: Based on all information, which option is correct? "
             f"Options: {opt_str}.\nAnswer:")
    return (description + "\n" + body) if receiver_shared is not None else (
        f"Question: which option is correct? Options: {opt_str}.\nAnswer:")

def agent_texts(task, m):
    hid = list(task["hidden_information"])
    shared = list(task["shared_information"])
    if not hid:
        return []
    while len(hid) < m:
        hid.append(hid[len(hid) % len(hid)])
    hid = hid[:m]
    return [(" ".join(shared) + " " + h).strip() for h in hid]

def match_answer(gen_text, correct, options):
    g = gen_text.strip().lower()
    c = correct.strip().lower()
    if c and c in g:
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
    print(f"[init] H={H} dtype={dtype} device={device}", flush=True)

    def enc(text):
        return tokenizer(text, return_tensors="pt").input_ids.to(device)

    tasks = load_tasks(args.data)
    if args.limit:
        tasks = tasks[: args.limit]

    # === Pass 0: collect all agent latents for PCA ===
    print(f"[pass0] Collecting agent latents for {len(tasks)} tasks...", flush=True)
    all_z = []
    task_z_agents = []  # per-task: list of z_i tensors
    for ti, task in enumerate(tasks):
        m = min(args.num_agents, max(1, len(task["hidden_information"])))
        facts = agent_texts(task, m)
        z_agents = []
        for f in facts:
            f_ids = enc(f)
            z = encode_prefix(model, f_ids, torch.ones_like(f_ids))[0].to(dtype)
            z_agents.append(z)
            all_z.append(z[0].float().cpu())
        task_z_agents.append(z_agents)
        if (ti + 1) % 20 == 0:
            torch.cuda.empty_cache()
            print(f"[pass0] {ti+1}/{len(tasks)} done", flush=True)

    z_all = torch.stack(all_z, 0).to(device)
    zbar = z_all.mean(0, keepdim=True)
    shared_dirs_K1, sv_K1, z_mean_pca, _ = pca_remove_shared(z_all, K=1)
    shared_dirs_K2, sv_K2, _, _ = pca_remove_shared(z_all, K=2)
    shared_dirs_K3, sv_K3, _, _ = pca_remove_shared(z_all, K=3)
    total_sv = sv_K3.sum()
    print(f"[PCA] sv1/total={float(sv_K1[0]/total_sv):.4f} "
          f"(sv1+2+3)/total={float(sv_K3[:3].sum()/total_sv):.4f}", flush=True)

    # === Conditions ===
    conds = ["local_only", "full_oracle", "text_chain", "text_budget",
             "state_carryover", "parallel_avg", "shuffled_state",
             "memory_only", "dual_resid", "dual_pca_k1", "dual_pca_k2", "dual_pca_k3"]
    receiver_modes = ["context_aware", "context_unaware"]

    agg = {rm: {c: 0 for c in conds} for rm in receiver_modes}
    ntask = {rm: 0 for rm in receiver_modes}
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

        # Get agent latents from pass 0
        z_agents = task_z_agents[ti]

        # Compute fused plan conditions
        w = 1.0 / m
        lam = args.lam

        # Residual fusion: cond = zbar + lam * sum(w_i * (z_i - zbar))
        cond_resid = zbar.clone()
        for z_i in z_agents:
            cond_resid = cond_resid + lam * w * (z_i[0:1].float() - zbar[0:1].float())
        cond_resid = cond_resid.to(dtype)

        # PCA fusion: remove top-K, then fuse residuals
        def pca_fuse(z_agents, shared_dirs, z_mean, lam=1.0):
            fused_resid = torch.zeros_like(z_mean[0].float())
            for z_i in z_agents:
                zi_c = z_i[0].float() - z_mean[0].float()
                proj = zi_c @ shared_dirs.T
                resid = zi_c - proj @ shared_dirs
                fused_resid += w * lam * resid
            return (z_mean[0].float() + fused_resid).to(dtype).unsqueeze(0)

        cond_pca_k1 = pca_fuse(z_agents, shared_dirs_K1, z_mean_pca, lam)
        cond_pca_k2 = pca_fuse(z_agents, shared_dirs_K2, z_mean_pca, lam)
        cond_pca_k3 = pca_fuse(z_agents, shared_dirs_K3, z_mean_pca, lam)

        # Sample trajectories from fused conditions
        Z_resid = sample_trajectory_cfg(model, cond_resid, args.steps, args.cfg_scale, device, dtype)
        plan_resid = [model.predict_states(Z_resid[:, h]) for h in range(H)]

        Z_pca_k1 = sample_trajectory_cfg(model, cond_pca_k1, args.steps, args.cfg_scale, device, dtype)
        plan_pca_k1 = [model.predict_states(Z_pca_k1[:, h]) for h in range(H)]

        Z_pca_k2 = sample_trajectory_cfg(model, cond_pca_k2, args.steps, args.cfg_scale, device, dtype)
        plan_pca_k2 = [model.predict_states(Z_pca_k2[:, h]) for h in range(H)]

        Z_pca_k3 = sample_trajectory_cfg(model, cond_pca_k3, args.steps, args.cfg_scale, device, dtype)
        plan_pca_k3 = [model.predict_states(Z_pca_k3[:, h]) for h in range(H)]

        # Memory states
        mems = [capture_final_state(model, enc(f)) for f in facts]

        for rm in receiver_modes:
            receiver_shared = shared if rm == "context_aware" else None
            q = build_answer_prompt(task["description"], options, receiver_shared)
            q_ids = enc(q)

            # Query plan (existing condition)
            z_q = encode_prefix(model, q_ids, torch.ones_like(q_ids))[0].to(dtype)
            Zq = sample_trajectory_cfg(model, z_q, args.steps, args.cfg_scale, device, dtype)
            plan_q = [model.predict_states(Zq[:, h]) for h in range(H)]

            # Sequential carryover state
            seq_state = seq_carryover_state(model, tokenizer, facts, device)

            # Shuffled state
            sh_facts = list(facts)
            donor = tasks[(ti + 1) % len(tasks)]
            donor_facts = agent_texts(donor, m)
            if donor_facts:
                sh_facts[-1] = donor_facts[-1]
            seq_state_shuf = seq_carryover_state(model, tokenizer, sh_facts, device)

            a = args.plan_weight

            # Existing conditions
            pred_local = raw_text(model, tokenizer, facts[0], q, args)
            full_ctx = " ".join(shared) + " " + full_hidden
            pred_full = raw_text(model, tokenizer, full_ctx, q, args)
            pred_chain = raw_text(model, tokenizer, " ".join(facts), q, args)
            pred_budget = raw_text(model, tokenizer, " ".join(facts)[:args.budget_chars], q, args)

            pred_carry = decode_chunkwise(model, tokenizer, q_ids,
                lambda h, pq=plan_q: fuse_states([(a, pq[h]), ((1-a), seq_state)]), H, args)
            pred_avg = decode_chunkwise(model, tokenizer, q_ids,
                lambda h, pq=plan_q: fuse_states([(a, pq[h])] + [((1-a)*w, mm) for mm in mems]), H, args)
            pred_shuf = decode_chunkwise(model, tokenizer, q_ids,
                lambda h, pq=plan_q: fuse_states([(a, pq[h]), ((1-a), seq_state_shuf)]), H, args)

            # NEW: memory_only (pure seq_state, no plan)
            pred_memory = decode_chunkwise(model, tokenizer, q_ids,
                lambda h: fuse_states([(1.0, seq_state)]), H, args)

            # NEW: dual_resid (agent-plan residual fusion + seq_state)
            pred_dual_resid = decode_chunkwise(model, tokenizer, q_ids,
                lambda h, pr=plan_resid: fuse_states([(a, pr[h]), ((1-a), seq_state)]), H, args)

            # NEW: dual PCA K1/K2/K3
            pred_dual_k1 = decode_chunkwise(model, tokenizer, q_ids,
                lambda h, pk=plan_pca_k1: fuse_states([(a, pk[h]), ((1-a), seq_state)]), H, args)
            pred_dual_k2 = decode_chunkwise(model, tokenizer, q_ids,
                lambda h, pk=plan_pca_k2: fuse_states([(a, pk[h]), ((1-a), seq_state)]), H, args)
            pred_dual_k3 = decode_chunkwise(model, tokenizer, q_ids,
                lambda h, pk=plan_pca_k3: fuse_states([(a, pk[h]), ((1-a), seq_state)]), H, args)

            preds = {
                "local_only": pred_local, "full_oracle": pred_full,
                "text_chain": pred_chain, "text_budget": pred_budget,
                "state_carryover": pred_carry, "parallel_avg": pred_avg,
                "shuffled_state": pred_shuf,
                "memory_only": pred_memory,
                "dual_resid": pred_dual_resid,
                "dual_pca_k1": pred_dual_k1,
                "dual_pca_k2": pred_dual_k2,
                "dual_pca_k3": pred_dual_k3,
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
              f"mem={'1' if ca['memory_only']['hit'] else '.'} "
              f"dual={'1' if ca['dual_resid']['hit'] else '.'} "
              f"k3={'1' if ca['dual_pca_k3']['hit'] else '.'}", flush=True)
        torch.cuda.empty_cache()

    # Aggregate
    out = {"ckpt_dir": args.ckpt_dir, "n_tasks": len(items_out),
           "num_agents": args.num_agents, "steps": args.steps,
           "cfg_scale": args.cfg_scale, "budget_chars": args.budget_chars,
           "plan_weight": args.plan_weight, "lam": args.lam,
           "accuracy": {}, "items": items_out}
    for rm in receiver_modes:
        n = max(1, ntask[rm])
        out["accuracy"][rm] = {c: round(agg[rm][c] / n, 3) for c in conds}
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(f"\n=== HiddenBench Dual+PCA ({len(items_out)} tasks, M={args.num_agents}) ===", flush=True)
    for rm in receiver_modes:
        print(f"  [{rm}]", flush=True)
        for c in conds:
            print(f"    {out['accuracy'][rm][c]*100:5.1f}%  {c}", flush=True)
    print(f"written: {args.output}", flush=True)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default=str(
        REPO / "outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000"))
    p.add_argument("--data", default=str(REPO / "data/hiddenbench/benchmark.json"))
    p.add_argument("--output", default=str(REPO / "results/hiddenbench/hiddenbench_dual_pca.json"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_agents", type=int, default=4)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--cfg_scale", type=float, default=3.0)
    p.add_argument("--budget_chars", type=int, default=120)
    p.add_argument("--plan_weight", type=float, default=0.7)
    p.add_argument("--lam", type=float, default=1.0)
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
