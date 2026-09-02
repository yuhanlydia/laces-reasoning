"""GRPO on S2 with MMLU answer correctness as reward.

Instead of using CE (which doesn't distinguish good vs bad Z), we use
whether the generated answer matches the ground truth. This directly
aligns the reward with downstream performance.

For each training step:
  1. Pick a random MMLU question
  2. Encode question as prefix → sample G=8 Z's
  3. For each Z: decode answer (greedy first token or full match)
  4. Reward = 1.0 if correct, 0.0 if wrong (or -1/+1 for contrastive)
  5. GRPO policy gradient to push S2 toward Z's that produce correct answers

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/train_s2_grpo_mmlu.py \
    --ckpt_dir outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-ddpm-condboundary-sft-maximal/step_00050000 \
    --num_steps 500 --group_size 8 --steps 20
"""
import argparse, json, sys, math, glob, random
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import eval.diag_loop1_common as Diag
from models.state_hijacking_dit import cosine_alpha_bar
from scripts.eval.sample_prefix_suffix_trajectory_cfg import encode_prefix


# ---- Stochastic sampler (same as train_s2_grpo.py) ----


def _denoise_eps(model, z, t_batch, cond, uncond, cfg_scale):
    eps_cond = model.trajectory_dit(z, t_batch, cond=cond)
    if cfg_scale == 1.0:
        return eps_cond
    eps_uncond = model.trajectory_dit(z, t_batch, cond=uncond)
    return eps_uncond + cfg_scale * (eps_cond - eps_uncond)


def stochastic_sample_with_logprob(model, cond, steps, cfg_scale, device, dtype, eta=0.3):
    H = int(model.trajectory_horizon)
    B = cond.shape[0]
    z = torch.randn(B, H, model.latent_dim, device=device, dtype=dtype)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)
    uncond = torch.zeros_like(cond)
    total_logprob = torch.zeros(B, device=device, dtype=torch.float32)
    for i in range(steps):
        t_cur, t_nxt = ts[i], ts[i + 1]
        ab_cur = cosine_alpha_bar(t_cur.unsqueeze(0)).to(dtype).clamp(min=1e-4)
        ab_nxt = cosine_alpha_bar(t_nxt.unsqueeze(0)).to(dtype).clamp(min=1e-4)
        t_batch = t_cur.expand(B)
        eps = _denoise_eps(model, z, t_batch, cond, uncond, cfg_scale)
        z0_pred = (z - (1 - ab_cur).sqrt() * eps) / ab_cur.sqrt()
        sigma = (
            eta * ((1 - ab_nxt) / (1 - ab_cur)).clamp(min=1e-6).sqrt()
            * (1 - ab_cur / ab_nxt).clamp(min=0).sqrt()
        )
        coef = (1 - ab_nxt - sigma**2).clamp(min=0).sqrt()
        mean = ab_nxt.sqrt() * z0_pred + coef * eps
        if i < steps - 1:
            noise = torch.randn_like(z)
            z_next = mean + sigma * noise
            std = sigma.clamp(min=1e-6)
            var = (std**2).clamp(min=1e-8)
            lp = -0.5 * (((z_next - mean) ** 2) / var + torch.log(2 * math.pi * var))
            total_logprob = total_logprob + lp.float().flatten(1).sum(dim=1)
            z = z_next
        else:
            z = mean
    return z, total_logprob


# ---- MMLU reward ----


def load_mmlu_questions(task_data_dir, num_questions=200):
    """Load MMLU questions as (prompt, answer_letter, answer_text) tuples."""
    questions = []
    mmlu_path = Path(task_data_dir) / "mmlu.jsonl"
    if not mmlu_path.exists():
        print(f"MMLU data not found at {mmlu_path}", flush=True)
        return questions
    with open(mmlu_path) as f:
        for line in f:
            d = json.loads(line.strip())
            prompt = d.get("prompt", "")
            choices = d.get("choices", [])
            answer = d.get("answer", d.get("ground_truth", ""))
            # Build full prompt
            choice_str = "\n".join(f"({c}) {t}" for c, t in zip(["A","B","C","D"], choices[:4]))
            full_prompt = f"{prompt}\n{choice_str}\nAnswer:"
            questions.append((full_prompt, str(answer).strip()))
            if len(questions) >= num_questions:
                break
    return questions


@torch.no_grad()
def decode_answer(model, tokenizer, input_ids, z_traj, max_tokens=32):
    """Decode answer from Z trajectory. Returns generated text after the prompt."""
    C = int(model.trajectory_chunk_size)
    H = z_traj.shape[1]
    states = model.predict_trajectory_states(z_traj)
    out = model.rwkv_model(input_ids=input_ids, use_cache=True, return_dict=True)
    pkv = out.past_key_values
    gen_ids = []
    for h in range(min(H, 4)):  # first 4 chunks = enough for MCQ answer
        states_h = [s[:, h] for s in states]
        pkv = model.inject_into_cache(pkv, states_h)
        for _ in range(min(C, max_tokens - len(gen_ids))):
            out2 = model.rwkv_model(
                input_ids=torch.tensor([[gen_ids[-1] if gen_ids else input_ids[0, -1].item()]],
                                       device=input_ids.device),
                past_key_values=pkv, use_cache=True, return_dict=True,
            )
            pkv = out2.past_key_values
            logits = out2.logits[0, -1]
            next_id = int(logits.argmax())
            gen_ids.append(next_id)
            if next_id == tokenizer.eos_token_id:
                break
    return tokenizer.decode(gen_ids).strip()


def extract_answer_letter(text: str) -> str:
    """Extract answer letter (A/B/C/D) from generated text."""
    text = text.strip().upper()
    for ch in text[:10]:  # look in first 10 chars
        if ch in "ABCD":
            return ch
    # Try to find "(A)" or "A)" pattern
    import re
    m = re.search(r'\(?([A-D])\)?', text)
    if m:
        return m.group(1)
    return ""


# ---- Main ----


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-ddpm-condboundary-sft-maximal/step_00050000")
    ap.add_argument("--task_data_dir", default="baseline/Cola-DLM/generate_task_data")
    ap.add_argument("--num_steps", type=int, default=500)
    ap.add_argument("--group_size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg_scale", type=float, default=3.0)
    ap.add_argument("--eta", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl_coef", type=float, default=0.1)
    ap.add_argument("--out", default="outputs_eval/s2_grpo_mmlu.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # Load MMLU questions
    questions = load_mmlu_questions(args.task_data_dir, num_questions=200)
    print(f"Loaded {len(questions)} MMLU questions", flush=True)
    if not questions:
        print("ERROR: No MMLU questions found", flush=True)
        return

    model, tok, dtype, pad = Diag.build_model(args.ckpt_dir, args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    for p in model.trajectory_dit.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(
        [p for p in model.trajectory_dit.parameters() if p.requires_grad], lr=args.lr
    )

    import copy
    ref_denoiser = copy.deepcopy(model.trajectory_dit).eval()
    for p in ref_denoiser.parameters():
        p.requires_grad = False

    class _RefShim:
        def __init__(self, base, denoiser):
            self._base = base
            self.trajectory_dit = denoiser
            self.trajectory_horizon = base.trajectory_horizon
            self.latent_dim = base.latent_dim

    ref_model = _RefShim(model, ref_denoiser)

    reward_hist = []
    correct_hist = []

    for step in range(1, args.num_steps + 1):
        # Pick a random question
        prompt_str, gt_answer = random.choice(questions)
        ids = tok(prompt_str, return_tensors="pt").input_ids.to(args.device)
        am = torch.ones_like(ids)
        z_prefix, _, _ = encode_prefix(model, ids, am)
        cond = z_prefix.detach().expand(args.group_size, -1)

        # Sample group of Z's
        z_group, logprob, kl = stochastic_sample_with_logprob(
            model, cond, args.steps, args.cfg_scale, args.device, dtype, eta=args.eta,
            ref_model=ref_model if args.kl_coef > 0 else None,
        )

        # Decode answers and compute rewards
        rewards = torch.zeros(args.group_size, device=args.device, dtype=torch.float32)
        for g in range(args.group_size):
            z_g = z_group[g:g+1]
            answer_text = decode_answer(model, tok, ids, z_g, max_tokens=16)
            pred_letter = extract_answer_letter(answer_text)
            is_correct = (pred_letter == gt_answer)
            rewards[g] = 1.0 if is_correct else 0.0

        correct_count = int(rewards.sum().item())
        correct_hist.append(correct_count)

        # GRPO loss
        if rewards.std() > 0:
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
        else:
            adv = torch.zeros_like(rewards)
        pg_loss = -(adv.detach() * logprob).mean()
        kl_loss = kl.mean() if args.kl_coef > 0 else torch.zeros((), device=args.device)
        loss = pg_loss + args.kl_coef * kl_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.trajectory_dit.parameters() if p.requires_grad], 1.0
        )
        opt.step()

        reward_hist.append(float(rewards.mean().item()))

        if step % 10 == 0:
            recent_r = np.mean(reward_hist[-20:]) if len(reward_hist) >= 20 else np.mean(reward_hist)
            recent_c = np.mean(correct_hist[-20:]) if len(correct_hist) >= 20 else np.mean(correct_hist)
            print(
                f"[step {step}] mean_reward={rewards.mean().item():.3f} "
                f"correct={correct_count}/{args.group_size} "
                f"best_r={rewards.max().item():.0f} "
                f"kl={float(kl_loss.item()):.2f} "
                f"running_r={recent_r:.3f} running_c={recent_c:.1f}",
                flush=True,
            )

        if step % 100 == 0:
            torch.save(
                {"trajectory_dit": model.trajectory_dit.state_dict(), "step": step},
                f"outputs_relay/s2-grpo-mmlu/grpo_step{step}.pt",
            )

    # Report
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    first20 = float(np.mean(reward_hist[:20])) if len(reward_hist) >= 20 else float(np.mean(reward_hist))
    last20 = float(np.mean(reward_hist[-20:]))
    res = {
        "ckpt": args.ckpt_dir, "num_steps": len(reward_hist), "group_size": args.group_size,
        "first20_reward": first20, "last20_reward": last20,
        "reward_improvement": last20 - first20,
        "reward_hist": reward_hist, "correct_hist": correct_hist,
    }
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n=== GRPO-MMLU ===")
    print(f"  first20 reward: {first20:.3f}  last20: {last20:.3f}  improvement: {last20 - first20:+.3f}")
    verdict = "REWARD RISES -> GRPO+MMLU works" if last20 - first20 > 0.02 else "flat -> tune"
    print(f"  verdict: {verdict}")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
