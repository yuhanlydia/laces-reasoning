"""Decode each iterated-relay latent into ACTUAL TEXT.

Shows what the anchor vs renorm latent-relay variants actually generate, per
relay step, so we can judge whether iterating the latent produces meaningful,
improving text (real "thinking") or degrades. For each relay step t, the latent
z_t is projected via S1 -> RWKV state, injected, and the frozen RWKV generates a
continuation. Zero training.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.sample_prefix_suffix_cfg import encode_prefix, sample_ddim_cfg, generate  # noqa: E402
from scripts.eval import diag_loop1_common as C  # noqa: E402


@torch.no_grad()
def run(ckpt_dir, device, prompt_idx, steps, cfg_scale, k_relay, anchor, renorm,
        max_new_tokens, output):
    model, tokenizer, _dtype, pad_id = C.build_model(ckpt_dir, device)
    model.eval()
    model._prefix_suffix_s2 = True
    dtype = next(model.latent_dit.parameters()).dtype

    gen_args = SimpleNamespace(
        max_new_tokens=max_new_tokens, temperature=0.7, top_k=30, top_p=0.9,
        repetition_penalty=1.3,
    )

    passage = C.PASSAGES[prompt_idx]
    prefix_ids, _suf, _ln = C.split_prefix_suffix(tokenizer, passage, model, pad_id)
    ids = torch.tensor([prefix_ids], device=device)
    am = torch.ones_like(ids)
    z_problem = encode_prefix(model, ids, am).to(dtype)

    print(f"PROMPT[{prompt_idx}]: {passage[:120]}...", flush=True)
    print(f"mode: anchor={anchor} renorm={renorm}\n", flush=True)

    records = []

    def decode_and_log(t, z):
        text, _ = generate(model, tokenizer, ids, am, z, gen_args)
        gen = text[len(tokenizer.decode(prefix_ids)):].strip()
        records.append({"step": t, "norm": float(z.float().norm(dim=-1).mean()), "text": gen})
        print(f"--- relay {t} (norm={float(z.float().norm(dim=-1).mean()):.1f}) ---", flush=True)
        print(gen[:400], flush=True)
        print("", flush=True)

    z_cur = z_problem.clone()
    decode_and_log(0, z_cur)
    for t in range(1, k_relay + 1):
        cond = z_cur
        if anchor > 0.0:
            cond = anchor * z_problem + (1.0 - anchor) * z_cur
        z_next = sample_ddim_cfg(model, cond, steps, cfg_scale, device, dtype)
        if renorm > 0.0:
            nrm = z_next.float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
            z_next = (z_next.float() * (renorm / nrm)).to(dtype)
        decode_and_log(t, z_next)
        z_cur = z_next

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(
        {"ckpt_dir": ckpt_dir, "prompt_idx": prompt_idx, "anchor": anchor,
         "renorm": renorm, "cfg_scale": cfg_scale, "records": records}, indent=2))
    print(f"written: {output}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="outputs_relay/test-v6-2.9B-s2-prefix-suffix-cfg/step_00150000")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prompt_idx", type=int, default=0)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cfg_scale", type=float, default=3.0)
    ap.add_argument("--k_relay", type=int, default=5)
    ap.add_argument("--anchor", type=float, default=0.0)
    ap.add_argument("--renorm", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--output", default="/tmp/diag_relay_decode.json")
    a = ap.parse_args()
    run(a.ckpt_dir, a.device, a.prompt_idx, a.steps, a.cfg_scale, a.k_relay,
        a.anchor, a.renorm, a.max_new_tokens, a.output)


if __name__ == "__main__":
    main()
