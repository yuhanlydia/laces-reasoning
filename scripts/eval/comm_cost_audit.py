#!/usr/bin/env python3
"""A00: Real communication-cost audit for multi-agent communication substrates.

Measures GENUINE per-hop and total payload BYTES (from actual tensor sizes, not
theoretical shape guesses), plus decoder-visible tokens, sender/receiver token
counts, latency, and peak GPU memory, for five communication objects:

  text        : natural-language facts concatenated / relayed (tokens grow with M, rounds)
  latent_plan : diffusion plan Z in R^[H, d]  (fixed per message)
  recurrent   : RWKV per-layer recurrent state (fixed per hop, independent of context len)
  kv_full     : Transformer-style KV cache 2*L*T*h_kv*d_head*bytes (grows with T)
  kv_equalbyte: KV compressed to match recurrent-state payload (reference point)

Key correction over the paper draft: the "73 chars" number is only the final
decoder-visible query, NOT the communication cost. Communication cost = bytes of
the transmitted object (recurrent state tensor / plan tensor / KV cache / text tokens).

Outputs:
  results/comm_cost/payload_summary.csv   -- one row per (method, M, rounds)
  results/comm_cost/fig_cost_vs_agents.pdf -- payload bytes vs number of agents M
  results/comm_cost/fig_kv_breakeven.pdf   -- KV bytes vs history tokens, with recurrent-state break-even line

Zero training; uses the champion checkpoint. GPU: set CUDA_VISIBLE_DEVICES to a free card.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from scripts.eval.relay_utils import load_relay_model  # noqa: E402
from scripts.eval.sample_prefix_suffix_trajectory_cfg import (  # noqa: E402
    encode_prefix, sample_trajectory_cfg,
)


# ---------- byte accounting primitives ----------

def tensor_bytes(x) -> int:
    """Genuine byte size of a tensor (numel * element_size). 0 for None."""
    if x is None or not isinstance(x, torch.Tensor):
        return 0
    return int(x.numel()) * int(x.element_size())


def state_payload_bytes(states) -> int:
    """Total bytes of a per-layer recurrent-state list (the transmitted object).

    states: list[Tensor|None] as produced by capture_final_state.
    """
    return sum(tensor_bytes(s) for s in states)


def plan_payload_bytes(Z) -> int:
    """Bytes of a latent plan Z in R^[H,d] (the transmitted object)."""
    return tensor_bytes(Z)


def text_payload_bytes(token_ids, id_bytes: int = 2) -> int:
    """Bytes to transmit a token-id message. Token ids as int (2 bytes for <=65k vocab).

    We report token-id bytes (the honest wire cost of a tokenized message); we ALSO
    report the raw token count and char count separately so reviewers can pick a
    convention. Text KV must be recomputed by the receiver, so its receiver-side cost
    is captured under kv_full for the same token stream.
    """
    n = len(token_ids)
    return n * id_bytes


def kv_cache_bytes(num_layers: int, seq_len: int, n_kv_heads: int,
                   head_dim: int, bytes_per_elem: int = 2) -> int:
    """Transformer KV cache size: 2 (K and V) * L * T * h_kv * d_head * bytes.

    This is the object a Transformer agent (LatentMAS / C2C) transmits; it GROWS
    linearly with the cached sequence length T.
    """
    return 2 * num_layers * seq_len * n_kv_heads * head_dim * bytes_per_elem


# ---------- state / plan capture (real model calls) ----------

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


def peak_mem_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


# ---------- reference Transformer dims for KV comparison ----------
# We report KV for a matched-scale Transformer (Qwen3-4B-like) so the comparison is
# against a real peer, and separately for the RWKV backbone's own attention-if-it-had-KV
# hypothetical. These are DECLARED reference constants (documented in the CSV), not
# measured from our recurrent model, because our model has no KV cache.
TRANSFORMER_REFS = {
    # name: (num_layers, n_kv_heads, head_dim)
    "qwen3_4b": (36, 8, 128),      # Qwen3-4B GQA (kv heads 8) -- LatentMAS main backbone
    "llama3_8b": (32, 8, 128),     # Llama-3 8B GQA
}


@torch.no_grad()
def run(args):
    device = args.device
    model, _rwkv, tokenizer, _ckpt, _cfg = load_relay_model(args.ckpt_dir, device)
    model.eval()
    model._prefix_suffix_trajectory_s2 = True
    dtype = next(model.alpha_heads.parameters()).dtype
    H = int(model.trajectory_horizon)

    def enc(text):
        return tokenizer(text, return_tensors="pt").input_ids.to(device)

    # A representative fact (Hidden-Profile style) for per-hop measurement.
    fact = "Agent 1 knows: the expedition's leader is Marlowe."
    query = "Question: what is the expedition's leader? Answer: the expedition's leader is"
    fact_ids = enc(fact)
    query_ids = enc(query)

    # --- measure ONE recurrent-state payload (fixed per hop) ---
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    states = capture_final_state(model, fact_ids)
    state_ms = (time.time() - t0) * 1000.0
    state_bytes = state_payload_bytes(states)
    state_bytes_bf16 = state_bytes // 2  # if transmitted in bf16 instead of fp32
    state_mem_mb = peak_mem_mb()

    # --- measure ONE latent-plan payload (fixed per message) ---
    t0 = time.time()
    z = encode_prefix(model, query_ids, torch.ones_like(query_ids))[0].to(dtype)
    Z = sample_trajectory_cfg(model, z, args.steps, args.cfg_scale, device, dtype)
    plan_ms = (time.time() - t0) * 1000.0
    plan_bytes = plan_payload_bytes(Z)          # native dtype (bf16)
    plan_bytes_fp16 = int(Z.numel()) * 2

    fact_tokens = fact_ids.shape[1]
    query_tokens = query_ids.shape[1]
    fact_chars = len(fact)
    query_chars = len(query)

    print(f"[dims] num_layers={model.num_layers} H={H} "
          f"trajectory Z shape={tuple(Z.shape)}")
    print(f"[per-hop] recurrent state fp32={state_bytes}B bf16={state_bytes_bf16}B "
          f"| plan={plan_bytes}B ({tuple(Z.shape)}) "
          f"| fact={fact_tokens}tok/{fact_chars}chars query={query_tokens}tok/{query_chars}chars")

    rows = []

    def add_row(method, M, rounds, per_hop_bytes, num_hops, decoder_tokens,
                sender_tokens, receiver_tokens, note=""):
        rows.append(dict(
            method=method, M=M, rounds=rounds,
            per_hop_payload_bytes=per_hop_bytes,
            total_payload_bytes=per_hop_bytes * num_hops,
            num_hops=num_hops,
            decoder_visible_tokens=decoder_tokens,
            sender_generated_tokens=sender_tokens,
            receiver_prefill_tokens=receiver_tokens,
            note=note,
        ))

    # Sweep number of agents M; each agent contributes one fact.
    for M in args.agents:
        rounds = 1
        # TEXT concat: receiver must see all M facts -> decoder tokens grow with M.
        text_total_tokens = fact_tokens * M + query_tokens
        add_row("text_concat", M, rounds,
                per_hop_bytes=text_payload_bytes(list(range(fact_tokens))),
                num_hops=M,
                decoder_tokens=text_total_tokens,
                sender_tokens=fact_tokens * M, receiver_tokens=text_total_tokens,
                note="decoder-visible context grows with M")

        # LATENT plan: one fixed-size plan per agent message.
        add_row("latent_plan", M, rounds,
                per_hop_bytes=plan_bytes, num_hops=M,
                decoder_tokens=query_tokens,
                sender_tokens=0, receiver_tokens=query_tokens,
                note=f"fixed plan {tuple(Z.shape)}, decoder sees query only")

        # RECURRENT state carryover: one fixed-size state per hop.
        add_row("recurrent_state_fp32", M, rounds,
                per_hop_bytes=state_bytes, num_hops=M,
                decoder_tokens=query_tokens,
                sender_tokens=0, receiver_tokens=query_tokens,
                note="fixed recurrent state per hop, decoder sees query only")
        add_row("recurrent_state_bf16", M, rounds,
                per_hop_bytes=state_bytes_bf16, num_hops=M,
                decoder_tokens=query_tokens,
                sender_tokens=0, receiver_tokens=query_tokens,
                note="bf16 recurrent state per hop")

        # KV full (reference Transformer): cache grows with cumulative tokens.
        for ref_name, (L, hkv, hd) in TRANSFORMER_REFS.items():
            # LatentMAS relays cumulative KV: agent i transmits KV over i facts.
            total_kv = 0
            for i in range(1, M + 1):
                total_kv += kv_cache_bytes(L, fact_tokens * i, hkv, hd)
            per_hop_avg = total_kv // M
            rows.append(dict(
                method=f"kv_full_{ref_name}", M=M, rounds=rounds,
                per_hop_payload_bytes=per_hop_avg,
                total_payload_bytes=total_kv,
                num_hops=M,
                decoder_visible_tokens=query_tokens,
                sender_generated_tokens=0,
                receiver_prefill_tokens=query_tokens,
                note=f"cumulative KV relay, {ref_name} dims L={L} hkv={hkv} hd={hd}",
            ))

    out_dir = REPO / "results" / "comm_cost"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "payload_summary.csv"
    fields = ["method", "M", "rounds", "per_hop_payload_bytes", "total_payload_bytes",
              "num_hops", "decoder_visible_tokens", "sender_generated_tokens",
              "receiver_prefill_tokens", "note"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[write] {csv_path} ({len(rows)} rows)")

    # break-even: at what history length T does one Transformer KV cache exceed one recurrent state?
    breakeven = {}
    for ref_name, (L, hkv, hd) in TRANSFORMER_REFS.items():
        per_tok = kv_cache_bytes(L, 1, hkv, hd)
        be = state_bytes / per_tok if per_tok else float("inf")
        be_bf16 = state_bytes_bf16 / per_tok if per_tok else float("inf")
        breakeven[ref_name] = dict(kv_bytes_per_token=per_tok,
                                   breakeven_tokens_fp32=be,
                                   breakeven_tokens_bf16=be_bf16)
    meta = dict(
        num_layers=model.num_layers, trajectory_shape=list(Z.shape),
        state_bytes_fp32=state_bytes, state_bytes_bf16=state_bytes_bf16,
        plan_bytes=plan_bytes, fact_tokens=fact_tokens, query_tokens=query_tokens,
        fact_chars=fact_chars, query_chars=query_chars,
        state_capture_ms=state_ms, plan_sample_ms=plan_ms,
        state_peak_mem_mb=state_mem_mb,
        transformer_refs=TRANSFORMER_REFS, breakeven=breakeven,
    )
    with open(out_dir / "audit_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[write] {out_dir / 'audit_meta.json'}")
    print("[break-even] history tokens where one Transformer KV > one recurrent state:")
    for k, v in breakeven.items():
        print(f"   {k}: fp32={v['breakeven_tokens_fp32']:.1f} tok, bf16={v['breakeven_tokens_bf16']:.1f} tok "
              f"(kv {v['kv_bytes_per_token']}B/tok)")

    _make_plots(rows, meta, out_dir)


def _make_plots(rows, meta, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fig 1: total payload bytes vs M
    Ms = sorted({r["M"] for r in rows})
    series = {}
    for r in rows:
        series.setdefault(r["method"], {})[r["M"]] = r["total_payload_bytes"]
    plt.figure(figsize=(5, 3.4))
    for method, d in series.items():
        ys = [d.get(m, None) for m in Ms]
        if any(y is None for y in ys):
            continue
        plt.plot(Ms, ys, marker="o", label=method, linewidth=1.4, markersize=4)
    plt.yscale("log")
    plt.xlabel("number of agents M")
    plt.ylabel("total communicated bytes (log)")
    plt.title("Communication cost vs agents")
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_cost_vs_agents.pdf")
    plt.close()

    # Fig 2: KV bytes vs history tokens with recurrent-state break-even line
    plt.figure(figsize=(5, 3.4))
    T = list(range(1, 2049))
    for ref_name, (L, hkv, hd) in meta["transformer_refs"].items():
        kv = [kv_cache_bytes(L, t, hkv, hd) for t in T]
        plt.plot(T, kv, label=f"KV {ref_name}", linewidth=1.4)
    plt.axhline(meta["state_bytes_fp32"], color="k", linestyle="--",
                label="recurrent state (fp32)", linewidth=1.0)
    plt.axhline(meta["state_bytes_bf16"], color="gray", linestyle=":",
                label="recurrent state (bf16)", linewidth=1.0)
    plt.yscale("log")
    plt.xlabel("history tokens T")
    plt.ylabel("payload bytes (log)")
    plt.title("KV cache vs fixed recurrent state (break-even)")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_kv_breakeven.pdf")
    plt.close()
    print(f"[write] {out_dir / 'fig_cost_vs_agents.pdf'}")
    print(f"[write] {out_dir / 'fig_kv_breakeven.pdf'}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default=str(
        REPO / "outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-joint-scratch/step_00026000"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--cfg_scale", type=float, default=3.0)
    p.add_argument("--agents", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
