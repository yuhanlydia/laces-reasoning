"""Convert BlinkDL RWKV-7 (.pth) → fla/HF (config.json + model.safetensors).

After conversion, the output directory can be loaded by
`AutoModelForCausalLM.from_pretrained(...)` just like fla-hub/rwkv7-7.2B-g0,
so it can be used by `train_state_hijacking_dit.py` and the active
`scripts/eval/sample_*.py` samplers.

The script auto-infers all architecture hyperparameters from the .pth itself
(num_hidden_layers, hidden_size, head_dim, gate_low_rank_dim, intermediate_size,
vocab_size), so it works for any BlinkDL RWKV-7 G0/G1/G1f checkpoint —
0.1B, 0.4B, 1.5B, 2.9B, 7.2B, 13.3B, future ones.

Usage:
    python scripts/tools/convert_blinkdl_to_hf.py \\
        --pth /path/to/rwkv7-g1f-13.3b-20260415-ctx8192.pth \\
        --out_dir /path/to/RWKV7-G1f-13.3B-HF \\
        --tokenizer_src fla-hub/rwkv7-7.2B-g0     # copy tokenizer from any g0/g1 repo
"""

import argparse
import json
import os
import re
import sys

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pth", required=True, help="Path to BlinkDL .pth file")
    p.add_argument("--out_dir", required=True, help="Output dir for HF-format model")
    p.add_argument("--tokenizer_src", default="fla-hub/rwkv7-7.2B-g0",
                   help="HF repo or local dir to copy tokenizer files from")
    p.add_argument("--max_position_embeddings", type=int, default=8192,
                   help="ctx length for config.json (BlinkDL .pth has no ctx info inside)")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--verify", action="store_true",
                   help="After saving, reload via AutoModelForCausalLM and run a dummy forward")
    return p.parse_args()


def infer_arch(state):
    """Auto-detect architecture hyperparameters from a BlinkDL state_dict."""
    # vocab + hidden
    emb = state["emb.weight"]
    vocab_size, hidden_size = emb.shape

    # layers — count blocks.N.*
    layer_ids = set()
    for k in state:
        m = re.match(r"blocks\.(\d+)\.", k)
        if m:
            layer_ids.add(int(m.group(1)))
    num_hidden_layers = max(layer_ids) + 1

    # head_dim — from r_k shape [num_heads, head_dim] in newer dumps, or
    # from k_k shape [hidden_size] (no head info) → fall back to ln_x
    head_dim = 64  # RWKV-7 standard
    if "blocks.0.att.r_k" in state:
        rk = state["blocks.0.att.r_k"]
        if rk.dim() == 2:
            num_heads, head_dim = rk.shape
        else:
            num_heads = hidden_size // head_dim
    else:
        num_heads = hidden_size // head_dim

    # gate_low_rank_dim — from att.g1 (BlinkDL stores as [hidden, rank])
    g1 = state["blocks.0.att.g1"]
    gate_low_rank_dim = g1.shape[1] if g1.dim() == 2 else g1.shape[0]

    # decay/iclr low-rank dims (w1 / a1 / v1)
    w1 = state["blocks.0.att.w1"]
    decay_low_rank_dim = w1.shape[1] if w1.dim() == 2 else w1.shape[0]
    a1 = state["blocks.0.att.a1"]
    a_low_rank_dim = a1.shape[1] if a1.dim() == 2 else a1.shape[0]
    # v_lora exists only for layers >= 1
    v_low_rank_dim = None
    if "blocks.1.att.v1" in state:
        v1 = state["blocks.1.att.v1"]
        v_low_rank_dim = v1.shape[1] if v1.dim() == 2 else v1.shape[0]

    # FFN intermediate_size — ffn.key is [intermediate, hidden]
    fk = state["blocks.0.ffn.key.weight"]
    intermediate_size = fk.shape[0]

    arch = dict(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        head_dim=head_dim,
        num_heads=num_heads,
        gate_low_rank_dim=gate_low_rank_dim,
        decay_low_rank_dim=decay_low_rank_dim,
        a_low_rank_dim=a_low_rank_dim,
        v_low_rank_dim=v_low_rank_dim,
        intermediate_size=intermediate_size,
    )
    return arch


def map_state(state, arch):
    """Map BlinkDL keys → fla RWKV7ForCausalLM keys with shape adjustments."""
    new_state = {}
    H = arch["hidden_size"]

    new_state["model.embeddings.weight"] = state["emb.weight"]
    new_state["model.norm.weight"] = state["ln_out.weight"]
    if "ln_out.bias" in state:
        new_state["model.norm.bias"] = state["ln_out.bias"]
    new_state["lm_head.weight"] = state["head.weight"]

    for n in range(arch["num_hidden_layers"]):
        b = f"blocks.{n}."
        L = f"model.layers.{n}."

        # block-0-only pre-norm (called ln0 in BlinkDL, pre_norm in fla)
        if n == 0 and f"{b}ln0.weight" in state:
            new_state[f"{L}pre_norm.weight"] = state[f"{b}ln0.weight"]
            if f"{b}ln0.bias" in state:
                new_state[f"{L}pre_norm.bias"] = state[f"{b}ln0.bias"]

        # attn_norm = ln1
        new_state[f"{L}attn_norm.weight"] = state[f"{b}ln1.weight"]
        if f"{b}ln1.bias" in state:
            new_state[f"{L}attn_norm.bias"] = state[f"{b}ln1.bias"]
        # ffn_norm = ln2
        new_state[f"{L}ffn_norm.weight"] = state[f"{b}ln2.weight"]
        if f"{b}ln2.bias" in state:
            new_state[f"{L}ffn_norm.bias"] = state[f"{b}ln2.bias"]

        # time-mix scalars (same names, no transpose)
        for sub in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"):
            new_state[f"{L}attn.{sub}"] = state[f"{b}att.{sub}"]

        # bonus / k_k / k_a / r_k
        new_state[f"{L}attn.k_k"] = state[f"{b}att.k_k"]
        new_state[f"{L}attn.k_a"] = state[f"{b}att.k_a"]
        new_state[f"{L}attn.r_k"] = state[f"{b}att.r_k"]

        # projections (BlinkDL stores [out, in], fla nn.Linear also [out, in] → no transpose)
        new_state[f"{L}attn.r_proj.weight"] = state[f"{b}att.receptance.weight"]
        new_state[f"{L}attn.k_proj.weight"] = state[f"{b}att.key.weight"]
        new_state[f"{L}attn.v_proj.weight"] = state[f"{b}att.value.weight"]
        new_state[f"{L}attn.o_proj.weight"] = state[f"{b}att.output.weight"]

        # group norm on values (ln_x → g_norm)
        new_state[f"{L}attn.g_norm.weight"] = state[f"{b}att.ln_x.weight"]
        if f"{b}att.ln_x.bias" in state:
            new_state[f"{L}attn.g_norm.bias"] = state[f"{b}att.ln_x.bias"]

        # ── LoRAs: BlinkDL stores w1=[D, rank], w2=[rank, D], w0=[1,1,D] ────
        # fla LoRA module uses nn.Linear(D, rank, bias=False) then
        # nn.Linear(rank, D, bias=True), so weights are stored as [rank, D]
        # and [D, rank] — i.e. transposed compared to BlinkDL.
        def _lora_in(t):    # w1: [D, rank] → lora.0.weight [rank, D]
            return t.t().contiguous() if t.dim() == 2 else t

        def _lora_out(t):   # w2: [rank, D] → lora.2.weight [D, rank]
            return t.t().contiguous() if t.dim() == 2 else t

        def _lora_bias(t):  # w0: [1,1,D] or [D] → [D]
            return t.reshape(-1).contiguous()

        # w_lora (decay)
        new_state[f"{L}attn.w_lora.lora.0.weight"] = _lora_in(state[f"{b}att.w1"])
        new_state[f"{L}attn.w_lora.lora.2.weight"] = _lora_out(state[f"{b}att.w2"])
        new_state[f"{L}attn.w_lora.lora.2.bias"] = _lora_bias(state[f"{b}att.w0"])

        # a_lora (in-context-LR mixing)
        new_state[f"{L}attn.a_lora.lora.0.weight"] = _lora_in(state[f"{b}att.a1"])
        new_state[f"{L}attn.a_lora.lora.2.weight"] = _lora_out(state[f"{b}att.a2"])
        new_state[f"{L}attn.a_lora.lora.2.bias"] = _lora_bias(state[f"{b}att.a0"])

        # v_lora (only for n >= 1)
        if n >= 1 and f"{b}att.v1" in state:
            new_state[f"{L}attn.v_lora.lora.0.weight"] = _lora_in(state[f"{b}att.v1"])
            new_state[f"{L}attn.v_lora.lora.2.weight"] = _lora_out(state[f"{b}att.v2"])
            new_state[f"{L}attn.v_lora.lora.2.bias"] = _lora_bias(state[f"{b}att.v0"])

        # g_lora (gate) — no bias in BlinkDL, only g1/g2
        new_state[f"{L}attn.g_lora.lora.0.weight"] = _lora_in(state[f"{b}att.g1"])
        new_state[f"{L}attn.g_lora.lora.2.weight"] = _lora_out(state[f"{b}att.g2"])

        # FFN (same names, no transpose)
        new_state[f"{L}ffn.x_k"] = state[f"{b}ffn.x_k"]
        new_state[f"{L}ffn.key.weight"] = state[f"{b}ffn.key.weight"]
        new_state[f"{L}ffn.value.weight"] = state[f"{b}ffn.value.weight"]

    return new_state


def main():
    args = parse_args()
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print(f"Loading BlinkDL .pth from {args.pth} ...")
    state = torch.load(args.pth, map_location="cpu", weights_only=True)
    print(f"  Loaded {len(state)} tensors, "
          f"total {sum(t.numel() for t in state.values())/1e9:.2f}B params")

    print("Inferring architecture from state_dict ...")
    arch = infer_arch(state)
    for k, v in arch.items():
        print(f"  {k:25s} = {v}")

    print("Mapping keys BlinkDL → fla ...")
    new_state = map_state(state, arch)
    print(f"  Mapped to {len(new_state)} target tensors")

    # Sanity: build fla model from inferred config and check key match
    from fla.models import RWKV7Config
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rwkv_cfg = RWKV7Config(
        vocab_size=arch["vocab_size"],
        hidden_size=arch["hidden_size"],
        num_hidden_layers=arch["num_hidden_layers"],
        head_dim=arch["head_dim"],
        num_heads=arch["num_heads"],
        gate_low_rank_dim=arch["gate_low_rank_dim"],
        decay_low_rank_dim=arch["decay_low_rank_dim"],
        a_low_rank_dim=arch["a_low_rank_dim"],
        v_low_rank_dim=arch["v_low_rank_dim"] or arch["a_low_rank_dim"],
        intermediate_size=arch["intermediate_size"],
        max_position_embeddings=args.max_position_embeddings,
    )
    # Ensure the compat attr fla-hub repos use
    rwkv_cfg.fuse_linear_cross_entropy = False

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dt = dtype_map[args.dtype]

    print(f"Building empty RWKV7ForCausalLM (dtype={args.dtype}) ...")
    model = AutoModelForCausalLM.from_config(rwkv_cfg, torch_dtype=dt)

    target_keys = set(model.state_dict().keys())
    have_keys = set(new_state.keys())
    missing = target_keys - have_keys
    extra = have_keys - target_keys

    if missing:
        print(f"⚠️  {len(missing)} keys expected by fla but NOT produced by mapping:")
        for k in sorted(missing)[:15]:
            print(f"    {k}")
        if len(missing) > 15:
            print(f"    ... ({len(missing)-15} more)")
    if extra:
        print(f"⚠️  {len(extra)} keys produced by mapping but NOT expected by fla:")
        for k in sorted(extra)[:15]:
            print(f"    {k}")

    sd = model.state_dict()
    shape_changed = 0
    for k in have_keys & target_keys:
        sv = new_state[k]; tv = sd[k]
        if sv.dim() >= 2 and tv.dim() == 1 and sv.shape[0] == 1:
            new_state[k] = sv.reshape(tv.shape); shape_changed += 1
        elif sv.dim() == 1 and tv.dim() >= 2 and tv.shape[0] == 1:
            new_state[k] = sv.reshape(tv.shape); shape_changed += 1
    if shape_changed:
        print(f"  Auto-fixed {shape_changed} shape mismatches (BlinkDL ↔ fla convention)")
    shape_mismatches = []
    for k in have_keys & target_keys:
        if new_state[k].shape != sd[k].shape:
            shape_mismatches.append((k, tuple(new_state[k].shape), tuple(sd[k].shape)))
    if shape_mismatches:
        print(f"⚠️  {len(shape_mismatches)} shape mismatches:")
        for k, src, tgt in shape_mismatches[:15]:
            print(f"    {k}: got {src}, expected {tgt}")
        raise RuntimeError("Shape mismatch — aborting before save. "
                           "Check head_dim / low_rank_dim inference, or BlinkDL transpose convention.")

    if missing:
        raise RuntimeError(f"Cannot save: {len(missing)} required keys missing.")

    # cast tensors to target dtype
    new_state = {k: v.to(dt) for k, v in new_state.items()}

    print("Loading mapped weights into model ...")
    res = model.load_state_dict(new_state, strict=False)
    if res.missing_keys:
        print(f"⚠️  load_state_dict still reports missing: {res.missing_keys[:10]}")
    if res.unexpected_keys:
        print(f"⚠️  load_state_dict reports unexpected: {res.unexpected_keys[:10]}")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Saving to {args.out_dir} ...")
    try:
        model.save_pretrained(args.out_dir, safe_serialization=True, max_shard_size="5GB")
    except Exception:
        print("  save_pretrained failed (tied-weights bug), saving manually ...")
        torch.save(model.state_dict(), os.path.join(args.out_dir, "pytorch_model.bin"))
        model.config.save_pretrained(args.out_dir)

    # Copy tokenizer files DIRECTLY (do NOT use AutoTokenizer.save_pretrained —
    # it does not preserve RWKV's custom rwkv_vocab_v20230424.txt format
    # correctly, leading to "substring not found" errors when the trie
    # tokenizer tries to parse blank or malformed lines).
    print(f"Copying tokenizer files from {args.tokenizer_src} ...")
    import shutil
    TOKENIZER_FILES = [
        "rwkv_vocab_v20230424.txt",
        "vocab.txt",
        "tokenizer_config.json",
        "hf_rwkv_tokenizer.py",
        "special_tokens_map.json",
        "added_tokens.json",
    ]
    if os.path.isdir(args.tokenizer_src):
        tok_src_dir = args.tokenizer_src
    else:
        # HF repo: snapshot_download with file allowlist
        from huggingface_hub import snapshot_download
        tok_src_dir = snapshot_download(
            repo_id=args.tokenizer_src,
            allow_patterns=TOKENIZER_FILES,
        )
    copied = []
    for fname in TOKENIZER_FILES:
        src = os.path.join(tok_src_dir, fname)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(args.out_dir, fname))
            copied.append(fname)
    print(f"  Copied {len(copied)} tokenizer files: {copied}")
    # Sanity check: verify vocab parses cleanly
    vocab_path = None
    for cand in ("rwkv_vocab_v20230424.txt", "vocab.txt"):
        if os.path.isfile(os.path.join(args.out_dir, cand)):
            vocab_path = os.path.join(args.out_dir, cand); break
    if vocab_path is None:
        print("  ⚠️  No vocab file found in copied set — tokenizer load WILL fail.")
    else:
        with open(vocab_path, encoding="utf-8") as f:
            n_bad = sum(1 for line in f if line.strip() and " " not in line)
        if n_bad > 0:
            print(f"  ⚠️  {n_bad} malformed lines in {vocab_path} (will trip RWKV trie tokenizer)")
        else:
            print(f"  ✓ Vocab file parses clean: {vocab_path}")

    # Patch config.json to include fuse_linear_cross_entropy for forward-compat
    cfg_path = os.path.join(args.out_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["fuse_linear_cross_entropy"] = False
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    n_total = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"✅ Saved {n_total:.2f}B-param RWKV-7 to {args.out_dir}")

    if args.verify:
        print("\n=== Verification: reload + dummy forward ===")
        del model, new_state, state
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        m = AutoModelForCausalLM.from_pretrained(args.out_dir, trust_remote_code=True,
                                                  torch_dtype=dt)
        m.eval()
        if torch.cuda.is_available():
            m = m.cuda()
        with torch.no_grad():
            ids = torch.tensor([[1, 2, 3, 4, 5]], device=next(m.parameters()).device)
            out = m(input_ids=ids, output_hidden_states=True, return_dict=True)
        h = out.hidden_states[-1]
        print(f"  Dummy forward OK: hidden_states[-1] shape = {tuple(h.shape)}, "
              f"mean={h.float().mean().item():+.4f}, std={h.float().std().item():.4f}")
        if not torch.isfinite(h).all():
            print("  ❌ Non-finite values in hidden_states — conversion likely broken")
        else:
            print("  ✅ All finite — conversion looks correct")


if __name__ == "__main__":
    main()
