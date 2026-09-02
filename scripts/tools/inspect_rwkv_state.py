"""Inspect RWKV-7's WKV state structure for state-hijacking implementation.

Loads a small RWKV-7 model, runs one forward with use_cache=True, prints
the structure of past_key_values so we know exactly what shape the
state-predictor needs to produce.

Usage:
    python scripts/tools/inspect_rwkv_state.py \\
        --rwkv_path /inspire/.../models/RWKV7-Goose-World3-2.9B-HF
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def describe(x, prefix=""):
    if isinstance(x, torch.Tensor):
        return f"{prefix}Tensor shape={tuple(x.shape)}, dtype={x.dtype}"
    if isinstance(x, (list, tuple)):
        return f"{prefix}{type(x).__name__}[len={len(x)}]\n" + "\n".join(
            describe(e, prefix + "  ") for e in x[:3]  # only first 3
        ) + ("\n  ..." if len(x) > 3 else "")
    if hasattr(x, "__dict__"):
        return f"{prefix}{type(x).__name__} attrs={list(x.__dict__.keys())[:10]}"
    return f"{prefix}{type(x).__name__} value={x}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rwkv_path", type=str, required=True)
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seq_len", type=int, default=16, help="Short seq for fast inspect")
    args = p.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    print(f"\nLoading RWKV-7 from {args.rwkv_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.rwkv_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
    ).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.rwkv_path, trust_remote_code=True, local_files_only=True
    )

    cfg = model.config
    print(f"\n{'='*60}")
    print(f"RWKV-7 config:")
    print(f"  num_hidden_layers : {cfg.num_hidden_layers}")
    print(f"  hidden_size       : {cfg.hidden_size}")
    print(f"  head_dim          : {getattr(cfg, 'head_dim', None)}")
    print(f"  num_heads (calc)  : {cfg.hidden_size // getattr(cfg, 'head_dim', 64)}")
    print(f"  vocab_size        : {cfg.vocab_size}")
    print(f"{'='*60}\n")

    # Build a dummy input
    x = tokenizer("Hello world.", return_tensors="pt").to(args.device)
    print(f"Forward pass with use_cache=True, seq_len={x['input_ids'].shape[1]}...")

    with torch.no_grad():
        out = model(
            input_ids=x["input_ids"],
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )

    print(f"\n{'='*60}")
    print(f"output keys: {list(out.keys()) if hasattr(out, 'keys') else out.__dict__}")
    print(f"{'='*60}\n")

    # past_key_values is what carries state
    pkv = out.past_key_values
    print(f"past_key_values: type={type(pkv).__name__}")
    print(describe(pkv, prefix="  "))
    print()

    # Handle HF Cache object (newer transformers): drill into .layers
    layers = getattr(pkv, "layers", None)
    if layers is not None:
        print(f"\n{'='*60}")
        print(f"Cache.layers : type={type(layers).__name__}, len={len(layers)}")
        print(f"{'='*60}\n")

        if len(layers) > 0:
            print(f"── Layer 0 inspection ──")
            l0 = layers[0]
            print(f"  type   : {type(l0).__name__}")
            print(f"  module : {type(l0).__module__}")
            print()

            # Probe FLA-specific methods
            print(f"  FLA-specific probes:")
            for method in ["is_initialized", "get_seq_length", "lazy_initialization"]:
                if hasattr(l0, method):
                    try:
                        m = getattr(l0, method)
                        r = m() if callable(m) else m
                        print(f"    .{method}{'()' if callable(m) else ''} → {r}")
                    except Exception as e:
                        print(f"    .{method} error: {e}")
            print()

            # .state and .keys() and .values() return what?
            for attr in ("state", "keys", "values"):
                if hasattr(l0, attr):
                    val = getattr(l0, attr)
                    if callable(val):
                        try:
                            r = val()
                            print(f"  .{attr}() → type={type(r).__name__}")
                            if isinstance(r, torch.Tensor):
                                print(f"      shape={tuple(r.shape)}, dtype={r.dtype}")
                            elif isinstance(r, dict):
                                print(f"      keys={list(r.keys())[:5]}")
                                for k, v in list(r.items())[:3]:
                                    if isinstance(v, torch.Tensor):
                                        print(f"      [{k}] shape={tuple(v.shape)}, dtype={v.dtype}")
                            elif isinstance(r, (list, tuple)):
                                print(f"      len={len(r)}")
                                for i, v in enumerate(r[:3]):
                                    if isinstance(v, torch.Tensor):
                                        print(f"      [{i}] shape={tuple(v.shape)}, dtype={v.dtype}")
                                    else:
                                        print(f"      [{i}] type={type(v).__name__}")
                        except Exception as e:
                            print(f"  .{attr}() error: {type(e).__name__}: {e}")
                    else:
                        # property access (not callable)
                        try:
                            r = val
                            print(f"  .{attr} → type={type(r).__name__}")
                            if isinstance(r, torch.Tensor):
                                print(f"      shape={tuple(r.shape)}, dtype={r.dtype}")
                            elif isinstance(r, (list, tuple)):
                                print(f"      len={len(r)}")
                                for i, v in enumerate(r[:3]):
                                    if isinstance(v, torch.Tensor):
                                        print(f"      [{i}] shape={tuple(v.shape)}, dtype={v.dtype}")
                                    else:
                                        print(f"      [{i}] type={type(v).__name__}")
                        except Exception as e:
                            print(f"  .{attr} error: {type(e).__name__}: {e}")
            print()

            # Dump every non-private attr's value type
            print(f"  All attrs (non-callable values shown):")
            for attr in sorted(dir(l0)):
                if attr.startswith("_"):
                    continue
                try:
                    val = getattr(l0, attr)
                except Exception:
                    continue
                if callable(val):
                    continue
                if isinstance(val, torch.Tensor):
                    print(f"    .{attr:25s}: Tensor shape={tuple(val.shape)}, dtype={val.dtype}")
                elif isinstance(val, (bool, int, float, str)):
                    print(f"    .{attr:25s}: {type(val).__name__} = {val}")
                elif val is None:
                    print(f"    .{attr:25s}: None")
                else:
                    print(f"    .{attr:25s}: {type(val).__name__}")
            print()

            # Drill into the dict at .state
            print(f"  ── .state dict contents ──")
            try:
                state_dict = l0.state
                if isinstance(state_dict, dict):
                    print(f"    dict keys: {list(state_dict.keys())}")
                    for k, v in state_dict.items():
                        if isinstance(v, torch.Tensor):
                            print(f"    [{k}] Tensor shape={tuple(v.shape)}, dtype={v.dtype}")
                        elif isinstance(v, (list, tuple)):
                            print(f"    [{k}] {type(v).__name__}[len={len(v)}]")
                            for i, sub in enumerate(v):
                                if isinstance(sub, torch.Tensor):
                                    print(f"        [{i}] shape={tuple(sub.shape)}, dtype={sub.dtype}")
                        else:
                            print(f"    [{k}] {type(v).__name__}: {v if not hasattr(v, 'shape') else 'shape='+str(v.shape)}")
                    if len(state_dict) == 0:
                        print(f"    (dict is empty — state may live elsewhere)")
            except Exception as e:
                print(f"    error: {type(e).__name__}: {e}")
            print()

            # Print fla.models.utils source
            try:
                import fla.models.utils as fmu
                import inspect as _inspect
                fla_src = fmu.__file__
                print(f"  FLALayer source file: {fla_src}")
                print()
                print(f"  ── FLALayer class source (next ~100 lines) ──")
                try:
                    src = _inspect.getsource(fmu.FLALayer)
                    # print at most first 100 lines
                    lines = src.split("\n")
                    for ln in lines[:100]:
                        print(f"    {ln}")
                    if len(lines) > 100:
                        print(f"    ... ({len(lines)-100} more lines)")
                except Exception as e:
                    print(f"    getsource error: {e}")
            except Exception:
                pass

    elif isinstance(pkv, (list, tuple)) and len(pkv) > 0:
        print(f"\n{'='*60}")
        print(f"FIRST LAYER STATE INSPECTION (list/tuple form)")
        print(f"{'='*60}")
        first = pkv[0]
        print(f"  type: {type(first).__name__}")
        if isinstance(first, (list, tuple)):
            for i, e in enumerate(first):
                print(f"  [{i}] {describe(e)}")
        elif isinstance(first, torch.Tensor):
            print(f"  Tensor shape={tuple(first.shape)}, dtype={first.dtype}")
        else:
            print(f"  obj: {first}")
            for attr in dir(first):
                if not attr.startswith("_"):
                    try:
                        val = getattr(first, attr)
                        if isinstance(val, torch.Tensor):
                            print(f"  .{attr} : {describe(val)}")
                    except Exception:
                        pass

    print(f"\n{'='*60}")
    print(f"INSTRUCTIONS FOR STATE-HIJACKING")
    print(f"{'='*60}")
    print(f"To inject custom initial state, build a structure matching the")
    print(f"shape of past_key_values printed above, then pass it as")
    print(f"past_key_values=<custom_state> when calling the model.")
    print(f"All weights, no past, no prefix tokens.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
