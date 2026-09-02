# pyright: reportAny=false, reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnusedCallResult=false, reportAttributeAccessIssue=false
"""Loop-1 diagnostic #2 (PRIORITY): reachability of planned states.

Builds a BANK of REAL RWKV-7 recurrent states by running many real text prefixes
through the FROZEN backbone (no injection) and recording
out.past_key_values.layers[l].state["recurrent_state"] per layer.

Then, for the target checkpoint, projects S1(z_clean) and S1(z_sampled) to per-layer
states and measures each state's nearest-neighbor L2 distance to the real-state bank.

CENTRAL RESULT: the RATIO nearest(sampled)/nearest(clean). If >> 1, the S2-sampled
latent produces states that are further from any reachable real RWKV state, i.e. the
clean->sampled gap is an OFF-MANIFOLD state problem (not just a logit-behavior one).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.eval.diag_loop1_common as C  # noqa: E402

DEFAULT_CKPT = "outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-ddpm-condboundary/step_00026000"


@torch.no_grad()
def build_real_state_bank(model, tokenizer, pad_id, device, bank_size, chunk_len):
    """Run `bank_size` real prefixes through the frozen RWKV; store final-position
    recurrent_state per layer. Returns list per layer of [bank_size, feat] on CPU (fp16).
    We chunk each passage into `chunk_len`-token windows to get many distinct real
    states from the 16 base passages (they are long enough to yield several windows)."""
    windows: list[list[int]] = []
    for passage in C.PASSAGES:
        ids = C._tokenize_without_specials(tokenizer, passage)
        if not ids:
            continue
        # slide over the passage to produce multiple real prefixes
        step = max(1, chunk_len // 2)
        for start in range(0, max(1, len(ids) - 8), step):
            w = ids[start:start + chunk_len]
            if len(w) >= 8:
                windows.append(w)
            if len(windows) >= bank_size:
                break
        if len(windows) >= bank_size:
            break
    # if not enough windows, repeat passages with offsets
    idx = 0
    while len(windows) < bank_size and C.PASSAGES:
        base = C._tokenize_without_specials(tokenizer, C.PASSAGES[idx % len(C.PASSAGES)])
        off = (idx * 7) % max(1, len(base))
        w = (base[off:] + base[:off])[:chunk_len]
        if len(w) >= 8:
            windows.append(w)
        idx += 1
    windows = windows[:bank_size]

    bank_per_layer = None
    for w in windows:
        ii = torch.tensor([w], device=device, dtype=torch.long)
        am = torch.ones_like(ii)
        out = model.rwkv_model(input_ids=ii, attention_mask=am.bool(),
                               use_cache=True, return_dict=True)
        cache = out.past_key_values
        num_layers = len(cache.layers)
        if bank_per_layer is None:
            bank_per_layer = [[] for _ in range(num_layers)]
        for l in range(num_layers):
            st = cache.layers[l].state.get("recurrent_state") if cache.layers[l].state is not None else None
            if isinstance(st, torch.Tensor):
                bank_per_layer[l].append(st[0].float().reshape(-1).half().cpu())
            else:
                bank_per_layer[l].append(None)
    # stack valid entries per layer
    stacked = []
    for l in range(len(bank_per_layer)):
        entries = [e for e in bank_per_layer[l] if e is not None]
        stacked.append(torch.stack(entries, dim=0) if entries else None)  # [N, feat] fp16 cpu
    return stacked, len(windows)


@torch.no_grad()
def nearest_dist_per_layer(states_flat_gpu, bank_per_layer, device, batch=64):
    """For each layer, min L2 distance from each of the H planned states to the bank.
    Returns mean over (chunks, layers) of the per-state nearest distance."""
    all_min = []
    per_layer_min = []
    for l, planned in enumerate(states_flat_gpu):  # planned: [H, feat] fp32 gpu
        bank = bank_per_layer[l]
        if bank is None:
            continue
        bank_g = bank.to(device).float()  # [N, feat]
        # distances [H, N] computed in batches over N to bound memory
        H = planned.shape[0]
        mins = torch.full((H,), float("inf"), device=device)
        for s in range(0, bank_g.shape[0], batch):
            chunk = bank_g[s:s + batch]  # [b, feat]
            d = torch.cdist(planned, chunk)  # [H, b]
            mins = torch.minimum(mins, d.min(dim=1).values)
        layer_mean = float(mins.mean().item())
        per_layer_min.append(layer_mean)
        all_min.append(layer_mean)
        del bank_g
    return (sum(all_min) / len(all_min) if all_min else float("nan")), per_layer_min


@torch.no_grad()
def run(ckpt_dir, device, num_samples, bank_size, steps, cfg_scale, chunk_len, output):
    model, tokenizer, dtype, pad_id = C.build_model(ckpt_dir, device)
    print(f"building real-state bank (target {bank_size} prefixes)...")
    bank_per_layer, n_bank = build_real_state_bank(model, tokenizer, pad_id, device, bank_size, chunk_len)
    print(f"bank built with {n_bank} real prefixes, {len(bank_per_layer)} layers")

    clean_nn, sampled_nn, ratios = [], [], []
    clean_pl_acc, sampled_pl_acc = None, None
    passages = C.PASSAGES[:num_samples]

    for i, passage in enumerate(passages):
        pre, suf, _ln = C.split_prefix_suffix(tokenizer, passage, model, pad_id)
        z_clean = C.get_z_clean(model, suf, device)
        z_sampled = C.get_z_sampled(model, pre, device, dtype, steps, cfg_scale)
        sc = C.per_layer_states_flat(model, z_clean)      # list [H,feat] gpu
        ss = C.per_layer_states_flat(model, z_sampled)
        c_mean, c_pl = nearest_dist_per_layer(sc, bank_per_layer, device)
        s_mean, s_pl = nearest_dist_per_layer(ss, bank_per_layer, device)
        clean_nn.append(c_mean)
        sampled_nn.append(s_mean)
        ratios.append(s_mean / c_mean if c_mean > 0 else float("nan"))
        if clean_pl_acc is None:
            clean_pl_acc = [[] for _ in c_pl]
            sampled_pl_acc = [[] for _ in s_pl]
        for li in range(len(c_pl)):
            clean_pl_acc[li].append(c_pl[li])
            sampled_pl_acc[li].append(s_pl[li])
        print(f"[{i+1}/{len(passages)}] nn_clean={c_mean:.3f} nn_sampled={s_mean:.3f} ratio={ratios[-1]:.3f}")

    def ms(x):
        xf = [v for v in x if v == v]  # drop nan
        return {"mean": statistics.mean(xf) if xf else float("nan"),
                "std": statistics.pstdev(xf) if len(xf) > 1 else 0.0}

    result = {
        "diagnostic": "loop1_reachability(#2)",
        "ckpt": ckpt_dir,
        "num_samples": len(passages),
        "bank_size": n_bank,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "nearest_real_state_dist_clean": ms(clean_nn),
        "nearest_real_state_dist_sampled": ms(sampled_nn),
        "sampled_over_clean_ratio": ms(ratios),
        "nn_clean_per_layer_mean": [statistics.mean(v) for v in clean_pl_acc] if clean_pl_acc else [],
        "nn_sampled_per_layer_mean": [statistics.mean(v) for v in sampled_pl_acc] if sampled_pl_acc else [],
        "interpretation": (
            "ratio >> 1 => S2-sampled latent produces states further from any reachable "
            "real RWKV state than the clean latent does => the clean->sampled downstream "
            "gap is (at least partly) an OFF-MANIFOLD state problem. ratio ~ 1 => sampled "
            "states are as reachable as clean; the gap is then a behavior/logit issue, not "
            "geometry."
        ),
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    print("\n=== SUMMARY (reachability #2) ===")
    print(f"nearest real-state dist  CLEAN  : {result['nearest_real_state_dist_clean']['mean']:.4f}")
    print(f"nearest real-state dist  SAMPLED: {result['nearest_real_state_dist_sampled']['mean']:.4f}")
    print(f"RATIO sampled/clean            : {result['sampled_over_clean_ratio']['mean']:.4f}")
    print(f"written: {output}")
    print(f"gpu peak MiB: {int(torch.cuda.max_memory_allocated()/1024/1024) if device.startswith('cuda') else 0}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_samples", type=int, default=16)
    ap.add_argument("--bank_size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cfg_scale", type=float, default=3.0)
    ap.add_argument("--chunk_len", type=int, default=64)
    ap.add_argument("--output", default="/tmp/diag_loop1_reachability.json")
    args = ap.parse_args()
    run(args.ckpt_dir, args.device, args.num_samples, args.bank_size,
        args.steps, args.cfg_scale, args.chunk_len, args.output)


if __name__ == "__main__":
    main()
