# pyright: reportAny=false, reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnusedCallResult=false, reportAttributeAccessIssue=false
"""Loop-1 diagnostic #8: mutual information between latents and topic labels.

PROXY LABELS: each of the 16 passages is treated as a distinct topic/category
(AI, biology, finance, history, ...). We sample the S2 latent trajectory multiple
times per passage (different seeds) and, for each chunk position h, estimate the
mutual information between the latent vector z_h and the topic label using
sklearn's mutual_info_classif. High MI => the latent at that chunk carries
topic-discriminative information (the planner is using the condition); low/flat MI
=> latents are topic-agnostic at that position.

Also computes MI for the CLEAN encoder latents as an upper reference.
Clearly a PROXY (topic identity, not task correctness), but it localizes which
chunks carry conditioning information and whether sampling degrades it vs clean.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.eval.diag_loop1_common as C  # noqa: E402

DEFAULT_CKPT = "outputs_relay/traj32x16-2.9B-singlez-bridge-v2-s2-birwkv-ddpm-condboundary/step_00026000"


def _ensure_sklearn():
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        print("sklearn missing; attempting install...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scikit-learn"], check=True)
            import sklearn  # noqa: F401
            return True
        except Exception as e:  # noqa: BLE001
            print(f"sklearn install failed: {e}")
            return False


@torch.no_grad()
def run(ckpt_dir, device, samples_per_topic, steps, cfg_scale, output):
    have_sklearn = _ensure_sklearn()
    model, tokenizer, dtype, pad_id = C.build_model(ckpt_dir, device)
    passages = C.PASSAGES
    H = int(model.trajectory_horizon)
    D = int(model.latent_dim)

    # collect: for each topic (passage), sample z multiple times
    sampled_rows = []   # [ (topic, z[H,D]) ]
    clean_rows = []
    for t, passage in enumerate(passages):
        pre, suf, _ln = C.split_prefix_suffix(tokenizer, passage, model, pad_id)
        zc = C.get_z_clean(model, suf, device)[0].float().cpu()
        clean_rows.append((t, zc))
        for s in range(samples_per_topic):
            torch.manual_seed(1000 * t + s)
            zs = C.get_z_sampled(model, pre, device, dtype, steps, cfg_scale)[0].float().cpu()
            sampled_rows.append((t, zs))
        print(f"[topic {t+1}/{len(passages)}] sampled {samples_per_topic}x")

    def mi_per_chunk(rows):
        if not have_sklearn:
            return None
        import numpy as np
        from sklearn.feature_selection import mutual_info_classif
        labels = np.asarray([r[0] for r in rows])
        n = len(labels)
        n_neighbors = max(1, min(3, n - 1))
        # need at least 2 samples total and >1 distinct label for a meaningful estimate
        if n < 2 or len(set(labels.tolist())) < 2:
            return None
        per_chunk = []
        for h in range(H):
            X = torch.stack([r[1][h] for r in rows], dim=0).numpy()  # [N, D]
            try:
                mi = mutual_info_classif(
                    X, labels, discrete_features=False,
                    n_neighbors=n_neighbors, random_state=0,
                )
                per_chunk.append(float(mi.mean()))  # mean MI over latent dims (nats)
            except ValueError:
                per_chunk.append(float("nan"))
        return per_chunk

    sampled_mi = mi_per_chunk(sampled_rows)
    clean_mi = mi_per_chunk(clean_rows)

    result = {
        "diagnostic": "loop1_mi(#8)",
        "ckpt": ckpt_dir,
        "note": "PROXY: labels are passage/topic identity, not task correctness.",
        "num_topics": len(passages),
        "samples_per_topic": samples_per_topic,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "sklearn_available": have_sklearn,
        "sampled_mi_per_chunk_nats": sampled_mi,
        "clean_mi_per_chunk_nats": clean_mi,
        "sampled_mi_mean": (statistics.mean(sampled_mi) if sampled_mi else None),
        "clean_mi_mean": (statistics.mean(clean_mi) if clean_mi else None),
        "interpretation": (
            "MI per chunk shows which latent positions carry topic-discriminative "
            "information. If sampled MI is much lower than clean MI, the S2 sampler "
            "loses conditioning information relative to the encoder. Flat-near-zero MI "
            "across chunks would indicate topic-agnostic (condition-ignoring) latents."
        ),
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    print("\n=== SUMMARY (MI #8, proxy topic labels) ===")
    if sampled_mi:
        print(f"sampled MI mean (nats): {result['sampled_mi_mean']:.4f}")
    if clean_mi:
        print(f"clean   MI mean (nats): {result['clean_mi_mean']:.4f}")
    if not have_sklearn:
        print("sklearn unavailable -> MI skipped (see JSON).")
    print(f"written: {output}")
    print(f"gpu peak MiB: {int(torch.cuda.max_memory_allocated()/1024/1024) if device.startswith('cuda') else 0}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--samples_per_topic", type=int, default=8)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cfg_scale", type=float, default=3.0)
    ap.add_argument("--output", default="/tmp/diag_loop1_mi.json")
    args = ap.parse_args()
    run(args.ckpt_dir, args.device, args.samples_per_topic, args.steps, args.cfg_scale, args.output)


if __name__ == "__main__":
    main()
