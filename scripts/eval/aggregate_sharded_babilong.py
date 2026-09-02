#!/usr/bin/env python3
"""Aggregate sharded-babilong sweep JSONs into accuracy/payload/latency tables."""
import json
import sys
from pathlib import Path

def main(res_dir="outputs_eval/sharded_babilong"):
    rows = []
    for f in sorted(Path(res_dir).glob("M*_L*_qa*.json")):
        d = json.load(open(f))
        for task, blob in d.items():
            s = blob["summary"]
            rows.append((f.name, task, s))
    if not rows:
        print("no results yet")
        return
    conds = ["first_shard", "oracle_shard", "text_full", "text_trunc",
             "carryover", "state_inject", "dual"]
    for task in sorted(set(t for _, t, _ in rows)):
        print(f"\n=== {task} ===")
        hdr = f"{'config':22s}" + "".join(f"{c[:11]:>12s}" for c in conds)
        print(hdr + f"{'carry-ms':>9s}{'payload':>10s}")
        for fname, t, s in rows:
            if t != task:
                continue
            label = f"M{s['num_agents']}_L{s['local_tokens']}_{s['babilong_length']}"
            acc = s["accuracy"]
            line = f"{label:22s}" + "".join(f"{acc.get(c, 0)*100:>11.1f}%" for c in conds)
            cms = s["mean_ms"].get("carryover", {})
            tot = cms.get("prefill_ms", 0) + cms.get("decode_ms", 0)
            pay = s["mean_payload_bytes"].get("carryover", 0)
            pay_s = f"{pay/1e6:.1f}MB" if pay > 1e6 else f"{pay/1e3:.0f}KB"
            line += f"{tot:>8.0f}{pay_s:>10s}"
            print(line)
        tf = next((s["mean_payload_bytes"].get("text_full", 0) for _, t, s in rows if t == task), 0)
        cp = next((s["mean_payload_bytes"].get("carryover", 0) for _, t, s in rows if t == task), 0)
        if tf and cp:
            print(f"  -> carryover payload = {cp/tf*100:.1f}% of text_full")

if __name__ == "__main__":
    main(*sys.argv[1:])
