"""Summarise run_e2e.py results: one row per (tag, mode), one column per episode + SR / mean distance.

Usage: python3 scripts/sim/verify/summarize_e2e.py scripts/sim/verify/out/e2e [--match clear]
"""
import json
import sys
from pathlib import Path


def main(root, match=""):
    rows = []
    for summ in sorted(Path(root).glob("*/*/summary.json")):
        tag, mode = summ.parts[-3], summ.parts[-2]
        if match and match not in tag:
            continue
        eps = json.loads(summ.read_text())
        rows.append((tag, mode, eps))
    if not rows:
        print("no results")
        return
    ids = sorted({e["id"] for _, _, eps in rows for e in eps})
    print("| tag | mode | " + " | ".join(ids) + " | SR | mean dist [m] |")
    print("|" + "---|" * (len(ids) + 4))
    for tag, mode, eps in rows:
        by = {e["id"]: e for e in eps}
        cells = []
        for i in ids:
            e = by.get(i)
            if e is None:
                cells.append("—")
                continue
            mark = "✅" if e.get("success") else ("⏱" if not e["stopped_by_model"] else "✗")
            cells.append(f"{mark} {e.get('final_dist_to_goal_m', float('nan')):.1f}")
        sr = sum(bool(e.get("success")) for e in eps) / len(eps)
        md = sum(e.get("final_dist_to_goal_m", 0) for e in eps) / len(eps)
        print(f"| {tag} | {mode} | " + " | ".join(cells) + f" | {sr:.2f} | {md:.1f} |")
    print("\n✅ success (stopped within radius)  ✗ stopped elsewhere  ⏱ timeout; number = final distance to goal")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="scripts/sim/verify/out/e2e")
    ap.add_argument("--match", default="")
    a = ap.parse_args()
    main(a.root, a.match)
