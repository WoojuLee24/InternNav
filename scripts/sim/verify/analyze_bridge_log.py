"""Summarise a vln_bridge_node CSV log (log_dir:=...):  V5 pixel-space hypothesis, V6 projection method mix,
V8 tf-at-stamp success, server latency.

Usage: python3 scripts/sim/verify/analyze_bridge_log.py scripts/sim/verify/out/<run>/vln_*.csv
"""
import csv
import json
import sys

import numpy as np


def main(path):
    rows = list(csv.DictReader(open(path)))
    f = lambda k: np.array([float(r[k]) if r[k] not in ("", "None", "nan") else np.nan for r in rows])  # noqa: E731
    s2 = [r for r in rows if r["pixel_r"] not in ("", "None")]
    res = {"n_replies": len(rows), "n_s2_pixel": len(s2)}

    d = {k: f(k) for k in ("dpx_384", "dpx_392", "dpx_proc")}
    valid = ~np.isnan(d["dpx_384"])
    res["V5_n"] = int(valid.sum())
    if valid.any():
        med = {k: float(np.median(v[valid])) for k, v in d.items()}
        res["V5_dpx_median"] = med
        best = min(med, key=med.get)
        others = [v for k, v in med.items() if k != best]
        res["V5_best"] = best
        res["V5_pass(best<=0.5*others)"] = bool(all(med[best] <= 0.5 * o for o in others))

    methods = [r["proj_method"] for r in s2]
    res["V6_methods"] = {m: methods.count(m) for m in set(methods)}
    tf_ok = f("tf_ok")
    res["V8_tf_at_stamp_success"] = float(np.nanmean(tf_ok)) if len(tf_ok) else None
    t = f("t_infer")
    is_s2 = np.array([bool(r["llm"]) for r in rows])
    if is_s2.any():
        res["latency_s2_p50_ms"] = float(np.median(t[is_s2]) * 1000)
    if (~is_s2).any():
        res["latency_s1_p50_ms"] = float(np.median(t[~is_s2]) * 1000)
    acts = [r["action"] for r in rows if r["kind"] == "action"]
    res["discrete_actions"] = {a: acts.count(a) for a in set(acts)}
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
