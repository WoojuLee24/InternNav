#!/usr/bin/env python3
"""Parse Gate 3l results and print paper-ready table rows."""
import json, math, os

LOG = "/tmp/gate3l_slope"
BAGS = ["073623", "061841", "063047"]
SLOPES = [0.005, 0.010, 0.015, 0.020, 0.030]

def load(path):
    with open(path) as f:
        return json.load(f)

def cramers_v(d, da):
    ft_a = da.get("fresh_traj_outputs", 0)
    fa_a = da.get("fresh_action_outputs", 0)
    ft_c = d.get("fresh_traj_outputs", 0)
    fa_c = d.get("fresh_action_outputs", 0)
    n_a = ft_a + fa_a; n_c = ft_c + fa_c
    if n_a == 0 or n_c == 0:
        return None
    n = n_a + n_c
    obs = [[ft_a, fa_a], [ft_c, fa_c]]
    rs = [n_a, n_c]; cs = [ft_a + ft_c, fa_a + fa_c]
    chi2 = sum(
        (obs[i][j] - rs[i] * cs[j] / n) ** 2 / max(rs[i] * cs[j] / n, 1e-9)
        for i in range(2) for j in range(2)
    )
    return math.sqrt(chi2 / n)

print("=== Gate 3l: Slope Predictive Refresh Sweep ===\n")

print("--- BASELINE (I-054=off, MH=15, τ=0.92) ---")
for b in BAGS:
    path = f"{LOG}/{b}_BASELINE/metrics.json"
    if os.path.exists(path):
        d = load(path)
        ft = d.get("fresh_traj_outputs", 0)
        fa = d.get("fresh_action_outputs", 0)
        skip = d.get("temporal_cache_skip_ratio", 0)
        ar = fa / max(1, ft+fa) * 100
        bg = d.get("background_s2_runs", 0)
        aa = d.get("action_aware_bypasses", 0)
        mh = d.get("max_hold_bypasses", 0)
        sp = d.get("slope_predict_bypasses", 0)
        print(f"  bag={b}: skip={skip:.1f}%  bg={bg}  AA={aa}  MH={mh}  SP={sp}  ar={ar:.1f}%")
    else:
        print(f"  bag={b}: NOT FOUND")

print()
print(f"{'Slope':>8} {'Skip%':>8} {'V_073623':>10} {'V_061841':>10} {'V_063047':>10} {'V_max':>8} {'Status':>8}")
print("-" * 75)

for sl in SLOPES:
    sl_tag = str(sl).replace(".", "_")
    skip_vals = []
    row_data = {}

    for b in BAGS:
        tag = f"{b}_SL{sl_tag}"
        path = f"{LOG}/{tag}/metrics.json"
        base_path = f"{LOG}/{b}_BASELINE/metrics.json"

        if not (os.path.exists(path) and os.path.exists(base_path)):
            continue

        d = load(path)
        da = load(base_path)
        v = cramers_v(d, da)
        skip = d.get("temporal_cache_skip_ratio", 0)

        if v is not None:
            skip_vals.append(skip)
            row_data[b] = v

    if not skip_vals:
        print(f"{sl:>8.3f}   N/A (not run)")
        continue

    v073 = row_data.get("073623")
    v061 = row_data.get("061841")
    v063 = row_data.get("063047")
    v_max = max(v for v in [v073, v061, v063] if v is not None)
    skip_mean = sum(skip_vals) / len(skip_vals)
    status = "PASS" if v_max <= 0.10 else "FAIL"

    v073s = f"{v073:.4f}" if v073 is not None else "N/A"
    v061s = f"{v061:.4f}" if v061 is not None else "N/A"
    v063s = f"{v063:.4f}" if v063 is not None else "N/A"
    print(f"{sl:>8.3f} {skip_mean:>7.1f}% {v073s:>10} {v061s:>10} {v063s:>10} {v_max:>8.4f} {status:>8}")

print()
print("=== LaTeX rows ===")
for sl in SLOPES:
    sl_tag = str(sl).replace(".", "_")
    skip_vals = []
    row_data = {}

    for b in BAGS:
        tag = f"{b}_SL{sl_tag}"
        path = f"{LOG}/{tag}/metrics.json"
        base_path = f"{LOG}/{b}_BASELINE/metrics.json"
        if not (os.path.exists(path) and os.path.exists(base_path)):
            continue
        d = load(path)
        da = load(base_path)
        v = cramers_v(d, da)
        skip = d.get("temporal_cache_skip_ratio", 0)
        if v is not None:
            skip_vals.append(skip)
            row_data[b] = v

    if not skip_vals:
        print(f"{sl} & \\emph{{pending}} & \\emph{{---}} & \\emph{{---}} & \\emph{{---}} & \\emph{{---}} & \\emph{{---}} \\\\")
        continue

    v073 = row_data.get("073623")
    v061 = row_data.get("061841")
    v063 = row_data.get("063047")
    v_max = max(v for v in [v073, v061, v063] if v is not None)
    skip_mean = sum(skip_vals) / len(skip_vals)
    status = "\\checkmark" if v_max <= 0.10 else "\\ding{55}"

    v073s = f"{v073:.4f}" if v073 is not None else "---"
    v061s = f"{v061:.4f}" if v061 is not None else "---"
    v063s = f"{v063:.4f}" if v063 is not None else "---"
    print(f"{sl} & {skip_mean:.1f}\\% & {v073s} & {v061s} & {v063s} & {v_max:.4f} & {status} \\\\")
