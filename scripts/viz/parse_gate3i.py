#!/usr/bin/env python3
"""Parse Gate 3i results and print paper-ready table rows."""
import json, math, os

LOG = "/tmp/gate3i_serve"
BAGS = ["073623", "061841", "063047"]
THRESHOLDS = [5, 10, 15, 20, 30]

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

print("=== Gate 3i: Serve-Count Hold Sweep ===\n")

print("--- BASELINE (I-052=off, MH=15, tau=0.92) ---")
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
        sc = d.get("serve_count_bypasses", 0)
        print(f"  bag={b}: skip={skip:.1f}%  bg={bg}  AA={aa}  MH={mh}  SC={sc}  ar={ar:.1f}%")
    else:
        print(f"  bag={b}: NOT FOUND")

print()
print(f"{'Thresh':>8} {'Skip%':>8} {'V_073623':>10} {'V_061841':>10} {'V_063047':>10} {'V_max':>8} {'Status':>8}")
print("-" * 75)

for thr in THRESHOLDS:
    skip_vals = []
    row_data = {}

    for b in BAGS:
        tag = f"{b}_SC{thr}"
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
        print(f"{'SC='+str(thr):>8}   N/A (not run yet)")
        continue

    v073 = row_data.get("073623")
    v061 = row_data.get("061841")
    v063 = row_data.get("063047")
    v_max = max(v for v in row_data.values())
    skip_mean = sum(skip_vals) / len(skip_vals)
    status = "PASS" if v_max <= 0.10 else "FAIL"

    def fmt(v): return f"{v:.4f}" if v is not None else "  N/A"
    print(f"{'SC='+str(thr):>8} {skip_mean:>7.1f}%  {fmt(v073):>10} {fmt(v061):>10} {fmt(v063):>10} {fmt(v_max):>8}  {status}")

print()
print("--- LaTeX table rows ---")
for thr in THRESHOLDS:
    skip_vals = []
    row_data = {}

    for b in BAGS:
        tag = f"{b}_SC{thr}"
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
        print(f"{thr} & \\emph{{pending}} & \\emph{{---}} & \\emph{{---}} & \\emph{{---}} & \\emph{{---}} & \\emph{{---}} \\\\")
        continue

    v073 = row_data.get("073623", 0)
    v061 = row_data.get("061841", 0)
    v063 = row_data.get("063047", 0)
    v_max = max(row_data.values())
    skip_mean = sum(skip_vals) / len(skip_vals)
    chk = "\\checkmark" if v_max <= 0.10 else "\\times"
    print(f"{thr} & {skip_mean:.1f}\\% & {v073:.4f} & {v061:.4f} & {v063:.4f} & {v_max:.4f} & {chk} \\\\")

print()
print("--- Detailed bypass counts ---")
for thr in THRESHOLDS:
    for b in BAGS:
        path = f"{LOG}/{b}_SC{thr}/metrics.json"
        if not os.path.exists(path):
            continue
        d = load(path)
        bg  = d.get("background_s2_runs", 0)
        aa  = d.get("action_aware_bypasses", 0)
        mh  = d.get("max_hold_bypasses", 0)
        sc  = d.get("serve_count_bypasses", 0)
        skip = d.get("temporal_cache_skip_ratio", 0)
        ft  = d.get("fresh_traj_outputs", 0)
        fa  = d.get("fresh_action_outputs", 0)
        ar  = fa / max(1, ft+fa) * 100
        nat = bg - aa - mh - sc
        print(f"  SC={thr:>2}  bag={b}  bg={bg:>4}  skip={skip:.1f}%  nat={nat:>4}  AA={aa:>4}  MH={mh:>4}  SC={sc:>4}  ar={ar:.1f}%")
