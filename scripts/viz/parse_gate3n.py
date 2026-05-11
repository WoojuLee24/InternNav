#!/usr/bin/env python3
"""Parse Gate 3n results and print paper-ready table rows."""
import json, math, os

LOG = "/tmp/gate3n_prod"
BAGS = ["073623", "061841", "063047"]

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

print("=== Gate 3n: Full Production Stack Validation ===\n")

print("--- Condition A: No-Cache Baseline ---")
for b in BAGS:
    path = f"{LOG}/{b}_NOCACHE/metrics.json"
    if os.path.exists(path):
        d = load(path)
        ft = d.get("fresh_traj_outputs", 0)
        fa = d.get("fresh_action_outputs", 0)
        skip = d.get("temporal_cache_skip_ratio", 0)
        ar = fa / max(1, ft+fa) * 100
        bg = d.get("background_s2_runs", 0)
        print(f"  bag={b}: bg={bg}  skip={skip:.1f}%  ar={ar:.1f}%")
    else:
        print(f"  bag={b}: NOT FOUND")

print()
print("--- Condition P: Full Production Stack ---")
for b in BAGS:
    path = f"{LOG}/{b}_PROD/metrics.json"
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
        nat = bg - aa - mh - sp
        print(f"  bag={b}: bg={bg}  skip={skip:.1f}%  nat={nat}  AA={aa}  MH={mh}  SP={sp}  ar={ar:.1f}%")
    else:
        print(f"  bag={b}: NOT FOUND")

print()
print(f"{'Bag':>10} {'Skip%':>8} {'V (P vs A)':>12} {'Status':>8}")
print("-" * 45)
vs = []
for b in BAGS:
    p_path = f"{LOG}/{b}_PROD/metrics.json"
    a_path = f"{LOG}/{b}_NOCACHE/metrics.json"
    if not (os.path.exists(p_path) and os.path.exists(a_path)):
        print(f"  {b:>10}   N/A")
        continue
    d = load(p_path)
    v = cramers_v(d, load(a_path))
    skip = d.get("temporal_cache_skip_ratio", 0)
    if v is not None:
        vs.append(v)
        status = "PASS" if v <= 0.10 else "FAIL"
        print(f"  {b:>10} {skip:>7.1f}%  {v:>10.4f}  {status}")

if vs:
    v_max = max(vs)
    overall = "PASS" if v_max <= 0.10 else "FAIL"
    print(f"\n  V_max = {v_max:.4f}  →  GATE 3n: {overall}")

print()
print("--- LaTeX rows (bag & skip% & AA & MH & SP & V & status) ---")
for b in BAGS:
    p_path = f"{LOG}/{b}_PROD/metrics.json"
    a_path = f"{LOG}/{b}_NOCACHE/metrics.json"
    if not (os.path.exists(p_path) and os.path.exists(a_path)):
        print(f"{b} & \\emph{{pending}} & --- & --- & --- & --- & --- \\\\")
        continue
    d = load(p_path)
    v = cramers_v(d, load(a_path))
    skip = d.get("temporal_cache_skip_ratio", 0)
    aa = d.get("action_aware_bypasses", 0)
    mh = d.get("max_hold_bypasses", 0)
    sp = d.get("slope_predict_bypasses", 0)
    if v is not None:
        chk = "\\checkmark" if v <= 0.10 else "\\times"
        print(f"{b} & {skip:.1f}\\% & {aa} & {mh} & {sp} & {v:.4f} & {chk} \\\\")
