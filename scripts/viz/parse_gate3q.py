#!/usr/bin/env python3
"""Parse Gate 3q single-run variance analysis results."""
import json, os, math, itertools

LOG = "/tmp/gate3q_variance"
SHORT = "073623"
N_REPS = 3


def load(path):
    with open(path) as f:
        return json.load(f)


def cramers_v(d1, d2):
    ft_a = d1.get("fresh_traj_outputs", 0)
    fa_a = d1.get("fresh_action_outputs", 0)
    ft_c = d2.get("fresh_traj_outputs", 0)
    fa_c = d2.get("fresh_action_outputs", 0)
    n_a = ft_a + fa_a
    n_c = ft_c + fa_c
    if n_a == 0 or n_c == 0:
        return None
    n = n_a + n_c
    obs = [[ft_a, fa_a], [ft_c, fa_c]]
    rs = [n_a, n_c]
    cs = [ft_a + ft_c, fa_a + fa_c]
    chi2 = sum(
        (obs[i][j] - rs[i] * cs[j] / n) ** 2 / max(rs[i] * cs[j] / n, 1e-9)
        for i in range(2)
        for j in range(2)
    )
    return math.sqrt(chi2 / n)


nocache_runs = []
prod_runs = []
for rep in range(1, N_REPS + 1):
    nc_path = f"{LOG}/{SHORT}_NOCACHE_R{rep}/metrics.json"
    pr_path = f"{LOG}/{SHORT}_PROD_R{rep}/metrics.json"
    if os.path.exists(nc_path):
        nocache_runs.append(load(nc_path))
    if os.path.exists(pr_path):
        prod_runs.append(load(pr_path))

print(f"\nLoaded: {len(nocache_runs)} NOCACHE runs, {len(prod_runs)} PROD runs")

print("\n--- Action rates per run ---")
for i, d in enumerate(nocache_runs, 1):
    ft = d.get("fresh_traj_outputs", 0)
    fa = d.get("fresh_action_outputs", 0)
    n = ft + fa
    ar = fa / max(1, n) * 100
    print(f"  NOCACHE R{i}: bg={d.get('background_s2_runs',0)}  action_rate={ar:.1f}%  (ft={ft} fa={fa})")
for i, d in enumerate(prod_runs, 1):
    ft = d.get("fresh_traj_outputs", 0)
    fa = d.get("fresh_action_outputs", 0)
    n = ft + fa
    ar = fa / max(1, n) * 100
    skip = d.get("temporal_cache_skip_ratio", 0)
    print(f"  PROD    R{i}: bg={d.get('background_s2_runs',0)}  skip={skip:.1f}%  action_rate={ar:.1f}%  (ft={ft} fa={fa})")

print("\n--- V_intra_nocache (pairs of independent no-cache runs) ---")
v_intra_nc = []
for i, j in itertools.combinations(range(len(nocache_runs)), 2):
    v = cramers_v(nocache_runs[i], nocache_runs[j])
    if v is not None:
        v_intra_nc.append(v)
        print(f"  NOCACHE R{i+1} vs R{j+1}: V={v:.4f}")
if v_intra_nc:
    print(f"  Mean V_intra_nocache = {sum(v_intra_nc)/len(v_intra_nc):.4f}")
    print(f"  Max  V_intra_nocache = {max(v_intra_nc):.4f}")

print("\n--- V_intra_prod (pairs of independent PROD runs) ---")
v_intra_pr = []
for i, j in itertools.combinations(range(len(prod_runs)), 2):
    v = cramers_v(prod_runs[i], prod_runs[j])
    if v is not None:
        v_intra_pr.append(v)
        print(f"  PROD R{i+1} vs R{j+1}: V={v:.4f}")
if v_intra_pr:
    print(f"  Mean V_intra_prod = {sum(v_intra_pr)/len(v_intra_pr):.4f}")
    print(f"  Max  V_intra_prod = {max(v_intra_pr):.4f}")

print("\n--- V_cross (PROD vs NOCACHE, each pair) ---")
v_cross = []
for i, dp in enumerate(prod_runs, 1):
    for j, dn in enumerate(nocache_runs, 1):
        v = cramers_v(dp, dn)
        if v is not None:
            v_cross.append(v)
            print(f"  PROD R{i} vs NOCACHE R{j}: V={v:.4f}")
if v_cross:
    print(f"  Mean V_cross = {sum(v_cross)/len(v_cross):.4f}")
    print(f"  Max  V_cross = {max(v_cross):.4f}")

print("\n--- Summary ---")
nc_max = max(v_intra_nc) if v_intra_nc else float("nan")
pr_max = max(v_intra_pr) if v_intra_pr else float("nan")
cr_max = max(v_cross) if v_cross else float("nan")
cr_mean = sum(v_cross) / len(v_cross) if v_cross else float("nan")
nc_mean = sum(v_intra_nc) / len(v_intra_nc) if v_intra_nc else float("nan")
pr_mean = sum(v_intra_pr) / len(v_intra_pr) if v_intra_pr else float("nan")
status = "PASS" if v_intra_nc and nc_max < 0.05 else "REVIEW"
print(f"  V_intra_nocache max = {nc_max:.4f}  {'< 0.05 PASS' if nc_max < 0.05 else '>= 0.05 REVIEW'}")
print(f"  V_intra_prod max    = {pr_max:.4f}")
print(f"  V_cross max         = {cr_max:.4f}")
print(f"  Status: {status}")
if v_intra_nc and v_cross:
    snr = cr_max / max(nc_max, 1e-6)
    print(f"  Signal-to-noise (V_cross_max / V_intra_nc_max) = {snr:.2f}x")

print("\n--- LaTeX variance table ---")
print(r"\begin{tabular}{lcc}")
print(r"\toprule")
print("Comparison & Mean $V$ & Max $V$ \\\\")
print(r"\midrule")
if v_intra_nc:
    print(f"Intra-baseline (NOCACHE vs NOCACHE) & {nc_mean:.4f} & {nc_max:.4f} \\\\")
if v_intra_pr:
    print(f"Intra-PROD (PROD vs PROD) & {pr_mean:.4f} & {pr_max:.4f} \\\\")
if v_cross:
    print(f"Cross-condition (PROD vs NOCACHE) & {cr_mean:.4f} & {cr_max:.4f} \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
