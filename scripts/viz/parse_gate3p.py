#!/usr/bin/env python3
"""Parse Gate 3p odometry-progress hold sweep results and print LaTeX table rows."""
import json, os, math

LOG = "/tmp/gate3p_odom"
NOCACHE_LOG = "/tmp/gate3n_prod"
BAGS = ["073623", "061841", "063047"]
THRESHOLDS = ["0.3", "0.5", "0.7", "1.0", "1.5"]


def load(path):
    with open(path) as f:
        return json.load(f)


def cramers_v(d, da):
    ft_a = da.get("fresh_traj_outputs", 0)
    fa_a = da.get("fresh_action_outputs", 0)
    ft_c = d.get("fresh_traj_outputs", 0)
    fa_c = d.get("fresh_action_outputs", 0)
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


print(f"\n{'θ_d (m)':>10} {'Skip%':>7} {'V_073623':>10} {'V_061841':>10} {'V_063047':>10} {'V_max':>8} {'OP_mean':>9} {'Status':>8}")
print("-" * 95)

for thr in THRESHOLDS:
    tag_thr = thr.replace(".", "_")
    skip_vals = []
    v_row = {}
    op_vals = []
    for b in BAGS:
        p_path = f"{LOG}/{b}_OD{tag_thr}/metrics.json"
        a_path = f"{NOCACHE_LOG}/{b}_NOCACHE/metrics.json"
        if not (os.path.exists(p_path) and os.path.exists(a_path)):
            continue
        d = load(p_path)
        da = load(a_path)
        v = cramers_v(d, da)
        skip = d.get("temporal_cache_skip_ratio", 0)
        op = d.get("odom_progress_bypasses", 0)
        if v is not None:
            skip_vals.append(skip)
            v_row[b] = v
            op_vals.append(op)
    if not skip_vals:
        print(f"  {thr:>8}m  NOT AVAILABLE")
        continue
    v_max = max(v_row.values())
    skip_mean = sum(skip_vals) / len(skip_vals)
    op_mean = sum(op_vals) / len(op_vals)
    status = "PASS" if v_max <= 0.10 else "FAIL"

    def fv(b):
        return f"{v_row[b]:.4f}" if b in v_row else "  N/A"

    print(
        f"  {thr:>8}m {skip_mean:>7.1f}%  {fv('073623'):>10} {fv('061841'):>10} {fv('063047'):>10} {v_max:>8.4f} {op_mean:>9.1f}  {status}"
    )

print()
print("--- LaTeX rows (θ_d & Skip% & V_073623 & V_061841 & V_063047 & V_max & OP & Status) ---")
best_thr = None
best_v = float("inf")
for thr in THRESHOLDS:
    tag_thr = thr.replace(".", "_")
    skip_vals = []
    v_row = {}
    op_vals = []
    for b in BAGS:
        p_path = f"{LOG}/{b}_OD{tag_thr}/metrics.json"
        a_path = f"{NOCACHE_LOG}/{b}_NOCACHE/metrics.json"
        if not (os.path.exists(p_path) and os.path.exists(a_path)):
            continue
        d = load(p_path)
        da = load(a_path)
        v = cramers_v(d, da)
        skip = d.get("temporal_cache_skip_ratio", 0)
        op = d.get("odom_progress_bypasses", 0)
        if v is not None:
            skip_vals.append(skip)
            v_row[b] = v
            op_vals.append(op)
    if not skip_vals:
        continue
    v_max = max(v_row.values())
    if v_max <= 0.10 and v_max < best_v:
        best_v = v_max
        best_thr = thr
    skip_mean = sum(skip_vals) / len(skip_vals)
    op_mean = sum(op_vals) / len(op_vals)
    chk = "\\checkmark" if v_max <= 0.10 else "\\times"
    v073 = v_row.get("073623", 0)
    v061 = v_row.get("061841", 0)
    v063 = v_row.get("063047", 0)
    bold = "\\mathbf{" if thr == best_thr else ""
    endbold = "}" if thr == best_thr else ""
    print(
        f"{bold}{thr}{endbold} & {skip_mean:.1f}\\% & {v073:.4f} & {v061:.4f} & {v063:.4f} & {v_max:.4f} & {op_mean:.0f} & {chk} \\\\"
    )

if best_thr:
    print(f"\nOptimal θ_d = {best_thr}m (V_max={best_v:.4f})")
