#!/usr/bin/env python3
"""Parse Gate 3o ablation results and print LaTeX table rows."""
import json, os, math

LOG = "/tmp/gate3o_ablation"
NOCACHE_LOG = "/tmp/gate3n_prod"
BAGS = ["073623", "061841", "063047"]
CONFIGS = ["B", "C", "D", "E"]
CONFIG_LABELS = {
    "B": "Gate~3g (MH+AA+\\tau)",
    "C": "B + TR-EMA \\alpha=0.10",
    "D": "B + slope \\delta_s=0.010",
    "E": "Full stack (B+C+D)",
}


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


print(f"\n{'Config':>28} {'Skip%':>7} {'V_073623':>10} {'V_061841':>10} {'V_063047':>10} {'V_max':>8} {'Status':>8}")
print("-" * 90)

for cfg in CONFIGS:
    skip_vals = []
    v_row = {}
    aa_vals = []
    mh_vals = []
    sp_vals = []
    for b in BAGS:
        p_path = f"{LOG}/{b}_{cfg}/metrics.json"
        a_path = f"{NOCACHE_LOG}/{b}_NOCACHE/metrics.json"
        if not (os.path.exists(p_path) and os.path.exists(a_path)):
            continue
        d = load(p_path)
        da = load(a_path)
        v = cramers_v(d, da)
        skip = d.get("temporal_cache_skip_ratio", 0)
        if v is not None:
            skip_vals.append(skip)
            v_row[b] = v
            aa_vals.append(d.get("action_aware_bypasses", 0))
            mh_vals.append(d.get("max_hold_bypasses", 0))
            sp_vals.append(d.get("slope_predict_bypasses", 0))
    if not skip_vals:
        print(f"  {CONFIG_LABELS[cfg]:>26}  NOT AVAILABLE")
        continue
    v_max = max(v_row.values())
    skip_mean = sum(skip_vals) / len(skip_vals)
    status = "PASS" if v_max <= 0.10 else "FAIL"

    def fv(b):
        return f"{v_row[b]:.4f}" if b in v_row else "  N/A"

    print(
        f"  {CONFIG_LABELS[cfg]:>26} {skip_mean:>7.1f}%  {fv('073623'):>10} {fv('061841'):>10} {fv('063047'):>10} {v_max:>8.4f}  {status}"
    )

print()
print("--- LaTeX rows (Config & Skip% & V_073623 & V_061841 & V_063047 & V_max & Status) ---")
for cfg in CONFIGS:
    skip_vals = []
    v_row = {}
    aa_vals = []
    mh_vals = []
    sp_vals = []
    for b in BAGS:
        p_path = f"{LOG}/{b}_{cfg}/metrics.json"
        a_path = f"{NOCACHE_LOG}/{b}_NOCACHE/metrics.json"
        if not (os.path.exists(p_path) and os.path.exists(a_path)):
            continue
        d = load(p_path)
        da = load(a_path)
        v = cramers_v(d, da)
        skip = d.get("temporal_cache_skip_ratio", 0)
        if v is not None:
            skip_vals.append(skip)
            v_row[b] = v
            aa_vals.append(d.get("action_aware_bypasses", 0))
            mh_vals.append(d.get("max_hold_bypasses", 0))
            sp_vals.append(d.get("slope_predict_bypasses", 0))
    if not skip_vals:
        continue
    v_max = max(v_row.values())
    skip_mean = sum(skip_vals) / len(skip_vals)
    chk = "\\checkmark" if v_max <= 0.10 else "\\times"
    v073 = v_row.get("073623", 0)
    v061 = v_row.get("061841", 0)
    v063 = v_row.get("063047", 0)
    aa_m = sum(aa_vals) / len(aa_vals) if aa_vals else 0
    mh_m = sum(mh_vals) / len(mh_vals) if mh_vals else 0
    sp_m = sum(sp_vals) / len(sp_vals) if sp_vals else 0
    print(
        f"{CONFIG_LABELS[cfg]} & {skip_mean:.1f}\\% & {v073:.4f} & {v061:.4f} & {v063:.4f} & {v_max:.4f} & {chk} \\\\ % AA={aa_m:.0f} MH={mh_m:.0f} SP={sp_m:.0f}"
    )
