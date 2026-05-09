#!/usr/bin/env python3
"""Plot 2D quality-efficiency frontier from Gate 3g (MH sweep) and Gate 3j (tau sweep).

Output: figures/frontier_2d.pdf  (skip% vs V_max, colored by parameter type)
"""
import json, math, os
import numpy as np

# --------------------------------------------------------------------------
# Gate 3g: max_hold sweep data (from experiment logs, τ=0.92 fixed)
# --------------------------------------------------------------------------
GATE3G_LOG = "/tmp/gate3g_maxhold"
MH_VALUES  = [5, 10, 15, 20, 25, 30]
BAGS       = ["073623", "061841", "063047"]

# Gate 3d Condition A reference paths (baseline for Cramér's V)
GATE3D_A = {
    "073623": "/tmp/gate3d_ablation/073623_A/metrics.json",
    "061841": "/tmp/gate3d_ablation/061841_A/metrics.json",
    "063047": "/tmp/gate3d_ablation/063047_A/metrics.json",
}

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

def load_gate3g():
    """Load Gate 3g MH sweep results."""
    points = []
    for mh in MH_VALUES:
        skip_vals = []
        v_vals = []
        for b in BAGS:
            tag = f"{b}_MH{mh}"
            path = f"{GATE3G_LOG}/{tag}/metrics.json"
            a_path = GATE3D_A.get(b, "")
            if not (os.path.exists(path) and os.path.exists(a_path)):
                continue
            d = load(path)
            da = load(a_path)
            v = cramers_v(d, da)
            skip = d.get("temporal_cache_skip_ratio", 0)
            if v is not None:
                skip_vals.append(skip)
                v_vals.append(v)
        if skip_vals:
            points.append({
                "param": "MH",
                "value": mh,
                "label": f"H={mh}",
                "skip": np.mean(skip_vals),
                "v_max": max(v_vals),
                "v_vals": v_vals,
                "pass": max(v_vals) <= 0.10,
            })
    return points

def load_gate3j():
    """Load Gate 3j tau sweep results."""
    GATE3J_LOG = "/tmp/gate3j_tau"
    TAU_VALUES = [0.85, 0.88, 0.92, 0.95, 0.97]
    points = []
    for tau in TAU_VALUES:
        tau_tag = str(tau).replace(".", "_")
        skip_vals = []
        v_vals = []
        for b in BAGS:
            tag = f"{b}_TAU{tau_tag}"
            path = f"{GATE3J_LOG}/{tag}/metrics.json"
            a_path = GATE3D_A.get(b, "")
            if not (os.path.exists(path) and os.path.exists(a_path)):
                continue
            d = load(path)
            da = load(a_path)
            v = cramers_v(d, da)
            skip = d.get("temporal_cache_skip_ratio", 0)
            if v is not None:
                skip_vals.append(skip)
                v_vals.append(v)
        if skip_vals:
            points.append({
                "param": "tau",
                "value": tau,
                "label": f"τ={tau}",
                "skip": np.mean(skip_vals),
                "v_max": max(v_vals),
                "v_vals": v_vals,
                "pass": max(v_vals) <= 0.10,
            })
    return points

def load_gate3k():
    """Load Gate 3k EMA sweep results (if available)."""
    GATE3K_LOG = "/tmp/gate3k_ema"
    ALPHA_VALUES = [0.05, 0.10, 0.15, 0.20, 0.30]
    points = []
    for alpha in ALPHA_VALUES:
        alpha_tag = str(alpha).replace(".", "_")
        skip_vals = []
        v_vals = []
        baseline_v_vals = []
        for b in BAGS:
            tag = f"{b}_EMA{alpha_tag}"
            base_tag = f"{b}_BASELINE"
            path = f"{GATE3K_LOG}/{tag}/metrics.json"
            base_path = f"{GATE3K_LOG}/{base_tag}/metrics.json"
            if not (os.path.exists(path) and os.path.exists(base_path)):
                continue
            d = load(path)
            da = load(base_path)
            v = cramers_v(d, da)
            skip = d.get("temporal_cache_skip_ratio", 0)
            if v is not None:
                skip_vals.append(skip)
                v_vals.append(v)
        if skip_vals:
            points.append({
                "param": "alpha",
                "value": alpha,
                "label": f"α={alpha}",
                "skip": np.mean(skip_vals),
                "v_max": max(v_vals),
                "v_vals": v_vals,
                "pass": max(v_vals) <= 0.10,
            })
    return points

def plot_frontier(g3g_pts, g3j_pts, g3k_pts, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(7, 5))

    COLORS = {
        "MH": "#1f77b4",    # blue
        "tau": "#d62728",   # red
        "alpha": "#2ca02c", # green
    }
    MARKERS = {
        "MH": "o",
        "tau": "s",
        "alpha": "^",
    }

    v_limit = 0.10
    ax.axhline(v_limit, color="gray", linestyle="--", linewidth=1, label=f"$V$ limit ({v_limit})", zorder=1)
    ax.axhline(0, color="lightgray", linewidth=0.5, zorder=0)

    all_points = g3g_pts + g3j_pts + g3k_pts

    for pt in all_points:
        p = pt["param"]
        color = COLORS[p]
        marker = MARKERS[p]
        edge = "black" if pt["pass"] else "red"
        ax.scatter(pt["skip"], pt["v_max"],
                   c=color, marker=marker, s=80, edgecolors=edge, linewidths=1.5, zorder=3)
        offset_x = -0.5 if pt["skip"] > 93 else 0.3
        ax.annotate(pt["label"], (pt["skip"], pt["v_max"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7.5)

    # Pareto frontier outline (minimum V for each skip%)
    pass_pts = [p for p in all_points if p["pass"]]
    if len(pass_pts) >= 2:
        pass_sorted = sorted(pass_pts, key=lambda x: x["skip"])
        px = [p["skip"] for p in pass_sorted]
        py = [p["v_max"] for p in pass_sorted]
        ax.plot(px, py, linestyle=":", color="gray", linewidth=1, alpha=0.5, zorder=2)

    legend_patches = [
        mpatches.Patch(color=COLORS["MH"],    label="$H_{\\max}$ sweep (Gate 3g)"),
        mpatches.Patch(color=COLORS["tau"],   label="$\\tau$ sweep (Gate 3j)"),
        mpatches.Patch(color=COLORS["alpha"], label="EMA $\\alpha$ sweep (Gate 3k)"),
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=9)

    ax.set_xlabel("Skip rate (%)", fontsize=11)
    ax.set_ylabel("$V_{\\max}$ (worst bag)", fontsize=11)
    ax.set_title("Quality–Efficiency Frontier: MAD Temporal S2 Cache", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()

def main():
    print("Loading Gate 3g results...")
    g3g = load_gate3g()
    print(f"  {len(g3g)} points loaded")
    print("Loading Gate 3j results...")
    g3j = load_gate3j()
    print(f"  {len(g3j)} points loaded")
    print("Loading Gate 3k results...")
    g3k = load_gate3k()
    print(f"  {len(g3k)} points loaded")

    print("\n--- All frontier points ---")
    print(f"{'Param':>8} {'Value':>8} {'Skip%':>8} {'V_max':>8} {'Pass':>6}")
    for pt in g3g + g3j + g3k:
        status = "✓" if pt["pass"] else "✗"
        print(f"{pt['param']:>8} {str(pt['value']):>8} {pt['skip']:>7.1f}% {pt['v_max']:>8.4f} {status:>6}")

    output = "figures/frontier_2d.pdf"
    plot_frontier(g3g, g3j, g3k, output)

if __name__ == "__main__":
    main()
