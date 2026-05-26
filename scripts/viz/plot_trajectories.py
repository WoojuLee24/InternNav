"""
plot_trajectories.py — robot trajectory and subgoal visualization for InternNav.

Reads waypoints + subgoal data from stats/{tag}/metrics.jsonl (written by
stats_recorder.py) and generates:
  1. Top-down 2D path  — robot position over time, colored by response type
  2. Subgoal overlay   — S2 pixel goals projected onto the path
  3. Trajectory quality comparison — multiple experiments side-by-side
  4. Response-type timeline bar — when trajectory vs discrete vs waiting

Usage:
    python scripts/viz/plot_trajectories.py                     # all tags in stats/
    python scripts/viz/plot_trajectories.py --tag my_exp        # single experiment
    python scripts/viz/plot_trajectories.py --compare A B C     # overlay multiple
    python scripts/viz/plot_trajectories.py --timeline my_exp   # response timeline
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import numpy as np

_REPO  = Path(__file__).resolve().parents[2]
_STATS = _REPO / "stats"
_FIGS  = _REPO / "figures" / "trajectories"
_FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

RTYPE_COLORS = {
    "trajectory": "#2ECC71",   # green
    "discrete":   "#F39C12",   # orange
    "waiting":    "#BDC3C7",   # grey
    "error":      "#E74C3C",   # red
}
RTYPE_LABELS = {
    "trajectory": "S1 Trajectory (quality)",
    "discrete":   "Discrete Action",
    "waiting":    "Waiting for S2",
}


def _load_records(tag: str) -> List[Dict]:
    p = _STATS / tag / "metrics.jsonl"
    if not p.exists():
        return []
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _integrate_path(records: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Reconstruct approximate 2D path from accumulated waypoints.

    Sums the first waypoint delta (dx, dy) from each trajectory response
    to build a dead-reckoning path. If no waypoints, uses sequential index.
    """
    x, y, types = [0.0], [0.0], []
    cx, cy = 0.0, 0.0
    for r in records:
        rtype = r.get("response_type", "waiting")
        wpts = r.get("waypoints")
        if wpts and len(wpts) > 0 and len(wpts[0]) >= 2:
            dx, dy = wpts[0][0], wpts[0][1]
        else:
            dx, dy = 0.0, 0.0
        cx += dx
        cy += dy
        x.append(cx)
        y.append(cy)
        types.append(rtype)
    return np.array(x), np.array(y), types


# ── Plot 1: 2D path colored by response type ─────────────────────────────────

def plot_path(tag: str, save: bool = True) -> Optional[plt.Figure]:
    records = _load_records(tag)
    if not records:
        print(f"No records for '{tag}'"); return None

    x, y, types = _integrate_path(records)
    summary_path = _STATS / tag / "summary.json"
    summary = json.load(open(summary_path)) if summary_path.exists() else {}

    fig, (ax_path, ax_bar) = plt.subplots(1, 2, figsize=(14, 6),
                                           gridspec_kw={"width_ratios": [2, 1]})
    fig.suptitle(f"Trajectory: {tag}", fontsize=13, fontweight="bold")

    # ── left: 2D path ──────────────────────────────────────────────────────────
    # Draw segments colored by response type
    points = np.column_stack([x, y])
    segments = np.array([(points[i], points[i+1]) for i in range(len(points)-1)])
    colors = [RTYPE_COLORS.get(t, "#888") for t in types]

    lc = LineCollection(segments, colors=colors, linewidths=2, alpha=0.85)
    ax_path.add_collection(lc)
    ax_path.autoscale()

    # Start / end markers
    ax_path.scatter([x[0]], [y[0]], s=120, c="black", marker="o", zorder=5,
                    label="Start")
    ax_path.scatter([x[-1]], [y[-1]], s=150, c="black", marker="*", zorder=5,
                    label="End")

    # Subgoal pixel goals (if recorded — projected as dots on path at that timestep)
    subgoal_steps = [(i, r.get("subgoal_px")) for i, r in enumerate(records)
                     if r.get("subgoal_px")]
    if subgoal_steps:
        sx = [x[i] for i, _ in subgoal_steps]
        sy = [y[i] for i, _ in subgoal_steps]
        ax_path.scatter(sx, sy, s=60, c="#9B59B6", marker="x", linewidths=2,
                        zorder=6, label=f"S2 Subgoal (n={len(sx)})")

    # Legend
    patches = [mpatches.Patch(color=c, label=RTYPE_LABELS.get(t, t))
               for t, c in RTYPE_COLORS.items() if t in set(types)]
    patches += ax_path.get_legend_handles_labels()[0]
    ax_path.legend(handles=patches, fontsize=9, loc="upper left", framealpha=0.8)
    ax_path.set_xlabel("X (accumulated Δx)")
    ax_path.set_ylabel("Y (accumulated Δy)")
    ax_path.set_title("Dead-reckoned Path (colored by response type)")
    ax_path.set_aspect("equal")
    ax_path.grid(True, alpha=0.3)

    # ── right: response type distribution bar ─────────────────────────────────
    type_counts = {t: 0 for t in RTYPE_COLORS}
    for t in types:
        type_counts[t] = type_counts.get(t, 0) + 1
    total = sum(type_counts.values()) or 1

    bar_labels = [RTYPE_LABELS.get(t, t) for t in RTYPE_COLORS if type_counts[t] > 0]
    bar_vals   = [type_counts[t] / total * 100 for t in RTYPE_COLORS if type_counts[t] > 0]
    bar_colors = [RTYPE_COLORS[t] for t in RTYPE_COLORS if type_counts[t] > 0]

    bars = ax_bar.barh(bar_labels, bar_vals, color=bar_colors, edgecolor="white",
                       linewidth=1, alpha=0.85)
    for bar, val in zip(bars, bar_vals):
        ax_bar.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=10)

    traj_ratio = summary.get("trajectory_ratio", type_counts.get("trajectory", 0) / total)
    gate = "✓ PASS (≥50%)" if traj_ratio >= 0.5 else "✗ FAIL (<50%)"
    ax_bar.set_xlabel("Fraction of requests (%)")
    ax_bar.set_title(f"Response Distribution\n trajectory_ratio={traj_ratio*100:.1f}% {gate}")
    ax_bar.set_xlim(0, 100)
    ax_bar.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    if save:
        out = _FIGS / f"path_{tag}.pdf"
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
        print(f"Saved: {out}")
    return fig


# ── Plot 2: Compare multiple trajectories overlaid ───────────────────────────

def plot_compare(tags: List[str], metric: str = "trajectory", save: bool = True) -> Optional[plt.Figure]:
    """Overlay paths from multiple experiments — useful for ablation comparisons."""
    colors = plt.cm.tab10(np.linspace(0, 1, len(tags)))
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle("Trajectory Comparison", fontsize=13, fontweight="bold")

    for tag, color in zip(tags, colors):
        records = _load_records(tag)
        if not records:
            print(f"  skip {tag}: no records"); continue
        x, y, types = _integrate_path(records)

        # Only show trajectory segments if requested
        if metric == "trajectory":
            traj_mask = [t == "trajectory" for t in types]
            for i, is_traj in enumerate(traj_mask):
                if is_traj and i < len(x) - 1:
                    ax.plot(x[i:i+2], y[i:i+2], color=color, linewidth=2, alpha=0.7)
        else:
            ax.plot(x, y, color=color, linewidth=1.5, alpha=0.7)

        ax.scatter([x[0]], [y[0]], s=80, color=color, marker="o", zorder=5)
        # Load trajectory ratio for legend
        summary_path = _STATS / tag / "summary.json"
        ratio = ""
        if summary_path.exists():
            s = json.load(open(summary_path))
            ratio = f" ({s.get('trajectory_ratio', 0)*100:.1f}%)"
        ax.plot([], [], color=color, linewidth=2, label=f"{tag}{ratio}")

    ax.set_xlabel("X (accumulated Δx)")
    ax.set_ylabel("Y (accumulated Δy)")
    ax.set_title("Overlaid Paths (colored by experiment)")
    ax.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        label = "_vs_".join(t[:12] for t in tags[:3])
        out = _FIGS / f"compare_{label}.pdf"
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
        print(f"Saved: {out}")
    return fig


# ── Plot 3: Response-type timeline ───────────────────────────────────────────

def plot_timeline(tag: str, save: bool = True) -> Optional[plt.Figure]:
    """Horizontal bar showing response type at each timestep."""
    records = _load_records(tag)
    if not records:
        print(f"No records for '{tag}'"); return None

    types = [r.get("response_type", "waiting") for r in records]
    t = [r.get("t_rel", i) for i, r in enumerate(records)]

    fig, (ax_tl, ax_lat) = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    fig.suptitle(f"Timeline: {tag}", fontsize=13, fontweight="bold")

    # ── response type timeline ─────────────────────────────────────────────────
    for i, (ti, rtype) in enumerate(zip(t[:-1], types)):
        color = RTYPE_COLORS.get(rtype, "#888")
        ax_tl.barh(0, t[i+1] - ti, left=ti, height=0.6,
                   color=color, edgecolor="none", alpha=0.85)

    patches = [mpatches.Patch(color=c, label=RTYPE_LABELS.get(tp, tp))
               for tp, c in RTYPE_COLORS.items() if tp in set(types)]
    ax_tl.legend(handles=patches, fontsize=9, loc="upper right", framealpha=0.8)
    ax_tl.set_yticks([])
    ax_tl.set_ylabel("Response")
    ax_tl.set_title("Response Type Over Time")

    # ── latency overlay ────────────────────────────────────────────────────────
    latencies = [r.get("joint_latency_ms", 0) for r in records]
    ax_lat.plot(t, latencies, color="#3498DB", linewidth=1.5, alpha=0.9, label="joint_latency_ms")
    ax_lat.axhline(20, color="#E74C3C", linewidth=1, linestyle="--", label="Gate 0 threshold (20 ms)")
    ax_lat.set_ylabel("Latency (ms)")
    ax_lat.set_xlabel("Time (s)")
    ax_lat.legend(fontsize=9, framealpha=0.8)
    ax_lat.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        out = _FIGS / f"timeline_{tag}.pdf"
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
        print(f"Saved: {out}")
    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", help="Single experiment tag")
    ap.add_argument("--compare", nargs="+", help="Compare multiple experiments")
    ap.add_argument("--timeline", help="Response-type timeline for one experiment")
    ap.add_argument("--all", action="store_true", help="Generate all plots for all tags")
    args = ap.parse_args()

    if args.timeline:
        plot_timeline(args.timeline)
        return

    if args.compare:
        plot_compare(args.compare)
        return

    if args.tag:
        plot_path(args.tag)
        plot_timeline(args.tag)
        return

    # Default: all available tags
    tags = sorted(p.parent.name for p in _STATS.glob("*/metrics.jsonl"))
    if not tags:
        print("No experiment records found in stats/*/metrics.jsonl")
        print("  Run: python scripts/realworld/stats_recorder.py parse <client.log> --tag <name>")
        return
    for tag in tags:
        print(f"\nGenerating plots for: {tag}")
        plot_path(tag)
        plot_timeline(tag)
    if len(tags) > 1:
        plot_compare(tags)

    print(f"\nAll figures saved to: {_FIGS}")


if __name__ == "__main__":
    main()
