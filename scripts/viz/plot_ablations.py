"""
plot_ablations.py — publication-quality ablation plots for InternNav experiments.

Reads stats/{experiment_tag}/summary.json files (written by stats_recorder.py).
Generates control-variable plots, 3-bag comparisons, and time-series traces.

Usage:
    python scripts/viz/plot_ablations.py                        # all plots
    python scripts/viz/plot_ablations.py --var temperature      # one variable
    python scripts/viz/plot_ablations.py --exp exp_a exp_b      # specific experiments
    python scripts/viz/plot_ablations.py --timeseries exp_tag   # time series
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # headless — saves to file; remove for interactive
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
_REPO  = Path(__file__).resolve().parents[2]
_STATS = _REPO / "stats"
_FIGS  = _REPO / "figures" / "ablations"
_FIGS.mkdir(parents=True, exist_ok=True)

# ── style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

PALETTE = {
    "trajectory_ratio":        "#2ECC71",
    "joint_req_hz":            "#3498DB",
    "joint_latency_ms_mean":   "#E74C3C",
    "s2_latency_ms_mean":      "#9B59B6",
    "discrete_ratio":          "#F39C12",
}
BASELINE = {
    "joint_req_hz": 2.49,
    "trajectory_ratio": 0.633,
    "joint_latency_ms_mean": 313.0,
    "s2_latency_ms_mean": 313.0,
}
METRIC_LABELS = {
    "trajectory_ratio":        "Trajectory Ratio",
    "joint_req_hz":            "Request Rate (Hz)",
    "joint_latency_ms_mean":   "Latency Mean (ms)",
    "s2_latency_ms_mean":      "S2 Latency (ms)",
}
METRIC_FORMAT = {
    "trajectory_ratio":  lambda v: f"{v*100:.1f}%",
    "joint_req_hz":      lambda v: f"{v:.2f} Hz",
    "joint_latency_ms_mean":  lambda v: f"{v:.0f} ms",
    "s2_latency_ms_mean":     lambda v: f"{v:.0f} ms",
}


def _load_summaries(tags: Optional[List[str]] = None) -> List[Dict]:
    dirs = sorted(_STATS.glob("*/summary.json"))
    out = []
    for p in dirs:
        with open(p) as f:
            d = json.load(f)
        if tags is None or d["tag"] in tags:
            out.append(d)
    return out


def _load_metrics_jsonl(tag: str) -> List[Dict]:
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


# ── Plot 1: control variable sweep (bar chart with baseline reference) ─────────

def plot_control_variable(
    summaries: List[Dict],
    control_var: str,
    metrics: Optional[List[str]] = None,
    save: bool = True,
) -> plt.Figure:
    """Bar chart: one control variable (x-axis) vs one or more metrics (y-axes).

    Groups experiments by their control_vars[control_var] value.
    Overlays a dashed SYNC baseline line.
    """
    if metrics is None:
        metrics = ["trajectory_ratio", "joint_req_hz"]

    # Filter experiments that have this control variable
    valid = [s for s in summaries if control_var in s.get("control_vars", {})]
    if not valid:
        print(f"No experiments with control_var='{control_var}'")
        return None

    x_vals = sorted(set(s["control_vars"][control_var] for s in valid))
    x_str  = [str(v) for v in x_vals]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    fig.suptitle(f"Ablation: {control_var}", fontsize=14, fontweight="bold", y=1.02)

    for ax, metric in zip(axes, metrics):
        color = PALETTE.get(metric, "#555555")
        y_vals, y_err = [], []

        for xv in x_vals:
            group = [s[metric] for s in valid
                     if s["control_vars"].get(control_var) == xv and metric in s]
            if group:
                y_vals.append(np.mean(group))
                y_err.append(np.std(group) if len(group) > 1 else 0)
            else:
                y_vals.append(np.nan)
                y_err.append(0)

        bars = ax.bar(x_str, y_vals, color=color, alpha=0.82,
                      edgecolor="white", linewidth=1.2,
                      yerr=y_err, capsize=4, error_kw={"linewidth": 1.5})

        # Baseline reference line
        if metric in BASELINE:
            bl = BASELINE[metric]
            ax.axhline(bl, color="black", linewidth=1.2, linestyle="--",
                       label=f"SYNC baseline ({METRIC_FORMAT[metric](bl)})")
            ax.legend(fontsize=9, framealpha=0.8)

        # Value annotations on bars
        fmt = METRIC_FORMAT.get(metric, lambda v: f"{v:.3f}")
        for bar, val in zip(bars, y_vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (max(y_vals, default=1) * 0.01),
                        fmt(val), ha="center", va="bottom", fontsize=9)

        ax.set_xlabel(control_var, fontsize=11)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: fmt(x)) if metric == "trajectory_ratio"
            else ticker.AutoFormatter()
        )

    plt.tight_layout()
    if save:
        out = _FIGS / f"ablation_{control_var}.pdf"
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
        print(f"Saved: {out}")
    return fig


# ── Plot 2: 3-bag comparison bar chart ────────────────────────────────────────

def plot_3bag_comparison(
    summaries: List[Dict],
    metric: str = "trajectory_ratio",
    save: bool = True,
) -> plt.Figure:
    """Grouped bar chart comparing experiments across 3 bags."""
    # Group by tag, sub-group by bag_id
    tags   = sorted(set(s["tag"] for s in summaries))
    bags   = sorted(set(s["control_vars"].get("bag_id", "unknown")
                        for s in summaries if "bag_id" in s.get("control_vars", {})))
    if not bags:
        bags = ["all"]

    x     = np.arange(len(bags))
    width = 0.8 / max(len(tags), 1)
    fig, ax = plt.subplots(figsize=(max(6, len(bags) * 2.5), 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(tags)))

    for i, (tag, color) in enumerate(zip(tags, colors)):
        vals = []
        for bag in bags:
            match = [s[metric] for s in summaries
                     if s["tag"] == tag and s.get("control_vars", {}).get("bag_id") == bag
                     and metric in s]
            vals.append(np.mean(match) if match else np.nan)

        offset = (i - len(tags) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9, label=tag,
                      color=color, alpha=0.85, edgecolor="white")
        fmt = METRIC_FORMAT.get(metric, lambda v: f"{v:.3f}")
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        fmt(val), ha="center", va="bottom", fontsize=8, rotation=45)

    if metric in BASELINE:
        bl = BASELINE[metric]
        ax.axhline(bl, color="black", linewidth=1.2, linestyle="--",
                   label=f"SYNC baseline")

    ax.set_xticks(x)
    ax.set_xticklabels([f"bag {b}" for b in bags])
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(f"3-Bag Comparison: {METRIC_LABELS.get(metric, metric)}")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

    plt.tight_layout()
    if save:
        out = _FIGS / f"3bag_{metric}.pdf"
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
        print(f"Saved: {out}")
    return fig


# ── Plot 3: time-series trace during a single experiment ─────────────────────

def plot_time_series(
    tag: str,
    metrics: Optional[List[str]] = None,
    rolling_window: int = 20,
    save: bool = True,
) -> plt.Figure:
    """Line plot of rolling metrics over time for one experiment."""
    rows = _load_metrics_jsonl(tag)
    if not rows:
        print(f"No JSONL data for experiment '{tag}'")
        return None

    if metrics is None:
        metrics = ["joint_latency_ms", "s2_latency_ms"]

    t = [r["t_rel"] for r in rows]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    fig.suptitle(f"Time Series: {tag}", fontsize=13, fontweight="bold")

    for ax, metric in zip(axes, metrics):
        raw = [r.get(metric, np.nan) for r in rows]
        # Rolling mean
        kernel  = np.ones(rolling_window) / rolling_window
        if len(raw) >= rolling_window:
            smooth = np.convolve(raw, kernel, mode="valid")
            t_sm   = t[rolling_window - 1:]
        else:
            smooth = np.array(raw)
            t_sm   = t

        ax.plot(t, raw, alpha=0.25, color=PALETTE.get(metric, "#888"),
                linewidth=0.8, label="raw")
        ax.plot(t_sm, smooth, color=PALETTE.get(metric, "#333"),
                linewidth=2, label=f"rolling-{rolling_window}")
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.legend(fontsize=9, framealpha=0.7)
        ax.grid(True, alpha=0.3)

        # Response-type coloring as background bands
        if metric == "joint_latency_ms":
            types = [r.get("response_type", "waiting") for r in rows]
            for i, rtype in enumerate(types):
                c = {"trajectory": "#2ECC7122", "discrete": "#F39C1222"}.get(rtype, "#00000000")
                if i < len(t) - 1:
                    ax.axvspan(t[i], t[i + 1], facecolor=c, edgecolor="none")

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    if save:
        out = _FIGS / f"timeseries_{tag}.pdf"
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
        print(f"Saved: {out}")
    return fig


# ── Plot 4: trajectory ratio vs request rate scatter (quality-speed tradeoff) ──

def plot_quality_speed_tradeoff(
    summaries: List[Dict],
    color_by: Optional[str] = None,
    save: bool = True,
) -> plt.Figure:
    """Scatter: trajectory_ratio (quality) vs joint_req_hz (speed), each experiment as a dot."""
    if not summaries:
        return None

    x = [s.get("joint_req_hz", 0) for s in summaries]
    y = [s.get("trajectory_ratio", 0) for s in summaries]
    labels = [s["tag"] for s in summaries]

    color_vals = None
    if color_by:
        color_vals = [s.get("control_vars", {}).get(color_by, 0) for s in summaries]

    fig, ax = plt.subplots(figsize=(8, 6))

    sc = ax.scatter(x, y, c=color_vals, cmap="viridis" if color_vals else None,
                    s=80, alpha=0.85, edgecolors="white", linewidths=0.8,
                    zorder=3)
    if color_vals and color_by:
        cbar = plt.colorbar(sc, ax=ax, pad=0.01)
        cbar.set_label(color_by)

    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), textcoords="offset points",
                    xytext=(5, 3), fontsize=7, alpha=0.8)

    # Baseline dot
    ax.scatter([BASELINE["joint_req_hz"]], [BASELINE["trajectory_ratio"]],
               marker="*", s=200, color="black", zorder=4, label="SYNC baseline")

    # Gate lines
    ax.axhline(0.50, color="#2ECC71", linestyle="--", linewidth=1.2,
               label="Gate: 50% traj ratio")
    ax.axhline(0.35, color="#E74C3C", linestyle=":", linewidth=1,
               label="Fail: <35%")
    ax.axvline(10.0, color="#3498DB", linestyle="--", linewidth=1,
               label="Gate: 10 Hz")

    ax.set_xlabel("Request Rate (Hz)")
    ax.set_ylabel("Trajectory Ratio")
    ax.set_title("Quality–Speed Tradeoff Space")
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        out = _FIGS / "quality_speed_tradeoff.pdf"
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
        print(f"Saved: {out}")
    return fig


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", help="Control variable to ablate (e.g., temperature)")
    ap.add_argument("--exp", nargs="+", help="Specific experiment tags to include")
    ap.add_argument("--metric", default="trajectory_ratio",
                    help="Primary metric for 3-bag comparison")
    ap.add_argument("--timeseries", help="Plot time series for this experiment tag")
    ap.add_argument("--tradeoff", action="store_true",
                    help="Plot quality–speed tradeoff scatter")
    ap.add_argument("--color-by", default=None,
                    help="Control variable to color tradeoff dots by")
    args = ap.parse_args()

    summaries = _load_summaries(args.exp)
    if not summaries:
        print("No experiment summaries found in stats/. Run an experiment first.")
        print("  Stats recorder: python scripts/realworld/stats_recorder.py parse <client.log> --tag my_exp")
        return

    if args.timeseries:
        plot_time_series(args.timeseries)
        return

    if args.tradeoff:
        plot_quality_speed_tradeoff(summaries, color_by=args.color_by)
        return

    if args.var:
        plot_control_variable(summaries, args.var,
                              metrics=["trajectory_ratio", "joint_req_hz", "joint_latency_ms_mean"])
    else:
        # Default: generate all ablation plots for every detected control variable
        all_cvars = set()
        for s in summaries:
            all_cvars.update(s.get("control_vars", {}).keys())
        all_cvars -= {"bag_id", "mode"}  # skip non-ablation vars

        if not all_cvars:
            print("No control variables detected. Generating 3-bag comparison only.")
        else:
            for cvar in sorted(all_cvars):
                plot_control_variable(summaries, cvar)

        plot_3bag_comparison(summaries, args.metric)
        plot_quality_speed_tradeoff(summaries)

    print(f"\nAll figures saved to: {_FIGS}")


if __name__ == "__main__":
    main()
