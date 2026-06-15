"""
Plot SR, SPL, OSS, NES ablation figures (bar + line charts).

Each experiment group defines an x-axis label, ordered conditions, and their results.
Results can be hardcoded or loaded from result JSON files.

Usage:
    # plot all groups in one figure (subplots)
    python scripts/eval/plot_ablation.py

    # plot a single group
    python scripts/eval/plot_ablation.py --group num_history

    # save to file
    python scripts/eval/plot_ablation.py --output figures/ablation.png

    # show only specific metrics
    python scripts/eval/plot_ablation.py --metrics SR SPL
    python scripts/eval/plot_ablation.py --metrics SR OSS NES
"""

import json
import os
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPLIT = "val_unseen"

# ── Set these to run the script directly ─────────────────────────────────────
GROUP   = "num_history"
OUTPUT  = None   # None → auto: figures/<group>.png
METRICS = ["SR", "SPL", "OSS", "NES"]   # any subset of these four


# ── Data definition ──────────────────────────────────────────────────────────

@dataclass
class Condition:
    label: str
    SR: float = 0.0
    SPL: float = 0.0
    OSS: float = 0.0
    NES: float = 0.0
    json_path: str | None = None      # relative to repo root; overrides SR/SPL/OSS/NES if file exists


@dataclass
class ExperimentGroup:
    name: str
    xlabel: str
    conditions: list[Condition] = field(default_factory=list)


# ── Fill in your experiments here ────────────────────────────────────────────

GROUPS: list[ExperimentGroup] = [
    ExperimentGroup(
        name="predict_step_num",
        xlabel="predict_step_num",
        conditions=[
            Condition("8",         SR=36.977, SPL=30.934, OSS=43.665, NES=5.68538),
            Condition("16",        SR=53.834, SPL=47.439, OSS=60.359, NES=4.72516),
            Condition("32\n(base)",SR=53.072, SPL=46.689, OSS=59.380, NES=4.82142),
            Condition("64",        SR=46.112, SPL=39.810, OSS=54.649, NES=5.21729),
        ],
    ),
    ExperimentGroup(
        name="sample_step",
        xlabel="sample_step",
        conditions=[
            Condition("2",        SR=55.519, SPL=49.315, OSS=62.262, NES=4.72123),
            Condition("4\n(base)",SR=52.637, SPL=46.063, OSS=59.869, NES=4.71466),
            Condition("8",        SR=50.571, SPL=44.376, OSS=57.912, NES=5.04033),
        ],
    ),
    ExperimentGroup(
        name="num_history",
        xlabel="num_history",
        conditions=[
            Condition("1",         SR=51.495, SPL=45.409, OSS=58.673, NES=4.80695),
            Condition("4\n(base)", SR=53.235, SPL=46.403, OSS=59.217, NES=4.90030),
            Condition("8",         SR=52.964, SPL=46.706, OSS=60.848, NES=4.66974),
            Condition("16",        SR=49.538, SPL=43.660, OSS=57.259, NES=4.99906),
        ],
    ),
    ExperimentGroup(
        name="learning_rate",
        xlabel="learning rate",
        conditions=[
            Condition("5e-5",          SR=48.722, SPL=42.119, OSS=55.628, NES=5.00197),
            Condition("1e-4\n(base)",  SR=49.864, SPL=43.527, OSS=56.987, NES=4.94820),
            Condition("2e-4",          SR=53.616, SPL=47.042, OSS=60.250, NES=4.66862),
        ],
    ),
    ExperimentGroup(
        name="lr_scheduler",
        xlabel="lr scheduler",
        conditions=[
            Condition("linear",                   SR=52.365, SPL=45.619, OSS=59.326, NES=4.84452),
            Condition("cosine_w/\nmin_lr\n(base)",SR=48.831, SPL=42.411, OSS=56.716, NES=4.99939),
            Condition("cosine",                   SR=49.810, SPL=43.645, OSS=57.368, NES=4.96110),
        ],
    ),
    ExperimentGroup(
        name="batch_size",
        xlabel="batch size",
        conditions=[
            Condition("2",        SR=51.115, SPL=44.397, OSS=58.401, NES=4.72243),
            Condition("4\n(base)",SR=51.822, SPL=45.571, OSS=59.598, NES=4.85956),
            Condition("8",        SR=50.408, SPL=44.732, OSS=58.021, NES=4.88225),
        ],
    ),
    ExperimentGroup(
        name="finetune_module",
        xlabel="fine-tuned module",
        conditions=[
            Condition("vision",        SR=7.613,  SPL=6.557,  OSS=20.827, NES=9.44190),
            Condition("mlp",           SR=47.471, SPL=43.029, OSS=53.616, NES=5.40390),
            Condition("base\n(full)",  SR=51.332, SPL=45.165, OSS=57.966, NES=4.90021),
        ],
    ),
    ExperimentGroup(
        name="visual_resolution",
        xlabel="resolution",
        conditions=[
            Condition("224",          SR=53.072, SPL=46.630, OSS=59.271, NES=4.76914),
            Condition("384\n(base)",  SR=49.701, SPL=43.896, OSS=56.987, NES=4.97302),
            Condition("512",          SR=54.432, SPL=47.536, OSS=60.196, NES=4.77843),
        ],
    ),
    ExperimentGroup(
        name="latent_query",
        xlabel="latent query",
        conditions=[
            Condition("2",        SR=52.855, SPL=46.334, OSS=58.891, NES=4.81709),
            Condition("4\n(base)",SR=53.844, SPL=47.446, OSS=60.305, NES=4.79420),
            Condition("8",        SR=53.562, SPL=47.428, OSS=59.815, NES=4.84289),
        ],
    ),
]


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_condition(cond: Condition) -> Condition:
    if cond.json_path is None:
        return cond
    full = os.path.join(REPO_ROOT, cond.json_path)
    if not os.path.exists(full):
        return cond
    with open(full) as f:
        data = json.load(f)
    d = data.get(SPLIT, data)
    return Condition(
        label=cond.label,
        SR=d["SR"], SPL=d["SPL"],
        OSS=d.get("OSS", cond.OSS),
        NES=d.get("NES", cond.NES),
        json_path=cond.json_path,
    )


def load_group(group: ExperimentGroup) -> ExperimentGroup:
    return ExperimentGroup(
        name=group.name,
        xlabel=group.xlabel,
        conditions=[_load_condition(c) for c in group.conditions],
    )


# ── Plotting ──────────────────────────────────────────────────────────────────

BAR_COLOR_SR  = "#2166AC"   # blue
BAR_COLOR_SPL = "#D6604D"   # red-orange
BAR_COLOR_OSS = "#4DAC26"   # green
LINE_COLOR_NES = "#7B2D8B"  # purple

FONT_FAMILY   = "serif"
LABEL_SIZE    = 9
TICK_SIZE     = 8
LEGEND_SIZE   = 7
VALUE_SIZE    = 6.0
COL_WIDTH_IN  = 3.33


def _apply_paper_style() -> None:
    plt.rcParams.update({
        "font.family":        FONT_FAMILY,
        "font.size":          LABEL_SIZE,
        "axes.labelsize":     LABEL_SIZE,
        "axes.titlesize":     LABEL_SIZE,
        "xtick.labelsize":    TICK_SIZE,
        "ytick.labelsize":    TICK_SIZE,
        "legend.fontsize":    LEGEND_SIZE,
        "axes.linewidth":     0.8,
        "xtick.major.width":  0.8,
        "ytick.major.width":  0.8,
        "xtick.major.size":   3,
        "ytick.major.size":   3,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
    })


_BAR_METRICS = [
    ("SR",  BAR_COLOR_SR,  "",      {}),
    ("SPL", BAR_COLOR_SPL, "////",  {}),
    ("OSS", BAR_COLOR_OSS, "xxxx",  {}),
]


def _draw_group(ax: plt.Axes, group: ExperimentGroup, metrics: list[str] | None = None) -> None:
    if metrics is None:
        metrics = ["SR", "SPL", "OSS", "NES"]

    labels = [c.label for c in group.conditions]
    x = np.arange(len(labels))

    bar_metrics = [m for m in metrics if m != "NES"]
    show_nes    = "NES" in metrics

    n_bars = len(bar_metrics)
    w = max(0.12, 0.7 / max(n_bars, 1) * 0.8)
    centers = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * w if n_bars > 1 else [0.0]

    all_bars = []
    all_pct  = []
    for (name, color, hatch, _), offset in zip(
        [bm for bm in _BAR_METRICS if bm[0] in bar_metrics],
        centers,
    ):
        vals = [getattr(c, name) for c in group.conditions]
        bars = ax.bar(x + offset, vals, w, label=name, color=color,
                      hatch=hatch, edgecolor="white", linewidth=0.5)
        all_bars.extend(bars)
        all_pct.extend(vals)

    for bar in all_bars:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.1f}", ha="center", va="bottom", fontsize=VALUE_SIZE, color="#333333",
            )

    ax.set_xlabel(group.xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel("Score (%)", fontsize=LABEL_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=TICK_SIZE)

    max_pct = max(all_pct, default=0)
    ax.set_ylim(0, min(100, max_pct * 1.25) if max_pct > 0 else 100)
    ax.spines["top"].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="gray", alpha=0.6)
    ax.set_axisbelow(True)

    if show_nes:
        nes_vals = [c.NES for c in group.conditions]
        ax2 = ax.twinx()
        ax2.plot(x, nes_vals, color=LINE_COLOR_NES, marker="o", markersize=4,
                 linewidth=1.2, linestyle="--", label="NES")
        for xi, v in zip(x, nes_vals):
            ax2.text(xi, v + 0.05, f"{v:.2f}", ha="center", va="bottom",
                     fontsize=VALUE_SIZE - 0.5, color=LINE_COLOR_NES)
        nes_min, nes_max = min(nes_vals), max(nes_vals)
        pad = (nes_max - nes_min) * 0.5 if nes_max > nes_min else 0.5
        ax2.set_ylim(nes_min - pad, nes_max + pad * 2)
        ax2.set_ylabel("NES (m) ↓", fontsize=LABEL_SIZE, color=LINE_COLOR_NES)
        ax2.tick_params(axis="y", labelcolor=LINE_COLOR_NES, labelsize=TICK_SIZE)
        ax2.spines["top"].set_visible(False)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=LEGEND_SIZE,
                  ncol=2, loc="upper right", handlelength=1.2, handletextpad=0.4)
    else:
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=LEGEND_SIZE, ncol=2,
                  loc="upper right", handlelength=1.2, handletextpad=0.4)


def plot_single(group: ExperimentGroup, output: str | None = None,
                metrics: list[str] | None = None) -> None:
    _apply_paper_style()
    group = load_group(group)
    n = len(group.conditions)
    fig, ax = plt.subplots(figsize=(max(COL_WIDTH_IN, n * 1.1), 2.8))
    _draw_group(ax, group, metrics=metrics)
    fig.tight_layout(pad=0.5)
    _save_or_show(fig, output)


def plot_all(groups: list[ExperimentGroup], output: str | None = None,
             metrics: list[str] | None = None) -> None:
    groups = [load_group(g) for g in groups]
    n = len(groups)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for ax, group in zip(axes_flat, groups):
        _draw_group(ax, group, metrics=metrics)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Ablation Study", fontsize=14, y=1.01)
    plt.tight_layout()
    _save_or_show(fig, output)


def _save_or_show(fig: plt.Figure, output: str | None) -> None:
    if output:
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved to {output}")
    else:
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--group",   default=GROUP,  help="experiment group name")
    parser.add_argument("--output",  default=OUTPUT, help="output file path (default: figures/<group>.png)")
    parser.add_argument("--metrics", nargs="+", default=METRICS,
                        choices=["SR", "SPL", "OSS", "NES"],
                        help="metrics to display (default: all four)")
    args = parser.parse_args()

    matched = [g for g in GROUPS if g.name == args.group]
    if not matched:
        print(f"Unknown group '{args.group}'. Available: {[g.name for g in GROUPS]}")
        return

    output = args.output or f"figures/{args.group}.png"
    plot_single(matched[0], output=output, metrics=args.metrics)


if __name__ == "__main__":
    main()
