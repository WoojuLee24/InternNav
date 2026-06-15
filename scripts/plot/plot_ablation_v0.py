"""
Plot SR and SPL ablation figures (bar charts).

Each experiment group defines an x-axis label, ordered conditions, and their results.
Results can be hardcoded or loaded from result JSON files.

Usage:
    # plot all groups in one figure (subplots)
    python scripts/eval/plot_ablation.py

    # plot a single group
    python scripts/eval/plot_ablation.py --group num_history

    # save to file
    python scripts/eval/plot_ablation.py --output figures/ablation.png
"""

import json
import os
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPLIT = "val_unseen"

# ── Set these to run the script directly ─────────────────────────────────────
GROUP  = "num_history"          # which experiment group to plot
OUTPUT = "figures/num_history.png"  # where to save


# ── Data definition ──────────────────────────────────────────────────────────

@dataclass
class Condition:
    label: str                        # x-axis tick label
    SR: float = 0.0
    SPL: float = 0.0
    json_path: str | None = None      # relative to repo root; overrides SR/SPL if file exists


@dataclass
class ExperimentGroup:
    name: str                         # group id (used with --group)
    xlabel: str                       # x-axis title
    conditions: list[Condition] = field(default_factory=list)


# ── Fill in your experiments here ────────────────────────────────────────────

GROUPS: list[ExperimentGroup] = [
    ExperimentGroup(
        name="num_history",
        xlabel="num_history",
        conditions=[
            Condition("1",  SR=51.495, SPL=45.409, json_path=None),
            Condition("4",  SR=53.235, SPL=46.403, json_path=None),
            Condition("16", SR=49.538, SPL=43.660, json_path=None),
        ],
    ),
    ExperimentGroup(
        name="num_future_steps",
        xlabel="num_future_steps",
        conditions=[
            Condition("2", SR=0.00, SPL=0.00, json_path=None),
            Condition("8", SR=0.00, SPL=0.00, json_path=None),
        ],
    ),
    ExperimentGroup(
        name="predict_step_num",
        xlabel="predict_step_num",
        conditions=[
            Condition("8",  SR=36.977, SPL=30.934, json_path=None),
            Condition("16", SR=0.00, SPL=0.00, json_path=None),
            Condition("32", SR=0.00, SPL=0.00, json_path=None),
        ],
    ),
    ExperimentGroup(
        name="sample_step",
        xlabel="sample_step",
        conditions=[
            Condition("2", SR=0.00, SPL=0.00, json_path=None),
            Condition("8", SR=0.00, SPL=0.00, json_path=None),
        ],
    ),
    ExperimentGroup(
        name="stop_confidence",
        xlabel="stop_weight",
        conditions=[
            Condition("1",  SR=0.00, SPL=0.00, json_path=None),
            Condition("3",  SR=0.00, SPL=0.00, json_path=None),
            Condition("10", SR=0.00, SPL=0.00, json_path=None),
        ],
    ),
    ExperimentGroup(
        name="finetune_module",
        xlabel="fine-tuned module",
        conditions=[
            Condition("vision", SR=0.00, SPL=0.00, json_path=None),
            Condition("mlp",    SR=0.00, SPL=0.00, json_path=None),
            Condition("llm",    SR=0.00, SPL=0.00, json_path=None),
        ],
    ),
    ExperimentGroup(
        name="visual_resolution",
        xlabel="resolution",
        conditions=[
            Condition("224", SR=0.00, SPL=0.00, json_path=None),
            Condition("336", SR=0.00, SPL=0.00, json_path=None),
            Condition("512", SR=0.00, SPL=0.00, json_path=None),
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
    return Condition(label=cond.label, SR=d["SR"], SPL=d["SPL"], json_path=cond.json_path)


def load_group(group: ExperimentGroup) -> ExperimentGroup:
    return ExperimentGroup(
        name=group.name,
        xlabel=group.xlabel,
        conditions=[_load_condition(c) for c in group.conditions],
    )


# ── Plotting ──────────────────────────────────────────────────────────────────

BAR_COLOR_SR  = "#4C72B0"
BAR_COLOR_SPL = "#DD8452"


def _draw_group(ax: plt.Axes, group: ExperimentGroup) -> None:
    labels = [c.label for c in group.conditions]
    sr_vals  = [c.SR  for c in group.conditions]
    spl_vals = [c.SPL for c in group.conditions]

    x = np.arange(len(labels))
    w = 0.35

    bars_sr  = ax.bar(x - w / 2, sr_vals,  w, label="SR",  color=BAR_COLOR_SR,  edgecolor="white")
    bars_spl = ax.bar(x + w / 2, spl_vals, w, label="SPL", color=BAR_COLOR_SPL, edgecolor="white")

    for bar in (*bars_sr, *bars_spl):
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.1f}", ha="center", va="bottom", fontsize=7,
            )

    ax.set_xlabel(group.xlabel, fontsize=11)
    ax.set_ylabel("(%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)


def plot_single(group: ExperimentGroup, output: str | None = None) -> None:
    group = load_group(group)
    fig, ax = plt.subplots(figsize=(max(5, len(group.conditions) * 1.6), 5))
    _draw_group(ax, group)
    plt.tight_layout()
    _save_or_show(fig, output)


def plot_all(groups: list[ExperimentGroup], output: str | None = None) -> None:
    groups = [load_group(g) for g in groups]
    n = len(groups)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for ax, group in zip(axes_flat, groups):
        _draw_group(ax, group)

    for ax in axes_flat[n:]:  # hide unused subplots
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
    matched = [g for g in GROUPS if g.name == GROUP]
    if not matched:
        print(f"Unknown GROUP '{GROUP}'. Available: {[g.name for g in GROUPS]}")
        return
    plot_single(matched[0], output=OUTPUT)


if __name__ == "__main__":
    main()
