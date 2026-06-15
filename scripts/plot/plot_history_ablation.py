"""
Plot SR and SPL vs num_history (bar chart).

Usage:
    python scripts/eval/plot_history_ablation.py
    python scripts/eval/plot_history_ablation.py --output figures/history_ablation.png

Data can be filled in directly below, or loaded from result JSON files.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

# ── Fill in your experiment results here ────────────────────────────────────
# Each entry: num_history → {"SR": float, "SPL": float}
RESULTS = {
    1: {"SR": 0.00, "SPL": 0.00},
    3: {"SR": 0.00, "SPL": 0.00},
    5: {"SR": 0.6274, "SPL": 0.5679},
    7: {"SR": 0.00, "SPL": 0.00},
    # Add more as needed
}

# Optionally load from JSON files (overrides RESULTS if path exists)
# Map num_history → path to result.json
JSON_PATHS: dict[int, str] = {
    # 1: "logs/history_1/result.json",
    # 3: "logs/history_3/result.json",
    5: "logs/test_n1/result.json",
    # 7: "logs/history_7/result.json",
}

SPLIT = "val_unseen"  # key inside result.json


def load_results() -> dict[int, dict]:
    results = dict(RESULTS)
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for num_hist, path in JSON_PATHS.items():
        full_path = os.path.join(repo_root, path)
        if os.path.exists(full_path):
            with open(full_path) as f:
                data = json.load(f)
            split_data = data.get(SPLIT, data)
            results[num_hist] = {"SR": split_data["SR"], "SPL": split_data["SPL"]}
    return results


def plot(results: dict[int, dict], output: str | None = None) -> None:
    num_histories = sorted(results.keys())
    sr_values = [results[h]["SR"] for h in num_histories]
    spl_values = [results[h]["SPL"] for h in num_histories]

    x = np.arange(len(num_histories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(num_histories) * 1.5), 5))

    bars_sr = ax.bar(x - width / 2, sr_values, width, label="SR", color="#4C72B0", edgecolor="white")
    bars_spl = ax.bar(x + width / 2, spl_values, width, label="SPL", color="#DD8452", edgecolor="white")

    # Value labels on top of each bar
    for bar in bars_sr:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars_spl:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("num_history", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("SR and SPL vs. num_history", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in num_histories])
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()

    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        plt.savefig(output, dpi=150)
        print(f"Saved to {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default=None, help="Save path (e.g. figures/ablation.png)")
    args = parser.parse_args()

    results = load_results()
    print("Loaded results:")
    for h in sorted(results):
        r = results[h]
        print(f"  num_history={h}: SR={r['SR']:.4f}, SPL={r['SPL']:.4f}")

    plot(results, output=args.output)


if __name__ == "__main__":
    main()
