"""Plot measured MovieLens ranking metrics without exposing source ratings."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark import ROOT

LABELS = {"popularity": "Popularity", "cosine": "Item cosine", "pagerank": "PageRank"}
COLORS = ["#55749e", "#139279", "#b27450"]
INK = "#16394b"


def canvas(title, note, *, paired=False):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5)) if paired else plt.subplots(figsize=(11, 5))
    fig.subplots_adjust(left=.105, right=.96, top=.77, bottom=.20, wspace=.34)
    fig.set_facecolor("#fbfcfd")
    fig.text(.045, .92, title, fontsize=18, weight="bold", color=INK)
    fig.text(.045, .045, note, fontsize=9, color="#4d6473")
    for ax in np.atleast_1d(axes):
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.2)
    return fig, axes


def save(fig, out):
    fig.savefig(out, dpi=175)
    plt.close(fig)


def main(report, out):
    data = json.loads(report.read_bytes())
    test = data["test"]
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = canvas("Ranking accuracy across the two temporal cutoffs",
                       "MovieLens latest-small · first rating ≥4 per user · every known unseen film is a candidate", paired=True)
    for ax, field, title in zip(axes, ("recall_at_10", "ndcg_at_10"), ("Recall@10", "nDCG@10")):
        for offset, (name, color) in enumerate(zip(LABELS, COLORS)):
            values = [data[period]["methods"][name]["all"][field] for period in ("validation", "test")]
            bars = ax.bar(np.arange(2) + (offset - 1) * .22, values, width=.19, label=LABELS[name], color=color)
            ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
        ax.set(xticks=[0, 1], xticklabels=["Validation", "Test"], ylim=(0, .47), title=title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(.97, .88), ncol=3, frameon=False)
    save(fig, out / "temporal-ranking-quality.png")

    fig, ax = canvas("Recommendation coverage stays narrow",
                     f"Share of {test['catalog']:,} candidate films recommended to at least one of {test['targets']} test users")
    values = [100 * test["methods"][name]["catalog_coverage"] for name in LABELS]
    bars = ax.bar(list(range(3)), values, width=.63, color=COLORS)
    ax.bar_label(bars, labels=[f"{val:.2f}% · {test['methods'][method]['unique_recommended']} films"
                               for method, val in zip(LABELS, values)], padding=5)
    ax.set(xticks=list(range(3)), xticklabels=list(LABELS.values()), ylabel="Catalog coverage (%)",
           ylim=(0, max(values) * 1.27))
    save(fig, out / "recommendation-coverage.png")

    fig, axes = canvas("Cold-start cases dominate the held-out user sample",
                       "MovieLens latest-small · test cutoff 19 Nov 2016 · 12 targets also enter the catalog after cutoff", paired=True)
    names = ["new_user", "1–20", "21+"]
    ticks = ["No history", "1–20 ratings", "21+ ratings"]
    activity = test["methods"]["popularity"]["by_activity"]
    totals = [activity.get(name, {}).get("users", 0) for name in names]
    bars = axes[0].bar(range(3), totals, width=.64, color="#587998")
    axes[0].bar_label(bars, padding=4)
    axes[0].set(xticks=range(3), xticklabels=ticks, ylabel="Test users", ylim=(0, max(totals)*1.25))
    for offset, (name, color) in enumerate(zip(LABELS, COLORS)):
        values = [test["methods"][name]["by_activity"].get(group, {}).get("recall_at_10", 0) for group in names]
        axes[1].bar(np.arange(3) + (offset - 1)*.22, values, width=.18, color=color, label=LABELS[name])
    axes[1].set(xticks=range(3), xticklabels=ticks, ylabel="Recall@10", ylim=(0, .65))
    axes[1].legend(ncol=3, loc="lower center", bbox_to_anchor=(.5, 1.01), frameon=False, fontsize=8)
    save(fig, out / "cold-start-breakdown.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "runs/moviebench/metrics.json")
    args = parser.parse_args()
    main(args.report, ROOT / "figures/results")
