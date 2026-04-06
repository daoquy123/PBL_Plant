from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    src = root / "reports" / "multi_seed_tl_summary_7seeds.json"
    out = root / "reports" / "figures" / "19_experimental_significance_7seed.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(src, encoding="utf-8") as f:
        payload = json.load(f)

    summary = payload["summary_mean_std"]["vgg16_cbam"]
    metrics = [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
        ("recall_la_sau", "Recall la_sau"),
        ("precision_la_sau", "Precision la_sau"),
        ("tl_score", "TL score"),
    ]

    means = np.array([summary[k]["mean"] for k, _ in metrics], dtype=np.float64)
    stds = np.array([summary[k]["std"] for k, _ in metrics], dtype=np.float64)

    per_seed = payload["per_seed_metrics"]["vgg16_cbam"]
    metric_values = {
        k: np.array([per_seed[str(s)][k] for s in payload["seeds"]], dtype=np.float64)
        for k, _ in metrics
    }
    mins = np.array([metric_values[k].min() for k, _ in metrics], dtype=np.float64)
    maxs = np.array([metric_values[k].max() for k, _ in metrics], dtype=np.float64)
    cv = (stds / np.maximum(means, 1e-12)) * 100.0

    x = np.arange(len(metrics))
    labels = [name for _, name in metrics]

    plt.style.use("ggplot")
    fig = plt.figure(figsize=(13, 6), dpi=160)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.0], wspace=0.25)

    ax = fig.add_subplot(gs[0, 0])
    bars = ax.bar(x, means, color="#4E79A7", width=0.62, label="Mean (7 seeds)")
    ax.errorbar(
        x,
        means,
        yerr=stds,
        fmt="none",
        ecolor="#D62728",
        elinewidth=2,
        capsize=6,
        label="± Std",
        zorder=5,
    )
    ax.errorbar(
        x,
        means,
        yerr=[means - mins, maxs - means],
        fmt="none",
        ecolor="#2CA02C",
        elinewidth=1.5,
        capsize=4,
        alpha=0.75,
        label="Min-Max range",
        zorder=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(0.58, 0.92)
    ax.set_ylabel("Score")
    ax.set_title("Experimental Significance (VGG16+CBAM, 7 seeds)")
    ax.legend(loc="lower right", fontsize=8, frameon=True)

    for i, b in enumerate(bars):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.008,
            f"{means[i]*100:.2f}% ± {stds[i]*100:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#1f1f1f",
        )

    # Right panel: compact stats table
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    ax2.set_title("Stability Summary", fontsize=11, pad=8)

    table_lines = ["Metric                 Mean±Std        Min-Max        CV%"]
    for i, (_, name) in enumerate(metrics):
        line = f"{name[:20]:20} {means[i]*100:5.2f}±{stds[i]*100:4.2f}   {mins[i]*100:5.2f}-{maxs[i]*100:5.2f}   {cv[i]:4.2f}"
        table_lines.append(line)

    ax2.text(
        0.02,
        0.98,
        "\n".join(table_lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=8.5,
        color="#222222",
        transform=ax2.transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f7f7f7", edgecolor="#cccccc"),
    )
    ax2.text(
        0.02,
        0.08,
        "Low std and low CV indicate\nhigh reproducibility across seeds.",
        va="bottom",
        ha="left",
        fontsize=8.5,
        color="#333333",
        transform=ax2.transAxes,
    )

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out}")


if __name__ == "__main__":
    main()
