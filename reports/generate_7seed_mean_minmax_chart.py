from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    reports = root / "reports"
    fig_dir = reports / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    seeds = [118, 119, 120, 121, 122, 123, 124]
    per_seed = []
    for s in seeds:
        p = reports / f"vgg16_cbam_seed{s}_metrics_summary.json"
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        f1_la_sau = float(d["f1_per_class"]["la_sau"])
        macro_f1 = float(d["macro_f1"])
        tl = 0.65 * f1_la_sau + 0.35 * macro_f1
        per_seed.append(
            {
                "accuracy": float(d["accuracy"]),
                "macro_f1": macro_f1,
                "recall_la_sau": float(d["recall_la_sau"]),
                "precision_la_sau": float(d["precision_la_sau"]),
                "tl_score": tl,
            }
        )

    metric_keys = ["accuracy", "macro_f1", "recall_la_sau", "precision_la_sau", "tl_score"]
    metric_labels = ["Accuracy", "Macro-F1", "Recall la_sau", "Precision la_sau", "TL score"]

    means, mins, maxs = [], [], []
    for k in metric_keys:
        vals = np.asarray([r[k] for r in per_seed], dtype=np.float64)
        means.append(float(vals.mean()))
        mins.append(float(vals.min()))
        maxs.append(float(vals.max()))

    x = np.arange(len(metric_labels))
    means_arr = np.asarray(means)
    lower = means_arr - np.asarray(mins)
    upper = np.asarray(maxs) - means_arr

    fig, ax = plt.subplots(figsize=(11, 5.4), dpi=150)
    bars = ax.bar(x, means_arr, color="#4E79A7", alpha=0.9, width=0.62, label="Trung bình (7 seed)")
    ax.errorbar(
        x,
        means_arr,
        yerr=[lower, upper],
        fmt="none",
        ecolor="#D62728",
        elinewidth=1.8,
        capsize=6,
        label="Khoảng min-max",
    )

    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"TB {means[i]:.4f}", ha="center", va="bottom", fontsize=9)
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() - 0.04, f"min {mins[i]:.4f}\nmax {maxs[i]:.4f}", ha="center", va="top", fontsize=8, color="#222")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=0)
    ax.set_ylim(0.55, 0.95)
    ax.set_ylabel("Score")
    ax.set_title("VGG16+CBAM (7 seed): Trung bình kèm min-max")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()

    out = fig_dir / "14_vgg16_cbam_7seed_mean_minmax_metrics.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out}")


if __name__ == "__main__":
    main()
