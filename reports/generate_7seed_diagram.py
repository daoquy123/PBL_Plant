from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
SRC_JSON = REPORTS / "multi_seed_tl_summary.json"
OUT_JSON = REPORTS / "multi_seed_tl_summary_7seeds.json"
OUT_MD = REPORTS / "MULTI_SEED_TL_SUMMARY_7SEEDS.md"
OUT_FIG = REPORTS / "figures" / "12_vgg16_cbam_7seed_tl_score.png"


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0)


def main() -> None:
    with open(SRC_JSON, encoding="utf-8") as f:
        payload = json.load(f)

    runs = payload["per_seed_metrics"]["vgg16_cbam"]
    keep_seeds = [118, 119, 120, 121, 122, 123, 124]
    filtered_runs = {str(s): runs[str(s)] for s in keep_seeds if str(s) in runs}

    metric_names = list(next(iter(filtered_runs.values())).keys())
    summary: dict[str, dict[str, float]] = {}
    for metric in metric_names:
        vals = [filtered_runs[str(s)][metric] for s in keep_seeds]
        m, s = mean_std(vals)
        summary[metric] = {"mean": m, "std": s}

    new_payload = {
        "models": ["vgg16_cbam"],
        "seeds": keep_seeds,
        "tl_score_formula": payload.get("tl_score_formula", "0.65 * f1_la_sau + 0.35 * macro_f1"),
        "per_seed_metrics": {"vgg16_cbam": filtered_runs},
        "summary_mean_std": {"vgg16_cbam": summary},
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(new_payload, f, indent=2, ensure_ascii=False)

    md = []
    md.append("# Multi-seed TL summary (7 seeds)\n")
    md.append(f"Seeds: {keep_seeds}\n")
    md.append(f"TL score formula: `{new_payload['tl_score_formula']}`\n")
    md.append(
        "| Model | Accuracy TB(std) | Macro-F1 TB(std) | Recall la_sau TB(std) | Precision la_sau TB(std) | TL score TB(std) |\n"
    )
    md.append("|---|---:|---:|---:|---:|---:|\n")
    md.append(
        "| vgg16_cbam | {acc_m:.4f} ({acc_s:.4f}) | {mf1_m:.4f} ({mf1_s:.4f}) | {rec_m:.4f} ({rec_s:.4f}) | {pre_m:.4f} ({pre_s:.4f}) | {tl_m:.4f} ({tl_s:.4f}) |\n".format(
            acc_m=summary["accuracy"]["mean"],
            acc_s=summary["accuracy"]["std"],
            mf1_m=summary["macro_f1"]["mean"],
            mf1_s=summary["macro_f1"]["std"],
            rec_m=summary["recall_la_sau"]["mean"],
            rec_s=summary["recall_la_sau"]["std"],
            pre_m=summary["precision_la_sau"]["mean"],
            pre_s=summary["precision_la_sau"]["std"],
            tl_m=summary["tl_score"]["mean"],
            tl_s=summary["tl_score"]["std"],
        )
    )
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("".join(md))

    seeds = np.asarray(keep_seeds, dtype=int)
    tl_scores = np.asarray([filtered_runs[str(s)]["tl_score"] for s in keep_seeds], dtype=np.float64)
    tl_mean = summary["tl_score"]["mean"]
    tl_std = summary["tl_score"]["std"]

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=140)
    ax.plot(seeds, tl_scores, marker="o", linewidth=2, color="#2E6FBA", label="TL score theo seed")
    for x, y in zip(seeds, tl_scores):
        ax.text(x, y + 0.004, f"{y:.4f}", ha="center", va="bottom", fontsize=8, color="#1f4e8c")
    max_idx = int(np.argmax(tl_scores))
    min_idx = int(np.argmin(tl_scores))
    x_max, y_max = int(seeds[max_idx]), float(tl_scores[max_idx])
    x_min, y_min = int(seeds[min_idx]), float(tl_scores[min_idx])
    ax.scatter([x_max], [y_max], color="#d62728", s=52, zorder=5, label="MAX")
    ax.scatter([x_min], [y_min], color="#2ca02c", s=52, zorder=5, label="MIN")
    ax.annotate(
        f"MAX: seed {x_max} = {y_max:.4f}",
        xy=(x_max, y_max),
        xytext=(x_max + 0.15, y_max + 0.018),
        fontsize=9,
        color="#b22222",
        arrowprops=dict(arrowstyle="->", color="#b22222", lw=1.0),
    )
    ax.annotate(
        f"MIN: seed {x_min} = {y_min:.4f}",
        xy=(x_min, y_min),
        xytext=(x_min + 0.15, y_min - 0.026),
        fontsize=9,
        color="#1f7a1f",
        arrowprops=dict(arrowstyle="->", color="#1f7a1f", lw=1.0),
    )
    ax.axhline(tl_mean, color="#D62728", linestyle="--", linewidth=1.8, label=f"TB = {tl_mean:.4f}")
    ax.fill_between(seeds, tl_mean - tl_std, tl_mean + tl_std, color="#D62728", alpha=0.15, label=f"TB ± std ({tl_std:.4f})")
    ax.set_title("VGG16+CBAM - TL score cho 7 seed")
    ax.set_xlabel("Seed")
    ax.set_ylabel("TL score")
    ax.set_xticks(seeds)
    ax.set_ylim(0.70, max(0.86, float(tl_scores.max() + 0.03)))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {OUT_JSON}")
    print(f"[SAVED] {OUT_MD}")
    print(f"[SAVED] {OUT_FIG}")


if __name__ == "__main__":
    main()
