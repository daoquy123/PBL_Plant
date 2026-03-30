from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"


def _load_metrics(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _draw_architecture_resnet50(out_path: Path, with_cbam: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(13, 3.6), dpi=160)
    ax.axis("off")

    boxes = [
        "Input\n224×224×3",
        "ResNet50\n(ImageNet)\nfreeze",
    ]
    if with_cbam:
        boxes.append("CBAM\nChannel +\nSpatial")
    boxes += [
        "GAP",
        "Dense\n256 + DO",
        "Softmax\n5 classes",
    ]

    x_positions = np.linspace(0.06, 0.94, len(boxes))
    y = 0.5
    for i, (x, text) in enumerate(zip(x_positions, boxes)):
        rect = plt.Rectangle(
            (x - 0.065, y - 0.18),
            0.13,
            0.36,
            fill=True,
            facecolor="#e6eef7",
            edgecolor="#2b6cb0",
            linewidth=1.2,
            joinstyle="round",
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=9)
        if i < len(boxes) - 1:
            ax.annotate(
                "",
                xy=(x_positions[i + 1] - 0.07, y),
                xytext=(x + 0.07, y),
                arrowprops=dict(arrowstyle="->", lw=1.0, color="#4a5568"),
            )

    title = "Kiến trúc: ResNet50 backbone + classifier (PBL5)"
    if with_cbam:
        title = "Kiến trúc: ResNet50 backbone + CBAM + classifier (PBL5)"
    ax.set_title(title, fontsize=12, pad=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _draw_residual_block_explain(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 3.6), dpi=160)
    ax.axis("off")
    ax.set_title("Residual Connection trong ResNet50", fontsize=12, pad=10)

    # Main path
    ax.text(0.08, 0.5, "x", fontsize=12, weight="bold")
    ax.annotate("", xy=(0.2, 0.5), xytext=(0.1, 0.5), arrowprops=dict(arrowstyle="->"))
    ax.text(0.24, 0.5, "F(x)\n(Conv-BN-ReLU blocks)", ha="center", va="center", fontsize=10)
    ax.annotate("", xy=(0.38, 0.5), xytext=(0.29, 0.5), arrowprops=dict(arrowstyle="->"))

    # Skip path
    ax.annotate("", xy=(0.38, 0.72), xytext=(0.1, 0.72), arrowprops=dict(arrowstyle="->", linestyle="--"))
    ax.text(0.24, 0.76, "Skip / Identity", ha="center", fontsize=9, color="#4a5568")
    ax.annotate("", xy=(0.38, 0.53), xytext=(0.38, 0.70), arrowprops=dict(arrowstyle="->"))

    # Add and output
    circle = plt.Circle((0.43, 0.5), 0.04, edgecolor="#2d3748", facecolor="#edf2f7", lw=1.2)
    ax.add_patch(circle)
    ax.text(0.43, 0.5, "+", ha="center", va="center", fontsize=12, weight="bold")
    ax.annotate("", xy=(0.55, 0.5), xytext=(0.47, 0.5), arrowprops=dict(arrowstyle="->"))
    ax.text(0.6, 0.5, "y = F(x) + x", fontsize=11, weight="bold")
    ax.text(0.6, 0.36, "Giúp gradient truyền tốt hơn,\ntrain mạng sâu ổn định hơn.", fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _draw_metric_comparison(vgg: dict, res: dict, out_path: Path) -> None:
    metrics = ["accuracy", "macro_f1", "recall_la_sau", "precision_la_sau"]
    labels = ["Accuracy", "Macro F1", "Recall la_sau", "Precision la_sau"]
    vgg_vals = [float(vgg[k]) for k in metrics]
    res_vals = [float(res[k]) for k in metrics]

    x = np.arange(len(metrics))
    w = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=160)
    b1 = ax.bar(x - w / 2, vgg_vals, w, label="VGG16+CBAM", color="#4C78A8")
    b2 = ax.bar(x + w / 2, res_vals, w, label="ResNet50", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("So sánh chỉ số test: VGG16+CBAM vs ResNet50")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _write_summary_md(vgg: dict, res: dict, out_path: Path) -> None:
    delta_acc = res["accuracy"] - vgg["accuracy"]
    delta_macro_f1 = res["macro_f1"] - vgg["macro_f1"]
    delta_recall = res["recall_la_sau"] - vgg["recall_la_sau"]
    delta_precision = res["precision_la_sau"] - vgg["precision_la_sau"]

    text = f"""# So sánh kiến trúc và kết quả: VGG16+CBAM vs ResNet50

## 1) Khác nhau về kiến trúc

- `VGG16+CBAM`: backbone VGG16 + attention CBAM (channel + spatial) + classifier.
- `ResNet50`: backbone sâu hơn, có residual connections giúp tối ưu ổn định hơn.
- `ResNet50+CBAM` (tuỳ chọn): kết hợp ưu điểm residual + attention.

## 2) Khác nhau về kết quả test (314 ảnh)

| Metric | VGG16+CBAM | ResNet50 | Delta (ResNet - VGG) |
|---|---:|---:|---:|
| Accuracy | {vgg["accuracy"]:.4f} | {res["accuracy"]:.4f} | {delta_acc:+.4f} |
| Macro F1 | {vgg["macro_f1"]:.4f} | {res["macro_f1"]:.4f} | {delta_macro_f1:+.4f} |
| Recall la_sau | {vgg["recall_la_sau"]:.4f} | {res["recall_la_sau"]:.4f} | {delta_recall:+.4f} |
| Precision la_sau | {vgg["precision_la_sau"]:.4f} | {res["precision_la_sau"]:.4f} | {delta_precision:+.4f} |

## 3) Diễn giải thực tế

- Nếu mục tiêu là **báo động không bỏ sót sâu**: VGG16+CBAM tốt hơn do recall `la_sau` cao hơn.
- Nếu mục tiêu là **độ chính xác tổng thể và giảm báo động giả sâu**: ResNet50 tốt hơn do accuracy/macro-F1/precision cao hơn.
- Với dữ liệu hiện tại, chọn mô hình phụ thuộc vào ưu tiên nghiệp vụ.

## 4) Hình minh hoạ đi kèm

- `reports/figures/08_resnet50_architecture.png`
- `reports/figures/09_resnet50_cbam_architecture.png`
- `reports/figures/10_resnet50_residual_block.png`
- `reports/figures/11_vgg_vs_resnet_metrics.png`
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")


def main() -> None:
    vgg_path = REPORTS_DIR / "vgg16_cbam_metrics_summary.json"
    res_path = REPORTS_DIR / "resnet50_metrics_summary.json"
    if not vgg_path.is_file() or not res_path.is_file():
        raise FileNotFoundError("Thiếu file metrics summary để vẽ so sánh.")

    vgg = _load_metrics(vgg_path)
    res = _load_metrics(res_path)

    _draw_architecture_resnet50(FIG_DIR / "08_resnet50_architecture.png", with_cbam=False)
    _draw_architecture_resnet50(FIG_DIR / "09_resnet50_cbam_architecture.png", with_cbam=True)
    _draw_residual_block_explain(FIG_DIR / "10_resnet50_residual_block.png")
    _draw_metric_comparison(vgg, res, FIG_DIR / "11_vgg_vs_resnet_metrics.png")
    _write_summary_md(vgg, res, REPORTS_DIR / "RESNET50_VS_VGG16CBAM.md")
    print("Generated ResNet50 comparison assets.")


if __name__ == "__main__":
    main()
