from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _extract_curve_y(region_rgb: np.ndarray, color: str) -> np.ndarray:
    """Extract one colored curve as y-pixel for each x-column."""
    r = region_rgb[:, :, 0].astype(np.int16)
    g = region_rgb[:, :, 1].astype(np.int16)
    b = region_rgb[:, :, 2].astype(np.int16)

    if color == "blue":
        mask = (b > 95) & (b - r > 18) & (b - g > 10)
    elif color == "orange":
        mask = (r > 145) & (g > 90) & (b < 110) & (r - b > 45)
    else:
        raise ValueError("Unsupported color")

    h, w = mask.shape
    ys = np.full(w, np.nan, dtype=np.float64)
    for x in range(w):
        y_idx = np.where(mask[:, x])[0]
        if y_idx.size > 0:
            ys[x] = float(np.median(y_idx))

    valid = np.where(~np.isnan(ys))[0]
    if valid.size < 2:
        raise RuntimeError(f"Không trích được đường {color} từ ảnh.")
    xs = np.arange(w, dtype=np.float64)
    ys = np.interp(xs, valid.astype(np.float64), ys[valid])
    return ys


def _to_relative_score(ys: np.ndarray, h: int) -> np.ndarray:
    # Top=1.0, bottom=0.0 (relative inside plot area)
    return 1.0 - (ys / float(max(h - 1, 1)))


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    fig_dir = root / "reports" / "figures"
    seeds = [118, 119, 120, 121, 122, 123, 124]
    paths = [fig_dir / f"vgg16_cbam_seed{s}_04_loss_accuracy.png" for s in seeds]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Thiếu ảnh seed:\n" + "\n".join(missing))

    # Region of two subplots in current chart layout (size ~1665x657)
    left = (70, 36, 790, 590)   # x1,y1,x2,y2 (loss)
    right = (875, 36, 1595, 590)  # (accuracy)

    loss_train_all, loss_val_all = [], []
    acc_train_all, acc_val_all = [], []

    for p in paths:
        arr = np.asarray(Image.open(p).convert("RGB"))
        lx1, ly1, lx2, ly2 = left
        rx1, ry1, rx2, ry2 = right
        loss_reg = arr[ly1:ly2, lx1:lx2, :]
        acc_reg = arr[ry1:ry2, rx1:rx2, :]

        loss_train_y = _extract_curve_y(loss_reg, "blue")
        loss_val_y = _extract_curve_y(loss_reg, "orange")
        acc_train_y = _extract_curve_y(acc_reg, "blue")
        acc_val_y = _extract_curve_y(acc_reg, "orange")

        loss_train_all.append(_to_relative_score(loss_train_y, loss_reg.shape[0]))
        loss_val_all.append(_to_relative_score(loss_val_y, loss_reg.shape[0]))
        acc_train_all.append(_to_relative_score(acc_train_y, acc_reg.shape[0]))
        acc_val_all.append(_to_relative_score(acc_val_y, acc_reg.shape[0]))

    def _stack(x: list[np.ndarray]) -> np.ndarray:
        min_w = min(v.shape[0] for v in x)
        return np.asarray([v[:min_w] for v in x], dtype=np.float64)

    ltr = _stack(loss_train_all)
    lva = _stack(loss_val_all)
    atr = _stack(acc_train_all)
    ava = _stack(acc_val_all)

    n = min(ltr.shape[1], lva.shape[1], atr.shape[1], ava.shape[1])
    ltr, lva, atr, ava = ltr[:, :n], lva[:, :n], atr[:, :n], ava[:, :n]
    x = np.linspace(1, 40, n)

    def _stats(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return a.mean(axis=0), a.min(axis=0), a.max(axis=0)

    ltr_m, ltr_min, ltr_max = _stats(ltr)
    lva_m, lva_min, lva_max = _stats(lva)
    atr_m, atr_min, atr_max = _stats(atr)
    ava_m, ava_min, ava_max = _stats(ava)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), dpi=150)

    ax = axes[0]
    ax.plot(x, ltr_m, color="#1f77b4", linewidth=2, label="train_loss (mean)")
    ax.fill_between(x, ltr_min, ltr_max, color="#1f77b4", alpha=0.18, label="train_loss (min-max)")
    ax.plot(x, lva_m, color="#ff7f0e", linewidth=2, label="val_loss (mean)")
    ax.fill_between(x, lva_min, lva_max, color="#ff7f0e", alpha=0.18, label="val_loss (min-max)")
    ax.set_title("Loss trung bình từ 7 sơ đồ hiện có")
    ax.set_xlabel("Epoch (xấp xỉ)")
    ax.set_ylabel("Relative score")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(x, atr_m, color="#1f77b4", linewidth=2, label="train_acc (mean)")
    ax.fill_between(x, atr_min, atr_max, color="#1f77b4", alpha=0.18, label="train_acc (min-max)")
    ax.plot(x, ava_m, color="#ff7f0e", linewidth=2, label="val_acc (mean)")
    ax.fill_between(x, ava_min, ava_max, color="#ff7f0e", alpha=0.18, label="val_acc (min-max)")
    ax.set_title("Accuracy trung bình từ 7 sơ đồ hiện có")
    ax.set_xlabel("Epoch (xấp xỉ)")
    ax.set_ylabel("Relative score")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle("VGG16+CBAM: Tổng hợp 7 seed từ ảnh hiện có (không train lại)", fontsize=12)
    fig.tight_layout()
    out = fig_dir / "16_vgg16_cbam_7seed_avg_from_existing_images.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out}")


if __name__ == "__main__":
    main()
