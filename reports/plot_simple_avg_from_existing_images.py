from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _extract_curve_y(region_rgb: np.ndarray, color: str) -> np.ndarray:
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
        idx = np.where(mask[:, x])[0]
        if idx.size:
            ys[x] = float(np.median(idx))
    valid = np.where(~np.isnan(ys))[0]
    if valid.size < 2:
        raise RuntimeError(f"Không trích được đường {color}.")
    xs = np.arange(w, dtype=np.float64)
    ys = np.interp(xs, valid.astype(np.float64), ys[valid])
    return ys


def _normalize_top1_bottom0(ys: np.ndarray, h: int) -> np.ndarray:
    return 1.0 - (ys / float(max(h - 1, 1)))


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    fig_dir = root / "reports" / "figures"
    seeds = [118, 119, 120, 121, 122, 123, 124]
    paths = [fig_dir / f"vgg16_cbam_seed{s}_04_loss_accuracy.png" for s in seeds]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Thiếu ảnh seed:\n" + "\n".join(missing))

    # Crop regions matching existing chart layout (1665x657)
    left = (70, 36, 790, 590)   # loss
    right = (875, 36, 1595, 590)  # accuracy

    loss_train, loss_val, acc_train, acc_val = [], [], [], []
    for p in paths:
        arr = np.asarray(Image.open(p).convert("RGB"))
        lx1, ly1, lx2, ly2 = left
        rx1, ry1, rx2, ry2 = right
        loss_region = arr[ly1:ly2, lx1:lx2, :]
        acc_region = arr[ry1:ry2, rx1:rx2, :]

        loss_train.append(_normalize_top1_bottom0(_extract_curve_y(loss_region, "blue"), loss_region.shape[0]))
        loss_val.append(_normalize_top1_bottom0(_extract_curve_y(loss_region, "orange"), loss_region.shape[0]))
        acc_train.append(_normalize_top1_bottom0(_extract_curve_y(acc_region, "blue"), acc_region.shape[0]))
        acc_val.append(_normalize_top1_bottom0(_extract_curve_y(acc_region, "orange"), acc_region.shape[0]))

    def _stack_mean(series: list[np.ndarray]) -> np.ndarray:
        m = min(len(s) for s in series)
        arr = np.asarray([s[:m] for s in series], dtype=np.float64)
        return arr.mean(axis=0)

    loss_train_m = _stack_mean(loss_train)
    loss_val_m = _stack_mean(loss_val)
    acc_train_m = _stack_mean(acc_train)
    acc_val_m = _stack_mean(acc_val)
    n = min(len(loss_train_m), len(loss_val_m), len(acc_train_m), len(acc_val_m))
    loss_train_m, loss_val_m = loss_train_m[:n], loss_val_m[:n]
    acc_train_m, acc_val_m = acc_train_m[:n], acc_val_m[:n]
    epochs = np.linspace(1, 40, n)

    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), dpi=150)

    axes[0].plot(epochs, loss_train_m, label="train_loss", linewidth=2)
    axes[0].plot(epochs, loss_val_m, label="val_loss", linewidth=2)
    axes[0].set_title("Loss (vgg16_cbam)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, acc_train_m, label="train_acc", linewidth=2)
    axes[1].plot(epochs, acc_val_m, label="val_acc", linewidth=2)
    axes[1].set_title("Accuracy (vgg16_cbam)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    out = fig_dir / "17_vgg16_cbam_7seed_simple_avg_loss_accuracy.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out}")


if __name__ == "__main__":
    main()
