from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_history(path: Path) -> dict[str, list[float]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _to_np(series: list[list[float]]) -> np.ndarray:
    min_len = min(len(x) for x in series)
    arr = np.asarray([x[:min_len] for x in series], dtype=np.float64)
    return arr


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    hist_dir = root / "reports" / "histories"
    fig_dir = root / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    seeds = [118, 119, 120, 121, 122, 123, 124]
    paths = [hist_dir / f"vgg16_cbam_seed{s}_training_history.json" for s in seeds]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Thiếu history theo seed. Chạy lại multi-seed để tạo files này:\n"
            + "\n".join(missing)
        )

    hs = [_load_history(p) for p in paths]
    loss = _to_np([h["loss"] for h in hs])
    val_loss = _to_np([h["val_loss"] for h in hs])
    acc = _to_np([h["accuracy"] for h in hs])
    val_acc = _to_np([h["val_accuracy"] for h in hs])

    n = min(loss.shape[1], val_loss.shape[1], acc.shape[1], val_acc.shape[1])
    loss, val_loss, acc, val_acc = loss[:, :n], val_loss[:, :n], acc[:, :n], val_acc[:, :n]
    epochs = np.arange(1, n + 1)

    def stats(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return a.mean(axis=0), a.min(axis=0), a.max(axis=0)

    loss_m, loss_min, loss_max = stats(loss)
    vloss_m, vloss_min, vloss_max = stats(val_loss)
    acc_m, acc_min, acc_max = stats(acc)
    vacc_m, vacc_min, vacc_max = stats(val_acc)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=150)

    ax = axes[0]
    ax.plot(epochs, loss_m, label="train_loss (mean)", linewidth=2, color="#1f77b4")
    ax.fill_between(epochs, loss_min, loss_max, color="#1f77b4", alpha=0.18, label="train_loss (min-max)")
    ax.plot(epochs, vloss_m, label="val_loss (mean)", linewidth=2, color="#ff7f0e")
    ax.fill_between(epochs, vloss_min, vloss_max, color="#ff7f0e", alpha=0.18, label="val_loss (min-max)")
    ax.set_title("Loss trung bình (7 seed)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(epochs, acc_m, label="train_acc (mean)", linewidth=2, color="#1f77b4")
    ax.fill_between(epochs, acc_min, acc_max, color="#1f77b4", alpha=0.18, label="train_acc (min-max)")
    ax.plot(epochs, vacc_m, label="val_acc (mean)", linewidth=2, color="#ff7f0e")
    ax.fill_between(epochs, vacc_min, vacc_max, color="#ff7f0e", alpha=0.18, label="val_acc (min-max)")
    ax.set_title("Accuracy trung bình (7 seed)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle("VGG16+CBAM: Mean Loss/Accuracy qua 7 seed (118-124)", fontsize=12)
    fig.tight_layout()
    out = fig_dir / "15_vgg16_cbam_7seed_avg_loss_accuracy.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out}")


if __name__ == "__main__":
    main()
