from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from image_io import load_image_rgb_from_path
from model_resnet50_cbam import (
    CLASS_NAMES as RESNET_CLASS_NAMES,
    IMG_SIZE as RESNET_IMG_SIZE,
    build_resnet50_model,
)
from model_vgg16_cbam import (
    CLASS_NAMES as VGG_CLASS_NAMES,
    IMG_SIZE as VGG_IMG_SIZE,
    build_vgg16_cbam_model,
)
from reporting import compute_gradcam, load_training_history, upsample_heatmap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT.parent / "dataset"
TEST_DIR = DATA_ROOT / "test"
VAL_DIR = DATA_ROOT / "val"
FIG_DIR_DEFAULT = PROJECT_ROOT.parent / "reports" / "figures"
REPORTS_DIR_DEFAULT = PROJECT_ROOT.parent / "reports"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff", ".heic", ".heif"}


def _build_model(model_name: str) -> tuple[tf.keras.Model, list[str], tuple[int, int]]:
    if model_name == "vgg16_cbam":
        return build_vgg16_cbam_model(), VGG_CLASS_NAMES, VGG_IMG_SIZE
    if model_name == "resnet50":
        return build_resnet50_model(use_cbam=False), RESNET_CLASS_NAMES, RESNET_IMG_SIZE
    if model_name == "resnet50_cbam":
        return build_resnet50_model(use_cbam=True), RESNET_CLASS_NAMES, RESNET_IMG_SIZE
    raise ValueError(f"Model không hỗ trợ: {model_name}")


def _collect_split_paths(split_dir: Path, class_names: list[str]) -> tuple[list[Path], list[int]]:
    paths: list[Path] = []
    labels: list[int] = []
    for idx, cls in enumerate(class_names):
        cls_dir = split_dir / cls
        if not cls_dir.is_dir():
            continue
        files = sorted([p for p in cls_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
        paths.extend(files)
        labels.extend([idx] * len(files))
    return paths, labels


def _predict_on_split(
    model: tf.keras.Model,
    split_dir: Path,
    class_names: list[str],
    img_size: tuple[int, int],
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Path]]:
    paths, y_true_list = _collect_split_paths(split_dir, class_names)
    if not paths:
        raise FileNotFoundError(f"Không tìm thấy ảnh trong: {split_dir}")

    y_pred_list: list[int] = []
    y_prob_list: list[np.ndarray] = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        batch_np = [load_image_rgb_from_path(p, img_size) for p in batch_paths]
        x = tf.convert_to_tensor(np.stack(batch_np, axis=0), dtype=tf.float32)
        preds = model.predict(x, verbose=0)
        y_prob_list.append(preds)
        y_pred_list.extend(np.argmax(preds, axis=-1).tolist())
    y_prob = np.concatenate(y_prob_list, axis=0)
    return np.asarray(y_true_list, dtype=np.int32), np.asarray(y_pred_list, dtype=np.int32), y_prob, paths


def _apply_la_sau_threshold(
    probs: np.ndarray,
    la_sau_idx: int,
    threshold: float,
) -> np.ndarray:
    preds = np.argmax(probs, axis=-1)
    forced = probs[:, la_sau_idx] >= threshold
    preds = np.where(forced, la_sau_idx, preds)
    return preds


def _choose_best_threshold_on_val(
    y_true: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
    min_recall: float,
) -> tuple[float, np.ndarray]:
    la_sau_idx = class_names.index("la_sau")
    best_t = 0.5
    best_score = -1.0
    best_pred = np.argmax(probs, axis=-1)

    for t in np.arange(0.35, 0.91, 0.01):
        y_pred = _apply_la_sau_threshold(probs, la_sau_idx, float(t))
        p, r, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            zero_division=0,
        )
        la_sau_recall = float(r[la_sau_idx])
        la_sau_f1 = float(f1[la_sau_idx])
        macro_f1 = float(np.mean(f1))
        if la_sau_recall < min_recall:
            continue
        score = 0.65 * la_sau_f1 + 0.35 * macro_f1
        if score > best_score:
            best_score = score
            best_t = float(t)
            best_pred = y_pred

    return best_t, best_pred


def _plot_loss_accuracy(history: dict[str, list[float]], out_path: Path, title_suffix: str = "") -> None:
    loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])
    acc = history.get("accuracy", [])
    val_acc = history.get("val_accuracy", [])
    if not loss:
        print("Không có history loss/accuracy để vẽ.")
        return

    n = len(loss)
    epochs = np.arange(1, n + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=140)
    axes[0].plot(epochs, loss, label="train_loss", linewidth=2)
    if val_loss:
        axes[0].plot(epochs[: len(val_loss)], val_loss, label="val_loss", linewidth=2)
    axes[0].set_title(f"Loss{title_suffix}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs[: len(acc)], acc, label="train_acc", linewidth=2)
    if val_acc:
        axes[1].plot(epochs[: len(val_acc)], val_acc, label="val_acc", linewidth=2)
    axes[1].set_title(f"Accuracy{title_suffix}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.6), dpi=140)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True, xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _plot_f1_per_class(f1_per_class: np.ndarray, class_names: list[str], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=140)
    bars = ax.bar(class_names, f1_per_class, color="#4C78A8")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("F1-score")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, f1_per_class):
        ax.text(bar.get_x() + bar.get_width() / 2.0, min(0.98, val + 0.02), f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _pick_gradcam_samples(paths: list[Path], labels: np.ndarray, class_names: list[str], per_class: int, seed: int) -> list[tuple[Path, int]]:
    rng = random.Random(seed)
    picked: list[tuple[Path, int]] = []
    by_class: dict[int, list[int]] = {i: [] for i in range(len(class_names))}
    for i, y in enumerate(labels.tolist()):
        by_class[y].append(i)
    for cls_idx in range(len(class_names)):
        ids = by_class.get(cls_idx, [])
        if not ids:
            continue
        rng.shuffle(ids)
        for id_ in ids[:per_class]:
            picked.append((paths[id_], cls_idx))
    return picked


def _plot_gradcam_grid(
    model: tf.keras.Model,
    samples: list[tuple[Path, int]],
    class_names: list[str],
    img_size: tuple[int, int],
    out_path: Path,
) -> None:
    if not samples:
        print("Không có mẫu để vẽ Grad-CAM.")
        return
    cols = 4
    rows = int(np.ceil(len(samples) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.4 * rows), dpi=140)
    axes = np.array(axes).reshape(rows, cols)

    for k in range(rows * cols):
        r, c = divmod(k, cols)
        ax = axes[r, c]
        if k >= len(samples):
            ax.axis("off")
            continue
        path, true_idx = samples[k]
        img = load_image_rgb_from_path(path, img_size)
        x = tf.convert_to_tensor(np.expand_dims(img, axis=0), dtype=tf.float32)
        preds = model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        heat = compute_gradcam(model, x, pred_index=pred_idx)
        heat_up = upsample_heatmap(heat, size=img_size)
        ax.imshow((img / 255.0).clip(0, 1))
        ax.imshow(heat_up, cmap="jet", alpha=0.4)
        ok = "OK" if pred_idx == true_idx else "WRONG"
        ax.set_title(f"T:{class_names[true_idx]} | P:{class_names[pred_idx]} ({ok})", fontsize=9)
        ax.axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def run(args: argparse.Namespace) -> None:
    model, class_names, img_size = _build_model(args.model)
    if class_names != VGG_CLASS_NAMES:
        raise ValueError("CLASS_NAMES của model không khớp bộ nhãn chuẩn.")

    weights_path = Path(args.weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy weights: {weights_path}")
    model.load_weights(str(weights_path))
    print(f"Loaded weights: {weights_path}")

    fig_dir = Path(args.fig_dir)
    reports_dir = Path(args.reports_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    history = load_training_history(args.history) if args.history else None
    suffix = f" ({args.model})"
    if history:
        out_hist = fig_dir / (f"{args.prefix}04_loss_accuracy.png" if args.prefix else "04_loss_accuracy.png")
        _plot_loss_accuracy(history, out_hist, title_suffix=suffix)
    else:
        print("Không có history JSON, bỏ qua biểu đồ loss/accuracy.")

    y_true_val, _, probs_val, _ = _predict_on_split(
        model,
        VAL_DIR,
        class_names,
        img_size,
        batch_size=args.batch_size,
    )
    best_threshold, _ = _choose_best_threshold_on_val(
        y_true_val,
        probs_val,
        class_names,
        min_recall=args.min_recall_la_sau,
    )
    print(f"Best la_sau threshold from val: {best_threshold:.2f}")

    y_true, y_pred_argmax, probs_test, paths = _predict_on_split(
        model,
        TEST_DIR,
        class_names,
        img_size,
        batch_size=args.batch_size,
    )
    y_pred = _apply_la_sau_threshold(probs_test, class_names.index("la_sau"), best_threshold)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    macro_f1 = float(np.mean(f1))
    acc = float(np.mean(y_true == y_pred))

    out_cm = fig_dir / (f"{args.prefix}05_confusion_matrix.png" if args.prefix else "05_confusion_matrix.png")
    _plot_confusion_matrix(cm, class_names, out_cm, title=f"Confusion Matrix Test{suffix}")

    out_f1 = fig_dir / (f"{args.prefix}06_f1_per_class.png" if args.prefix else "06_f1_per_class.png")
    _plot_f1_per_class(f1, class_names, out_f1, title=f"F1-score theo lớp (test){suffix}")

    samples = _pick_gradcam_samples(paths, y_true, class_names, per_class=args.gradcam_per_class, seed=args.seed)
    out_gc = fig_dir / (f"{args.prefix}07_gradcam_samples.png" if args.prefix else "07_gradcam_samples.png")
    _plot_gradcam_grid(model, samples, class_names, img_size, out_gc)

    cls_report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    # Lưu thêm baseline argmax để so sánh trước/sau tune threshold.
    p0, r0, f10, _ = precision_recall_fscore_support(
        y_true,
        y_pred_argmax,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    stats = {
        "model": args.model,
        "weights": str(weights_path),
        "history": str(args.history) if args.history else None,
        "la_sau_threshold": best_threshold,
        "test_samples": int(len(y_true)),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "recall_la_sau": float(recall[class_names.index("la_sau")]),
        "precision_la_sau": float(precision[class_names.index("la_sau")]),
        "argmax_macro_f1": float(np.mean(f10)),
        "argmax_recall_la_sau": float(r0[class_names.index("la_sau")]),
        "argmax_precision_la_sau": float(p0[class_names.index("la_sau")]),
        "f1_per_class": {class_names[i]: float(f1[i]) for i in range(len(class_names))},
        "support_per_class": {class_names[i]: int(support[i]) for i in range(len(class_names))},
    }

    json_out = reports_dir / (f"{args.prefix}metrics_summary.json" if args.prefix else "metrics_summary.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Saved: {json_out}")

    txt_out = reports_dir / (f"{args.prefix}classification_report.txt" if args.prefix else "classification_report.txt")
    with open(txt_out, "w", encoding="utf-8") as f:
        f.write(cls_report)
        f.write("\n")
        f.write(f"\nAccuracy: {acc:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n")
    print(f"Saved: {txt_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vẽ biểu đồ báo cáo đầy đủ tương tự notebook.")
    parser.add_argument("--model", type=str, default="resnet50", choices=["vgg16_cbam", "resnet50", "resnet50_cbam"])
    parser.add_argument("--weights", type=str, default=str(CHECKPOINT_DIR / "resnet50_best.weights.h5"))
    parser.add_argument("--history", type=str, default=str(CHECKPOINT_DIR / "resnet50_training_history.json"))
    parser.add_argument("--fig-dir", type=str, default=str(FIG_DIR_DEFAULT))
    parser.add_argument("--reports-dir", type=str, default=str(REPORTS_DIR_DEFAULT))
    parser.add_argument("--prefix", type=str, default="", help="Prefix file output, ví dụ: resnet50_")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradcam-per-class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--min-recall-la-sau", type=float, default=0.75)
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    args = parse_args()
    run(args)
