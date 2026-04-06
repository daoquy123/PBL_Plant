from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    fig_dir = root / "reports" / "figures"
    reports_dir = root / "reports"
    seeds = [118, 119, 120, 121, 122, 123, 124]
    img_paths = [fig_dir / f"vgg16_cbam_seed{s}_04_loss_accuracy.png" for s in seeds]

    missing = [p for p in img_paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Thiếu file: " + ", ".join(str(m) for m in missing))

    images = [Image.open(p).convert("RGB") for p in img_paths]
    metrics_by_seed: dict[int, dict[str, float]] = {}
    history_by_seed: dict[int, dict[str, object]] = {}
    for seed in seeds:
        p = reports_dir / f"vgg16_cbam_seed{seed}_metrics_summary.json"
        with open(p, encoding="utf-8") as f:
            m = json.load(f)
        f1_la_sau = float(m["f1_per_class"]["la_sau"])
        macro_f1 = float(m["macro_f1"])
        tl = 0.65 * f1_la_sau + 0.35 * macro_f1
        metrics_by_seed[seed] = {
            "accuracy": float(m["accuracy"]),
            "macro_f1": macro_f1,
            "recall_la_sau": float(m["recall_la_sau"]),
            "precision_la_sau": float(m["precision_la_sau"]),
            "tl_score": tl,
        }
        hp = reports_dir / "histories" / f"vgg16_cbam_seed{seed}_training_history.json"
        if hp.exists():
            with open(hp, encoding="utf-8") as f:
                h = json.load(f)
            acc = [float(v) for v in h.get("accuracy", [])]
            loss = [float(v) for v in h.get("loss", [])]
            if acc and loss:
                acc_max_i = max(range(len(acc)), key=lambda i: acc[i])
                acc_min_i = min(range(len(acc)), key=lambda i: acc[i])
                loss_max_i = max(range(len(loss)), key=lambda i: loss[i])
                loss_min_i = min(range(len(loss)), key=lambda i: loss[i])
                history_by_seed[seed] = {
                    "acc_max": (acc[acc_max_i], acc_max_i + 1),
                    "acc_min": (acc[acc_min_i], acc_min_i + 1),
                    "loss_max": (loss[loss_max_i], loss_max_i + 1),
                    "loss_min": (loss[loss_min_i], loss_min_i + 1),
                }

    max_w = max(im.width for im in images)
    max_h = max(im.height for im in images)

    cols = 2
    rows = 4
    pad = 24
    label_h = 132
    canvas_w = cols * (max_w + pad) + pad
    canvas_h = rows * (max_h + label_h + pad) + pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 247, 250))
    draw = ImageDraw.Draw(canvas)

    for i, (seed, im) in enumerate(zip(seeds, images)):
        r = i // cols
        c = i % cols
        x = pad + c * (max_w + pad)
        y = pad + r * (max_h + label_h + pad)
        mm = metrics_by_seed[seed]
        draw.text((x, y), f"Seed {seed}", fill=(20, 30, 45))
        draw.text((x, y + 20), f"Acc: {mm['accuracy']:.4f} | Macro-F1: {mm['macro_f1']:.4f}", fill=(25, 45, 70))
        draw.text(
            (x, y + 40),
            f"TL: {mm['tl_score']:.4f} | R_sau: {mm['recall_la_sau']:.4f} | P_sau: {mm['precision_la_sau']:.4f}",
            fill=(25, 45, 70),
        )
        hh = history_by_seed.get(seed)
        if hh:
            acc_max, e_acc_max = hh["acc_max"]
            acc_min, e_acc_min = hh["acc_min"]
            loss_min, e_loss_min = hh["loss_min"]
            loss_max, e_loss_max = hh["loss_max"]
            draw.text(
                (x, y + 60),
                f"Acc max/min: {acc_max:.4f}@e{e_acc_max} / {acc_min:.4f}@e{e_acc_min}",
                fill=(90, 30, 20),
            )
            draw.text(
                (x, y + 80),
                f"Loss min/max: {loss_min:.4f}@e{e_loss_min} / {loss_max:.4f}@e{e_loss_max}",
                fill=(90, 30, 20),
            )
        else:
            draw.text((x, y + 70), "Thiếu history seed này -> chưa có acc/loss max-min theo epoch", fill=(140, 30, 30))
        y_img = y + label_h
        if im.size != (max_w, max_h):
            im = ImageOps.pad(im, (max_w, max_h), color=(255, 255, 255))
        canvas.paste(im, (x, y_img))

    out_path = fig_dir / "13_vgg16_cbam_7seed_loss_accuracy_grid.png"
    canvas.save(out_path)
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
