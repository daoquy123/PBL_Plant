from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    fig_dir = root / "reports" / "figures"
    seeds = [118, 119, 120, 121, 122, 123, 124]
    paths = [fig_dir / f"vgg16_cbam_seed{s}_05_confusion_matrix.png" for s in seeds]

    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Thiếu ảnh confusion matrix:\n" + "\n".join(missing))

    images = [Image.open(p).convert("RGB") for p in paths]
    w = max(im.width for im in images)
    h = max(im.height for im in images)

    cols = 2
    rows = 4
    pad = 16
    label_h = 28
    bg = (246, 246, 246)
    canvas_w = cols * w + (cols + 1) * pad
    canvas_h = rows * (h + label_h + pad) + pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg)
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    for i, (seed, im) in enumerate(zip(seeds, images)):
        r = i // cols
        c = i % cols
        x = pad + c * (w + pad)
        y = pad + r * (h + label_h + pad)
        draw.text((x, y), f"Seed {seed}", fill=(30, 30, 30), font=font)
        y_img = y + label_h
        if im.size != (w, h):
            im = ImageOps.pad(im, (w, h), color=(255, 255, 255))
        canvas.paste(im, (x, y_img))

    out = fig_dir / "20_vgg16_cbam_7seed_confusion_matrix_grid.png"
    canvas.save(out)
    print(f"[SAVED] {out}")


if __name__ == "__main__":
    main()
