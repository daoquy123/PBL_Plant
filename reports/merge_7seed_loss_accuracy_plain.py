from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    fig_dir = root / "reports" / "figures"
    seeds = [118, 119, 120, 121, 122, 123, 124]
    paths = [fig_dir / f"vgg16_cbam_seed{s}_04_loss_accuracy.png" for s in seeds]

    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Thiếu ảnh:\n" + "\n".join(missing))

    images = [Image.open(p).convert("RGB") for p in paths]
    w = max(im.width for im in images)
    h = max(im.height for im in images)

    cols = 2
    rows = 4
    pad = 16
    bg = (246, 246, 246)
    canvas_w = cols * w + (cols + 1) * pad
    canvas_h = rows * h + (rows + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg)

    for i, im in enumerate(images):
        r = i // cols
        c = i % cols
        x = pad + c * (w + pad)
        y = pad + r * (h + pad)
        if im.size != (w, h):
            im = ImageOps.pad(im, (w, h), color=(255, 255, 255))
        canvas.paste(im, (x, y))

    out = fig_dir / "18_vgg16_cbam_7seed_loss_accuracy_plain_grid.png"
    canvas.save(out)
    print(f"[SAVED] {out}")


if __name__ == "__main__":
    main()
