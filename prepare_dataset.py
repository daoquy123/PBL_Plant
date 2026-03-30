"""
Chuẩn hóa ảnh thô thành dataset train/val/test cho mô hình lá cải.

Đầu vào:
  raw_images/
    healthy/
    yellow/
    pest/
    yellow_pest/

Đầu ra (đã resize, chia train/val/test theo tỉ lệ):
  dataset/train/la_khoe/
  dataset/train/la_vang/
  dataset/train/la_sau/
  dataset/train/la_sau_va_vang/
  dataset/val/...
  dataset/test/...
"""

from pathlib import Path
from typing import Tuple

from PIL import Image
import imagehash
import random

from app.ml.model_vgg16_cbam import IMG_SIZE


RAW_ROOT = Path("raw_images")
DATASET_ROOT = Path("dataset")

MAP_RAW_TO_CLASS = {
    "healthy": "la_khoe",
    "yellow": "la_vang",
    "pest": "la_sau",
    "yellow_pest": "la_sau_va_vang",
}

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2


def ensure_dirs():
    for split in ["train", "val", "test"]:
        for cls in MAP_RAW_TO_CLASS.values():
            path = DATASET_ROOT / split / cls
            path.mkdir(parents=True, exist_ok=True)


def collect_images(raw_subdir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in raw_subdir.rglob("*") if p.suffix.lower() in exts]


def resize_and_save(src: Path, dst: Path):
    with Image.open(src) as img:
        img = img.convert("RGB")
        img = img.resize(IMG_SIZE)
        img.save(dst, format="JPEG", quality=90)


def remove_duplicates(files):
    """Lọc bớt ảnh trùng bằng perceptual hash đơn giản."""
    seen = set()
    unique_files = []
    for f in files:
        try:
            with Image.open(f) as img:
                img = img.convert("RGB")
                h = imagehash.phash(img)
        except Exception:
            continue

        if h in seen:
            continue
        seen.add(h)
        unique_files.append(f)
    return unique_files


def process_group(raw_group: str, class_name: str):
    src_dir = RAW_ROOT / raw_group
    if not src_dir.exists():
        print(f"Bỏ qua {raw_group}, không tìm thấy thư mục {src_dir}")
        return

    files = collect_images(src_dir)
    print(f"{raw_group}: tìm thấy {len(files)} ảnh thô")
    files = remove_duplicates(files)
    print(f"{raw_group}: sau khi lọc trùng còn {len(files)} ảnh")

    random.shuffle(files)
    n_total = len(files)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    n_test = n_total - n_train - n_val

    # Giữ tập test đủ dữ liệu để đánh giá:
    # nếu dữ liệu lớp có từ 3 ảnh trở lên thì cố giữ tối thiểu 1 ảnh test.
    if n_total >= 3 and n_test == 0:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        elif n_val > 0:
            n_val -= 1
        else:
            n_train = max(0, n_train - 1)

    train_files = files[:n_train]
    val_files = files[n_train : n_train + n_val]
    test_files = files[n_train + n_val :]

    for split, split_files in [
        ("train", train_files),
        ("val", val_files),
        ("test", test_files),
    ]:
        out_dir = DATASET_ROOT / split / class_name
        for idx, src in enumerate(split_files):
            dst = out_dir / f"{raw_group}_{idx:05d}.jpg"
            try:
                resize_and_save(src, dst)
            except Exception as e:
                print(f"Lỗi với ảnh {src}: {e}")

    print(
        f"{raw_group}: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}"
    )


def main():
    ensure_dirs()
    for raw_group, cls in MAP_RAW_TO_CLASS.items():
        process_group(raw_group, cls)


if __name__ == "__main__":
    main()

