"""
Tach lai dataset hien co theo ti le 6/2/2.

Muc tieu:
- Gom du lieu tu dataset/train+val+test/<class> lam nguon
- Chia lai thanh train/val/test theo ti le mong muon
- Giu theo ti le gan dung train/val/test = 60/20/20 tren tung lop
- Co seed de tai lap ket qua
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


CLASS_NAMES = ["la_khoe", "la_vang", "la_sau", "sau", "co"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".heif"}


def _collect_files(class_dir: Path) -> list[Path]:
    return [
        p
        for p in class_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]


def _split_counts(n_total: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    # Neu lop co >= 5 anh, dam bao val va test khong rong.
    if n_total >= 5:
        if n_val == 0:
            n_val = 1
            n_train = max(0, n_train - 1)
        if n_test == 0:
            n_test = 1
            if n_train > 0:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1
    return n_train, n_val, n_test


def _clear_split_class(split_root: Path, class_name: str) -> None:
    d = split_root / class_name
    d.mkdir(parents=True, exist_ok=True)
    for p in d.rglob("*"):
        if p.is_file():
            p.unlink()


def _safe_name(dst_dir: Path, src: Path, suffix_tag: str) -> Path:
    dst = dst_dir / src.name
    if not dst.exists():
        return dst
    return dst_dir / f"{src.stem}_{suffix_tag}{src.suffix}"


def _safe_name_by_str(dst_dir: Path, filename: str, suffix_tag: str) -> Path:
    p = Path(filename)
    dst = dst_dir / p.name
    if not dst.exists():
        return dst
    return dst_dir / f"{p.stem}_{suffix_tag}{p.suffix}"


def split_dataset(
    dataset_root: Path,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 123,
    dry_run: bool = False,
    source_splits: tuple[str, ...] = ("train", "val", "test"),
) -> None:
    train_root = dataset_root / "train"
    val_root = dataset_root / "val"
    test_root = dataset_root / "test"

    for root in [train_root, val_root, test_root]:
        root.mkdir(parents=True, exist_ok=True)
    tmp_root = dataset_root / ".split_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    print(f"[INFO] dataset_root = {dataset_root}")
    print(f"[INFO] ratios train/val/test = {train_ratio:.2f}/{val_ratio:.2f}/{1-train_ratio-val_ratio:.2f}")
    print(f"[INFO] seed = {seed}")
    print(f"[INFO] dry_run = {dry_run}")
    print(f"[INFO] source_splits = {','.join(source_splits)}")

    for cls in CLASS_NAMES:
        # Gom du lieu tu cac split duoc chi dinh.
        files: list[Path] = []
        split_roots_map = {"train": train_root, "val": val_root, "test": test_root}
        for split_name in source_splits:
            split_root = split_roots_map[split_name]
            cls_dir = split_root / cls
            if cls_dir.is_dir():
                files.extend(_collect_files(cls_dir))

        if not files:
            print(f"[WARN] bo qua class {cls}: khong tim thay file nao trong train/val/test")
            continue

        rng.shuffle(files)
        n_total = len(files)
        n_train, n_val, n_test = _split_counts(n_total, train_ratio, val_ratio)

        train_files = files[:n_train]
        val_files = files[n_train : n_train + n_val]
        test_files = files[n_train + n_val :]

        print(
            f"[CLASS] {cls}: total={n_total}, train={len(train_files)}, val={len(val_files)}, test={len(test_files)}"
        )

        if dry_run:
            continue

        # B1) move tat ca file cua class vao vung tam de tranh xoa nham nguon.
        cls_tmp = tmp_root / cls
        if cls_tmp.exists():
            shutil.rmtree(cls_tmp)
        cls_tmp.mkdir(parents=True, exist_ok=True)

        staged_paths: list[Path] = []
        for idx, p in enumerate(files):
            staged_name = f"{idx:06d}{p.suffix.lower()}"
            staged_dst = cls_tmp / staged_name
            shutil.move(str(p), str(staged_dst))
            staged_paths.append(staged_dst)

        _clear_split_class(train_root, cls)
        _clear_split_class(val_root, cls)
        _clear_split_class(test_root, cls)

        (train_root / cls).mkdir(parents=True, exist_ok=True)
        (val_root / cls).mkdir(parents=True, exist_ok=True)
        (test_root / cls).mkdir(parents=True, exist_ok=True)

        # B2) chia lai tu staged_paths theo index split.
        train_idx_end = len(train_files)
        val_idx_end = train_idx_end + len(val_files)
        train_staged = staged_paths[:train_idx_end]
        val_staged = staged_paths[train_idx_end:val_idx_end]
        test_staged = staged_paths[val_idx_end:]

        for p in train_staged:
            dst = _safe_name_by_str(train_root / cls, p.name, "train")
            shutil.move(str(p), str(dst))

        for p in val_staged:
            dst = _safe_name_by_str(val_root / cls, p.name, "val")
            shutil.move(str(p), str(dst))

        for p in test_staged:
            dst = _safe_name_by_str(test_root / cls, p.name, "test")
            shutil.move(str(p), str(dst))

        # B3) don dep folder tam cua class.
        shutil.rmtree(cls_tmp, ignore_errors=True)

    if tmp_root.exists() and not any(tmp_root.iterdir()):
        tmp_root.rmdir()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tach dataset/train thanh train/val/test = 6/2/2")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--source-splits",
        type=str,
        default="train,val,test",
        help="Danh sach split nguon, vd: train,val,test hoac train",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_splits = tuple(s.strip() for s in args.source_splits.split(",") if s.strip())
    invalid = [s for s in source_splits if s not in {"train", "val", "test"}]
    if invalid:
        raise ValueError(f"source-splits khong hop le: {invalid}")
    if not source_splits:
        raise ValueError("source-splits khong duoc rong")

    split_dataset(
        dataset_root=args.dataset_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        dry_run=args.dry_run,
        source_splits=source_splits,
    )


if __name__ == "__main__":
    main()
