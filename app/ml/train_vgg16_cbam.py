import math
import os
import random
from pathlib import Path
import numpy as np
import tensorflow as tf

from model_vgg16_cbam import build_vgg16_cbam_model, IMG_SIZE, CLASS_NAMES
from reporting import save_training_history
from image_io import load_image_rgb_from_path


# ========= CẤU HÌNH CƠ BẢN =========
# dataset/train và dataset/val — mỗi lớp một thư mục con:
#   la_khoe, la_vang, la_sau, sau, co
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# dataset nằm ngoài thư mục PROJECT_ROOT một cấp
DATA_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "dataset"))

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")

BATCH_SIZE = 32
EPOCHS = 30
AUTOTUNE = tf.data.AUTOTUNE
VAL_SPLIT = 0.2
SPLIT_SEED = 123
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff", ".heic", ".heif"}

# Nơi lưu trọng số/best model
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
BEST_WEIGHTS_PATH = os.path.join(CHECKPOINT_DIR, "vgg16_cbam_best.weights.h5")
HISTORY_JSON_PATH = os.path.join(CHECKPOINT_DIR, "training_history.json")


def _count_train_images_per_class() -> dict[int, int]:
    counts: dict[int, int] = {}
    for idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = Path(TRAIN_DIR) / cls_name
        if not cls_dir.is_dir():
            counts[idx] = 0
            continue
        counts[idx] = sum(1 for p in cls_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return counts


def _compute_class_weight(class_counts: dict[int, int]) -> dict[int, float]:
    """
    Balanced class weight: N / (K * n_c).
    Nếu lớp không có ảnh -> weight 0 để tránh chia 0.
    """
    total = sum(class_counts.values())
    num_classes = len(CLASS_NAMES)
    weights: dict[int, float] = {}
    for c, n in class_counts.items():
        if n <= 0:
            weights[c] = 0.0
        else:
            weights[c] = float(total) / float(num_classes * n)
    return weights


def _decode_resize(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Decode ảnh bằng PIL để hỗ trợ nhiều định dạng (jpg/png/tif/webp/heic nếu có decoder),
    sau đó resize về IMG_SIZE.
    """
    def _load_with_pil(p: tf.Tensor) -> np.ndarray:
        p_str = p.numpy().decode("utf-8")
        return load_image_rgb_from_path(p_str, IMG_SIZE)

    img = tf.py_function(_load_with_pil, [path], Tout=tf.float32)
    img.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
    return img, label


def _build_dataset_from_paths(paths: list[str], labels: list[int], shuffle: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        # Dùng seed cố định để tái lập.
        ds = ds.shuffle(buffer_size=max(len(paths), 1), seed=SPLIT_SEED, reshuffle_each_iteration=True)
    ds = ds.map(_decode_resize, num_parallel_calls=AUTOTUNE)
    # Bỏ qua file ảnh lỗi/hỏng còn sót lại để không crash toàn bộ quá trình train.
    ds = ds.ignore_errors()
    ds = ds.batch(BATCH_SIZE)
    return ds


def _collect_paths_from_split(split_dir: str) -> tuple[list[str], list[int], dict[int, int]]:
    """
    Quét ảnh theo CLASS_NAMES trong một split dir (train/val).
    Trả về: paths, labels, class_counts.
    """
    paths: list[str] = []
    labels: list[int] = []
    counts: dict[int, int] = {i: 0 for i in range(len(CLASS_NAMES))}

    for label, cls_name in enumerate(CLASS_NAMES):
        cls_dir = Path(split_dir) / cls_name
        if not cls_dir.is_dir():
            continue
        files = [str(p) for p in cls_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        counts[label] = len(files)
        paths.extend(files)
        labels.extend([label] * len(files))
    return paths, labels, counts


def _stratified_split_from_train_dir() -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Tách train/val theo từng lớp để đảm bảo lớp hiếm (vd. la_vang) luôn có mặt ở validation.
    """
    rng = random.Random(SPLIT_SEED)
    train_paths: list[str] = []
    train_labels: list[int] = []
    val_paths: list[str] = []
    val_labels: list[int] = []

    for label, cls_name in enumerate(CLASS_NAMES):
        cls_dir = Path(TRAIN_DIR) / cls_name
        files = [str(p) for p in cls_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        rng.shuffle(files)
        n_total = len(files)
        if n_total == 0:
            continue
        n_val = int(math.ceil(n_total * VAL_SPLIT))
        # Tránh rơi vào val rỗng hoặc train rỗng khi lớp có ít ảnh.
        n_val = min(max(1, n_val), max(1, n_total - 1)) if n_total > 1 else 1
        cls_val = files[:n_val]
        cls_train = files[n_val:] if n_total > 1 else files

        val_paths.extend(cls_val)
        val_labels.extend([label] * len(cls_val))
        train_paths.extend(cls_train)
        train_labels.extend([label] * len(cls_train))

    train_ds = _build_dataset_from_paths(train_paths, train_labels, shuffle=True)
    val_ds = _build_dataset_from_paths(val_paths, val_labels, shuffle=False)
    print(f"Stratified split từ TRAIN_DIR: train={len(train_paths)} ảnh, val={len(val_paths)} ảnh")
    return train_ds, val_ds


def load_datasets() -> tuple[tf.data.Dataset, tf.data.Dataset, dict[int, float]]:
    """
    Tạo tf.data.Dataset từ thư mục.
    Trả về: train_ds, val_ds, class_weight.
    """
    print(f"Đang load dữ liệu từ:\n  TRAIN_DIR = {TRAIN_DIR}\n  VAL_DIR   = {VAL_DIR}")

    train_paths_all, train_labels_all, train_class_counts = _collect_paths_from_split(TRAIN_DIR)
    class_weight = _compute_class_weight(train_class_counts)
    print("Số ảnh train theo lớp:", {CLASS_NAMES[k]: v for k, v in train_class_counts.items()})
    print("Class weight:", {CLASS_NAMES[k]: round(v, 3) for k, v in class_weight.items()})

    # Dùng pipeline quét đường dẫn + PIL decode để hỗ trợ nhiều định dạng hơn
    # (bao gồm HEIC/HEIF nếu môi trường có decoder tương ứng).
    val_paths_all, val_labels_all, _ = _collect_paths_from_split(VAL_DIR)

    if len(val_paths_all) > 0 and len(train_paths_all) > 0:
        train_ds = _build_dataset_from_paths(train_paths_all, train_labels_all, shuffle=True)
        val_ds = _build_dataset_from_paths(val_paths_all, val_labels_all, shuffle=False)
        print(f"Dùng trực tiếp TRAIN_DIR/VAL_DIR: train={len(train_paths_all)} ảnh, val={len(val_paths_all)} ảnh")
    else:
        # Nếu VAL_DIR trống hoặc không có ảnh hợp lệ,
        # fallback: tách validation theo từng lớp từ TRAIN_DIR.
        print(
            "Không tìm thấy ảnh trong VAL_DIR, tự động tách validation theo lớp "
            f"({int(VAL_SPLIT * 100)}%) từ TRAIN_DIR."
        )
        train_ds, val_ds = _stratified_split_from_train_dir()

    # Tối ưu pipeline
    train_ds = train_ds.cache().shuffle(1000, seed=SPLIT_SEED).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_weight


def train():
    train_ds, val_ds, class_weight = load_datasets()

    model = build_vgg16_cbam_model()
    model.summary()

    # Callback: early stopping + lưu best weights
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        BEST_WEIGHTS_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stopping, checkpoint],
        class_weight=class_weight,
    )

    save_training_history(history, HISTORY_JSON_PATH)
    print(f"Đã lưu biểu đồ huấn luyện (raw): {HISTORY_JSON_PATH}")

    # File .keras có thể lỗi với Lambda layer — chỉ thử lưu khi cần
    saved_model_path = os.path.join(PROJECT_ROOT, "saved_models", "vgg16_cbam.keras")
    os.makedirs(os.path.dirname(saved_model_path), exist_ok=True)
    try:
        model.save(saved_model_path)
        print(f"Full model: {saved_model_path}")
    except Exception as e:
        print(f"Bỏ qua lưu .keras (dùng weights .h5): {e}")

    print(f"\nĐã train xong. Best weights: {BEST_WEIGHTS_PATH}")

    return model, history


if __name__ == "__main__":
    model, history = train()

