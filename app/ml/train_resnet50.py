import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from image_io import load_image_rgb_from_path
from model_resnet50_cbam import CLASS_NAMES, IMG_SIZE, build_resnet50_model
from reporting import save_training_history

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "dataset"))

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")

BATCH_SIZE = 32
EPOCHS_STAGE1 = 30
EPOCHS_STAGE2 = 20
AUTOTUNE = tf.data.AUTOTUNE
VAL_SPLIT = 0.2
SPLIT_SEED = 123
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff", ".heic", ".heif"}

LA_KHOE_IDX = CLASS_NAMES.index("la_khoe")
LA_SAU_IDX = CLASS_NAMES.index("la_sau")
LA_VANG_IDX = CLASS_NAMES.index("la_vang")


class _ClassSpecificBase(tf.keras.metrics.Metric):
    def __init__(self, class_id: int, name: str, dtype=tf.float32):
        super().__init__(name=name, dtype=dtype)
        self.class_id = int(class_id)
        self.eps = tf.constant(1e-8, dtype=dtype)

    def _to_label(self, y_pred: tf.Tensor) -> tf.Tensor:
        return tf.argmax(y_pred, axis=-1, output_type=tf.int32)

    def _to_true(self, y_true: tf.Tensor) -> tf.Tensor:
        return tf.cast(tf.reshape(y_true, [-1]), tf.int32)


class RecallLaSau(_ClassSpecificBase):
    def __init__(self, class_id: int, name: str = "recall_la_sau"):
        super().__init__(class_id=class_id, name=name)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        y_t = self._to_true(y_true)
        y_p = self._to_label(y_pred)
        tp_mask = tf.logical_and(tf.equal(y_t, self.class_id), tf.equal(y_p, self.class_id))
        fn_mask = tf.logical_and(tf.equal(y_t, self.class_id), tf.not_equal(y_p, self.class_id))
        self.tp.assign_add(tf.reduce_sum(tf.cast(tp_mask, self.dtype)))
        self.fn.assign_add(tf.reduce_sum(tf.cast(fn_mask, self.dtype)))

    def result(self):
        return self.tp / (self.tp + self.fn + self.eps)

    def reset_state(self):
        self.tp.assign(0.0)
        self.fn.assign(0.0)


class PrecisionLaSau(_ClassSpecificBase):
    def __init__(self, class_id: int, name: str = "precision_la_sau"):
        super().__init__(class_id=class_id, name=name)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        y_t = self._to_true(y_true)
        y_p = self._to_label(y_pred)
        tp_mask = tf.logical_and(tf.equal(y_t, self.class_id), tf.equal(y_p, self.class_id))
        fp_mask = tf.logical_and(tf.not_equal(y_t, self.class_id), tf.equal(y_p, self.class_id))
        self.tp.assign_add(tf.reduce_sum(tf.cast(tp_mask, self.dtype)))
        self.fp.assign_add(tf.reduce_sum(tf.cast(fp_mask, self.dtype)))

    def result(self):
        return self.tp / (self.tp + self.fp + self.eps)

    def reset_state(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)


class F1LaSau(_ClassSpecificBase):
    def __init__(self, class_id: int, name: str = "f1_la_sau"):
        super().__init__(class_id=class_id, name=name)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        y_t = self._to_true(y_true)
        y_p = self._to_label(y_pred)
        tp_mask = tf.logical_and(tf.equal(y_t, self.class_id), tf.equal(y_p, self.class_id))
        fp_mask = tf.logical_and(tf.not_equal(y_t, self.class_id), tf.equal(y_p, self.class_id))
        fn_mask = tf.logical_and(tf.equal(y_t, self.class_id), tf.not_equal(y_p, self.class_id))
        self.tp.assign_add(tf.reduce_sum(tf.cast(tp_mask, self.dtype)))
        self.fp.assign_add(tf.reduce_sum(tf.cast(fp_mask, self.dtype)))
        self.fn.assign_add(tf.reduce_sum(tf.cast(fn_mask, self.dtype)))

    def result(self):
        precision = self.tp / (self.tp + self.fp + self.eps)
        recall = self.tp / (self.tp + self.fn + self.eps)
        return 2.0 * precision * recall / (precision + recall + self.eps)

    def reset_state(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)


def _augment_train_batch(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    x = tf.cast(images, tf.float32)
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_brightness(x, max_delta=0.08)
    x = tf.image.random_contrast(x, lower=0.9, upper=1.1)
    x = tf.clip_by_value(x, 0.0, 255.0)
    return x, labels


def _build_cost_matrix() -> tf.Tensor:
    n = len(CLASS_NAMES)
    matrix = np.ones((n, n), dtype=np.float32)
    np.fill_diagonal(matrix, 1.0)
    # Ưu tiên không bỏ sót la_sau nhưng cũng chống "spam la_sau".
    matrix[LA_SAU_IDX, LA_KHOE_IDX] = 3.0
    matrix[LA_KHOE_IDX, LA_SAU_IDX] = 2.4
    matrix[LA_VANG_IDX, LA_SAU_IDX] = 1.8
    matrix[LA_SAU_IDX, LA_VANG_IDX] = 1.2
    return tf.constant(matrix, dtype=tf.float32)


def _make_cost_sensitive_loss(cost_matrix: tf.Tensor):
    base_loss = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE
    )

    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_i = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        ce = base_loss(y_true_i, y_pred)
        row_costs = tf.gather(cost_matrix, y_true_i)
        expected_cost = tf.reduce_sum(row_costs * y_pred, axis=-1)
        return ce * expected_cost

    return loss_fn


def _compute_class_weight(class_counts: dict[int, int]) -> dict[int, float]:
    total = sum(class_counts.values())
    num_classes = len(CLASS_NAMES)
    weights: dict[int, float] = {}
    for c, n in class_counts.items():
        weights[c] = 0.0 if n <= 0 else float(total) / float(num_classes * n)
    return weights


def _decode_resize(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    def _load_with_pil(p: tf.Tensor) -> np.ndarray:
        p_str = p.numpy().decode("utf-8")
        return load_image_rgb_from_path(p_str, IMG_SIZE)

    img = tf.py_function(_load_with_pil, [path], Tout=tf.float32)
    img.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
    return img, label


def _build_dataset_from_paths(paths: list[str], labels: list[int], shuffle: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=max(len(paths), 1), seed=SPLIT_SEED, reshuffle_each_iteration=True)
    ds = ds.map(_decode_resize, num_parallel_calls=AUTOTUNE)
    ds = ds.ignore_errors()
    ds = ds.batch(BATCH_SIZE)
    if shuffle:
        ds = ds.map(_augment_train_batch, num_parallel_calls=AUTOTUNE)
    return ds


def _collect_paths_from_split(split_dir: str) -> tuple[list[str], list[int], dict[int, int]]:
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


def load_datasets() -> tuple[tf.data.Dataset, tf.data.Dataset, dict[int, float], int | None, int | None]:
    print(f"Đang load dữ liệu từ:\n  TRAIN_DIR = {TRAIN_DIR}\n  VAL_DIR   = {VAL_DIR}")
    train_paths_all, train_labels_all, train_class_counts = _collect_paths_from_split(TRAIN_DIR)
    class_weight = _compute_class_weight(train_class_counts)
    print("Số ảnh train theo lớp:", {CLASS_NAMES[k]: v for k, v in train_class_counts.items()})
    print("Class weight:", {CLASS_NAMES[k]: round(v, 3) for k, v in class_weight.items()})

    val_paths_all, val_labels_all, _ = _collect_paths_from_split(VAL_DIR)
    train_steps: int | None = None
    val_steps: int | None = None
    if len(val_paths_all) > 0 and len(train_paths_all) > 0:
        train_ds = _build_dataset_from_paths(train_paths_all, train_labels_all, shuffle=True)
        val_ds = _build_dataset_from_paths(val_paths_all, val_labels_all, shuffle=False)
        print(f"Dùng trực tiếp TRAIN_DIR/VAL_DIR: train={len(train_paths_all)} ảnh, val={len(val_paths_all)} ảnh")
        train_steps = max(1, math.ceil(len(train_paths_all) / BATCH_SIZE))
        val_steps = max(1, math.ceil(len(val_paths_all) / BATCH_SIZE))
    else:
        print(
            "Không tìm thấy ảnh trong VAL_DIR, tự động tách validation theo lớp "
            f"({int(VAL_SPLIT * 100)}%) từ TRAIN_DIR."
        )
        train_ds, val_ds = _stratified_split_from_train_dir()

    # repeat() + steps_per_epoch giúp tránh cảnh báo "input ran out of data"
    # khi có một số file bị bỏ qua bởi ignore_errors().
    train_ds = train_ds.cache().shuffle(1000, seed=SPLIT_SEED).repeat().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().repeat().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds, class_weight, train_steps, val_steps


def _build_paths(model_tag: str) -> tuple[str, str, str]:
    checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    saved_models_dir = os.path.join(PROJECT_ROOT, "saved_models")
    os.makedirs(saved_models_dir, exist_ok=True)
    return (
        os.path.join(checkpoint_dir, f"{model_tag}_best.weights.h5"),
        os.path.join(checkpoint_dir, f"{model_tag}_training_history.json"),
        os.path.join(saved_models_dir, f"{model_tag}.keras"),
    )


def train(use_cbam: bool = False):
    model_tag = "resnet50_cbam" if use_cbam else "resnet50"
    best_weights_path, history_json_path, saved_model_path = _build_paths(model_tag)
    train_ds, val_ds, class_weight, train_steps, val_steps = load_datasets()

    model = build_resnet50_model(use_cbam=use_cbam)
    cost_sensitive_loss = _make_cost_sensitive_loss(_build_cost_matrix())
    metrics = [
        "accuracy",
        RecallLaSau(class_id=LA_SAU_IDX, name="recall_la_sau"),
        PrecisionLaSau(class_id=LA_SAU_IDX, name="precision_la_sau"),
        F1LaSau(class_id=LA_SAU_IDX, name="f1_la_sau"),
    ]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=cost_sensitive_loss,
        metrics=metrics,
    )
    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_f1_la_sau",
        patience=8,
        restore_best_weights=True,
        mode="max",
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        best_weights_path,
        monitor="val_f1_la_sau",
        save_best_only=True,
        save_weights_only=True,
        mode="max",
        verbose=1,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )

    print(f"\n=== Stage 1: train head với ResNet50 frozen ({EPOCHS_STAGE1} epochs) ===")
    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE1,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=[early_stopping, checkpoint, reduce_lr],
        class_weight=class_weight,
    )

    print(f"\n=== Stage 2: fine-tune conv5_x ResNet50 ({EPOCHS_STAGE2} epochs) ===")
    resnet_layer = next((l for l in model.layers if "resnet50" in l.name.lower()), None)
    if resnet_layer is not None:
        for layer in resnet_layer.layers:
            layer.trainable = layer.name.startswith("conv5")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(2e-5),
        loss=cost_sensitive_loss,
        metrics=metrics,
    )
    fine_tune_start = len(history_stage1.history.get("loss", []))
    history_stage2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=fine_tune_start + EPOCHS_STAGE2,
        initial_epoch=fine_tune_start,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=[early_stopping, checkpoint, reduce_lr],
        class_weight=class_weight,
    )

    merged_history: dict[str, list[float]] = {}
    for k, v in history_stage1.history.items():
        merged_history[k] = list(v)
    for k, v in history_stage2.history.items():
        merged_history.setdefault(k, [])
        if len(merged_history[k]) < fine_tune_start:
            merged_history[k].extend([float("nan")] * (fine_tune_start - len(merged_history[k])))
        merged_history[k].extend(list(v))

    history = tf.keras.callbacks.History()
    history.history = merged_history
    save_training_history(history, history_json_path)
    print(f"Đã lưu biểu đồ huấn luyện (raw): {history_json_path}")

    try:
        model.save(saved_model_path)
        print(f"Full model: {saved_model_path}")
    except Exception as e:
        print(f"Bỏ qua lưu .keras (dùng weights .h5): {e}")

    print(f"\nĐã train xong. Best weights: {best_weights_path}")
    return model, history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet50 baseline hoặc ResNet50+CBAM")
    parser.add_argument(
        "--use-cbam",
        action="store_true",
        help="Bật CBAM trên top feature map của ResNet50",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model, history = train(use_cbam=args.use_cbam)
