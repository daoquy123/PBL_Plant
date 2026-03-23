"""
Tiện ích đánh giá mô hình: gom nhãn dự đoán, lưu history huấn luyện.
Dùng từ notebook báo cáo hoặc script evaluate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def save_training_history(history: tf.keras.callbacks.History, path: str | Path) -> None:
    """Lưu history.fit() ra JSON (để notebook vẽ loss/accuracy)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        k: [float(x) for x in v] for k, v in history.history.items()
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def load_training_history(path: str | Path) -> dict[str, list[float]] | None:
    path = Path(path)
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dataset_to_predictions(
    model: tf.keras.Model,
    ds: tf.data.Dataset,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Chạy model trên dataset (y, x) với label_mode int.
    Trả về y_true, y_pred (cả hai shape [N]).
    """
    y_true_list: list[np.ndarray] = []
    y_pred_list: list[np.ndarray] = []

    for i, batch in enumerate(ds):
        if max_batches is not None and i >= max_batches:
            break
        x, y = batch
        preds = model.predict(x, verbose=0)
        y_pred_list.append(np.argmax(preds, axis=-1))
        y_true_list.append(np.asarray(y))

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    return y_true, y_pred


def find_vgg_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    for layer in model.layers:
        if "vgg" in layer.name.lower():
            return layer
    raise ValueError("Không tìm thấy lớp backbone VGG16 trong model.")


def _spatial_tensor_before_gap(model: tf.keras.Model):
    """
    Tensor feature map không gian cuối (trước GAP): với VGG16+CBAM là (H,W,C) sau CBAM.

    Dùng `model.layers[i-1].output` (lớp đi vào GAP), không dùng `GlobalAveragePooling2D.input`:
    trên Keras 3 `layer.input` đôi khi là KerasTensor không nằm trong graph cha → KeyError khi tạo Model.
    """
    layers = model.layers
    for i, layer in enumerate(layers):
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            if i < 1:
                raise ValueError("GlobalAveragePooling2D ở đầu model — không hợp lệ cho Grad-CAM.")
            return layers[i - 1].output
    raise ValueError("Không tìm thấy GlobalAveragePooling2D — không xác định được feature map cho Grad-CAM.")


def clear_gradcam_cache() -> None:
    """Giữ API tương thích; Grad-CAM không còn cache submodel toàn cục."""
    return None


def _build_gradcam_submodel(model: tf.keras.Model) -> tf.keras.Model:
    spatial = _spatial_tensor_before_gap(model)
    # Dùng output lớp cuối thay vì model.output — tránh tensor “lạ” trong graph Keras 3.
    logits_tensor = model.layers[-1].output
    return tf.keras.Model(inputs=model.input, outputs=[spatial, logits_tensor])


def _gradcam_fallback_input_saliency(
    model: tf.keras.Model,
    img_batch: tf.Tensor,
    pred_index: int | None,
) -> np.ndarray:
    """Khi submodel 2 đầu ra lỗi KeyError: saliency |d(score)/d(input)| (minh hoạ vùng nhạy)."""
    with tf.GradientTape() as tape:
        tape.watch(img_batch)
        preds = model(img_batch, training=False)
        if pred_index is None:
            pred_index = int(tf.argmax(preds[0]))
        class_channel = preds[:, pred_index]
    g = tape.gradient(class_channel, img_batch)
    if g is None:
        raise RuntimeError("Gradient None trên ảnh đầu vào — không thể vẽ heatmap.")
    heatmap = tf.reduce_mean(tf.abs(g[0]), axis=-1)
    heatmap = tf.nn.relu(heatmap)
    maxv = tf.reduce_max(heatmap)
    heatmap = heatmap / (maxv + 1e-8)
    return heatmap.numpy()


def compute_gradcam(
    model: tf.keras.Model,
    img_batch: tf.Tensor,
    pred_index: int | None = None,
) -> np.ndarray:
    """
    Grad-CAM trên feature map không gian cuối (đầu vào của GAP: sau CBAM, trước phân loại).
    Trả về heatmap 2D (H', W') đã chuẩn hoá [0, 1].

    Một số bản Keras 3 gặp KeyError khi gọi submodel 2 đầu ra; khi đó tự động dùng saliency
    trên ảnh đầu vào (heatmap 224×224 — notebook vẫn upsample được).
    """
    img_batch = tf.convert_to_tensor(img_batch, dtype=tf.float32)
    # Không cache submodel toàn cục: tránh kernel Jupyter giữ bản graph cũ sau khi sửa code/weights.
    grad_model = _build_gradcam_submodel(model)

    prev_eager = tf.config.functions_run_eagerly()
    try:
        tf.config.run_functions_eagerly(True)
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_batch)
            if pred_index is None:
                pred_index = int(tf.argmax(preds[0]))
            class_channel = preds[:, pred_index]
        grads = tape.gradient(class_channel, conv_out)
        if grads is None:
            raise RuntimeError("Gradient None — kiểm tra graph model.")
        pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
        conv_first = conv_out[0]
        pg = pooled_grads[0]
        heatmap = tf.reduce_sum(tf.multiply(conv_first, pg), axis=-1)
        heatmap = tf.nn.relu(heatmap)
        maxv = tf.reduce_max(heatmap)
        heatmap = heatmap / (maxv + 1e-8)
        return heatmap.numpy()
    except KeyError:
        return _gradcam_fallback_input_saliency(model, img_batch, pred_index)
    finally:
        tf.config.run_functions_eagerly(prev_eager)


def upsample_heatmap(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Nội suy heatmap lên kích thước ảnh (H, W)."""
    t = tf.constant(heatmap, dtype=tf.float32)
    t = tf.reshape(t, (1, heatmap.shape[0], heatmap.shape[1], 1))
    t = tf.image.resize(t, size, method="bilinear")
    return tf.squeeze(t, [0, -1]).numpy()
