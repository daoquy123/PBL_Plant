import os
import sys
from typing import Any

import numpy as np
import gradio as gr
import tensorflow as tf


# Thiết lập đường dẫn để có thể import được module trong thư mục app/ml
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.model_vgg16_cbam import IMG_SIZE, CLASS_NAMES, build_vgg16_cbam_model
from ml.labels import CLASS_LABELS_VI
from ml.image_io import load_image_rgb_from_path

WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "vgg16_cbam_best.weights.h5")


def load_model():
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Không tìm thấy trọng số tại {WEIGHTS_PATH}. "
            f"Hãy chắc chắn bạn đã train và file này tồn tại."
        )
    # Khởi tạo kiến trúc model rồi nạp best weights
    model = build_vgg16_cbam_model()
    model.load_weights(WEIGHTS_PATH)
    return model


_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model()
    return _MODEL


def predict_leaf(file_obj) -> tuple[str, Any]:
    """
    Nhận file ảnh upload (ưu tiên hỗ trợ nhiều định dạng), trả về nhãn tiếng Việt.
    """
    if file_obj is None:
        return "Chưa có ảnh.", None

    file_path = getattr(file_obj, "name", None)
    if not file_path:
        return "Không đọc được file ảnh.", None

    try:
        model = get_model()
        arr = load_image_rgb_from_path(file_path, IMG_SIZE)
        img_batch = tf.convert_to_tensor(np.expand_dims(arr, axis=0), dtype=tf.float32)
        preds = model.predict(img_batch, verbose=0)[0]
    except Exception as e:
        return f"Lỗi đọc/dự đoán ảnh: {e}", None

    class_idx = int(np.argmax(preds))
    class_name = CLASS_NAMES[class_idx]
    conf = float(preds[class_idx])
    label_vi = CLASS_LABELS_VI.get(class_name, class_name)
    # Trả ảnh đã decode để frontend luôn hiển thị được (kể cả TIFF/HEIC).
    preview = np.clip(arr, 0, 255).astype(np.uint8)
    return f"{label_vi} ({conf:.2%})", preview


def preview_image(file_obj):
    """
    Hiển thị ảnh ngay khi upload để tránh phụ thuộc khả năng preview của widget gốc.
    """
    if file_obj is None:
        return None
    file_path = getattr(file_obj, "name", None)
    if not file_path:
        return None
    try:
        arr = load_image_rgb_from_path(file_path, IMG_SIZE)
        return np.clip(arr, 0, 255).astype(np.uint8)
    except Exception:
        return None


def build_interface() -> gr.Blocks:
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.File(
                    label="Ảnh lá cây",
                    # Dùng đuôi file cụ thể để trình chọn file nhận cả HEIC/HEIF.
                    file_types=[
                        ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp",
                        ".tif", ".tiff", ".heic", ".heif"
                    ],
                )
                submit_btn = gr.Button("Dự đoán")

            with gr.Column(scale=1):
                label_out = gr.Textbox(
                    label="Nhãn dự đoán",
                    interactive=False,
                )
                preview_out = gr.Image(
                    label="Ảnh hiển thị",
                    type="numpy",
                    interactive=False,
                )

        submit_btn.click(
            fn=predict_leaf,
            inputs=[image_in],
            outputs=[label_out, preview_out],
        )
        image_in.change(
            fn=preview_image,
            inputs=[image_in],
            outputs=[preview_out],
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch()

