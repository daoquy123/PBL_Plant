from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from .model_vgg16_cbam import load_trained_model, IMG_SIZE, CLASS_NAMES
from .labels import EXPLANATIONS_VI, CLASS_LABELS_VI
from .image_io import load_image_rgb_from_bytes

_APP_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_WEIGHTS = _APP_DIR / "checkpoints" / "vgg16_cbam_best.weights.h5"


class LeafHealthPredictor:
    """Tải model VGG16+CBAM và trả về nhãn + xác suất + giải thích tiếng Việt."""

    def __init__(self, weights_path: str | Path | None = None):
        path = Path(weights_path) if weights_path is not None else _DEFAULT_WEIGHTS
        if not path.is_file():
            legacy = _APP_DIR / "models" / "leaf_vgg16_cbam_best.h5"
            if legacy.is_file():
                path = legacy

        if not path.is_file():
            raise FileNotFoundError(
                f"Chưa có file trọng số. Hãy train: python app/ml/train_vgg16_cbam.py "
                f"(kỳ vọng: {_DEFAULT_WEIGHTS})"
            )

        self.weights_path = str(path)
        self.model = load_trained_model(self.weights_path)

    def preprocess_image(self, file_bytes: bytes) -> tf.Tensor:
        arr = load_image_rgb_from_bytes(file_bytes, IMG_SIZE)
        img = tf.convert_to_tensor(arr, dtype=tf.float32)
        img = tf.expand_dims(img, 0)
        return img

    def predict(self, file_bytes: bytes) -> dict[str, Any]:
        img_batch = self.preprocess_image(file_bytes)
        preds = self.model.predict(img_batch, verbose=0)[0]

        idx = int(np.argmax(preds))
        prob = float(preds[idx])
        label = CLASS_NAMES[idx]

        return {
            "label": label,
            "probability": prob,
            "label_vietnamese": CLASS_LABELS_VI.get(label, label),
            "explanation": EXPLANATIONS_VI.get(label, ""),
            "raw_probs": {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))},
        }


_global_predictor: LeafHealthPredictor | None = None


def get_predictor() -> LeafHealthPredictor:
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = LeafHealthPredictor()
    return _global_predictor
