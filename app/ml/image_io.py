from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

# Đăng ký decoder HEIF/HEIC nếu thư viện có sẵn trong môi trường.
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:
    # Không bắt buộc; khi thiếu pillow-heif thì các file HEIC có thể lỗi decode.
    pass


def load_image_rgb_from_path(path: str | Path, img_size: tuple[int, int]) -> np.ndarray:
    """
    Đọc ảnh từ đường dẫn, convert RGB, resize về (H, W), trả về float32.
    """
    with Image.open(str(path)) as im:
        im = im.convert("RGB")
        im = im.resize((img_size[1], img_size[0]), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32)
    return arr


def load_image_rgb_from_bytes(file_bytes: bytes, img_size: tuple[int, int]) -> np.ndarray:
    """
    Đọc ảnh từ bytes (upload API), convert RGB, resize về (H, W), trả về float32.
    """
    with Image.open(BytesIO(file_bytes)) as im:
        im = im.convert("RGB")
        im = im.resize((img_size[1], img_size[0]), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32)
    return arr

