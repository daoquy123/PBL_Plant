"""
Tối ưu runtime TensorFlow cho huấn luyện: GPU memory growth, mixed precision (khi có GPU),
XLA, tùy chọn tf.data.
"""

from __future__ import annotations

import os

import tensorflow as tf


def with_data_perf_options(ds: tf.data.Dataset) -> tf.data.Dataset:
    """Bật song song hóa map và tối ưu pipeline (an toàn cho train/val)."""
    opts = tf.data.Options()
    try:
        opts.experimental_optimization.map_parallelization = True
    except AttributeError:
        pass
    return ds.with_options(opts)


def configure_training_runtime(
    *,
    mixed_precision: bool = True,
    xla: bool = True,
) -> dict[str, bool | str]:
    """
    Gọi một lần trước khi build model / compile.

    - GPU (CUDA): bật memory growth; mixed_float16 nếu mixed_precision và có ít nhất 1 GPU.
    - XLA: bật trừ khi đặt biến môi trường TF_TRAIN_DISABLE_XLA=1 hoặc xla=False.

    Trả về dict để in log (không phụ thuộc thứ tự key).
    """
    xla = xla and os.environ.get("TF_TRAIN_DISABLE_XLA", "").strip() not in ("1", "true", "True")
    used_mixed = False

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except (ValueError, RuntimeError):
            pass

    if mixed_precision and gpus:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            used_mixed = True
        except (ValueError, RuntimeError):
            used_mixed = False

    if xla:
        try:
            tf.config.optimizer.set_jit(True)
        except (ValueError, RuntimeError):
            xla = False

    logical = tf.config.list_logical_devices()
    summary = ", ".join(f"{d.device_type}:{d.name}" for d in logical[:8])
    if len(logical) > 8:
        summary += ", ..."

    return {
        "mixed_float16": used_mixed,
        "xla_jit": xla,
        "physical_gpus": len(gpus),
        "devices_preview": summary,
    }
