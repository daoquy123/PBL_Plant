import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16

# Kích thước ảnh đầu vào
IMG_SIZE = (224, 224)

# 5 lớp: lá cải (3) + sâu thực thể + cỏ/nhiễu (ảnh vườn thực tế)
# Thư mục train/val phải trùng tên: la_khoe, la_vang, la_sau, sau, co
CLASS_NAMES = ["la_khoe", "la_vang", "la_sau", "sau", "co"]
NUM_CLASSES = len(CLASS_NAMES)


def channel_attention(input_feature, reduction_ratio: int = 8):
    """Channel Attention: trả lời 'Cái gì quan trọng?'."""
    channel = int(input_feature.shape[-1])

    # Dùng Lambda để tránh lỗi KerasTensor với tf.reduce_*
    avg_pool = layers.Lambda(
        lambda x: tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    )(input_feature)
    max_pool = layers.Lambda(
        lambda x: tf.reduce_max(x, axis=[1, 2], keepdims=True)
    )(input_feature)

    shared_mlp = models.Sequential(
        [
            layers.Dense(channel // reduction_ratio, activation="relu", use_bias=True),
            layers.Dense(channel, activation=None, use_bias=True),
        ]
    )

    avg_pool_flat = layers.Flatten()(avg_pool)
    max_pool_flat = layers.Flatten()(max_pool)

    mlp_avg = shared_mlp(avg_pool_flat)
    mlp_max = shared_mlp(max_pool_flat)

    cbam_feature = layers.Activation("sigmoid")(layers.Add()([mlp_avg, mlp_max]))
    cbam_feature = layers.Reshape((1, 1, channel))(cbam_feature)
    return layers.Multiply()([input_feature, cbam_feature])


def spatial_attention(input_feature, kernel_size: int = 7):
    """Spatial Attention: trả lời 'Ở đâu quan trọng?'."""
    avg_pool = layers.Lambda(
        lambda x: tf.reduce_mean(x, axis=-1, keepdims=True)
    )(input_feature)
    max_pool = layers.Lambda(
        lambda x: tf.reduce_max(x, axis=-1, keepdims=True)
    )(input_feature)

    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])

    cbam_feature = layers.Conv2D(
        filters=1,
        kernel_size=kernel_size,
        strides=1,
        padding="same",
        activation="sigmoid",
        use_bias=False,
    )(concat)

    return layers.Multiply()([input_feature, cbam_feature])


def cbam_block(input_feature, reduction_ratio: int = 8, kernel_size: int = 7):
    x = channel_attention(input_feature, reduction_ratio)
    x = spatial_attention(x, kernel_size)
    return x


def build_vgg16_cbam_model(img_size=IMG_SIZE, num_classes: int = NUM_CLASSES):
    """Xây dựng mô hình VGG16 + CBAM theo tài liệu đề tài."""
    base_model = VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=img_size + (3,),
    )

    for layer in base_model.layers:
        layer.trainable = False

    inputs = layers.Input(shape=img_size + (3,))
    x = tf.keras.applications.vgg16.preprocess_input(inputs)
    x = base_model(x)
    x = cbam_block(x, reduction_ratio=8, kernel_size=7)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_trained_model(weights_path: str | None = None):
    """Khởi tạo mô hình và nạp trọng số nếu có."""
    model = build_vgg16_cbam_model()
    if weights_path:
        model.load_weights(weights_path)
    return model


if __name__ == "__main__":
    m = build_vgg16_cbam_model()
    m.summary()

