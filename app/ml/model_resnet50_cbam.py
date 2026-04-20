import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

# Đồng bộ với model VGG hiện tại để dùng chung pipeline.
IMG_SIZE = (224, 224)
CLASS_NAMES = ["la_khoe", "la_vang", "la_sau", "sau", "co"]
NUM_CLASSES = len(CLASS_NAMES)


def channel_attention(input_feature, reduction_ratio: int = 8):
    channel = int(input_feature.shape[-1])
    avg_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2], keepdims=True))(input_feature)
    max_pool = layers.Lambda(lambda x: tf.reduce_max(x, axis=[1, 2], keepdims=True))(input_feature)

    shared_mlp = models.Sequential(
        [
            layers.Dense(max(channel // reduction_ratio, 1), activation="relu", use_bias=True),
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
    avg_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(input_feature)
    max_pool = layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(input_feature)
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


def build_resnet50_model(
    img_size=IMG_SIZE,
    num_classes: int = NUM_CLASSES,
    use_cbam: bool = False,
):
    base_model = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=img_size + (3,),
    )
    for layer in base_model.layers:
        layer.trainable = False

    inputs = layers.Input(shape=img_size + (3,))
    x = tf.keras.applications.resnet50.preprocess_input(inputs)
    x = base_model(x)
    if use_cbam:
        x = cbam_block(x, reduction_ratio=8, kernel_size=7)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = models.Model(inputs, outputs, name="resnet50_cbam" if use_cbam else "resnet50")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_trained_model(weights_path: str | None = None, use_cbam: bool = False):
    model = build_resnet50_model(use_cbam=use_cbam)
    if weights_path:
        model.load_weights(weights_path)
    return model
