import tensorflow as tf
from pathlib import Path

from app.ml.model_vgg16_cbam import (
    build_vgg16_cbam_model,
    IMG_SIZE,
    CLASS_NAMES,
)

DATA_ROOT = Path("dataset")
BATCH_SIZE = 16
EPOCHS = 20


def get_datasets():
    train_dir = DATA_ROOT / "train"
    val_dir = DATA_ROOT / "val"

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
        ]
    )

    def augment(x, y):
        return data_augmentation(x), y

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.map(augment, num_parallel_calls=autotune).prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    return train_ds, val_ds


def main():
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    train_ds, val_ds = get_datasets()
    model = build_vgg16_cbam_model()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(models_dir / "leaf_vgg16_cbam_best.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    model.save_weights(str(models_dir / "leaf_vgg16_cbam_final.h5"))


if __name__ == "__main__":
    main()

