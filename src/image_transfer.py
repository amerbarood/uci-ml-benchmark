"""Image classification by transfer learning from an ImageNet backbone.

Used for the Jute Pest (ResNet50) and RealWaste (MobileNetV2) datasets.
The backbone is frozen and only the new classification head is trained.

Folder layout: one sub-folder per class. If the data folder contains
train/ val/ test/ sub-folders those splits are used, otherwise 20% of the
images are held out for validation.

Usage:
    python src/image_transfer.py --data data/images/jute_pest --backbone resnet50
    python src/image_transfer.py --data data/images/realwaste --backbone mobilenetv2
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import applications, layers, models

IMG_SIZE = (224, 224)
BACKBONES = {
    "resnet50": (applications.ResNet50, applications.resnet50.preprocess_input),
    "mobilenetv2": (applications.MobileNetV2, applications.mobilenet_v2.preprocess_input),
}


def load_splits(root: str, batch_size: int):
    kw = dict(image_size=IMG_SIZE, batch_size=batch_size, label_mode="int")
    if os.path.isdir(os.path.join(root, "train")):
        train = tf.keras.utils.image_dataset_from_directory(os.path.join(root, "train"), shuffle=True, **kw)
        val = tf.keras.utils.image_dataset_from_directory(os.path.join(root, "val"), shuffle=False, **kw)
        test_dir = os.path.join(root, "test")
        test = tf.keras.utils.image_dataset_from_directory(test_dir, shuffle=False, **kw) if os.path.isdir(test_dir) else val
    else:
        train = tf.keras.utils.image_dataset_from_directory(root, validation_split=0.2, subset="training", seed=123, **kw)
        val = tf.keras.utils.image_dataset_from_directory(root, validation_split=0.2, subset="validation", seed=123, **kw)
        test = val
    return train, val, test


def build_model(backbone: str, num_classes: int) -> tf.keras.Model:
    base_cls, preprocess = BACKBONES[backbone]
    base = base_cls(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
    base.trainable = False

    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = augment(inputs)
    x = preprocess(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--backbone", default="mobilenetv2", choices=BACKBONES)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", default="results")
    args = p.parse_args()

    train, val, test = load_splits(args.data, args.batch_size)
    class_names = train.class_names
    autotune = tf.data.AUTOTUNE
    train = train.cache().shuffle(1000).prefetch(autotune)
    val = val.cache().prefetch(autotune)

    model = build_model(args.backbone, len(class_names))
    history = model.fit(train, validation_data=val, epochs=args.epochs)

    y_true, y_pred = [], []
    for images, labels in test:
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(model.predict(images, verbose=0), axis=1))
    print(classification_report(y_true, y_pred, target_names=class_names))

    tag = os.path.basename(os.path.normpath(args.data))
    fig_dir = os.path.join(args.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion matrix: {tag} ({args.backbone})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{tag}_cm.png"), dpi=110)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(history.history["accuracy"], label="train")
    a1.plot(history.history["val_accuracy"], label="val")
    a1.set_title("Accuracy")
    a1.legend()
    a2.plot(history.history["loss"], label="train")
    a2.plot(history.history["val_loss"], label="val")
    a2.set_title("Loss")
    a2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{tag}_training.png"), dpi=110)


if __name__ == "__main__":
    main()
