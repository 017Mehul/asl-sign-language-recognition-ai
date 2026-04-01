"""Train a deep learning model for ASL alphabet classification."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.feature_extraction import HandLandmarkExtractor  # noqa: E402

LOGGER = logging.getLogger("train_model")
DEFAULT_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def parse_labels(value: str) -> List[str]:
    cleaned = value.strip().upper()
    if not cleaned:
        return DEFAULT_LABELS

    if "," in cleaned:
        labels = [item.strip() for item in cleaned.split(",") if item.strip()]
    else:
        labels = [char for char in cleaned if char.strip()]

    return [label[0] for label in labels if label]


def load_csv_dataset(csv_path: Path, include_labels: set[str]) -> Tuple[np.ndarray, np.ndarray]:
    LOGGER.info("Loading CSV dataset: %s", csv_path)
    dataframe = pd.read_csv(csv_path)
    if "label" not in dataframe.columns:
        raise ValueError("CSV dataset must include a 'label' column.")

    dataframe["label"] = dataframe["label"].astype(str).str.upper()
    dataframe = dataframe[dataframe["label"].isin(include_labels)]
    if dataframe.empty:
        raise ValueError("CSV dataset contains no rows for selected labels.")

    labels = dataframe["label"].values
    features = dataframe.drop(columns=["label"]).values.astype(np.float32)
    return features, labels


def load_image_dataset(image_dataset_dir: Path, include_labels: set[str]) -> Tuple[np.ndarray, np.ndarray]:
    LOGGER.info("Loading image dataset from: %s", image_dataset_dir)
    extractor = HandLandmarkExtractor(static_image_mode=True, min_detection_confidence=0.5)

    features: List[np.ndarray] = []
    labels: List[str] = []

    try:
        for label_dir in sorted(image_dataset_dir.iterdir()):
            if not label_dir.is_dir():
                continue

            label = label_dir.name.upper()
            if label not in include_labels:
                continue

            image_files = sorted(
                [
                    path
                    for path in label_dir.iterdir()
                    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
                ]
            )

            if not image_files:
                LOGGER.warning("No image files found under %s", label_dir)
                continue

            for image_path in image_files:
                image = cv2.imread(str(image_path))
                if image is None:
                    LOGGER.warning("Unreadable image skipped: %s", image_path)
                    continue

                detection = extractor.process_frame(image)
                if detection.features is None:
                    continue

                features.append(detection.features)
                labels.append(label)
    finally:
        extractor.close()

    if not features:
        raise ValueError("No valid hand landmark samples found in image dataset.")

    return np.asarray(features, dtype=np.float32), np.asarray(labels)


def build_model(input_dim: int, class_count: int, learning_rate: float) -> tf.keras.Model:
    normalizer = tf.keras.layers.Normalization(axis=-1, name="feature_normalizer")
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,), name="input_features"),
            normalizer,
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(class_count, activation="softmax", name="class_probs"),
        ],
        name="asl_alphabet_classifier",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def run_training(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    selected_labels = parse_labels(args.labels)
    include_labels = set(selected_labels)
    if not include_labels:
        raise ValueError("No target labels provided for training.")

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        LOGGER.info("Detected GPU devices: %s", [gpu.name for gpu in gpus])
    else:
        LOGGER.warning("No GPU detected. Training will run on CPU.")

    csv_path = Path(args.csv_path)
    image_dataset_dir = Path(args.image_dataset_dir)
    model_output = Path(args.model_output)
    labels_output = Path(args.labels_output)
    metrics_output = Path(args.metrics_output)
    history_output = Path(args.history_output)

    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []

    if csv_path.exists():
        x_csv, y_csv = load_csv_dataset(csv_path, include_labels=include_labels)
        x_parts.append(x_csv)
        y_parts.append(y_csv)
        LOGGER.info("Loaded %d samples from CSV dataset.", len(y_csv))
    else:
        LOGGER.info("CSV dataset not found at %s. Skipping CSV source.", csv_path)

    if image_dataset_dir.exists():
        try:
            x_img, y_img = load_image_dataset(image_dataset_dir, include_labels=include_labels)
            x_parts.append(x_img)
            y_parts.append(y_img)
            LOGGER.info("Loaded %d samples from image dataset.", len(y_img))
        except ValueError as exc:
            LOGGER.warning("Skipping image dataset source: %s", exc)
    else:
        LOGGER.info("Image dataset directory not found at %s. Skipping image source.", image_dataset_dir)

    if not x_parts:
        raise FileNotFoundError(
            "No training data found. Provide dataset/landmarks.csv and/or dataset/asl_alphabet_images."
        )

    x = np.vstack(x_parts).astype(np.float32)
    y = np.concatenate(y_parts).astype(str)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    classes = encoder.classes_.tolist()
    LOGGER.info("Final training dataset: %d samples across %d classes.", len(y), len(classes))

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y_encoded,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y_encoded,
    )

    model = build_model(input_dim=x.shape[1], class_count=len(classes), learning_rate=args.learning_rate)
    normalizer = model.get_layer("feature_normalizer")
    normalizer.adapt(x_train)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    LOGGER.info("Starting training for up to %d epochs...", args.epochs)
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    probabilities = model.predict(x_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    accuracy = float(accuracy_score(y_test, predictions))
    report_dict = classification_report(y_test, predictions, target_names=classes, output_dict=True)

    model_output.parent.mkdir(parents=True, exist_ok=True)
    labels_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    history_output.parent.mkdir(parents=True, exist_ok=True)

    model.save(model_output)
    labels_output.write_text(json.dumps(classes, indent=2), encoding="utf-8")
    history_frame = pd.DataFrame(history.history)
    history_frame.to_csv(history_output, index=False)

    metrics_payload = {
        "test_accuracy": accuracy,
        "num_classes": len(classes),
        "classes": classes,
        "classification_report": report_dict,
    }
    metrics_output.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    LOGGER.info("Training complete.")
    LOGGER.info("Test accuracy: %.4f", accuracy)
    LOGGER.info("Saved model to: %s", model_output)
    LOGGER.info("Saved label mapping to: %s", labels_output)
    LOGGER.info("Saved metrics to: %s", metrics_output)
    LOGGER.info("Saved training history to: %s", history_output)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ASL alphabet classifier.")
    parser.add_argument(
        "--csv_path",
        type=str,
        default=str(ROOT_DIR / "dataset" / "landmarks.csv"),
        help="Path to CSV dataset with label + landmark features.",
    )
    parser.add_argument(
        "--image_dataset_dir",
        type=str,
        default=str(ROOT_DIR / "dataset" / "asl_alphabet_images"),
        help="Directory containing image subfolders named by labels.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        help="Labels to train. Examples: ABC...Z or A,B,C",
    )
    parser.add_argument(
        "--model_output",
        type=str,
        default=str(ROOT_DIR / "models" / "asl_model.h5"),
        help="Output path for trained model.",
    )
    parser.add_argument(
        "--labels_output",
        type=str,
        default=str(ROOT_DIR / "models" / "labels.json"),
        help="Output path for class labels JSON.",
    )
    parser.add_argument(
        "--metrics_output",
        type=str,
        default=str(ROOT_DIR / "models" / "training_metrics.json"),
        help="Output path for evaluation metrics JSON.",
    )
    parser.add_argument(
        "--history_output",
        type=str,
        default=str(ROOT_DIR / "models" / "training_history.csv"),
        help="Output path for per-epoch training history CSV.",
    )
    parser.add_argument("--epochs", type=int, default=80, help="Maximum training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        run_training(args)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Training failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
