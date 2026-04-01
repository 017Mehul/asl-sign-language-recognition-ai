"""Run real-time ASL recognition from webcam using a trained model."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from collections import Counter, deque
from pathlib import Path
from typing import Deque, Optional, Tuple

import cv2
import numpy as np
import pyttsx3
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.feature_extraction import HandLandmarkExtractor  # noqa: E402

LOGGER = logging.getLogger("realtime_prediction")


class PredictionSmoother:
    """Smooths per-frame predictions using majority vote in a sliding window."""

    def __init__(self, window_size: int = 6) -> None:
        self.buffer: Deque[Tuple[str, float]] = deque(maxlen=window_size)

    def update(self, letter: Optional[str], confidence: float) -> Tuple[Optional[str], float]:
        if letter is None:
            return None, 0.0

        self.buffer.append((letter, confidence))
        letters = [item[0] for item in self.buffer]
        if not letters:
            return None, 0.0

        most_common_letter, count = Counter(letters).most_common(1)[0]
        min_votes = max(2, len(self.buffer) // 2 + 1)
        if count < min_votes:
            return None, 0.0

        confidences = [item[1] for item in self.buffer if item[0] == most_common_letter]
        return most_common_letter, float(np.mean(confidences))


class TTSEngine:
    """Simple async wrapper around pyttsx3."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.engine = None
        self.enabled = False
        try:
            self.engine = pyttsx3.init()
            self.enabled = True
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("TTS disabled. Failed to initialize pyttsx3: %s", exc)

    def speak_async(self, text: str) -> None:
        if not self.enabled or not text.strip():
            return

        threading.Thread(target=self._speak, args=(text,), daemon=True).start()

    def _speak(self, text: str) -> None:
        with self.lock:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.error("TTS error: %s", exc)


def load_labels(labels_path: Path) -> list[str]:
    with labels_path.open("r", encoding="utf-8") as handle:
        labels = json.load(handle)
    if not isinstance(labels, list) or not labels:
        raise ValueError("labels.json must contain a non-empty list of class names.")
    return [str(label) for label in labels]


def run_realtime(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path)
    labels_path = Path(args.labels_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    model = tf.keras.models.load_model(model_path)
    labels = load_labels(labels_path)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to access webcam index {args.camera_index}.")

    extractor = HandLandmarkExtractor(
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    smoother = PredictionSmoother(window_size=args.smoothing_window)
    tts = TTSEngine()

    sentence = ""
    current_letter: Optional[str] = None
    current_confidence: float = 0.0

    LOGGER.info("Real-time inference started.")
    LOGGER.info("Controls: a=append, space=space, b=backspace, c=clear, t=speak, q=quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                LOGGER.warning("Failed to read frame from webcam.")
                continue

            frame = cv2.flip(frame, 1)
            detection = extractor.process_frame(frame)
            HandLandmarkExtractor.draw_landmarks(frame, detection.raw_landmarks)

            raw_letter: Optional[str] = None
            raw_confidence = 0.0

            if detection.features is not None:
                probs = model.predict(detection.features[np.newaxis, ...], verbose=0)[0]
                top_idx = int(np.argmax(probs))
                raw_letter = labels[top_idx]
                raw_confidence = float(probs[top_idx])

                if raw_confidence < args.confidence_threshold:
                    raw_letter = None

            smooth_letter, smooth_conf = smoother.update(raw_letter, raw_confidence)
            if smooth_letter is not None:
                current_letter = smooth_letter
                current_confidence = smooth_conf

            letter_display = current_letter if current_letter else "--"
            conf_display = current_confidence * 100.0

            cv2.putText(
                frame,
                f"Predicted: {letter_display} ({conf_display:.1f}%)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (70, 255, 70),
                2,
            )
            cv2.putText(
                frame,
                f"Sentence: {sentence or '-'}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "Keys: a=append, space=space, b=backspace, c=clear, t=speak, q=quit",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (180, 180, 180),
                2,
            )

            cv2.imshow("ASL Real-Time Recognition", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("a"):
                if current_letter:
                    sentence += current_letter
            elif key == 32:  # Space key
                sentence += " "
            elif key == ord("b"):
                sentence = sentence[:-1]
            elif key == ord("c"):
                sentence = ""
            elif key == ord("t"):
                tts.speak_async(sentence.strip())
            elif key == ord("q"):
                break

    finally:
        extractor.close()
        cap.release()
        cv2.destroyAllWindows()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real-time ASL alphabet inference from webcam.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(ROOT_DIR / "models" / "asl_model.h5"),
        help="Path to trained model file.",
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        default=str(ROOT_DIR / "models" / "labels.json"),
        help="Path to labels JSON file.",
    )
    parser.add_argument("--camera_index", type=int, default=0, help="Webcam index.")
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.60,
        help="Minimum confidence to accept a prediction.",
    )
    parser.add_argument(
        "--smoothing_window",
        type=int,
        default=6,
        help="Number of recent predictions used for smoothing.",
    )
    parser.add_argument(
        "--min_detection_confidence",
        type=float,
        default=0.6,
        help="MediaPipe minimum detection confidence.",
    )
    parser.add_argument(
        "--min_tracking_confidence",
        type=float,
        default=0.6,
        help="MediaPipe minimum tracking confidence.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        run_realtime(args)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Real-time prediction failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
