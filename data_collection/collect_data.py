"""Collect ASL hand landmark training data from a webcam feed."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Iterable, List

import cv2

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.feature_extraction import HandLandmarkExtractor  # noqa: E402

LOGGER = logging.getLogger("collect_data")
DEFAULT_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def parse_labels(value: str) -> List[str]:
    cleaned = value.strip().upper()
    if not cleaned:
        return DEFAULT_LABELS

    if "," in cleaned:
        labels = [item.strip() for item in cleaned.split(",") if item.strip()]
    else:
        labels = [char for char in cleaned if char.strip()]

    return [label[0] for label in labels if label]


def write_rows(csv_path: Path, rows: Iterable[List[float]], append: bool) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    mode = "a" if append and file_exists else "w"
    with csv_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if mode == "w":
            header = ["label"] + [f"f{i}" for i in range(HandLandmarkExtractor.VECTOR_SIZE)]
            writer.writerow(header)
        writer.writerows(rows)


def run_collection(args: argparse.Namespace) -> None:
    labels = parse_labels(args.labels)
    if not labels:
        raise ValueError("No labels were provided for data collection.")

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to access webcam index {args.camera_index}.")

    extractor = HandLandmarkExtractor(
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    collected_rows: List[List[float]] = []
    quit_requested = False

    LOGGER.info("Starting data collection for labels: %s", labels)
    LOGGER.info("Controls: [s] start/pause, [n] skip label, [q] quit")

    try:
        for label in labels:
            count = 0
            capture_on = False
            last_capture = 0.0
            LOGGER.info("Preparing label '%s'. Press 's' to start capture.", label)

            while count < args.samples_per_label:
                ok, frame = cap.read()
                if not ok:
                    LOGGER.warning("Failed to read frame from webcam. Retrying...")
                    continue

                frame = cv2.flip(frame, 1)
                detection = extractor.process_frame(frame)
                HandLandmarkExtractor.draw_landmarks(frame, detection.raw_landmarks)

                now = time.time()
                if (
                    capture_on
                    and detection.features is not None
                    and (now - last_capture) >= args.capture_interval
                ):
                    collected_rows.append([label, *detection.features.tolist()])
                    count += 1
                    last_capture = now

                status = "CAPTURING" if capture_on else "PAUSED"
                guidance = "Show hand sign clearly in frame."
                if detection.features is None:
                    guidance = "No hand detected. Move hand fully into frame."

                cv2.putText(
                    frame,
                    f"Label: {label} | Samples: {count}/{args.samples_per_label}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (50, 220, 50),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Status: {status}",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 180, 50),
                    2,
                )
                cv2.putText(
                    frame,
                    guidance,
                    (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "Keys: s=start/pause, n=skip label, q=quit",
                    (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (180, 180, 180),
                    2,
                )

                cv2.imshow("ASL Data Collection", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("s"):
                    capture_on = not capture_on
                elif key == ord("n"):
                    LOGGER.info("Skipping label '%s' at %d samples.", label, count)
                    break
                elif key == ord("q"):
                    quit_requested = True
                    break

            if quit_requested:
                break

    finally:
        extractor.close()
        cap.release()
        cv2.destroyAllWindows()

    if not collected_rows:
        LOGGER.warning("No samples were collected. CSV file not updated.")
        return

    write_rows(Path(args.output_csv), collected_rows, append=args.append)
    LOGGER.info("Collection complete. Saved %d rows to %s", len(collected_rows), args.output_csv)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect ASL landmark data using webcam.")
    parser.add_argument(
        "--output_csv",
        type=str,
        default=str(ROOT_DIR / "dataset" / "landmarks.csv"),
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        help="Labels to collect. Examples: ABCD or A,B,C,D",
    )
    parser.add_argument(
        "--samples_per_label",
        type=int,
        default=200,
        help="Number of samples to collect per label.",
    )
    parser.add_argument(
        "--capture_interval",
        type=float,
        default=0.10,
        help="Seconds between automatic sample captures while running.",
    )
    parser.add_argument(
        "--camera_index",
        type=int,
        default=0,
        help="Webcam index for OpenCV capture.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append rows to existing CSV instead of overwriting.",
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
        run_collection(args)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Data collection failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
