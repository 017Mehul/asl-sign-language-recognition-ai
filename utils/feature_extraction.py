"""Shared feature extraction utilities for ASL hand landmark pipelines."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

LOGGER = logging.getLogger(__name__)
MP_HANDS = mp.solutions.hands
MP_DRAWING = mp.solutions.drawing_utils


@dataclass
class DetectionOutput:
    """Container for hand detection results."""

    features: Optional[np.ndarray]
    landmarks: Optional[np.ndarray]
    handedness: Optional[str]
    raw_landmarks: Optional[object]


class HandLandmarkExtractor:
    """Extracts MediaPipe hand landmarks and converts them to normalized vectors."""

    VECTOR_SIZE = 63  # 21 landmarks * (x, y, z)

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.hands = MP_HANDS.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.hands.close()

    def process_frame(self, frame_bgr: np.ndarray) -> DetectionOutput:
        """Detect hand landmarks from a BGR frame and return normalized features."""
        if frame_bgr is None or frame_bgr.size == 0:
            LOGGER.warning("Received an empty frame for landmark extraction.")
            return DetectionOutput(None, None, None, None)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if not results.multi_hand_landmarks:
            return DetectionOutput(None, None, None, None)

        first_hand = results.multi_hand_landmarks[0]
        landmarks = self._landmarks_to_array(first_hand)
        features = self._normalize_and_flatten(landmarks).astype(np.float32)

        handedness = None
        if results.multi_handedness:
            handedness = results.multi_handedness[0].classification[0].label

        return DetectionOutput(features, landmarks, handedness, first_hand)

    @staticmethod
    def draw_landmarks(frame_bgr: np.ndarray, raw_landmarks: object) -> np.ndarray:
        """Draw detected landmarks on a frame for visualization."""
        if raw_landmarks is None:
            return frame_bgr

        MP_DRAWING.draw_landmarks(
            frame_bgr,
            raw_landmarks,
            MP_HANDS.HAND_CONNECTIONS,
        )
        return frame_bgr

    @staticmethod
    def _landmarks_to_array(hand_landmarks: object) -> np.ndarray:
        points = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
        return np.asarray(points, dtype=np.float32)

    @staticmethod
    def _normalize_and_flatten(landmarks: np.ndarray) -> np.ndarray:
        # Translate landmarks to wrist origin for translation invariance.
        centered = landmarks - landmarks[0]

        # Use maximum XY radial distance as a scale factor for size invariance.
        scale = float(np.max(np.linalg.norm(centered[:, :2], axis=1)))
        if scale < 1e-6:
            scale = float(np.max(np.linalg.norm(centered, axis=1)))
        if scale < 1e-6:
            return np.zeros(HandLandmarkExtractor.VECTOR_SIZE, dtype=np.float32)

        normalized = centered / scale
        return normalized.flatten()


def extract_features_from_image(
    image_path: str,
    extractor: Optional[HandLandmarkExtractor] = None,
) -> Optional[np.ndarray]:
    """Load an image file and return hand landmark feature vector."""
    local_extractor = extractor or HandLandmarkExtractor(static_image_mode=True)
    created_extractor = extractor is None

    try:
        image = cv2.imread(image_path)
        if image is None:
            LOGGER.warning("Failed to read image file: %s", image_path)
            return None
        detection = local_extractor.process_frame(image)
        return detection.features
    finally:
        if created_extractor:
            local_extractor.close()
