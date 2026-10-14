"""FastAPI backend for real-time ASL inference and sentence management."""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Dict, Literal, Optional

import cv2
import numpy as np
import pyttsx3
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.feature_extraction import HandLandmarkExtractor  # noqa: E402

LOGGER = logging.getLogger("asl_backend")
MAX_IMAGE_DATA_URL_CHARS = 5_000_000
DEFAULT_ALLOWED_ORIGINS = "http://127.0.0.1:8000,http://localhost:8000"


class PredictRequest(BaseModel):
    image: str = Field(..., min_length=1, max_length=MAX_IMAGE_DATA_URL_CHARS,
                       description="Base64 image or data URL; payload size is limited.")


class PredictResponse(BaseModel):
    hand_detected: bool
    top_letter: Optional[str] = None
    predicted_letter: Optional[str] = None
    confidence: float = 0.0


class SentenceActionRequest(BaseModel):
    session_id: str = Field(default="default", min_length=1, max_length=128)
    action: Literal["append", "space", "backspace", "clear", "speak"]
    letter: Optional[str] = Field(default=None, min_length=1, max_length=1, pattern="^[A-Za-z]$")


class SentenceResponse(BaseModel):
    session_id: str
    sentence: str
    spoken: bool = False


class SentenceStore:
    """Thread-safe in-process sentence storage; not persistent across restarts."""

    def __init__(self) -> None:
        self._state: Dict[str, str] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> str:
        with self._lock:
            return self._state.get(session_id, "")

    def apply_action(self, session_id: str, action: str, letter: Optional[str]) -> str:
        with self._lock:
            sentence = self._state.get(session_id, "")
            if action == "append":
                if not letter or len(letter) != 1 or not letter.isalpha() or not letter.isascii():
                    raise ValueError("Action 'append' requires one ASCII alphabet letter.")
                sentence += letter.upper()
            elif action == "space":
                sentence += " "
            elif action == "backspace":
                sentence = sentence[:-1]
            elif action == "clear":
                sentence = ""
            self._state[session_id] = sentence
            return sentence


class TTSService:
    def __init__(self) -> None:
        self._engine = None
        self._lock = threading.Lock()
        self.enabled = False
        try:
            self._engine = pyttsx3.init()
            self.enabled = True
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("TTS unavailable: %s", exc)

    def speak_async(self, text: str) -> bool:
        if not self.enabled or not text.strip():
            return False
        threading.Thread(target=self._speak, args=(text,), daemon=True).start()
        return True

    def _speak(self, text: str) -> None:
        with self._lock:
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.error("TTS playback failed: %s", exc)


class ModelService:
    def __init__(self, model_path: Path, labels_path: Path, min_confidence: float) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("ASL_MIN_CONFIDENCE must be between 0 and 1.")
        self.model_path = model_path
        self.labels_path = labels_path
        self.min_confidence = min_confidence
        self.model: Optional[tf.keras.Model] = None
        self.labels: list[str] = []
        self.extractor: Optional[HandLandmarkExtractor] = None
        self._lock = threading.Lock()

    def load(self) -> None:
        if not self.model_path.exists():
            LOGGER.warning("Model file not found: %s", self.model_path)
            return
        if not self.labels_path.exists():
            LOGGER.warning("Labels file not found: %s", self.labels_path)
            return

        model = tf.keras.models.load_model(self.model_path)
        labels_raw = json.loads(self.labels_path.read_text(encoding="utf-8"))
        if not isinstance(labels_raw, list) or not labels_raw:
            raise ValueError("labels.json must contain a non-empty list.")
        labels = [str(item) for item in labels_raw]
        if len(set(labels)) != len(labels):
            raise ValueError("labels.json must not contain duplicate labels.")
        if len(model.output_shape) != 2 or model.output_shape[-1] != len(labels):
            raise ValueError(
                f"Model output count ({model.output_shape[-1]}) does not match "
                f"label count ({len(labels)})."
            )
        if len(model.input_shape) != 2 or model.input_shape[-1] != HandLandmarkExtractor.VECTOR_SIZE:
            raise ValueError("Model input must accept 63 normalized hand-landmark features.")

        extractor = HandLandmarkExtractor(
            max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6
        )
        self.model = model
        self.labels = labels
        self.extractor = extractor
        LOGGER.info("Model loaded with %d labels.", len(self.labels))

    def unload(self) -> None:
        if self.extractor:
            self.extractor.close()

    @property
    def ready(self) -> bool:
        return bool(self.model is not None and self.labels and self.extractor is not None)

    def predict(self, frame: np.ndarray) -> PredictResponse:
        if not self.ready:
            raise RuntimeError("Model service is not ready.")

        detection = self.extractor.process_frame(frame)
        if detection.features is None:
            return PredictResponse(hand_detected=False, confidence=0.0)

        with self._lock:
            probs = self.model.predict(detection.features[np.newaxis, ...], verbose=0)[0]

        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])
        top_letter = self.labels[top_idx]
        predicted_letter = top_letter if confidence >= self.min_confidence else None
        return PredictResponse(
            hand_detected=True,
            top_letter=top_letter,
            predicted_letter=predicted_letter,
            confidence=confidence,
        )


def decode_base64_image(image_data: str) -> np.ndarray:
    """Decode a bounded, validated base64 image into an OpenCV BGR frame."""
    try:
        payload = image_data.split(",", 1)[1] if image_data.startswith("data:") and "," in image_data else image_data
        raw_bytes = base64.b64decode(payload, validate=True)
        if not raw_bytes:
            raise ValueError("Image payload is empty.")
        np_buffer = np.frombuffer(raw_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            raise ValueError("Decoded image is invalid or empty.")
        return frame
    except (ValueError, binascii.Error) as exc:
        raise ValueError(f"Invalid base64 image payload: {exc}") from exc


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

model_path = Path(os.getenv("ASL_MODEL_PATH", str(ROOT_DIR / "models" / "asl_model.h5"))).expanduser()
labels_path = Path(os.getenv("ASL_LABELS_PATH", str(ROOT_DIR / "models" / "labels.json"))).expanduser()
frontend_path = ROOT_DIR / "frontend" / "index.html"
min_confidence = float(os.getenv("ASL_MIN_CONFIDENCE", "0.60"))
allowed_origins = [
    origin.strip()
    for origin in os.getenv("ASL_ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS).split(",")
    if origin.strip()
]

app = FastAPI(
    title="Real-Time ASL Recognition API",
    description="FastAPI backend for American Sign Language alphabet recognition.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

model_service = ModelService(model_path=model_path, labels_path=labels_path, min_confidence=min_confidence)
sentence_store = SentenceStore()
tts_service = TTSService()


@app.on_event("startup")
def startup_event() -> None:
    model_service.load()
    if not model_service.ready:
        LOGGER.warning(
            "Prediction endpoint will return 503 until model artifacts exist at %s and %s.",
            model_path,
            labels_path,
        )


@app.on_event("shutdown")
def shutdown_event() -> None:
    model_service.unload()


@app.get("/", include_in_schema=False)
def serve_frontend() -> FileResponse:
    if not frontend_path.exists():
        raise HTTPException(status_code=404, detail="Frontend file not found.")
    return FileResponse(frontend_path)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_ready": model_service.ready,
        "tts_enabled": tts_service.enabled,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if not model_service.ready:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first or configure ASL_MODEL_PATH and ASL_LABELS_PATH.",
        )
    try:
        frame = decode_base64_image(request.image)
        return model_service.predict(frame)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction failed due to server error.") from exc


@app.get("/sentence/{session_id}", response_model=SentenceResponse)
def get_sentence(session_id: str) -> SentenceResponse:
    sentence = sentence_store.get(session_id)
    return SentenceResponse(session_id=session_id, sentence=sentence, spoken=False)


@app.post("/sentence/action", response_model=SentenceResponse)
def sentence_action(request: SentenceActionRequest) -> SentenceResponse:
    if request.action == "speak":
        text = sentence_store.get(request.session_id)
        spoken = tts_service.speak_async(text)
        return SentenceResponse(session_id=request.session_id, sentence=text, spoken=spoken)

    try:
        sentence = sentence_store.apply_action(
            session_id=request.session_id,
            action=request.action,
            letter=request.letter,
        )
        return SentenceResponse(session_id=request.session_id, sentence=sentence, spoken=False)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
