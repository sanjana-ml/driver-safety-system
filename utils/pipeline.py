"""
pipeline.py
Ties every stage of the implementation strategy together into a single
per-frame call:

Webcam frame -> Face + landmark detection -> Frame quality check ->
Adaptive cue selection -> CNN+CBAM prediction -> Sliding window ->
Confidence calculation -> Threshold check -> DetectionResult

Both predict.py (console mode) and gui/app.py (GUI mode) drive the camera
loop themselves and call DriverSafetyPipeline.process_frame() once per
frame, so the decision logic lives in exactly one place.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

import config
from cbam.cbam import CUSTOM_OBJECTS
from utils.cue_selector import CueReadings, compute_cue_readings, cue_based_drowsiness_score
from utils.exceptions import ModelNotFoundError
from utils.image_utils import preprocess_face
from utils.landmarks import FaceBox, LandmarkDetector
from utils.logger import get_logger
from utils.quality import QualityReport, assess_frame_quality
from utils.sliding_window import SlidingWindow

logger = get_logger(__name__)


@dataclass
class DetectionResult:
    frame_quality_ok: bool
    quality_label: str
    face_box: Optional[FaceBox] = None
    landmarks: Optional[np.ndarray] = None
    cue_readings: Optional[CueReadings] = None
    cnn_probability_drowsy: float = 0.0
    cue_score: float = 0.0
    confidence: float = 0.0
    window_drowsy_ratio: float = 0.0
    status: str = "Insufficient Data"
    alert_triggered: bool = False
    fps: float = 0.0
    active_cues_label: str = "none"


class DriverSafetyPipeline:
    def __init__(
        self,
        model_path: Optional[str] = None,
        window_size: int = config.SLIDING_WINDOW_SIZE,
        probability_threshold: float = config.PROBABILITY_THRESHOLD,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
    ) -> None:
        self.probability_threshold = probability_threshold
        self.confidence_threshold = confidence_threshold

        self.landmark_detector = LandmarkDetector()
        self.sliding_window = SlidingWindow(window_size=window_size)
        self.model = self._load_model(model_path)

        self._last_alert_time = 0.0
        self._no_face_since: Optional[float] = None
        self._prev_tick = time.time()

    # ------------------------------------------------------------------ #
    def _load_model(self, model_path: Optional[str]):
        import tensorflow as tf

        path = model_path or str(config.MODEL_PATH)
        if not config.MODEL_PATH.exists() and model_path is None:
            raise ModelNotFoundError(
                f"No trained model found at '{path}'. Run 'python train.py' "
                f"first, or place a compatible .h5 file at that path."
            )
        model = tf.keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS)
        logger.info("Loaded trained model from %s", path)
        return model

    # ------------------------------------------------------------------ #
    def _update_fps(self) -> float:
        now = time.time()
        dt = now - self._prev_tick
        self._prev_tick = now
        if dt <= 0:
            return 0.0
        return 1.0 / dt

    def process_frame(self, frame_bgr: np.ndarray) -> DetectionResult:
        fps = self._update_fps()

        # MediaPipe Face Mesh needs the color (RGB-convertible) frame; the
        # grayscale copy is still used for brightness/quality checks and
        # for the CNN input crop.
        face_box, landmarks = self.landmark_detector.detect(frame_bgr)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if face_box is None:
            # Track how long the driver has been completely absent from
            # frame -- distinct from "present but poor quality", which
            # should never itself trigger an alert. Sustained absence is
            # treated as its own alert condition (looking away, slumped
            # out of frame, camera blocked, etc.).
            if self._no_face_since is None:
                self._no_face_since = time.time()
            absence_duration = time.time() - self._no_face_since

            status = "Insufficient Data"
            alert_triggered = False
            if absence_duration >= config.NO_FACE_ALERT_SECONDS:
                status = "Driver Not Visible"
                now = time.time()
                if now - self._last_alert_time >= config.ALERT_COOLDOWN_SEC:
                    alert_triggered = True
                    self._last_alert_time = now

            return DetectionResult(
                frame_quality_ok=False,
                quality_label="Insufficient Data (no face detected)",
                status=status,
                alert_triggered=alert_triggered,
                fps=fps,
            )

        self._no_face_since = None
        quality = assess_frame_quality(gray, face_box, landmarks)

        if not quality.overall_ok:
            return DetectionResult(
                frame_quality_ok=False,
                quality_label=quality.as_label(),
                face_box=face_box,
                landmarks=landmarks,
                status="Insufficient Data",
                fps=fps,
            )

        cue_reading = compute_cue_readings(gray, landmarks, quality)
        cue_score = cue_based_drowsiness_score(cue_reading)

        face_input = preprocess_face(gray, face_box)
        batch = np.expand_dims(face_input, axis=0)
        probs = self.model.predict(batch, verbose=0)[0]
        drowsy_index = config.CLASS_NAMES.index("Drowsy")
        prob_drowsy = float(probs[drowsy_index])

        # Blend CNN visual probability with the interpretable rule-based
        # cue score to form the confidence metric. Confidence is high when
        # both lines of evidence agree; it drops when they disagree, which
        # is exactly the "weak/incomplete evidence" case that must not
        # trigger a false alarm.
        agreement = 1.0 - abs(prob_drowsy - cue_score)
        confidence = float(np.clip(0.5 * agreement + 0.5 * max(prob_drowsy, cue_score) * agreement, 0.0, 1.0))

        frame_is_drowsy = prob_drowsy >= 0.5
        self.sliding_window.push(frame_is_drowsy, prob_drowsy)

        window_ratio = self.sliding_window.drowsy_vote_ratio()
        smoothed_probability = self.sliding_window.mean_probability()

        alert_triggered = False
        status = "Not Drowsy"

        majority_drowsy = self.sliding_window.majority_vote()
        if (
            majority_drowsy
            and smoothed_probability >= self.probability_threshold
            and confidence >= self.confidence_threshold
        ):
            status = "Drowsy"
            now = time.time()
            if now - self._last_alert_time >= config.ALERT_COOLDOWN_SEC:
                alert_triggered = True
                self._last_alert_time = now

        return DetectionResult(
            frame_quality_ok=True,
            quality_label=quality.as_label(),
            face_box=face_box,
            landmarks=landmarks,
            cue_readings=cue_reading,
            cnn_probability_drowsy=prob_drowsy,
            cue_score=cue_score,
            confidence=confidence,
            window_drowsy_ratio=window_ratio,
            status=status,
            alert_triggered=alert_triggered,
            fps=fps,
            active_cues_label=cue_reading.label(),
        )

    def reset(self) -> None:
        self.sliding_window.reset()
        self._last_alert_time = 0.0
        self._no_face_since = None

    def shutdown(self) -> None:
        self.landmark_detector.close()