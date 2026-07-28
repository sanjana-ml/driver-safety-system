"""
pipeline.py
Ties every stage of the implementation strategy together into a single
per-frame call:

Webcam frame -> Face + landmark detection -> Frame quality gate (with
hysteresis) -> Driver calibration (once per session) -> Adaptive cue
selection (personalized EAR/MAR) -> CNN+CBAM prediction -> PERCLOS ->
Head-pose temporal monitoring -> Sliding window -> Multi-signal fusion ->
DetectionResult.

Both predict.py (console mode) and gui/app.py (GUI mode) drive the camera
loop themselves and call DriverSafetyPipeline.process_frame() once per
frame, so the decision logic lives in exactly one place.

Session lifecycle
------------------
Every time reset() is called (a brand new DriverSafetyPipeline, or an
explicit reset() at the start of a new "Start Monitoring" session), the
pipeline:
  1. Clears the sliding window, cue selector, PERCLOS tracker, head-pose
     monitor, and quality gate.
  2. Starts a fresh CalibrationSession. Until it finishes (5-10 configurable
     seconds, config.CALIBRATION_DURATION_SEC), process_frame() reports
     status="Calibrating" and never raises a Drowsy/Not Drowsy verdict.
  3. Once calibration finishes, personalized EAR/MAR thresholds (or the
     fixed config defaults, if calibration could not collect enough
     reliable samples) are installed into the cue selector for the rest of
     the session.

Frame quality
-------------
Every frame's raw quality is still assessed by
utils.quality.assess_frame_quality() exactly as before, but the decision of
whether to actually trust it is now debounced by a utils.quality.QualityGate
per config.QUALITY_GRACE_FRAMES -- a brief, isolated bad-quality frame
(momentary landmark jitter, a slight out-of-frame movement) is absorbed
instead of immediately being treated as a quality failure. During an
absorbed blip, the pipeline does not run cue/CNN/PERCLOS/head-pose scoring
on that frame's (untrustworthy) landmarks; it instead returns the last
known-good reading so downstream displays/logs don't flicker or reset, and
always reports status="Insufficient Data" for that frame rather than ever
propagating a stale "Drowsy" verdict.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

import config
from cbam.cbam import CUSTOM_OBJECTS
from utils.calibration import CalibrationSession, PersonalizedThresholds
from utils.cue_selector import CueReadings, CueSelector
from utils.exceptions import ModelNotFoundError
from utils.head_pose import HeadPoseMonitor, HeadPoseStatus
from utils.image_utils import preprocess_face, enhance_low_light
from utils.landmarks import FaceBox, LandmarkDetector
from utils.logger import get_logger
from utils.perclos import PERCLOSCalculator
from utils.quality import QualityGate, QualityReport, assess_frame_quality
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
    perclos_value: float = 0.0
    perclos_ready: bool = False
    perclos_drowsy: bool = False
    head_pose_prolonged_tilt: bool = False
    head_pose_repeated_nod: bool = False
    status: str = "Insufficient Data"
    alert_triggered: bool = False
    fps: float = 0.0
    active_cues_label: str = "none"
    calibration_in_progress: bool = False
    calibration_progress: float = 0.0
    calibration_remaining: float = 0.0
    personalized_thresholds_active: bool = False


class DriverSafetyPipeline:
    def __init__(
        self,
        model_path: Optional[str] = None,
        window_size: int = config.SLIDING_WINDOW_SIZE,
        probability_threshold: float = config.PROBABILITY_THRESHOLD,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
        calibration_duration_sec: float = config.CALIBRATION_DURATION_SEC,
    ) -> None:
        self.probability_threshold = probability_threshold
        self.confidence_threshold = confidence_threshold

        self.landmark_detector = LandmarkDetector()
        self.sliding_window = SlidingWindow(window_size=window_size)
        self.cue_selector = CueSelector()
        self.quality_gate = QualityGate()
        self.calibration = CalibrationSession(duration_sec=calibration_duration_sec)
        self.perclos = PERCLOSCalculator()
        self.head_pose_monitor = HeadPoseMonitor()
        self.model = self._load_model(model_path)

        self._last_alert_time = 0.0
        self._no_face_since: Optional[float] = None
        self._prev_tick = time.time()
        self._thresholds: Optional[PersonalizedThresholds] = None
        self._last_good_result: Optional[DetectionResult] = None

        # A fresh pipeline always starts with a calibration phase.
        self.calibration.start()

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

    def _update_fps(self) -> float:
        now = time.time()
        dt = now - self._prev_tick
        self._prev_tick = now
        if dt <= 0:
            return 0.0
        return 1.0 / dt

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def process_frame(self, frame_bgr: np.ndarray) -> DetectionResult:
        fps = self._update_fps()
        now = time.time()

        frame_bgr = enhance_low_light(frame_bgr)
        face_box, landmarks = self.landmark_detector.detect(frame_bgr)

        if face_box is None:
            return self._handle_no_face(fps, now)

        self._no_face_since = None
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        raw_quality = assess_frame_quality(gray, face_box, landmarks, frame_bgr)
        stable_ok = self.quality_gate.update(raw_quality.overall_ok)

        if not stable_ok:
            # Sustained bad quality -- a genuine, non-transient failure.
            return self._insufficient_data_result(fps, raw_quality, face_box, landmarks)

        if not raw_quality.overall_ok:
            # This particular frame's landmarks/orientation are not
            # trustworthy, but the grace period is absorbing the blip.
            # Do not run cue/CNN scoring on unreliable data for this frame --
            # freeze on the last known-good reading instead.
            return self._frozen_result(fps, raw_quality, face_box, landmarks)

        # -- Quality is genuinely good this frame -- #
        cue_reading = self.cue_selector.compute(frame_bgr, landmarks, raw_quality)

        if self.calibration.in_progress:
            if cue_reading.eyes_available and cue_reading.mouth_available:
                self.calibration.add_sample(cue_reading.corrected_ear, cue_reading.mar)

            if not self.calibration.is_time_elapsed():
                return self._calibration_result(fps, raw_quality, face_box, landmarks, cue_reading)

            # Calibration duration has just elapsed on this frame: finalize
            # and immediately re-derive the cue reading with the freshly
            # personalized thresholds so this frame is scored consistently
            # with every frame that follows it.
            thresholds = self.calibration.finalize()
            self._thresholds = thresholds
            self.cue_selector.set_thresholds(thresholds)
            cue_reading = self.cue_selector.compute(frame_bgr, landmarks, raw_quality)

        return self._process_detection_frame(
            fps, raw_quality, face_box, landmarks, cue_reading, frame_bgr, gray, now
        )

    # ------------------------------------------------------------------ #
    # Branches
    # ------------------------------------------------------------------ #
    def _handle_no_face(self, fps: float, now: float) -> DetectionResult:
        if self._no_face_since is None:
            self._no_face_since = now
        absence_duration = now - self._no_face_since

        status = "Insufficient Data"
        alert_triggered = False
        if absence_duration >= config.FACE_NOT_DETECTED_GRACE_SEC:
            status = "Face Not Detected"
            if now - self._last_alert_time >= config.ALERT_COOLDOWN_SEC:
                alert_triggered = True
                self._last_alert_time = now

        return DetectionResult(
            frame_quality_ok=False,
            quality_label="Insufficient Data (no face detected)",
            status=status,
            alert_triggered=alert_triggered,
            fps=fps,
            calibration_in_progress=self.calibration.in_progress,
            calibration_progress=self.calibration.progress_ratio,
            calibration_remaining=self.calibration.remaining,
        )

    def _insufficient_data_result(
        self,
        fps: float,
        quality: QualityReport,
        face_box: FaceBox,
        landmarks: np.ndarray,
    ) -> DetectionResult:
        return DetectionResult(
            frame_quality_ok=False,
            quality_label=quality.as_label(),
            face_box=face_box,
            landmarks=landmarks,
            status="Insufficient Data",
            fps=fps,
            calibration_in_progress=self.calibration.in_progress,
            calibration_progress=self.calibration.progress_ratio,
            calibration_remaining=self.calibration.remaining,
        )

    def _frozen_result(
        self,
        fps: float,
        quality: QualityReport,
        face_box: FaceBox,
        landmarks: np.ndarray,
    ) -> DetectionResult:
        base = self._last_good_result
        if base is None:
            # No prior good reading to freeze on yet (e.g. very start of a
            # session) -- fall back to a plain "Insufficient Data" result.
            return self._insufficient_data_result(fps, quality, face_box, landmarks)

        return DetectionResult(
            frame_quality_ok=False,
            quality_label=f"{quality.as_label()} (transient, holding last reading)",
            face_box=face_box,
            landmarks=landmarks,
            cue_readings=base.cue_readings,
            cnn_probability_drowsy=base.cnn_probability_drowsy,
            cue_score=base.cue_score,
            confidence=base.confidence,
            window_drowsy_ratio=base.window_drowsy_ratio,
            perclos_value=base.perclos_value,
            perclos_ready=base.perclos_ready,
            perclos_drowsy=base.perclos_drowsy,
            head_pose_prolonged_tilt=base.head_pose_prolonged_tilt,
            head_pose_repeated_nod=base.head_pose_repeated_nod,
            # Never propagate a "Drowsy" verdict from a frame we don't
            # actually trust -- tracking is unreliable, so the safe status
            # is always "Insufficient Data" here.
            status="Insufficient Data",
            alert_triggered=False,
            fps=fps,
            active_cues_label=base.active_cues_label,
            calibration_in_progress=False,
            calibration_progress=1.0,
            calibration_remaining=0.0,
            personalized_thresholds_active=base.personalized_thresholds_active,
        )

    def _calibration_result(
        self,
        fps: float,
        quality: QualityReport,
        face_box: FaceBox,
        landmarks: np.ndarray,
        cue_reading: CueReadings,
    ) -> DetectionResult:
        return DetectionResult(
            frame_quality_ok=True,
            quality_label=quality.as_label(),
            face_box=face_box,
            landmarks=landmarks,
            cue_readings=cue_reading,
            status="Calibrating",
            fps=fps,
            active_cues_label=cue_reading.label(),
            calibration_in_progress=True,
            calibration_progress=self.calibration.progress_ratio,
            calibration_remaining=self.calibration.remaining,
        )

    def _process_detection_frame(
        self,
        fps: float,
        quality: QualityReport,
        face_box: FaceBox,
        landmarks: np.ndarray,
        cue_reading: CueReadings,
        frame_bgr: np.ndarray,
        gray: np.ndarray,
        now: float,
    ) -> DetectionResult:
        # -- CNN + CBAM prediction -- #
        face_input = preprocess_face(gray, face_box)
        batch = np.expand_dims(face_input, axis=0)
        probs = self.model.predict(batch, verbose=0)[0]
        drowsy_index = config.CLASS_NAMES.index("Drowsy")
        prob_drowsy = float(probs[drowsy_index])

        cue_score = self.cue_selector.score(cue_reading)

        # -- PERCLOS (rolling time-window eyelid closure) -- #
        if cue_reading.eyes_available:
            self.perclos.update(cue_reading.eye_closed, now)
        perclos_value = self.perclos.value()
        perclos_ready = self.perclos.has_sufficient_data
        perclos_drowsy = self.perclos.is_drowsy()

        # -- Head-pose temporal monitoring (prolonged tilt / repeated nod) -- #
        if cue_reading.head_pose_available:
            head_status = self.head_pose_monitor.update(cue_reading.pitch, cue_reading.roll, now)
        else:
            head_status = self.head_pose_monitor.current_status()

        # -- CNN/cue agreement confidence (unchanged formula) -- #
        agreement = 1.0 - abs(prob_drowsy - cue_score)
        confidence = float(
            np.clip(0.5 * agreement + 0.5 * max(prob_drowsy, cue_score) * agreement, 0.0, 1.0)
        )

        # -- Sliding window -- #
        frame_is_drowsy = prob_drowsy >= 0.5 or cue_reading.prolonged_eye_closure or cue_reading.yawning
        self.sliding_window.push(frame_is_drowsy, prob_drowsy)
        window_ratio = self.sliding_window.drowsy_vote_ratio()
        smoothed_probability = self.sliding_window.mean_probability()
        majority_drowsy = self.sliding_window.majority_vote()

        status, alert_triggered = self._fuse_decision(
            majority_drowsy=majority_drowsy,
            smoothed_probability=smoothed_probability,
            confidence=confidence,
            perclos_ready=perclos_ready,
            perclos_drowsy=perclos_drowsy,
            head_status=head_status,
            cue_reading=cue_reading,
            now=now,
        )

        result = DetectionResult(
            frame_quality_ok=True,
            quality_label=quality.as_label(),
            face_box=face_box,
            landmarks=landmarks,
            cue_readings=cue_reading,
            cnn_probability_drowsy=prob_drowsy,
            cue_score=cue_score,
            confidence=confidence,
            window_drowsy_ratio=window_ratio,
            perclos_value=perclos_value,
            perclos_ready=perclos_ready,
            perclos_drowsy=perclos_drowsy,
            head_pose_prolonged_tilt=head_status.prolonged_downward_tilt,
            head_pose_repeated_nod=head_status.repeated_nodding,
            status=status,
            alert_triggered=alert_triggered,
            fps=fps,
            active_cues_label=cue_reading.label(),
            calibration_in_progress=False,
            calibration_progress=1.0,
            calibration_remaining=0.0,
            personalized_thresholds_active=self.cue_selector.thresholds_personalized,
        )
        self._last_good_result = result
        return result

    # ------------------------------------------------------------------ #
    # Multi-signal fusion
    # ------------------------------------------------------------------ #
    def _fuse_decision(
        self,
        majority_drowsy: bool,
        smoothed_probability: float,
        confidence: float,
        perclos_ready: bool,
        perclos_drowsy: bool,
        head_status: HeadPoseStatus,
        cue_reading: CueReadings,
        now: float,
    ) -> Tuple[str, bool]:
        """Combines the CNN+CBAM prediction, personalized EAR/MAR cue
        evidence, PERCLOS, head pose, and the sliding-window vote into a
        single final status. Mirrors the two-gate design this project
        started with (CNN probability AND confidence must both clear their
        thresholds) but additionally requires secondary, independent
        physiological signals to agree before the alert-worthy "Drowsy"
        status is raised -- one noisy signal alone can no longer trigger a
        false alarm.

        Returns (status, alert_triggered).
        """
        if not self.sliding_window.has_enough_data():
            # Not enough temporal history yet to trust any vote (e.g. right
            # after a session (re)start) -- stay conservative.
            return "Insufficient Data", False

        # Core two-gate CNN/window/confidence check, unchanged from the
        # original design.
        cnn_gate = (
            majority_drowsy
            and smoothed_probability >= self.probability_threshold
            and confidence >= self.confidence_threshold
        )

        if not cnn_gate:
            return "Not Drowsy", False

        # The CNN gate agrees a Drowsy verdict is plausible. Now check how
        # many of the other, independent signals corroborate it.
        secondary_votes = 0
        secondary_total = 0

        if perclos_ready:
            secondary_total += 1
            if perclos_drowsy:
                secondary_votes += 1

        secondary_total += 1
        if head_status.is_fatigue_signal:
            secondary_votes += 1

        secondary_total += 1
        if cue_reading.prolonged_eye_closure or cue_reading.yawning:
            secondary_votes += 1

        agreement_ratio = secondary_votes / secondary_total if secondary_total else 0.0

        if agreement_ratio >= config.FUSION_AGREEMENT_RATIO:
            alert_triggered = False
            if now - self._last_alert_time >= config.ALERT_COOLDOWN_SEC:
                alert_triggered = True
                self._last_alert_time = now
            return "Drowsy", alert_triggered

        # CNN gate alone wasn't corroborated by enough secondary evidence --
        # stay in "Not Drowsy" rather than raising a false alarm off a
        # single signal.
        return "Not Drowsy", False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Resets all per-session state and starts a fresh calibration
        phase. Call this every time a new monitoring session begins (e.g.
        the GUI's "Start Monitoring" button, or predict.py's startup)."""
        self.sliding_window.reset()
        self.cue_selector.reset()
        self.quality_gate.reset()
        self.perclos.reset()
        self.head_pose_monitor.reset()
        self.calibration.reset()
        self.calibration.start()
        self._thresholds = None
        self._last_alert_time = 0.0
        self._no_face_since = None
        self._last_good_result = None

    def shutdown(self) -> None:
        self.landmark_detector.close()