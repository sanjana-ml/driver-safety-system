"""
cue_selector.py
Stateful, adaptive cue selection: decides which facial cues (eye closure,
blink/prolonged-closure duration, yawning, head nod/tilt) are currently
trustworthy, and turns raw per-frame measurements into duration-aware
drowsiness evidence.

Kept as a class because two required behaviours are inherently temporal:
- Sunglasses/eye-occlusion is voted over a rolling window
  (config.EYE_OCCLUSION_VOTE_FRAMES) instead of trusted on one frame, so a
  single noisy frame (blur, blink, glare) can't flip "eyes available" on
  and off every frame.
- "Prolonged eye closure" and "yawning" are about *duration* -- a single
  closed-eye frame is a blink, not drowsiness, and one wide-mouth frame
  could just be talking. Consecutive-frame streak counters track this.

One instance should live for the lifetime of a camera session and be
reset() alongside the sliding window when that session restarts.

EAR/MAR comparisons use adaptive, personalized thresholds rather than the
fixed config.EAR_THRESHOLD / config.MAR_THRESHOLD constants directly: the
constants are still the *default* (used before calibration completes, or
if it falls back due to insufficient samples), but utils/pipeline.py calls
set_thresholds() once utils.calibration.CalibrationSession finishes, after
which every subsequent compute() call in the session uses the driver's own
baseline-derived thresholds.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List

import numpy as np

import config
from utils.calibration import PersonalizedThresholds
from utils.landmarks import (
    LEFT_EYE_EAR_IDX,
    MOUTH_MAR_IDX,
    RIGHT_EYE_EAR_IDX,
    eye_aspect_ratio,
    get_points,
    mouth_aspect_ratio,
    pose_compensated_ear,
)
from utils.quality import QualityReport, eye_region_occluded


@dataclass
class CueReadings:
    left_ear: float = 0.0
    right_ear: float = 0.0
    avg_ear: float = 0.0
    corrected_ear: float = 0.0
    mar: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    eyes_available: bool = True
    mouth_available: bool = True
    head_pose_available: bool = True
    eye_closed: bool = False
    prolonged_eye_closure: bool = False
    yawning: bool = False
    repeated_yawning: bool = False
    head_nod: bool = False
    head_tilt: bool = False
    prolonged_head_tilt: bool = False
    ear_threshold_used: float = config.EAR_THRESHOLD
    mar_threshold_used: float = config.MAR_THRESHOLD
    active_cues: List[str] = field(default_factory=list)

    def label(self) -> str:
        return "+".join(self.active_cues) if self.active_cues else "none"


class CueSelector:
    def __init__(self) -> None:
        self._left_votes: Deque[bool] = deque(maxlen=config.EYE_OCCLUSION_VOTE_FRAMES)
        self._right_votes: Deque[bool] = deque(maxlen=config.EYE_OCCLUSION_VOTE_FRAMES)
        # Tracked as wall-clock timestamps (when the current closure/open/
        # tilt streak started), not frame counts -- frame counts silently
        # assumed ~30fps and became far too strict on machines where the
        # actual processing rate is much lower (e.g. ~3fps due to CNN
        # inference cost).
        self._eye_closed_since: float | None = None
        self._mouth_open_since: float | None = None
        self._head_tilt_since: float | None = None
        # Timestamps of completed yawns (mouth-open streak that cleared
        # YAWN_CONSEC_SEC and then closed again), trimmed to a rolling
        # window -- same edge-triggered counting pattern as
        # HeadPoseMonitor._nod_events. A single yawn should not by itself
        # be treated as drowsiness evidence; only repeated yawns should.
        self._yawn_events: Deque[float] = deque()
        self._yawn_counted_this_streak: bool = False
        # Adaptive EAR/MAR thresholds -- start out at the fixed config
        # defaults and are replaced once calibration finishes (see
        # set_thresholds()). Kept as plain floats (not the full
        # PersonalizedThresholds object) so compute() has a simple, always-
        # valid pair of numbers to compare against.
        self._ear_threshold: float = config.EAR_THRESHOLD
        self._mar_threshold: float = config.MAR_THRESHOLD
        self._thresholds_personalized: bool = False

    def reset(self) -> None:
        self._left_votes.clear()
        self._right_votes.clear()
        self._eye_closed_since = None
        self._mouth_open_since = None
        self._head_tilt_since = None
        self._yawn_events.clear()
        self._yawn_counted_this_streak = False
        # A new session may have a different driver / lighting, so drop back
        # to the fixed defaults until the new session's calibration finishes.
        self._ear_threshold = config.EAR_THRESHOLD
        self._mar_threshold = config.MAR_THRESHOLD
        self._thresholds_personalized = False

    def set_thresholds(self, thresholds: PersonalizedThresholds) -> None:
        """Called by utils.pipeline.DriverSafetyPipeline once the session's
        CalibrationSession finishes. Every compute() call from this point on
        uses these (possibly personalized, possibly fallback-default)
        thresholds instead of the raw config constants."""
        self._ear_threshold = thresholds.ear_threshold
        self._mar_threshold = thresholds.mar_threshold
        self._thresholds_personalized = thresholds.calibrated

    @property
    def thresholds_personalized(self) -> bool:
        return self._thresholds_personalized

    @property
    def ear_threshold(self) -> float:
        return self._ear_threshold

    @property
    def mar_threshold(self) -> float:
        return self._mar_threshold

    def _eyes_available(
        self, frame_bgr: np.ndarray, left_eye_points: np.ndarray,
        right_eye_points: np.ndarray, quality: QualityReport,
    ) -> bool:
        left_occluded = eye_region_occluded(
            frame_bgr, left_eye_points, frame_bgr.shape,
            quality.skin_brightness_ref, quality.skin_saturation_ref,
        )
        right_occluded = eye_region_occluded(
            frame_bgr, right_eye_points, frame_bgr.shape,
            quality.skin_brightness_ref, quality.skin_saturation_ref,
        )
        self._left_votes.append(left_occluded)
        self._right_votes.append(right_occluded)

        if len(self._left_votes) < config.EYE_OCCLUSION_VOTE_FRAMES:
            return not (left_occluded and right_occluded)

        both_occluded_votes = sum(
            1 for l, r in zip(self._left_votes, self._right_votes) if l and r
        )
        occluded_ratio = both_occluded_votes / len(self._left_votes)
        return occluded_ratio < config.EYE_OCCLUSION_VOTE_RATIO

    def compute(
        self,
        frame_bgr: np.ndarray,
        landmarks: np.ndarray,
        quality: QualityReport,
        now: float,
    ) -> CueReadings:
        left_eye_points = get_points(landmarks, LEFT_EYE_EAR_IDX)
        right_eye_points = get_points(landmarks, RIGHT_EYE_EAR_IDX)
        mouth_points = get_points(landmarks, MOUTH_MAR_IDX)

        reading = CueReadings()
        reading.left_ear = eye_aspect_ratio(left_eye_points)
        reading.right_ear = eye_aspect_ratio(right_eye_points)
        reading.avg_ear = (reading.left_ear + reading.right_ear) / 2.0
        reading.mar = mouth_aspect_ratio(mouth_points)
        reading.pitch, reading.yaw, reading.roll = quality.pitch, quality.yaw, quality.roll

        pose_trustworthy_for_ear = (
            abs(reading.pitch) <= config.CUE_TRUST_MAX_PITCH_DEG
            and abs(reading.yaw) <= config.CUE_TRUST_MAX_YAW_DEG
        )
        reading.eyes_available = (
            self._eyes_available(frame_bgr, left_eye_points, right_eye_points, quality)
            and pose_trustworthy_for_ear
        )
        reading.mouth_available = quality.landmarks_sufficient
        reading.head_pose_available = quality.face_detected

        reading.ear_threshold_used = self._ear_threshold
        reading.mar_threshold_used = self._mar_threshold

        if reading.eyes_available:
            reading.corrected_ear = pose_compensated_ear(reading.avg_ear, reading.pitch, reading.yaw)
            reading.eye_closed = reading.corrected_ear < self._ear_threshold
            if reading.eye_closed:
                if self._eye_closed_since is None:
                    self._eye_closed_since = now
                closed_duration = now - self._eye_closed_since
            else:
                self._eye_closed_since = None
                closed_duration = 0.0
            reading.prolonged_eye_closure = closed_duration >= config.EAR_CONSEC_SEC_DROWSY
        else:
            reading.corrected_ear = 0.0
            self._eye_closed_since = None

        if reading.mouth_available:
            mouth_open = reading.mar > self._mar_threshold
            if mouth_open:
                if self._mouth_open_since is None:
                    self._mouth_open_since = now
                open_duration = now - self._mouth_open_since
            else:
                self._mouth_open_since = None
                open_duration = 0.0
            reading.yawning = open_duration >= config.YAWN_CONSEC_SEC

            # Count a completed yawn once (on the frame it first clears the
            # duration gate, not every frame the mouth stays open), then
            # track repeats in a rolling window -- a single yawn is one
            # normal fatigue event, not evidence on its own.
            if reading.yawning and not self._yawn_counted_this_streak:
                self._yawn_events.append(now)
                self._yawn_counted_this_streak = True
            elif not mouth_open:
                self._yawn_counted_this_streak = False
        else:
            self._mouth_open_since = None

        yawn_cutoff = now - config.YAWN_WINDOW_SEC
        while self._yawn_events and self._yawn_events[0] < yawn_cutoff:
            self._yawn_events.popleft()
        reading.repeated_yawning = len(self._yawn_events) >= config.YAWN_COUNT_THRESHOLD

        if reading.head_pose_available:
            reading.head_nod = abs(reading.pitch) > config.HEAD_NOD_PITCH_DELTA_DEG
            reading.head_tilt = abs(reading.roll) > config.HEAD_TILT_ROLL_DELTA_DEG
            if reading.head_tilt:
                if self._head_tilt_since is None:
                    self._head_tilt_since = now
                tilt_duration = now - self._head_tilt_since
            else:
                self._head_tilt_since = None
                tilt_duration = 0.0
            reading.prolonged_head_tilt = tilt_duration >= config.HEAD_TILT_CONSEC_SEC
        else:
            self._head_tilt_since = None

        # Only list cues that are actually triggered right now (not just the
        # categories being tracked) -- otherwise this label is misleading,
        # e.g. showing "yawning+head_pose" even when nothing real is
        # happening yet.
        active: List[str] = []
        if reading.eyes_available and reading.eye_closed:
            active.append("eye_closure")
        if reading.eyes_available and reading.prolonged_eye_closure:
            active.append("blink_duration")
        if reading.mouth_available and reading.yawning:
            active.append("yawning")
        if reading.head_pose_available and (reading.head_nod or reading.head_tilt):
            active.append("head_pose")
        reading.active_cues = active

        return reading

    def score(self, reading: CueReadings) -> float:
        """Rule-based auxiliary score in [0, 1], blended with the CNN+CBAM
        visual prediction in utils/pipeline.py."""
        votes: List[float] = []

        if reading.eyes_available:
            if reading.prolonged_eye_closure:
                votes.append(1.0)
            elif reading.eye_closed:
                votes.append(0.5)
            else:
                votes.append(0.0)

        if reading.mouth_available:
            votes.append(1.0 if reading.yawning else 0.0)

        if reading.head_pose_available:
            votes.append(1.0 if (reading.head_nod or reading.prolonged_head_tilt) else 0.0)

        if not votes:
            return 0.0
        return float(np.mean(votes))