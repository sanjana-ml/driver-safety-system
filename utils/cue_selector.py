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
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List

import numpy as np

import config
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
    head_nod: bool = False
    head_tilt: bool = False
    active_cues: List[str] = field(default_factory=list)

    def label(self) -> str:
        return "+".join(self.active_cues) if self.active_cues else "none"


class CueSelector:
    def __init__(self) -> None:
        self._left_votes: Deque[bool] = deque(maxlen=config.EYE_OCCLUSION_VOTE_FRAMES)
        self._right_votes: Deque[bool] = deque(maxlen=config.EYE_OCCLUSION_VOTE_FRAMES)
        self._eye_closed_streak: int = 0
        self._mouth_open_streak: int = 0

    def reset(self) -> None:
        self._left_votes.clear()
        self._right_votes.clear()
        self._eye_closed_streak = 0
        self._mouth_open_streak = 0

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

        reading.eyes_available = self._eyes_available(
            frame_bgr, left_eye_points, right_eye_points, quality
        )
        reading.mouth_available = quality.landmarks_sufficient
        reading.head_pose_available = quality.face_detected

        if reading.eyes_available:
            reading.corrected_ear = pose_compensated_ear(reading.avg_ear, reading.pitch, reading.yaw)
            reading.eye_closed = reading.corrected_ear < config.EAR_THRESHOLD
            if reading.eye_closed:
                self._eye_closed_streak += 1
            else:
                self._eye_closed_streak = 0
            reading.prolonged_eye_closure = self._eye_closed_streak >= config.EAR_CONSEC_FRAMES_DROWSY
        else:
            reading.corrected_ear = 0.0
            self._eye_closed_streak = 0

        if reading.mouth_available:
            mouth_open = reading.mar > config.MAR_THRESHOLD
            if mouth_open:
                self._mouth_open_streak += 1
            else:
                self._mouth_open_streak = 0
            reading.yawning = self._mouth_open_streak >= config.YAWN_CONSEC_FRAMES
        else:
            self._mouth_open_streak = 0

        if reading.head_pose_available:
            reading.head_nod = abs(reading.pitch) > config.HEAD_NOD_PITCH_DELTA_DEG
            reading.head_tilt = abs(reading.roll) > config.HEAD_TILT_ROLL_DELTA_DEG

        active: List[str] = []
        if reading.eyes_available:
            active.append("eye_closure")
            active.append("blink_duration")
        if reading.mouth_available:
            active.append("yawning")
        if reading.head_pose_available:
            active.append("head_pose")
        if not active:
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
            votes.append(1.0 if (reading.head_nod or reading.head_tilt) else 0.0)

        if not votes:
            return 0.0
        return float(np.mean(votes))