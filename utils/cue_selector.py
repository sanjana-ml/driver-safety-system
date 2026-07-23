"""
cue_selector.py
Adaptive cue selection: decides which facial cues (eye closure, blink
duration, yawning, head pose) are currently trustworthy and should be
weighted, based on the frame quality report and per-cue occlusion checks.
If eyes are unavailable (e.g. sunglasses), the system automatically shifts
weight onto mouth movement and head pose instead of forcing an unreliable
eye-based estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

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
    mar: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    eyes_available: bool = True
    mouth_available: bool = True
    head_pose_available: bool = True
    active_cues: List[str] = field(default_factory=list)

    def label(self) -> str:
        return "+".join(self.active_cues) if self.active_cues else "none"


def compute_cue_readings(
    gray_frame: np.ndarray,
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

    left_occluded = eye_region_occluded(gray_frame, left_eye_points, gray_frame.shape)
    right_occluded = eye_region_occluded(gray_frame, right_eye_points, gray_frame.shape)
    reading.eyes_available = not (left_occluded and right_occluded)

    # Mouth is considered available as long as the landmarks themselves are
    # reliable (the frame quality gate already ensures enough landmarks).
    reading.mouth_available = quality.landmarks_sufficient

    # Head pose is available whenever solvePnP produced a sane orientation,
    # i.e. whenever quality assessment ran at all.
    reading.head_pose_available = quality.face_detected

    active: List[str] = []
    if reading.eyes_available:
        active.append("eye_closure")
        active.append("blink_duration")
    if reading.mouth_available:
        active.append("yawning")
    if reading.head_pose_available:
        active.append("head_pose")

    if not active:
        # Absolute fallback -- should not normally happen if quality gate
        # already passed, but guarantees the pipeline never divides by zero.
        active.append("head_pose")

    reading.active_cues = active
    return reading


def cue_based_drowsiness_score(reading: CueReadings) -> float:
    """Rule-based auxiliary score in [0, 1] combining whichever cues are
    currently active. This is blended with the CNN+CBAM visual prediction
    to form the final confidence score (see utils/pipeline.py), giving the
    system a second, interpretable line of evidence beyond the CNN alone."""
    votes: List[float] = []

    if reading.eyes_available:
        corrected_ear = pose_compensated_ear(reading.avg_ear, reading.pitch, reading.yaw)
        eye_score = 1.0 if corrected_ear < config.EAR_THRESHOLD else 0.0
        votes.append(eye_score)

    if reading.mouth_available:
        mouth_score = 1.0 if reading.mar > config.MAR_THRESHOLD else 0.0
        votes.append(mouth_score)

    if reading.head_pose_available:
        nod_score = 1.0 if abs(reading.pitch) > config.HEAD_NOD_PITCH_DELTA_DEG else 0.0
        votes.append(nod_score)

    if not votes:
        return 0.0
    return float(np.mean(votes))