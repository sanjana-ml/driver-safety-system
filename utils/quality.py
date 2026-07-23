"""
quality.py
Frame quality gating: before any drowsiness inference happens, verify that
the frame is trustworthy (face present, lighting adequate, orientation
close to frontal, enough landmarks visible). If not, the pipeline must
never force a prediction -- it reports "Insufficient Data" instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

import config
from utils.landmarks import FaceBox, head_pose_angles


@dataclass
class QualityReport:
    face_detected: bool = False
    brightness_ok: bool = False
    orientation_ok: bool = False
    landmarks_sufficient: bool = False
    brightness_value: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    num_landmarks: int = 0
    reason: str = ""

    @property
    def overall_ok(self) -> bool:
        return (
            self.face_detected
            and self.brightness_ok
            and self.orientation_ok
            and self.landmarks_sufficient
        )

    def as_label(self) -> str:
        return "Good" if self.overall_ok else f"Insufficient Data ({self.reason})"


def assess_frame_quality(
    gray_frame: np.ndarray,
    face_box: Optional[FaceBox],
    landmarks: Optional[np.ndarray],
) -> QualityReport:
    report = QualityReport()

    if face_box is None or landmarks is None:
        report.reason = "no face detected"
        return report
    report.face_detected = True

    report.num_landmarks = int(landmarks.shape[0])
    report.landmarks_sufficient = report.num_landmarks >= config.MIN_LANDMARKS_REQUIRED
    if not report.landmarks_sufficient:
        report.reason = "too few landmarks visible"

    # Brightness computed over the face ROI only, so background lighting
    # does not mask a poorly lit face (or vice versa).
    x1, y1, x2, y2 = face_box.as_tuple()
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(gray_frame.shape[1], x2)
    y2 = min(gray_frame.shape[0], y2)
    face_roi = gray_frame[y1:y2, x1:x2]
    if face_roi.size == 0:
        report.reason = "invalid face region"
        return report

    report.brightness_value = float(np.mean(face_roi))
    report.brightness_ok = config.MIN_BRIGHTNESS <= report.brightness_value <= config.MAX_BRIGHTNESS
    if not report.brightness_ok and not report.reason:
        report.reason = "poor lighting"

    pitch, yaw, roll = head_pose_angles(landmarks, gray_frame.shape)
    report.pitch, report.yaw, report.roll = pitch, yaw, roll
    report.orientation_ok = (
        abs(pitch) <= config.MAX_PITCH_DEG
        and abs(yaw) <= config.MAX_YAW_DEG
        and abs(roll) <= config.MAX_ROLL_DEG
    )
    if not report.orientation_ok and not report.reason:
        report.reason = "head not sufficiently frontal"

    if report.overall_ok:
        report.reason = ""

    return report


def eye_region_occluded(
    gray_frame: np.ndarray, eye_points: np.ndarray, frame_shape: Tuple[int, int]
) -> bool:
    """Heuristic sunglasses / heavy-occlusion detector: a genuinely visible
    eye region has meaningful pixel-intensity variance (iris, sclera,
    eyelid edges). Sunglasses or a hand covering the eyes tend to produce a
    much flatter, low-variance patch. eye_points can be any Nx2 array of
    landmark points covering the eye region (e.g. the 6-point EAR set)."""
    from utils.landmarks import region_bounding_box

    x1, y1, x2, y2 = region_bounding_box(eye_points, frame_shape, padding=4)
    patch = gray_frame[y1:y2, x1:x2]
    if patch.size == 0:
        return True
    return float(np.var(patch)) < config.EYE_OCCLUSION_VARIANCE_THRESHOLD
