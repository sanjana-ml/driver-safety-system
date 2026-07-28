"""
quality.py
Frame quality gating: before any drowsiness inference happens, verify that
the frame is trustworthy (face present, lighting adequate, orientation
close to frontal, enough landmarks visible). If not, the pipeline must
never force a prediction -- it reports "Insufficient Data" instead.

assess_frame_quality() evaluates a single frame in isolation and is
unchanged in behaviour. QualityGate wraps it with temporal hysteresis: a
single bad-quality frame (a moment of landmark jitter, a brief partial
out-of-frame movement) is absorbed for up to config.QUALITY_GRACE_FRAMES
consecutive frames before the gate actually reports "not ok" -- this is
what stops slight, transient tracking noise from being mistaken for a real
quality failure (and, under the old per-frame-only gate, potentially for a
"Drowsy"/alert-worthy signal). Recovery from bad to good quality is always
immediate; only degradation is debounced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

import config
from utils.landmarks import FaceBox, POSE_LANDMARK_IDX, head_pose_angles


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
    skin_brightness_ref: float = 0.0
    skin_saturation_ref: float = 0.0

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


class QualityGate:
    """Stateful hysteresis wrapper around a stream of per-frame
    `QualityReport.overall_ok` values. One instance should live for the
    lifetime of a monitoring session and be reset() alongside the other
    per-session state (sliding window, cue selector, etc.).

    Behaviour:
    - A "good" frame immediately marks the gate stable-ok and clears the
      bad-frame streak (recovery is instant).
    - A "bad" frame increments a streak counter. While that streak is at or
      below config.QUALITY_GRACE_FRAMES, the gate keeps reporting whatever
      its last stable state was (absorbing the blip). Only once the streak
      exceeds the grace period does the gate flip to stable-not-ok.
    """

    def __init__(self, grace_frames: int = config.QUALITY_GRACE_FRAMES) -> None:
        self.grace_frames = grace_frames
        self._bad_streak = 0
        self._stable_ok = False  # no frames observed yet -> not trusted

    def reset(self) -> None:
        self._bad_streak = 0
        self._stable_ok = False

    def update(self, raw_ok: bool) -> bool:
        """Feed in this frame's raw QualityReport.overall_ok. Returns the
        debounced ("stable") quality decision to actually act on."""
        if raw_ok:
            self._bad_streak = 0
            self._stable_ok = True
        else:
            self._bad_streak += 1
            if self._bad_streak > self.grace_frames:
                self._stable_ok = False
            # else: within grace -- keep the previous stable_ok, absorbing
            # this frame's blip.
        return self._stable_ok

    @property
    def is_within_grace(self) -> bool:
        """True when the gate is currently absorbing a bad-quality streak
        that hasn't yet exceeded the grace period (i.e. stable_ok is still
        True despite this frame's raw quality being bad)."""
        return 0 < self._bad_streak <= self.grace_frames


def assess_frame_quality(
    gray_frame: np.ndarray,
    face_box: Optional[FaceBox],
    landmarks: Optional[np.ndarray],
    frame_bgr: Optional[np.ndarray] = None,
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

    x1, y1, x2, y2 = face_box.as_tuple()
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(gray_frame.shape[1], x2)
    y2 = min(gray_frame.shape[0], y2)
    face_roi = gray_frame[y1:y2, x1:x2]
    if face_roi.size == 0:
        report.reason = "invalid face region"
        return report

    # Sunglasses drag *whole-face* brightness below MIN_BRIGHTNESS even in a
    # well-lit room, producing a false "poor lighting" result. Brightness is
    # instead sampled from the lower face (nose tip down to chin), which
    # stays visible regardless of eyewear.
    nose_tip_idx, chin_idx = POSE_LANDMARK_IDX[0], POSE_LANDMARK_IDX[1]
    lower_y1 = int(landmarks[nose_tip_idx][1])
    lower_y1 = max(y1, min(lower_y1, y2 - 1))
    lower_face_roi = gray_frame[lower_y1:y2, x1:x2]

    brightness_roi = lower_face_roi if lower_face_roi.size > 0 else face_roi
    report.brightness_value = float(np.mean(brightness_roi))
    report.brightness_ok = config.MIN_BRIGHTNESS <= report.brightness_value <= config.MAX_BRIGHTNESS
    if not report.brightness_ok and not report.reason:
        report.reason = "poor lighting"

    # Same lower-face region doubles as the skin-tone reference for eye
    # occlusion checks -- the one part of the face guaranteed visible
    # whether or not the driver is wearing sunglasses.
    report.skin_brightness_ref = report.brightness_value
    if frame_bgr is not None:
        lower_face_roi_bgr = frame_bgr[lower_y1:y2, x1:x2]
        if lower_face_roi_bgr.size > 0:
            hsv_roi = cv2.cvtColor(lower_face_roi_bgr, cv2.COLOR_BGR2HSV)
            report.skin_saturation_ref = float(np.mean(hsv_roi[:, :, 1]))

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
    frame_bgr: np.ndarray,
    eye_points: np.ndarray,
    frame_shape: Tuple[int, int],
    skin_brightness_ref: float,
    skin_saturation_ref: float,
) -> bool:
    """Sunglasses / heavy-occlusion detector for a single frame. Compares the
    eye patch's saturation and brightness against the same-frame skin
    reference rather than a fixed constant, since absolute pixel values swing
    with lighting and lens glare. Either signal alone can vote "occluded".
    This is a single-frame vote -- CueSelector applies the temporal
    smoothing that turns repeated votes into a stable decision."""
    from utils.landmarks import region_bounding_box

    x1, y1, x2, y2 = region_bounding_box(eye_points, frame_shape, padding=4)
    patch_bgr = frame_bgr[y1:y2, x1:x2]
    if patch_bgr.size == 0:
        return True

    hsv_patch = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    eye_brightness = float(np.mean(hsv_patch[:, :, 2]))
    eye_saturation = float(np.mean(hsv_patch[:, :, 1]))

    saturation_vote = (
        skin_saturation_ref > 0
        and (eye_saturation / skin_saturation_ref) < config.EYE_OCCLUSION_SATURATION_RATIO
    )

    if skin_brightness_ref <= 0:
        brightness_vote = False
    else:
        brightness_ratio = eye_brightness / skin_brightness_ref
        brightness_vote = (
            brightness_ratio < config.EYE_OCCLUSION_DARK_RATIO
            or brightness_ratio > config.EYE_OCCLUSION_BRIGHT_RATIO
        )

    return saturation_vote or brightness_vote