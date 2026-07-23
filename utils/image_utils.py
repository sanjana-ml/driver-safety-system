"""
image_utils.py
Image preprocessing for the CNN (resize/normalize) and drawing helpers used
by the GUI and console predictor to overlay bounding boxes, landmarks, and
eye/mouth regions on the live video feed.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

import config
from utils.landmarks import FaceBox, LEFT_EYE_CONTOUR_IDX, LIPS_CONTOUR_IDX, RIGHT_EYE_CONTOUR_IDX


def preprocess_face(
    gray_frame: np.ndarray,
    face_box: FaceBox,
    target_size: int = config.IMG_SIZE,
) -> np.ndarray:
    """Crops the face ROI, resizes to target_size x target_size, and
    normalizes pixel values to [0, 1]. Returns an array shaped
    (target_size, target_size, 1) ready to be batched for model.predict."""
    x1, y1, x2, y2 = face_box.as_tuple()
    h, w = gray_frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    face_roi = gray_frame[y1:y2, x1:x2]
    if face_roi.size == 0:
        face_roi = np.zeros((target_size, target_size), dtype=np.uint8)

    resized = cv2.resize(face_roi, (target_size, target_size), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=-1)
def enhance_low_light(frame_bgr: np.ndarray) -> np.ndarray:
    """Boosts local contrast/brightness in dim frames using CLAHE (Contrast
    Limited Adaptive Histogram Equalization) applied to the L (lightness)
    channel in LAB colour space, then converts back to BGR.

    This runs on every frame before face/landmark detection and before the
    quality-gate brightness check, so a driver in a dim cabin or wearing
    dark sunglasses is far less likely to trip a false "poor lighting" /
    "Insufficient Data" result, and MediaPipe has a much better-contrast
    image to find landmarks in. CLAHE is local (tile-based), so it lifts
    shadowed regions without blowing out already-bright ones -- unlike a
    flat brightness/gamma boost, which either leaves dark areas too dark or
    over-brightens well-lit areas.
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT, tileGridSize=config.CLAHE_TILE_GRID_SIZE
    )
    l_enhanced = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

def draw_face_box(frame_bgr: np.ndarray, face_box: FaceBox, color=(0, 255, 0)) -> None:
    cv2.rectangle(
        frame_bgr, (face_box.left, face_box.top), (face_box.right, face_box.bottom), color, 2
    )


def draw_landmarks(frame_bgr: np.ndarray, landmarks: np.ndarray, color=(0, 255, 255)) -> None:
    for (x, y) in landmarks:
        cv2.circle(frame_bgr, (int(x), int(y)), 1, color, -1)


def draw_region_outline(
    frame_bgr: np.ndarray, points: np.ndarray, color=(255, 0, 0), closed: bool = True
) -> None:
    pts = points.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(frame_bgr, [pts], isClosed=closed, color=color, thickness=1)


def draw_eye_and_mouth_regions(frame_bgr: np.ndarray, landmarks: np.ndarray) -> None:
    left_eye = landmarks[LEFT_EYE_CONTOUR_IDX]
    right_eye = landmarks[RIGHT_EYE_CONTOUR_IDX]
    mouth = landmarks[LIPS_CONTOUR_IDX]
    draw_region_outline(frame_bgr, left_eye, color=(255, 200, 0))
    draw_region_outline(frame_bgr, right_eye, color=(255, 200, 0))
    draw_region_outline(frame_bgr, mouth, color=(0, 128, 255))


def put_status_text(
    frame_bgr: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    color=(255, 255, 255),
    scale: float = 0.6,
    thickness: int = 2,
) -> None:
    cv2.putText(
        frame_bgr, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA
    )
