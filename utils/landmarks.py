"""
landmarks.py
Face detection + 468-point facial landmark localisation using MediaPipe
Face Mesh, and the geometric cue computations (EAR, MAR, head pose)
derived from them.

MediaPipe was chosen over dlib because its landmark model ships bundled
inside the pip package -- no manual model-file download needed, and no
C++ build toolchain required to install on Windows.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "mediapipe is required for face and landmark detection. "
        "Install it with 'pip install mediapipe' (see requirements.txt)."
    ) from exc

import config
from utils.logger import get_logger

logger = get_logger(__name__)

_mp_face_mesh = mp.solutions.face_mesh


# --------------------------------------------------------------------------- #
# MediaPipe Face Mesh landmark index sets
# (indices are fixed by the 468-point topology and are stable across images)
# --------------------------------------------------------------------------- #

# 6-point EAR sets, ordered to match the classic Soukupova & Cech formula:
# [outer_corner, upper_1, upper_2, inner_corner, lower_2, lower_1]
RIGHT_EYE_EAR_IDX: Tuple[int, int, int, int, int, int] = (33, 160, 158, 133, 153, 144)
LEFT_EYE_EAR_IDX: Tuple[int, int, int, int, int, int] = (362, 385, 387, 263, 373, 380)

# Simple 4-point MAR set: [left_corner, upper_inner_lip, right_corner, lower_inner_lip]
MOUTH_MAR_IDX: Tuple[int, int, int, int] = (61, 13, 291, 14)

# 6-point set used for solvePnP head-pose estimation (nose tip, chin,
# left/right eye outer corners, left/right mouth corners).
POSE_LANDMARK_IDX: Tuple[int, int, int, int, int, int] = (1, 152, 33, 263, 61, 291)


def _ordered_loop_from_connections(connections) -> List[int]:
    """MediaPipe exposes eye/lip outlines as an unordered set of edge pairs
    (a closed loop). This walks the loop to produce an ordered list of
    point indices suitable for cv2.polylines. Computed once at import
    time since the topology is fixed."""
    adjacency: Dict[int, List[int]] = defaultdict(list)
    for a, b in connections:
        adjacency[a].append(b)
        adjacency[b].append(a)

    start = next(iter(adjacency))
    order = [start]
    visited = {start}
    prev, current = None, start
    while True:
        candidates = [n for n in adjacency[current] if n != prev]
        next_node = next((n for n in candidates if n not in visited), None)
        if next_node is None:
            break
        order.append(next_node)
        visited.add(next_node)
        prev, current = current, next_node
    return order


LEFT_EYE_CONTOUR_IDX: List[int] = _ordered_loop_from_connections(_mp_face_mesh.FACEMESH_LEFT_EYE)
RIGHT_EYE_CONTOUR_IDX: List[int] = _ordered_loop_from_connections(_mp_face_mesh.FACEMESH_RIGHT_EYE)
LIPS_CONTOUR_IDX: List[int] = _ordered_loop_from_connections(_mp_face_mesh.FACEMESH_LIPS)


@dataclass
class FaceBox:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom


class LandmarkDetector:
    """Wraps MediaPipe's FaceMesh solution. max_num_faces is fixed at 1 in
    config (the driver), so no separate 'largest face' selection logic is
    needed the way it was with dlib's multi-face detector."""

    def __init__(self) -> None:
        self._face_mesh = _mp_face_mesh.FaceMesh(
            static_image_mode=config.MEDIAPIPE_STATIC_IMAGE_MODE,
            max_num_faces=config.MEDIAPIPE_MAX_NUM_FACES,
            refine_landmarks=config.MEDIAPIPE_REFINE_LANDMARKS,
            min_detection_confidence=config.MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
        )
        logger.info("LandmarkDetector initialised with MediaPipe Face Mesh (468 points).")

    def detect(self, frame_bgr: np.ndarray) -> Tuple[Optional[FaceBox], Optional[np.ndarray]]:
        """Runs Face Mesh on a BGR frame and returns (face_box, landmarks)
        in pixel coordinates, or (None, None) if no face is found. Never
        raises -- callers should treat a None return as 'insufficient
        data' rather than a hard failure."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None, None

        face_landmarks = results.multi_face_landmarks[0]
        coords = np.array(
            [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark], dtype=np.int32
        )

        pad = config.FACE_BOX_PADDING_PX
        x1 = max(0, int(np.min(coords[:, 0])) - pad)
        y1 = max(0, int(np.min(coords[:, 1])) - pad)
        x2 = min(w, int(np.max(coords[:, 0])) + pad)
        y2 = min(h, int(np.max(coords[:, 1])) + pad)
        face_box = FaceBox(x1, y1, x2, y2)

        return face_box, coords

    def close(self) -> None:
        self._face_mesh.close()

    def __enter__(self) -> "LandmarkDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# Geometric cue computations
# --------------------------------------------------------------------------- #
def _euclidean(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))


def get_points(landmarks: np.ndarray, indices) -> np.ndarray:
    return landmarks[list(indices)]


def eye_aspect_ratio(eye_points: np.ndarray) -> float:
    """Standard EAR formula (Soukupova & Cech, 2016). eye_points: 6x2 array
    ordered [outer_corner, upper_1, upper_2, inner_corner, lower_2, lower_1]."""
    if eye_points.shape[0] != 6:
        return 0.0
    vertical_1 = _euclidean(eye_points[1], eye_points[5])
    vertical_2 = _euclidean(eye_points[2], eye_points[4])
    horizontal = _euclidean(eye_points[0], eye_points[3])
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def mouth_aspect_ratio(mouth_points: np.ndarray) -> float:
    """MAR from 4 points ordered [left_corner, upper_lip, right_corner, lower_lip]."""
    if mouth_points.shape[0] != 4:
        return 0.0
    horizontal = _euclidean(mouth_points[0], mouth_points[2])
    vertical = _euclidean(mouth_points[1], mouth_points[3])
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


# 3D model points for a generic face, used for solvePnP head-pose estimation.
# Ordered to match POSE_LANDMARK_IDX: nose tip, chin, left eye outer corner,
# right eye outer corner, left mouth corner, right mouth corner.
_MODEL_POINTS_3D = np.array(
    [
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),  # Left eye outer corner
        (225.0, 170.0, -135.0),   # Right eye outer corner
        (-150.0, -150.0, -125.0), # Left mouth corner
        (150.0, -150.0, -125.0),  # Right mouth corner
    ],
    dtype=np.float64,
)


def head_pose_angles(
    landmarks: np.ndarray, frame_shape: Tuple[int, int]
) -> Tuple[float, float, float]:
    """Returns (pitch, yaw, roll) in degrees using OpenCV's solvePnP."""
    h, w = frame_shape[:2]
    focal_length = w
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1))  # assume no lens distortion

    image_points = np.array(
        [landmarks[i] for i in POSE_LANDMARK_IDX], dtype=np.float64
    )

    success, rotation_vector, _ = cv2.solvePnP(
        _MODEL_POINTS_3D,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return 0.0, 0.0, 0.0

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = np.arctan2(-rotation_matrix[2, 0], sy)
        roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw = np.arctan2(-rotation_matrix[2, 0], sy)
        roll = 0.0

    to_deg = 180.0 / np.pi
    pitch_deg = pitch * to_deg
    yaw_deg = yaw * to_deg
    roll_deg = roll * to_deg

    # This 6-point (nose/chin/eyes/mouth) solvePnP configuration has a
    # well-known ~180-degree ambiguity on the pitch and roll axes: the
    # object-space Z axis points out of the face toward the camera, which
    # is antiparallel to the camera's own forward Z axis, so a genuinely
    # frontal face decomposes to pitch/roll near +-180 degrees instead of
    # near 0. Wrap both back into a human-readable range where 0 degrees
    # means "looking straight at the camera".
    if pitch_deg > 90:
        pitch_deg -= 180
    elif pitch_deg < -90:
        pitch_deg += 180

    if roll_deg > 90:
        roll_deg -= 180
    elif roll_deg < -90:
        roll_deg += 180

    return pitch_deg, yaw_deg, roll_deg


def region_bounding_box(
    points: np.ndarray, frame_shape: Tuple[int, int], padding: int = 8
) -> Tuple[int, int, int, int]:
    """Axis-aligned bounding box around a set of landmark points, padded and
    clamped to the frame boundaries. Returns (x1, y1, x2, y2)."""
    h, w = frame_shape[:2]
    x1 = max(0, int(np.min(points[:, 0])) - padding)
    y1 = max(0, int(np.min(points[:, 1])) - padding)
    x2 = min(w, int(np.max(points[:, 0])) + padding)
    y2 = min(h, int(np.max(points[:, 1])) + padding)
    return x1, y1, x2, y2