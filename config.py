"""
config.py
Central configuration for the Smart Vision-Based Driver Safety Monitoring System.
Every tunable path, threshold and hyperparameter lives here so the rest of the
codebase never hard-codes a magic number.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


BASE_DIR: Path = Path(__file__).resolve().parent

DATASET_DIR: Path = BASE_DIR / "dataset"
MODELS_DIR: Path = BASE_DIR / "models"
WEIGHTS_DIR: Path = BASE_DIR / "weights"
LOGS_DIR: Path = BASE_DIR / "logs"
SCREENSHOTS_DIR: Path = BASE_DIR / "screenshots"
ALARM_DIR: Path = BASE_DIR / "alarm"
TENSORBOARD_DIR: Path = LOGS_DIR / "tensorboard"
PREDICTION_LOG_CSV: Path = LOGS_DIR / "prediction_log.csv"
APP_LOG_FILE: Path = LOGS_DIR / "app.log"

MODEL_PATH: Path = WEIGHTS_DIR / "drowsiness_cnn_cbam.h5"
BEST_CHECKPOINT_PATH: Path = WEIGHTS_DIR / "best_checkpoint.h5"
TRAINING_HISTORY_PATH: Path = LOGS_DIR / "training_history.json"

ALARM_SOUND_PATH: Path = ALARM_DIR / "alarm.wav"


def ensure_directories() -> None:
    for directory in (
        DATASET_DIR, MODELS_DIR, WEIGHTS_DIR, LOGS_DIR,
        SCREENSHOTS_DIR, ALARM_DIR, TENSORBOARD_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #
CAMERA_INDEX: int = 0
CAMERA_FRAME_WIDTH: int = 640
CAMERA_FRAME_HEIGHT: int = 480
CAMERA_FPS_TARGET: int = 30
CAMERA_MAX_INIT_RETRIES: int = 3
CAMERA_RETRY_DELAY_SEC: float = 1.0


# --------------------------------------------------------------------------- #
# Face / landmark detection (MediaPipe Face Mesh)
# --------------------------------------------------------------------------- #
NUM_LANDMARKS: int = 468
MIN_LANDMARKS_REQUIRED: int = 468
MEDIAPIPE_MAX_NUM_FACES: int = 1
MEDIAPIPE_STATIC_IMAGE_MODE: bool = False
MEDIAPIPE_REFINE_LANDMARKS: bool = False
MEDIAPIPE_MIN_DETECTION_CONFIDENCE: float = 0.5
MEDIAPIPE_MIN_TRACKING_CONFIDENCE: float = 0.5
FACE_BOX_PADDING_PX: int = 10


# --------------------------------------------------------------------------- #
# Frame quality gate
# --------------------------------------------------------------------------- #
MIN_BRIGHTNESS: float = 25.0
MAX_BRIGHTNESS: float = 230.0
MAX_YAW_DEG: float = 45.0
MAX_PITCH_DEG: float = 40.0
MAX_ROLL_DEG: float = 40.0
EYE_OCCLUSION_VARIANCE_THRESHOLD: float = 15.0  # legacy, kept for reference
# Sunglasses/occlusion detector: compares each eye patch against a same-frame
# skin reference (lower face) instead of a fixed brightness/variance constant,
# since absolute pixel values swing wildly with lighting and lens glare.
EYE_OCCLUSION_SATURATION_RATIO: float = 0.72
EYE_OCCLUSION_DARK_RATIO: float = 0.6
EYE_OCCLUSION_BRIGHT_RATIO: float = 1.5
EYE_OCCLUSION_VOTE_FRAMES: int = 8
EYE_OCCLUSION_VOTE_RATIO: float = 0.6
CLAHE_CLIP_LIMIT: float = 3.0
CLAHE_TILE_GRID_SIZE: Tuple[int, int] = (8, 8)

# How long the face must be continuously missing before the system escalates
# from "Insufficient Data" to "Face Not Detected" (and raises an alert).
# (Previously named NO_FACE_ALERT_SECONDS.)
FACE_NOT_DETECTED_GRACE_SEC: float = 3.0

# Frame-quality hysteresis: a single bad-quality frame (momentary landmark
# jitter, a brief partial out-of-frame move) is absorbed for up to this many
# CONSECUTIVE frames before the system actually reports "Insufficient Data".
# This is what prevents slight, transient tracking noise from being treated
# as a real quality failure / false alarm. Quality recovery (bad -> good) is
# always immediate -- only the bad direction is debounced.
QUALITY_GRACE_FRAMES: int = 6


# --------------------------------------------------------------------------- #
# Facial cue thresholds
# --------------------------------------------------------------------------- #
EAR_THRESHOLD: float = 0.21
EAR_CONSEC_FRAMES_BLINK: int = 2
EAR_CONSEC_FRAMES_DROWSY: int = 25   # ~0.8s @30fps of continuous closure
MAR_THRESHOLD: float = 0.6
YAWN_CONSEC_FRAMES: int = 15
HEAD_NOD_PITCH_DELTA_DEG: float = 15.0
HEAD_TILT_ROLL_DELTA_DEG: float = 20.0


# --------------------------------------------------------------------------- #
# Driver calibration (personalized EAR/MAR thresholds)
# --------------------------------------------------------------------------- #
# A short calibration phase runs at the start of every session (see
# utils/calibration.py): the driver looks naturally at the camera for
# CALIBRATION_DURATION_SEC seconds while baseline EAR/MAR statistics are
# collected, then personalized thresholds replace the fixed EAR_THRESHOLD /
# MAR_THRESHOLD defaults above for the rest of the session. If calibration
# does not collect enough usable samples (e.g. the driver's face was not
# reliably tracked for most of the phase), the system safely falls back to
# the fixed defaults instead of using noisy personalized values.
CALIBRATION_DURATION_SEC: float = 7.0          # within the required 5-10s window
CALIBRATION_MIN_SAMPLES: int = 30              # ~1s of good frames at 30fps
CALIBRATION_EAR_STD_MULTIPLIER: float = 0.8    # personal_ear = mean_ear - k * std_ear
CALIBRATION_MAR_STD_MULTIPLIER: float = 1.0    # personal_mar = mean_mar + k * std_mar
# Sane clamps so a short/noisy calibration can never produce a threshold
# that is unreasonably strict or unreasonably loose.
CALIBRATION_EAR_MIN: float = 0.12
CALIBRATION_EAR_MAX: float = 0.30
CALIBRATION_MAR_MIN: float = 0.35
CALIBRATION_MAR_MAX: float = 0.85


# --------------------------------------------------------------------------- #
# PERCLOS (percentage of eyelid closure over a rolling time window)
# --------------------------------------------------------------------------- #
PERCLOS_WINDOW_SEC: float = 60.0     # rolling time window used to compute PERCLOS
PERCLOS_MIN_WINDOW_SEC: float = 10.0  # minimum span of data before PERCLOS is trusted
PERCLOS_DROWSY_THRESHOLD: float = 0.30  # >=30% of the window with eyes closed => drowsy signal


# --------------------------------------------------------------------------- #
# Head-pose temporal monitoring (prolonged tilt / repeated nodding)
# --------------------------------------------------------------------------- #
# +1 if pitch increases as the head drops forward/down given this rig's
# solvePnP convention, -1 otherwise. Flip this if prolonged-tilt / nod
# detection appears inverted for your camera setup during testing.
PITCH_DOWN_SIGN: int = 1
DOWNWARD_TILT_PITCH_DEG: float = 15.0       # degrees of (signed) downward pitch to count as "tilted"
HEAD_TILT_MIN_DURATION_SEC: float = 1.5     # must be sustained this long to be "prolonged" (not a glance)
NOD_PITCH_DELTA_DEG: float = 12.0           # degrees of downward pitch that counts as a nod's "down" phase
NOD_WINDOW_SEC: float = 12.0                # rolling window for counting repeated nods
NOD_COUNT_THRESHOLD: int = 3                # this many down->up cycles within the window => repeated nodding


# --------------------------------------------------------------------------- #
# CNN + CBAM model
# --------------------------------------------------------------------------- #
IMG_SIZE: int = 64
IMG_CHANNELS: int = 1
INPUT_SHAPE: Tuple[int, int, int] = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS)
NUM_CLASSES: int = 2
CLASS_NAMES: Tuple[str, str] = ("Not Drowsy", "Drowsy")
CBAM_REDUCTION_RATIO: int = 8
DROPOUT_RATE: float = 0.4


# --------------------------------------------------------------------------- #
# Training hyperparameters
# --------------------------------------------------------------------------- #
BATCH_SIZE: int = 32
EPOCHS: int = 60
LEARNING_RATE: float = 1e-3
TRAIN_SPLIT: float = 0.8
VAL_SPLIT: float = 0.1
TEST_SPLIT: float = 0.1
RANDOM_SEED: int = 42

EARLY_STOPPING_PATIENCE: int = 10
REDUCE_LR_PATIENCE: int = 5
REDUCE_LR_FACTOR: float = 0.5
MIN_LEARNING_RATE: float = 1e-6

AUGMENTATION_ROTATION_RANGE: int = 10
AUGMENTATION_WIDTH_SHIFT: float = 0.1
AUGMENTATION_HEIGHT_SHIFT: float = 0.1
AUGMENTATION_ZOOM_RANGE: float = 0.1
AUGMENTATION_BRIGHTNESS_RANGE: Tuple[float, float] = (0.7, 1.3)
AUGMENTATION_HORIZONTAL_FLIP: bool = True


# --------------------------------------------------------------------------- #
# Sliding window / temporal smoothing
# --------------------------------------------------------------------------- #
SLIDING_WINDOW_SIZE: int = 30
DROWSY_VOTE_RATIO_THRESHOLD: float = 0.6
# The window must hold at least this fraction of SLIDING_WINDOW_SIZE frames
# before its vote is trusted for a final decision; before that, the system
# reports "Insufficient Data" rather than guessing off a mostly-empty window.
MIN_WINDOW_FILL_RATIO_FOR_DECISION: float = 0.5


# --------------------------------------------------------------------------- #
# Confidence / alert decision (multi-signal fusion)
# --------------------------------------------------------------------------- #
PROBABILITY_THRESHOLD: float = 0.65
CONFIDENCE_THRESHOLD: float = 0.60
ALERT_COOLDOWN_SEC: float = 3.0
FRAME_QUALITY_ALERT_COOLDOWN_SEC: float = 5.0
# Final fusion (utils/pipeline.py) combines the CNN+window+confidence gate
# with PERCLOS, head-pose events, and personalized EAR/MAR cue evidence.
# The CNN+window+confidence gate must always agree, AND at least this
# fraction of the other available secondary signals must agree too, before
# a "Drowsy" status/alert is raised.
FUSION_AGREEMENT_RATIO: float = 0.5


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_MAX_BYTES: int = 5 * 1024 * 1024
LOG_BACKUP_COUNT: int = 3


@dataclass
class RuntimeConfig:
    camera_index: int = CAMERA_INDEX
    probability_threshold: float = PROBABILITY_THRESHOLD
    confidence_threshold: float = CONFIDENCE_THRESHOLD
    sliding_window_size: int = SLIDING_WINDOW_SIZE
    headless: bool = False
    save_screenshots: bool = True
    log_predictions: bool = True


ensure_directories()