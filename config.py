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


# --------------------------------------------------------------------------- #
# Base paths
# --------------------------------------------------------------------------- #
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

# Trained artefacts
MODEL_PATH: Path = WEIGHTS_DIR / "drowsiness_cnn_cbam.h5"
BEST_CHECKPOINT_PATH: Path = WEIGHTS_DIR / "best_checkpoint.h5"
TRAINING_HISTORY_PATH: Path = LOGS_DIR / "training_history.json"

ALARM_SOUND_PATH: Path = ALARM_DIR / "alarm.wav"


def ensure_directories() -> None:
    """Create every directory this project writes to, if missing."""
    for directory in (
        DATASET_DIR,
        MODELS_DIR,
        WEIGHTS_DIR,
        LOGS_DIR,
        SCREENSHOTS_DIR,
        ALARM_DIR,
        TENSORBOARD_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #
CAMERA_INDEX: int = 0                # built-in laptop webcam
CAMERA_FRAME_WIDTH: int = 640
CAMERA_FRAME_HEIGHT: int = 480
CAMERA_FPS_TARGET: int = 30
CAMERA_MAX_INIT_RETRIES: int = 3
CAMERA_RETRY_DELAY_SEC: float = 1.0


# --------------------------------------------------------------------------- #
# Face / landmark detection (MediaPipe Face Mesh)
# --------------------------------------------------------------------------- #
NUM_LANDMARKS: int = 468             # MediaPipe Face Mesh point count
MIN_LANDMARKS_REQUIRED: int = 468    # MediaPipe returns all-or-nothing per face
MEDIAPIPE_MAX_NUM_FACES: int = 1
MEDIAPIPE_STATIC_IMAGE_MODE: bool = False   # False = video mode, enables tracking between frames
MEDIAPIPE_REFINE_LANDMARKS: bool = False    # True adds iris landmarks (not needed here)
MEDIAPIPE_MIN_DETECTION_CONFIDENCE: float = 0.5
MEDIAPIPE_MIN_TRACKING_CONFIDENCE: float = 0.5
FACE_BOX_PADDING_PX: int = 10        # padding around the min/max landmark bounding box


# --------------------------------------------------------------------------- #
# Frame quality gate
# --------------------------------------------------------------------------- #
MIN_BRIGHTNESS: float = 40.0         # mean grayscale intensity, 0-255
MAX_BRIGHTNESS: float = 230.0
MAX_YAW_DEG: float = 45.0            # head must be roughly frontal
MAX_PITCH_DEG: float = 40.0
MAX_ROLL_DEG: float = 40.0
EYE_OCCLUSION_VARIANCE_THRESHOLD: float = 15.0  # low variance -> sunglasses/occlusion

# How long the driver's face must be completely absent from the frame
# before it's treated as its own alert condition (distinct from "poor
# quality" -- this covers the driver looking far away, slumping out of
# frame, or the camera being blocked).
NO_FACE_ALERT_SECONDS: float = 3.0


# --------------------------------------------------------------------------- #
# Facial cue thresholds
# --------------------------------------------------------------------------- #
EAR_THRESHOLD: float = 0.21          # below -> eye considered closed
EAR_CONSEC_FRAMES_BLINK: int = 2
MAR_THRESHOLD: float = 0.6           # above -> mouth considered open (yawn candidate)
YAWN_CONSEC_FRAMES: int = 15
HEAD_NOD_PITCH_DELTA_DEG: float = 15.0


# --------------------------------------------------------------------------- #
# CNN + CBAM model
# --------------------------------------------------------------------------- #
IMG_SIZE: int = 64                   # face ROI is resized to IMG_SIZE x IMG_SIZE
IMG_CHANNELS: int = 1                # grayscale
INPUT_SHAPE: Tuple[int, int, int] = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS)
NUM_CLASSES: int = 2                 # Drowsy, Not Drowsy
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
DROWSY_VOTE_RATIO_THRESHOLD: float = 0.6   # fraction of window that must vote drowsy


# --------------------------------------------------------------------------- #
# Confidence / alert decision (two-gate design)
# --------------------------------------------------------------------------- #
PROBABILITY_THRESHOLD: float = 0.65
CONFIDENCE_THRESHOLD: float = 0.60
ALERT_COOLDOWN_SEC: float = 3.0


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_MAX_BYTES: int = 5 * 1024 * 1024
LOG_BACKUP_COUNT: int = 3


@dataclass
class RuntimeConfig:
    """Small mutable bundle for values that may be overridden at runtime
    (e.g. via CLI flags) without touching the module-level constants above."""

    camera_index: int = CAMERA_INDEX
    probability_threshold: float = PROBABILITY_THRESHOLD
    confidence_threshold: float = CONFIDENCE_THRESHOLD
    sliding_window_size: int = SLIDING_WINDOW_SIZE
    headless: bool = False
    save_screenshots: bool = True
    log_predictions: bool = True


ensure_directories()