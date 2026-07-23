"""
camera.py
Threaded webcam capture wrapper around cv2.VideoCapture, with retries and
clear error reporting if the laptop webcam cannot be opened or a read
fails repeatedly (in use by another app, disconnected, permission denied).
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np

import config
from utils.exceptions import CameraError
from utils.logger import get_logger

logger = get_logger(__name__)


class WebcamStream:
    """Continuously reads frames from the webcam on a background thread so
    the GUI / prediction loop is never blocked waiting on I/O."""

    def __init__(
        self,
        camera_index: int = config.CAMERA_INDEX,
        width: int = config.CAMERA_FRAME_WIDTH,
        height: int = config.CAMERA_FRAME_HEIGHT,
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_error: Optional[str] = None

    def _open_capture(self) -> cv2.VideoCapture:
        last_exc: Optional[Exception] = None
        for attempt in range(1, config.CAMERA_MAX_INIT_RETRIES + 1):
            try:
                cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW) \
                    if _is_windows() else cv2.VideoCapture(self.camera_index)
                if not cap.isOpened():
                    raise CameraError(
                        f"Could not open webcam at index {self.camera_index} "
                        f"(attempt {attempt}/{config.CAMERA_MAX_INIT_RETRIES})."
                    )
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS_TARGET)
                logger.info("Webcam opened at index %d", self.camera_index)
                return cap
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning("Webcam open attempt %d failed: %s", attempt, exc)
                time.sleep(config.CAMERA_RETRY_DELAY_SEC)

        raise CameraError(
            f"Webcam could not be opened after {config.CAMERA_MAX_INIT_RETRIES} "
            f"attempts. Check that no other application is using the camera, "
            f"that camera privacy permissions are granted, and that "
            f"CAMERA_INDEX in config.py matches your device."
        ) from last_exc

    def start(self) -> "WebcamStream":
        self._cap = self._open_capture()
        self._running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()
        return self

    def _update_loop(self) -> None:
        consecutive_failures = 0
        while self._running:
            if self._cap is None:
                break
            ok, frame = self._cap.read()
            if not ok or frame is None:
                consecutive_failures += 1
                self._last_error = "Failed to read frame from webcam."
                logger.warning(
                    "Frame read failed (%d consecutive failures).", consecutive_failures
                )
                if consecutive_failures >= 30:
                    logger.error("Too many consecutive frame read failures; stopping stream.")
                    self._running = False
                    break
                time.sleep(0.05)
                continue

            consecutive_failures = 0
            with self._lock:
                self._frame = frame

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            logger.info("Webcam released.")

    def __enter__(self) -> "WebcamStream":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


def _is_windows() -> bool:
    import platform

    return platform.system().lower() == "windows"
