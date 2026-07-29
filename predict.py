"""
predict.py
Console-mode real-time drowsiness detection: opens the webcam, runs a short
driver calibration phase, then runs the full pipeline (face/landmark
detection -> quality gate with hysteresis -> calibrated adaptive cues ->
CNN+CBAM -> PERCLOS -> head-pose monitoring -> sliding window -> multi-
signal fusion -> alert), and displays an OpenCV window with overlays. This
is a lighter-weight alternative to the Tkinter GUI in gui/app.py, and is
also useful for quickly testing the pipeline after training.

Usage:
    python predict.py
    python predict.py --headless                 # no display window, console + CSV log only
    python predict.py --calibration-seconds 5     # override the default calibration duration
Press 'q' to quit.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import cv2

import config
from camera import WebcamStream
from utils.alert import AlarmManager
from utils.exceptions import CameraError, DriverSafetyError
from utils.image_utils import (
    draw_eye_and_mouth_regions,
    draw_face_box,
    draw_landmarks,
    put_status_text,
)
from utils.logger import PredictionCSVLogger, get_logger
from utils.pipeline import DriverSafetyPipeline

logger = get_logger(__name__)

_STATUS_COLORS = {
    "Drowsy": (0, 0, 255),
    "Not Drowsy": (0, 200, 0),
    "Insufficient Data": (0, 200, 255),
    "Face Not Detected": (0, 0, 255),
    "Calibrating": (255, 200, 0),
}

# Alarm-worthy statuses: the continuous alarm should be active on these and
# stopped on everything else.
_ALARM_STATUSES = ("Drowsy", "Face Not Detected")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time driver drowsiness detection (console mode).")
    parser.add_argument("--headless", action="store_true", help="Run without an OpenCV display window.")
    parser.add_argument("--camera-index", type=int, default=config.CAMERA_INDEX)
    parser.add_argument("--no-alarm", action="store_true", help="Disable audio alarm playback.")
    parser.add_argument(
        "--calibration-seconds",
        type=float,
        default=config.CALIBRATION_DURATION_SEC,
        help=(
            "How long the initial driver calibration phase lasts "
            f"(default: {config.CALIBRATION_DURATION_SEC}s)."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        pipeline = DriverSafetyPipeline(calibration_duration_sec=args.calibration_seconds)
    except DriverSafetyError as exc:
        logger.error("Could not start pipeline: %s", exc)
        return 1

    alarm = None if args.no_alarm else AlarmManager()
    csv_logger = PredictionCSVLogger()

    try:
        stream = WebcamStream(camera_index=args.camera_index).start()
    except CameraError as exc:
        logger.error("Camera error: %s", exc)
        return 1

    logger.info(
        "Starting real-time monitoring. Calibrating for %.1fs -- look naturally at the camera. "
        "Press 'q' to quit (display mode only).",
        args.calibration_seconds,
    )

    try:
        # Give the camera thread a moment to deliver the first frame.
        for _ in range(50):
            if stream.read() is not None:
                break
            time.sleep(0.05)

        while True:
            frame = stream.read()
            if frame is None:
                if stream.last_error:
                    logger.error(stream.last_error)
                    break
                time.sleep(0.01)
                continue

            result = pipeline.process_frame(frame)

            if alarm is not None:
                if result.status in _ALARM_STATUSES:
                    alarm.start()
                else:
                    alarm.stop()

            csv_logger.log_row(
                {
                    "status": result.status,
                    "probability": f"{result.cnn_probability_drowsy:.3f}",
                    "confidence": f"{result.confidence:.3f}",
                    "active_cues": result.active_cues_label,
                    "frame_quality": result.quality_label,
                    "alert_triggered": result.alert_triggered,
                }
            )

            if result.alert_triggered and result.status == "Drowsy":
                _save_alert_screenshot(frame)
                logger.warning("DROWSINESS ALERT triggered (%s).", result.status)
            elif result.alert_triggered:
                logger.warning("Alert triggered (%s) -- no screenshot (not a drowsiness event).", result.status)

            if not args.headless:
                display = frame.copy()
                if result.face_box is not None:
                    draw_face_box(display, result.face_box)
                if result.landmarks is not None:
                    draw_landmarks(display, result.landmarks)
                    draw_eye_and_mouth_regions(display, result.landmarks)

                color = _STATUS_COLORS.get(result.status, (255, 255, 255))

                if result.calibration_in_progress:
                    put_status_text(
                        display,
                        f"Calibrating... {result.calibration_remaining:.1f}s remaining",
                        (10, 25),
                        color=color,
                        scale=0.7,
                    )
                    put_status_text(
                        display, "Please look naturally at the camera", (10, 50), color=(255, 255, 255)
                    )
                else:
                    put_status_text(display, f"Status: {result.status}", (10, 25), color=color)
                    put_status_text(display, f"Confidence: {result.confidence:.2f}", (10, 50))

                put_status_text(display, f"FPS: {result.fps:.1f}", (10, 75))
                put_status_text(display, f"Cues: {result.active_cues_label}", (10, 100))
                put_status_text(display, f"Quality: {result.quality_label}", (10, 125))
                put_status_text(
                    display,
                    f"PERCLOS: {result.perclos_value:.2f}"
                    + (" (drowsy)" if result.perclos_drowsy else ""),
                    (10, 150),
                )
                if result.alert_triggered:
                    put_status_text(display, "ALERT!", (10, 180), color=(0, 0, 255), scale=1.0)

                cv2.imshow("Driver Safety Monitor", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        stream.stop()
        pipeline.shutdown()
        if alarm is not None:
            alarm.shutdown()
        cv2.destroyAllWindows()

    return 0


def _save_alert_screenshot(frame) -> None:
    session_dir = config.current_screenshot_dir()
    filename = session_dir / f"drowsy_{int(time.time() * 1000)}.png"
    try:
        cv2.imwrite(str(filename), frame)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not save alert screenshot: %s", exc)


if __name__ == "__main__":
    raise SystemExit(main())