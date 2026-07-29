"""
gui/app.py
Professional Tkinter GUI for the driver safety system: live webcam feed
with bounding box / landmark / eye / mouth overlays, plus panels for
status, confidence, FPS, active cue set, frame quality, and alert state.
"""

from __future__ import annotations

import json
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Optional, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
from PIL import Image, ImageTk

import config
from camera import WebcamStream
from utils.alert import AlarmManager
from utils.exceptions import CameraError, DriverSafetyError
from utils.image_utils import draw_eye_and_mouth_regions, draw_face_box, draw_landmarks
from utils.logger import PredictionCSVLogger, get_logger
from utils.pipeline import DetectionResult, DriverSafetyPipeline

logger = get_logger(__name__)

_STATUS_COLORS = {
    "Drowsy": "#e53935",
    "Not Drowsy": "#2e7d32",
    "Insufficient Data": "#f9a825",
    "Face Not Detected": "#e53935",
    "Calibrating": "#42a5f5",
}


class DriverSafetyGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Smart Vision-Based Driver Safety Monitoring System")
        self.root.geometry("1040x680")
        self.root.minsize(900, 600)
        self.root.configure(bg="#111318")

        self.pipeline: Optional[DriverSafetyPipeline] = None
        self.stream: Optional[WebcamStream] = None
        self.alarm = AlarmManager()
        self.csv_logger = PredictionCSVLogger()

        self.running = False
        self._last_alert_screenshot_ts = 0.0
        self._last_quality_alert_time = 0.0
        self._was_calibrating = False
        self._calibration_popup_shown = False

        self._build_layout()
        self._try_init_pipeline()

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #
    def _build_layout(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TLabel", background="#111318", foreground="#e8e8e8", font=("Segoe UI", 11))
        style.configure("Header.TLabel", background="#111318", foreground="#ffffff", font=("Segoe UI", 16, "bold"))
        style.configure("Value.TLabel", background="#1b1e26", foreground="#ffffff", font=("Segoe UI", 13, "bold"))
        style.configure("TButton", font=("Segoe UI", 11))

        header = ttk.Label(self.root, text="Driver Safety Monitor", style="Header.TLabel")
        header.pack(side=tk.TOP, anchor="w", padx=16, pady=(12, 4))

        body = tk.Frame(self.root, bg="#111318")
        body.pack(fill=tk.BOTH, expand=True, padx=16, pady=8)

        # Video panel
        video_frame = tk.Frame(body, bg="#000000", bd=2, relief=tk.RIDGE)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # CRITICAL: without this, the frame auto-resizes to fit whatever
        # image the label inside it is showing, which then grows the next
        # image, which grows the frame again, and so on -- eventually
        # overrunning the sidebar. With propagation off, video_frame's
        # size is dictated purely by the pack layout (i.e. "however much
        # space is left after the fixed-width sidebar"), never by its
        # child's content.
        video_frame.pack_propagate(False)
        self.video_label = tk.Label(video_frame, bg="#000000")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        self._video_panel_size: Tuple[int, int] = (760, 570)
        video_frame.bind("<Configure>", self._on_video_frame_resize)

        # Info panel -- wrapped in a scrollable canvas. Adding the new
        # Calibration/PERCLOS/Thresholds rows made the sidebar taller than
        # the window in some sizes; a fixed-height Frame would silently
        # clip Start/Stop and the metrics section below it. Scrolling keeps
        # every row reachable regardless of window size, without changing
        # any of the row widgets, styling, or order below.
        info_outer = tk.Frame(body, bg="#111318", width=300)
        info_outer.pack(side=tk.RIGHT, fill=tk.Y, padx=(16, 0))
        info_outer.pack_propagate(False)

        info_canvas = tk.Canvas(info_outer, bg="#111318", highlightthickness=0, bd=0)
        info_scrollbar = ttk.Scrollbar(info_outer, orient=tk.VERTICAL, command=info_canvas.yview)
        info_canvas.configure(yscrollcommand=info_scrollbar.set)
        info_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        info_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        info_frame = tk.Frame(info_canvas, bg="#111318")
        info_canvas_window = info_canvas.create_window((0, 0), window=info_frame, anchor="nw")

        def _on_info_frame_configure(_event: tk.Event) -> None:
            info_canvas.configure(scrollregion=info_canvas.bbox("all"))

        def _on_info_canvas_configure(event: tk.Event) -> None:
            info_canvas.itemconfigure(info_canvas_window, width=event.width)

        def _on_info_mousewheel(event: tk.Event) -> None:
            info_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        info_frame.bind("<Configure>", _on_info_frame_configure)
        info_canvas.bind("<Configure>", _on_info_canvas_configure)
        info_canvas.bind("<Enter>", lambda _e: info_canvas.bind_all("<MouseWheel>", _on_info_mousewheel))
        info_canvas.bind("<Leave>", lambda _e: info_canvas.unbind_all("<MouseWheel>"))

        self.status_value = self._add_info_row(info_frame, "Status")
        self.confidence_value = self._add_info_row(info_frame, "Confidence")
        self.probability_value = self._add_info_row(info_frame, "Probability (Drowsy)")
        self.fps_value = self._add_info_row(info_frame, "FPS")
        self.cue_value = self._add_info_row(info_frame, "Active Cue(s)")
        self.quality_value = self._add_info_row(info_frame, "Frame Quality")
        self.alert_value = self._add_info_row(info_frame, "Alert Status")
        self.calibration_value = self._add_info_row(info_frame, "Calibration")
        self.perclos_value = self._add_info_row(info_frame, "PERCLOS")
        self.thresholds_value = self._add_info_row(info_frame, "Thresholds")
        self.ear_value = self._add_info_row(info_frame, "Current EAR")
        self.mar_value = self._add_info_row(info_frame, "Current MAR")
        self.head_pose_value = self._add_info_row(info_frame, "Head Pose (P/Y/R)")
        self.ear_threshold_value = self._add_info_row(info_frame, "EAR Threshold")
        self.mar_threshold_value = self._add_info_row(info_frame, "MAR Threshold")

        controls = tk.Frame(info_frame, bg="#111318")
        controls.pack(fill=tk.X, pady=(24, 0))

        self.show_landmarks_var = tk.BooleanVar(value=True)
        landmarks_check = tk.Checkbutton(
            controls,
            text="Show Landmarks & Overlays",
            variable=self.show_landmarks_var,
            bg="#111318",
            fg="#e8e8e8",
            selectcolor="#1b1e26",
            activebackground="#111318",
            activeforeground="#ffffff",
            font=("Segoe UI", 10),
        )
        landmarks_check.pack(fill=tk.X, pady=(0, 8), anchor="w")

        self.start_btn = ttk.Button(controls, text="Start Monitoring", command=self.start)
        self.start_btn.pack(fill=tk.X, pady=4)
        self.stop_btn = ttk.Button(controls, text="Stop", command=self.stop, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=4)

        # --- Model performance summary (loaded from logs/test_metrics.json,
        # produced by train.py / testing/run_tests.py) -------------------- #
        metrics_frame = tk.Frame(info_frame, bg="#111318")
        metrics_frame.pack(fill=tk.X, pady=(24, 0))
        ttk.Label(metrics_frame, text="Model Performance", style="Header.TLabel").pack(
            anchor="w", pady=(0, 6)
        )
        self.test_accuracy_value = self._add_info_row(metrics_frame, "Test Accuracy")
        self.roc_auc_value = self._add_info_row(metrics_frame, "ROC AUC")
        self.view_report_btn = ttk.Button(
            metrics_frame, text="View Full Report", command=self._open_metrics_report
        )
        self.view_report_btn.pack(fill=tk.X, pady=(8, 0))
        self._load_model_metrics()

        self.footer_label = ttk.Label(
            self.root, text="Model not loaded.", foreground="#f9a825", background="#111318"
        )
        self.footer_label.pack(side=tk.BOTTOM, anchor="w", padx=16, pady=8)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_video_frame_resize(self, event: tk.Event) -> None:
        if event.width > 10 and event.height > 10:
            self._video_panel_size = (event.width, event.height)

    def _add_info_row(self, parent: tk.Frame, label: str) -> tk.Label:
        row = tk.Frame(parent, bg="#111318")
        row.pack(fill=tk.X, pady=6)
        ttk.Label(row, text=label).pack(anchor="w")
        value = tk.Label(
            row, text="--", bg="#1b1e26", fg="#ffffff", font=("Segoe UI", 13, "bold"),
            anchor="w", padx=8, pady=6,
        )
        value.pack(fill=tk.X)
        return value

    # ------------------------------------------------------------------ #
    # Model performance metrics (from training/evaluation output)
    # ------------------------------------------------------------------ #
    def _load_model_metrics(self) -> None:
        metrics_path = config.LOGS_DIR / "test_metrics.json"
        if not metrics_path.exists():
            self.test_accuracy_value.configure(text="No report yet")
            self.roc_auc_value.configure(text="No report yet")
            self.view_report_btn.configure(state=tk.DISABLED)
            return
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            acc = metrics.get("test_accuracy")
            roc_auc = metrics.get("roc_auc")
            self.test_accuracy_value.configure(text=f"{acc * 100:.2f}%" if acc is not None else "--")
            self.roc_auc_value.configure(text=f"{roc_auc:.3f}" if roc_auc is not None else "--")
            self.view_report_btn.configure(state=tk.NORMAL)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read test_metrics.json: %s", exc)
            self.test_accuracy_value.configure(text="Error")
            self.roc_auc_value.configure(text="Error")
            self.view_report_btn.configure(state=tk.DISABLED)

    def _open_metrics_report(self) -> None:
        """Opens a separate window showing the full classification report
        plus the confusion matrix / ROC curve / training curve images
        produced by train.py and testing/run_tests.py."""
        window = tk.Toplevel(self.root)
        window.title("Model Performance Report")
        window.geometry("900x700")
        window.configure(bg="#111318")

        canvas = tk.Canvas(window, bg="#111318", highlightthickness=0)
        scrollbar = ttk.Scrollbar(window, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg="#111318")

        scroll_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        report_path = config.LOGS_DIR / "classification_report.txt"
        if report_path.exists():
            ttk.Label(scroll_frame, text="Classification Report", style="Header.TLabel").pack(
                anchor="w", padx=12, pady=(12, 4)
            )
            report_text = tk.Text(
                scroll_frame, height=10, bg="#1b1e26", fg="#ffffff", font=("Consolas", 10),
                relief=tk.FLAT, wrap=tk.NONE,
            )
            report_text.insert("1.0", report_path.read_text(encoding="utf-8"))
            report_text.configure(state=tk.DISABLED)
            report_text.pack(fill=tk.X, padx=12, pady=(0, 12))

        self._embedded_images = []  # keep references so Tk doesn't garbage-collect them
        for label, filename in (
            ("Confusion Matrix", "confusion_matrix.png"),
            ("ROC Curve", "roc_curve.png"),
            ("Training Accuracy", "accuracy_graph.png"),
            ("Training Loss", "loss_graph.png"),
        ):
            image_path = config.LOGS_DIR / filename
            if not image_path.exists():
                continue
            ttk.Label(scroll_frame, text=label, style="Header.TLabel").pack(
                anchor="w", padx=12, pady=(12, 4)
            )
            img = Image.open(image_path)
            img.thumbnail((820, 500))
            photo = ImageTk.PhotoImage(img)
            self._embedded_images.append(photo)
            tk.Label(scroll_frame, image=photo, bg="#111318").pack(padx=12, pady=(0, 8))

        if not report_path.exists() and not any(
            (config.LOGS_DIR / f).exists()
            for f in ("confusion_matrix.png", "roc_curve.png", "accuracy_graph.png", "loss_graph.png")
        ):
            ttk.Label(
                scroll_frame,
                text="No report artefacts found in logs/. Run 'python train.py' "
                "or 'python -m testing.run_tests' first.",
            ).pack(padx=12, pady=12)

    # ------------------------------------------------------------------ #
    # Pipeline lifecycle
    # ------------------------------------------------------------------ #
    def _try_init_pipeline(self) -> None:
        try:
            self.pipeline = DriverSafetyPipeline()
            self.footer_label.configure(
                text="Model loaded. Click 'Start Monitoring' to begin.", foreground="#66bb6a"
            )
        except DriverSafetyError as exc:
            logger.error("Pipeline initialisation failed: %s", exc)
            self.footer_label.configure(text=str(exc), foreground="#e53935")
            messagebox.showerror("Initialisation Error", str(exc))

    def start(self) -> None:
        if self.pipeline is None:
            self._try_init_pipeline()
            if self.pipeline is None:
                return

        try:
            self.stream = WebcamStream().start()
        except CameraError as exc:
            logger.error("Camera error: %s", exc)
            messagebox.showerror("Camera Error", str(exc))
            return

        self.pipeline.reset()
        self.running = True
        self._was_calibrating = True
        self._calibration_popup_shown = False
        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.footer_label.configure(text="Monitoring active.", foreground="#66bb6a")
        self._load_model_metrics()
        messagebox.showinfo(
            "Calibration Starting",
            f"Please stay steady and look naturally at the camera for the next "
            f"~{config.CALIBRATION_DURATION_SEC:.0f} seconds.\n\n"
            "This lets the system learn your personal eye/mouth baseline "
            "instead of using generic fixed thresholds.",
        )
        self._update_frame()

    def stop(self) -> None:
        self.running = False
        if self.stream is not None:
            self.stream.stop()
            self.stream = None
        self.alarm.stop()
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.footer_label.configure(text="Monitoring stopped.", foreground="#f9a825")

    def _on_close(self) -> None:
        self.stop()
        if self.pipeline is not None:
            self.pipeline.shutdown()
        self.alarm.shutdown()
        self.root.destroy()

    # ------------------------------------------------------------------ #
    # Frame loop
    # ------------------------------------------------------------------ #
    def _update_frame(self) -> None:
        if not self.running or self.stream is None or self.pipeline is None:
            return

        frame = self.stream.read()
        if frame is None:
            if self.stream.last_error:
                messagebox.showerror("Camera Error", self.stream.last_error)
                self.stop()
                return
            self.root.after(15, self._update_frame)
            return

        try:
            result = self._process(frame)
        except Exception as exc:  # noqa: BLE001 -- never let the GUI crash
            logger.exception("Unexpected error while processing frame: %s", exc)
            self.root.after(15, self._update_frame)
            return

        self._render(frame, result)
        self.root.after(15, self._update_frame)

    def _process(self, frame) -> DetectionResult:
        result = self.pipeline.process_frame(frame)

        if self._was_calibrating and not result.calibration_in_progress and not self._calibration_popup_shown:
            self._calibration_popup_shown = True
            self.root.after(0, lambda r=result: self._show_calibration_done_popup(r))
        self._was_calibrating = result.calibration_in_progress

        if result.calibration_in_progress:
            # Never alarm mid-calibration -- no verdict has been formed yet.
            self.alarm.stop()
        elif result.status in ("Drowsy", "Face Not Detected"):
            self.alarm.start()
        else:
            self.alarm.stop()

            # Frame-quality issues (poor lighting, head not frontal, too
            # few landmarks, etc.) get their own lighter-weight alert: a
            # single chime plus a pop-up, rate-limited so it doesn't spam
            # while the condition persists across many frames.
            if not result.frame_quality_ok and result.status == "Insufficient Data":
                now = time.time()
                if now - self._last_quality_alert_time >= config.FRAME_QUALITY_ALERT_COOLDOWN_SEC:
                    self._last_quality_alert_time = now
                    self.alarm.play_warning_once()
                    self.root.after(
                        0, lambda label=result.quality_label: self._show_quality_popup(label)
                    )

        # Only a genuine "Drowsy" verdict counts as a drowsiness event for
        # screenshot purposes -- "Face Not Detected" also alerts, but it is
        # not itself evidence of drowsiness.
        if result.alert_triggered and result.status == "Drowsy":
            self._save_alert_screenshot(frame)

        self.csv_logger.log_row(
            {
                "status": result.status,
                "probability": f"{result.cnn_probability_drowsy:.3f}",
                "confidence": f"{result.confidence:.3f}",
                "active_cues": result.active_cues_label,
                "frame_quality": result.quality_label,
                "alert_triggered": result.alert_triggered,
            }
        )
        return result

    def _show_calibration_done_popup(self, result: DetectionResult) -> None:
        cues = result.cue_readings
        if cues is None or not result.personalized_thresholds_active:
            messagebox.showinfo(
                "Calibration Complete",
                "Calibration finished, but not enough reliable data was collected "
                "to personalize your thresholds -- using the default fixed "
                "thresholds for this session instead.",
            )
            return
        messagebox.showinfo(
            "Calibration Complete",
            "Your personalized thresholds for this session:\n\n"
            f"EAR threshold: {cues.ear_threshold_used:.3f}\n"
            f"MAR threshold: {cues.mar_threshold_used:.3f}\n\n"
            "Monitoring will now use these instead of the generic defaults.",
        )

    def _show_quality_popup(self, quality_label: str) -> None:
        messagebox.showwarning(
            "Frame Quality Warning",
            f"Monitoring paused for this frame: {quality_label}.\n\n"
            "Please face the camera directly with good lighting so "
            "drowsiness detection can resume.",
        )

    def _render(self, frame, result: DetectionResult) -> None:
        display = frame.copy()
        if result.face_box is not None:
            draw_face_box(display, result.face_box)
        if result.landmarks is not None and self.show_landmarks_var.get():
            draw_landmarks(display, result.landmarks)
            draw_eye_and_mouth_regions(display, result.landmarks)

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        panel_w, panel_h = self._video_panel_size
        if panel_w > 10 and panel_h > 10:
            image = image.resize((panel_w, panel_h))

        photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=photo)
        self.video_label.image = photo  # keep a reference, avoid GC

        if result.calibration_in_progress:
            status_text = f"Calibrating ({result.calibration_remaining:.1f}s left)"
        else:
            status_text = result.status
        self.status_value.configure(
            text=status_text, fg=_STATUS_COLORS.get(result.status, "#ffffff")
        )
        self.confidence_value.configure(text=f"{result.confidence:.2f}")
        self.probability_value.configure(text=f"{result.cnn_probability_drowsy:.2f}")
        self.fps_value.configure(text=f"{result.fps:.1f}")
        self.cue_value.configure(text=result.active_cues_label)
        self.quality_value.configure(text=result.quality_label)

        if result.calibration_in_progress:
            self.calibration_value.configure(
                text=f"{result.calibration_progress * 100:.0f}%", fg="#42a5f5"
            )
        else:
            self.calibration_value.configure(text="Complete", fg="#66bb6a")

        if result.perclos_ready:
            perclos_color = "#e53935" if result.perclos_drowsy else "#66bb6a"
            self.perclos_value.configure(text=f"{result.perclos_value * 100:.0f}%", fg=perclos_color)
        else:
            self.perclos_value.configure(text="Warming up...", fg="#f9a825")

        self.thresholds_value.configure(
            text="Personalized" if result.personalized_thresholds_active else "Default",
            fg="#66bb6a" if result.personalized_thresholds_active else "#e8e8e8",
        )

        cues = result.cue_readings
        if cues is not None:
            self.ear_value.configure(text=f"{cues.corrected_ear:.3f}")
            self.mar_value.configure(text=f"{cues.mar:.3f}")
            self.head_pose_value.configure(
                text=f"{cues.pitch:.0f} / {cues.yaw:.0f} / {cues.roll:.0f}"
            )
            self.ear_threshold_value.configure(text=f"{cues.ear_threshold_used:.3f}")
            self.mar_threshold_value.configure(text=f"{cues.mar_threshold_used:.3f}")
        else:
            for value_label in (
                self.ear_value, self.mar_value, self.head_pose_value,
                self.ear_threshold_value, self.mar_threshold_value,
            ):
                value_label.configure(text="--")

        is_alert_state = (
            not result.calibration_in_progress
            and (result.alert_triggered or result.status in ("Drowsy", "Face Not Detected"))
        )
        self.alert_value.configure(
            text="ALERT" if is_alert_state else "Normal",
            fg="#e53935" if is_alert_state else "#66bb6a",
        )

    def _save_alert_screenshot(self, frame) -> None:
        now = time.time()
        if now - self._last_alert_screenshot_ts < 2.0:
            return
        self._last_alert_screenshot_ts = now
        session_dir = config.current_screenshot_dir()
        filename = session_dir / f"drowsy_{int(now * 1000)}.png"
        try:
            cv2.imwrite(str(filename), frame)
            logger.info("Saved alert screenshot: %s", filename)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not save alert screenshot: %s", exc)


def launch() -> None:
    root = tk.Tk()
    DriverSafetyGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch()