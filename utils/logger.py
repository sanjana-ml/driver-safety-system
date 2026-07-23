"""Centralised logging configuration. Every module calls get_logger(__name__)
to obtain a logger that writes to both the console and a rotating log file
under logs/app.log."""

from __future__ import annotations

import csv
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

import config

_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    config.ensure_directories()
    root = logging.getLogger("driver_safety")
    root.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
    root.propagate = False

    formatter = logging.Formatter(config.LOG_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    try:
        file_handler = RotatingFileHandler(
            config.APP_LOG_FILE,
            maxBytes=config.LOG_MAX_BYTES,
            backupCount=config.LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError:
        # Filesystem might be read-only or permission denied; console-only
        # logging is still functional so we do not crash the app.
        console_handler.stream.write(
            "WARNING: could not open log file, continuing with console logging only\n"
        )

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child logger, e.g. get_logger(__name__)."""
    _configure_root()
    return logging.getLogger(f"driver_safety.{name}")


class PredictionCSVLogger:
    """Appends one row per prediction to logs/prediction_log.csv.
    Creates the file with a header the first time it is used."""

    _FIELDNAMES = [
        "timestamp",
        "status",
        "probability",
        "confidence",
        "active_cues",
        "frame_quality",
        "alert_triggered",
    ]

    def __init__(self, csv_path: Optional[Path] = None) -> None:
        self.csv_path = csv_path or config.PREDICTION_LOG_CSV
        self._logger = get_logger(self.__class__.__name__)
        self._ensure_header()

    def _ensure_header(self) -> None:
        try:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.csv_path.exists():
                with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=self._FIELDNAMES)
                    writer.writeheader()
        except OSError as exc:
            self._logger.warning("Could not initialise prediction CSV log: %s", exc)

    def log_row(self, row: Dict[str, Any]) -> None:
        row = dict(row)
        row.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
        try:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._FIELDNAMES)
                writer.writerow({k: row.get(k, "") for k in self._FIELDNAMES})
        except OSError as exc:
            self._logger.warning("Could not write prediction row to CSV: %s", exc)
