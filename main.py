"""
main.py
Top-level entry point. Launches the professional GUI (gui/app.py). Run
'python predict.py' instead if you prefer a lightweight OpenCV-window
console mode.

Usage:
    python main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import config
from utils.logger import get_logger

logger = get_logger(__name__)


def _check_prerequisites() -> bool:
    ok = True
    if not config.MODEL_PATH.exists():
        logger.error(
            "Missing trained model at %s -- run 'python train.py' first "
            "(after adding your dataset under dataset/ -- see dataset/README.md).",
            config.MODEL_PATH,
        )
        ok = False
    return ok


def main() -> int:
    config.ensure_directories()
    if not _check_prerequisites():
        return 1

    from gui.app import launch

    launch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
