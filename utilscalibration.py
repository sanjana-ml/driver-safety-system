"""
calibration.py
Driver calibration phase: at the start of every monitoring session, the
driver looks naturally at the camera for a short window
(config.CALIBRATION_DURATION_SEC, 5-10 seconds) while baseline EAR (eye
aspect ratio) and MAR (mouth aspect ratio) statistics are collected from
good-quality frames only. Once the phase ends, personalized thresholds are
derived from the driver's own baseline instead of using the fixed
config.EAR_THRESHOLD / config.MAR_THRESHOLD defaults for the rest of the
session -- a driver with naturally narrower or wider eyes, or a different
resting mouth shape, gets thresholds tuned to them rather than one
one-size-fits-all constant.

If calibration cannot collect enough usable samples (e.g. the driver's face
was poorly tracked through most of the phase), it safely reports
`calibrated=False` and callers should keep using the fixed config defaults.

Usage (see utils/pipeline.py):
    session = CalibrationSession()
    session.start()
    ...
    while session.in_progress:
        session.add_sample(ear, mar)   # only for good-quality frames
        if session.is_time_elapsed():
            thresholds = session.finalize()
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

import config


@dataclass
class PersonalizedThresholds:
    """Personalized EAR/MAR thresholds derived from a driver's own
    calibration baseline. `calibrated` is False when the session fell back
    to the fixed config defaults because too few usable samples were
    collected -- callers can use this to inform the user, but the
    thresholds themselves are always safe to use directly either way."""

    ear_threshold: float
    mar_threshold: float
    baseline_ear: float
    baseline_mar: float
    ear_std: float
    mar_std: float
    sample_count: int
    calibrated: bool = True

    @classmethod
    def defaults(cls) -> "PersonalizedThresholds":
        """Fixed fallback thresholds, used before calibration completes or
        when calibration could not collect enough reliable data."""
        return cls(
            ear_threshold=config.EAR_THRESHOLD,
            mar_threshold=config.MAR_THRESHOLD,
            baseline_ear=0.0,
            baseline_mar=0.0,
            ear_std=0.0,
            mar_std=0.0,
            sample_count=0,
            calibrated=False,
        )


class CalibrationSession:
    """Stateful calibration phase covering a single monitoring session.
    One instance should be created per DriverSafetyPipeline and start()-ed
    every time a new session begins (mirroring how SlidingWindow / CueSelector
    are reset() per session), since lighting and the driver themselves may
    have changed."""

    def __init__(self, duration_sec: float = config.CALIBRATION_DURATION_SEC) -> None:
        self.duration_sec = duration_sec
        self._ear_samples: List[float] = []
        self._mar_samples: List[float] = []
        self._start_time: Optional[float] = None
        self._done = False
        self._result: Optional[PersonalizedThresholds] = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Begins (or restarts) the calibration phase."""
        self._start_time = time.time()
        self._ear_samples = []
        self._mar_samples = []
        self._done = False
        self._result = None

    def reset(self) -> None:
        """Clears all state; calibration is neither started nor finished
        until start() is called again."""
        self._start_time = None
        self._ear_samples = []
        self._mar_samples = []
        self._done = False
        self._result = None

    # ------------------------------------------------------------------ #
    # Progress
    # ------------------------------------------------------------------ #
    @property
    def in_progress(self) -> bool:
        return self._start_time is not None and not self._done

    @property
    def elapsed(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @property
    def remaining(self) -> float:
        return max(0.0, self.duration_sec - self.elapsed)

    @property
    def progress_ratio(self) -> float:
        if self.duration_sec <= 0:
            return 1.0
        return float(np.clip(self.elapsed / self.duration_sec, 0.0, 1.0))

    @property
    def sample_count(self) -> int:
        return min(len(self._ear_samples), len(self._mar_samples))

    def is_time_elapsed(self) -> bool:
        return self._start_time is not None and self.elapsed >= self.duration_sec

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    def add_sample(self, ear: float, mar: float) -> None:
        """Records one EAR/MAR sample from a good-quality, well-tracked
        frame. Callers should only call this when the eyes and mouth are
        both currently trustworthy (not occluded, landmarks sufficient)."""
        if self._done or self._start_time is None:
            return
        if ear > 0.0:
            self._ear_samples.append(float(ear))
        if mar >= 0.0:
            self._mar_samples.append(float(mar))

    # ------------------------------------------------------------------ #
    # Finalization
    # ------------------------------------------------------------------ #
    def finalize(self) -> PersonalizedThresholds:
        """Computes personalized thresholds from whatever samples were
        collected and marks the session done. Safe to call more than once
        -- subsequent calls just return the cached result."""
        if self._result is not None:
            self._done = True
            return self._result

        have_enough_ear = len(self._ear_samples) >= config.CALIBRATION_MIN_SAMPLES
        have_enough_mar = len(self._mar_samples) >= config.CALIBRATION_MIN_SAMPLES

        if have_enough_ear:
            baseline_ear = float(np.mean(self._ear_samples))
            ear_std = float(np.std(self._ear_samples))
            ear_threshold = baseline_ear - config.CALIBRATION_EAR_STD_MULTIPLIER * ear_std
            ear_threshold = float(
                np.clip(ear_threshold, config.CALIBRATION_EAR_MIN, config.CALIBRATION_EAR_MAX)
            )
        else:
            baseline_ear, ear_std = 0.0, 0.0
            ear_threshold = config.EAR_THRESHOLD

        if have_enough_mar:
            baseline_mar = float(np.mean(self._mar_samples))
            mar_std = float(np.std(self._mar_samples))
            mar_threshold = baseline_mar + config.CALIBRATION_MAR_STD_MULTIPLIER * mar_std
            mar_threshold = float(
                np.clip(mar_threshold, config.CALIBRATION_MAR_MIN, config.CALIBRATION_MAR_MAX)
            )
        else:
            baseline_mar, mar_std = 0.0, 0.0
            mar_threshold = config.MAR_THRESHOLD

        self._result = PersonalizedThresholds(
            ear_threshold=ear_threshold,
            mar_threshold=mar_threshold,
            baseline_ear=baseline_ear,
            baseline_mar=baseline_mar,
            ear_std=ear_std,
            mar_std=mar_std,
            sample_count=self.sample_count,
            calibrated=have_enough_ear and have_enough_mar,
        )
        self._done = True
        return self._result