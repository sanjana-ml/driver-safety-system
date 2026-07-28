"""
perclos.py
PERCLOS (PERcentage of eyelid CLOSure over time) -- one of the most
well-established fatigue metrics in driver-monitoring research: the
proportion of a rolling time window during which the eyes are classified
as closed.

This is deliberately a *time*-based rolling window (config.PERCLOS_WINDOW_SEC
seconds), not a frame-count window, so the result stays correct even if the
camera's FPS drifts or frames are dropped -- unlike a fixed-size frame
buffer, whose "60 frames" might represent very different amounts of wall
time depending on FPS.

PERCLOS is one of several signals fused together in utils/pipeline.py
alongside the CNN+CBAM prediction, personalized EAR/MAR cues, head-pose
events, and the sliding-window vote.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import config


@dataclass
class _Sample:
    timestamp: float
    eye_closed: bool


class PERCLOSCalculator:
    """Rolling time-window PERCLOS tracker. One instance should live for
    the lifetime of a monitoring session and be reset() alongside the other
    per-session state (sliding window, cue selector, etc.)."""

    def __init__(
        self,
        window_seconds: float = config.PERCLOS_WINDOW_SEC,
        min_window_seconds: float = config.PERCLOS_MIN_WINDOW_SEC,
    ) -> None:
        self.window_seconds = window_seconds
        self.min_window_seconds = min_window_seconds
        self._samples: Deque[_Sample] = deque()

    def reset(self) -> None:
        self._samples.clear()

    def update(self, eye_closed: bool, timestamp: Optional[float] = None) -> float:
        """Records one sample (only call this for frames where eye state is
        actually trustworthy -- e.g. not occluded by sunglasses) and returns
        the current PERCLOS value."""
        ts = timestamp if timestamp is not None else time.time()
        self._samples.append(_Sample(timestamp=ts, eye_closed=eye_closed))
        self._trim(ts)
        return self.value()

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._samples and self._samples[0].timestamp < cutoff:
            self._samples.popleft()

    def value(self) -> float:
        """Fraction of the window's wall-clock time spent with eyes closed,
        in [0, 1]. Time-weighted between consecutive samples so it is not
        skewed by uneven sampling intervals."""
        if len(self._samples) < 2:
            if len(self._samples) == 1:
                return 1.0 if self._samples[0].eye_closed else 0.0
            return 0.0

        samples = list(self._samples)
        closed_time = 0.0
        total_time = 0.0
        for prev, curr in zip(samples, samples[1:]):
            dt = curr.timestamp - prev.timestamp
            if dt <= 0:
                continue
            total_time += dt
            if prev.eye_closed:
                closed_time += dt

        if total_time <= 0:
            closed_count = sum(1 for s in samples if s.eye_closed)
            return closed_count / len(samples)
        return closed_time / total_time

    @property
    def span_seconds(self) -> float:
        if len(self._samples) < 2:
            return 0.0
        return self._samples[-1].timestamp - self._samples[0].timestamp

    @property
    def has_sufficient_data(self) -> bool:
        """True once enough wall-clock time has been observed for the
        PERCLOS value to be a meaningful fatigue signal rather than noise
        from a handful of samples."""
        required = min(self.window_seconds, self.min_window_seconds)
        return self.span_seconds >= required

    def is_drowsy(self, threshold: float = config.PERCLOS_DROWSY_THRESHOLD) -> bool:
        """True if PERCLOS has enough data AND its value clears the drowsy
        threshold. Returns False (never drowsy) when data is insufficient,
        so a fresh/short session can't trigger a false positive from PERCLOS
        alone."""
        return self.has_sufficient_data and self.value() >= threshold