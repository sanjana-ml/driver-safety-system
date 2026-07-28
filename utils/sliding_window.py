"""
sliding_window.py
Temporal smoothing over a rolling buffer of recent frame predictions.
Instead of deciding drowsiness from a single frame, the system evaluates
the proportion of frames in the window showing fatigue indicators
(majority voting), which filters out a normal blink or brief yawn from
triggering a false alarm.

The sliding-window vote is one of several signals fused together in
utils/pipeline.py, alongside the CNN+CBAM prediction, personalized EAR/MAR
cue evidence, PERCLOS, and head-pose events. has_enough_data() lets the
fusion logic hold off on any decision at all until the window has actually
accumulated a meaningful amount of history, rather than voting off a
mostly-empty buffer right after a session (re)starts.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List

import config


@dataclass
class WindowVote:
    is_drowsy: bool
    probability: float


class SlidingWindow:
    def __init__(self, window_size: int = config.SLIDING_WINDOW_SIZE) -> None:
        self.window_size = window_size
        self._votes: Deque[WindowVote] = deque(maxlen=window_size)

    def push(self, is_drowsy: bool, probability: float) -> None:
        self._votes.append(WindowVote(is_drowsy=is_drowsy, probability=probability))

    def reset(self) -> None:
        self._votes.clear()

    @property
    def is_full(self) -> bool:
        return len(self._votes) == self.window_size

    @property
    def fill_ratio(self) -> float:
        return len(self._votes) / float(self.window_size)

    def drowsy_vote_ratio(self) -> float:
        if not self._votes:
            return 0.0
        drowsy_count = sum(1 for v in self._votes if v.is_drowsy)
        return drowsy_count / len(self._votes)

    def mean_probability(self) -> float:
        if not self._votes:
            return 0.0
        return sum(v.probability for v in self._votes) / len(self._votes)

    def majority_vote(
        self, ratio_threshold: float = config.DROWSY_VOTE_RATIO_THRESHOLD
    ) -> bool:
        """Returns True if the fraction of drowsy votes in the window meets
        or exceeds the configured ratio threshold. A single blink or short
        yawn (a small minority of frames) will not flip this to True."""
        return self.drowsy_vote_ratio() >= ratio_threshold

    def has_enough_data(
        self, min_fill_ratio: float = config.MIN_WINDOW_FILL_RATIO_FOR_DECISION
    ) -> bool:
        """Returns True once the window holds at least `min_fill_ratio` of
        its capacity. Used by the pipeline's fusion decision to report
        "Insufficient Data" instead of trusting a vote from a window that
        has barely started filling (e.g. right after a session start or a
        reset())."""
        return self.fill_ratio >= min_fill_ratio