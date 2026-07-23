"""
sliding_window.py
Temporal smoothing over a rolling buffer of recent frame predictions.
Instead of deciding drowsiness from a single frame, the system evaluates
the proportion of frames in the window showing fatigue indicators
(majority voting), which filters out a normal blink or brief yawn from
triggering a false alarm.
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
