"""
head_pose.py
Temporal head-pose fatigue monitoring, layered on top of the per-frame pitch
/ roll angles already computed by utils.landmarks.head_pose_angles().

utils/cue_selector.py already flags a *single frame's* pitch/roll against
fixed degree thresholds (head_nod / head_tilt booleans) -- useful as one of
several per-frame cue votes, but not enough on its own to distinguish a
driver briefly checking a mirror or glancing at the dashboard from genuine
fatigue behaviour. This module adds the temporal pattern analysis that makes
that distinction:

- Prolonged downward tilt: the head must stay tilted down continuously for
  at least config.HEAD_TILT_MIN_DURATION_SEC seconds. A brief 200ms glance
  down resets the timer and never counts.
- Repeated nodding: individual down->up cycles are counted within a rolling
  time window (config.NOD_WINDOW_SEC); only when several cycles repeat in a
  short span (config.NOD_COUNT_THRESHOLD) is it flagged as "repeated
  nodding" -- a single nod (e.g. a nod of agreement, or one head-bob from a
  road bump) is ignored.

Both signals feed into the final multi-signal fusion in utils/pipeline.py.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import config


@dataclass
class HeadPoseStatus:
    prolonged_downward_tilt: bool = False
    repeated_nodding: bool = False
    tilt_duration_sec: float = 0.0
    nod_count_in_window: int = 0

    @property
    def is_fatigue_signal(self) -> bool:
        return self.prolonged_downward_tilt or self.repeated_nodding


class HeadPoseMonitor:
    """Stateful temporal head-pose analyzer. One instance should live for
    the lifetime of a monitoring session and be reset() alongside the other
    per-session state."""

    def __init__(
        self,
        tilt_min_duration_sec: float = config.HEAD_TILT_MIN_DURATION_SEC,
        nod_window_sec: float = config.NOD_WINDOW_SEC,
        nod_count_threshold: int = config.NOD_COUNT_THRESHOLD,
        downward_tilt_pitch_deg: float = config.DOWNWARD_TILT_PITCH_DEG,
        nod_pitch_delta_deg: float = config.NOD_PITCH_DELTA_DEG,
        pitch_down_sign: int = config.PITCH_DOWN_SIGN,
    ) -> None:
        self.tilt_min_duration_sec = tilt_min_duration_sec
        self.nod_window_sec = nod_window_sec
        self.nod_count_threshold = nod_count_threshold
        self.downward_tilt_pitch_deg = downward_tilt_pitch_deg
        self.nod_pitch_delta_deg = nod_pitch_delta_deg
        self.pitch_down_sign = pitch_down_sign

        self._tilt_start: Optional[float] = None
        self._nod_state: str = "up"  # "up" (neutral/raised) or "down" (nodded forward)
        self._nod_events: Deque[float] = deque()
        self._last_status = HeadPoseStatus()

    def reset(self) -> None:
        self._tilt_start = None
        self._nod_state = "up"
        self._nod_events.clear()
        self._last_status = HeadPoseStatus()

    def current_status(self) -> HeadPoseStatus:
        """Returns the most recently computed status without advancing any
        timers -- used for frames where head pose isn't currently available
        (e.g. this frame's landmarks weren't reliable) so the monitor's
        internal state isn't corrupted by a bad reading."""
        return self._last_status

    def update(self, pitch_deg: float, roll_deg: float, timestamp: Optional[float] = None) -> HeadPoseStatus:
        ts = timestamp if timestamp is not None else time.time()
        signed_pitch = pitch_deg * self.pitch_down_sign

        # -- Prolonged downward tilt: must be sustained, not momentary -- #
        is_tilted_down = signed_pitch > self.downward_tilt_pitch_deg
        if is_tilted_down:
            if self._tilt_start is None:
                self._tilt_start = ts
            tilt_duration = ts - self._tilt_start
        else:
            self._tilt_start = None
            tilt_duration = 0.0

        prolonged_tilt = is_tilted_down and tilt_duration >= self.tilt_min_duration_sec

        # -- Repeated nodding: count down->up cycles within a time window -- #
        is_nod_down = signed_pitch > self.nod_pitch_delta_deg
        if self._nod_state == "up" and is_nod_down:
            self._nod_state = "down"
        elif self._nod_state == "down" and not is_nod_down:
            self._nod_state = "up"
            self._nod_events.append(ts)  # one completed down->up cycle

        cutoff = ts - self.nod_window_sec
        while self._nod_events and self._nod_events[0] < cutoff:
            self._nod_events.popleft()

        repeated_nodding = len(self._nod_events) >= self.nod_count_threshold

        status = HeadPoseStatus(
            prolonged_downward_tilt=prolonged_tilt,
            repeated_nodding=repeated_nodding,
            tilt_duration_sec=tilt_duration,
            nod_count_in_window=len(self._nod_events),
        )
        self._last_status = status
        return status