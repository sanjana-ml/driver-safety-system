"""
alert.py
Alarm playback manager. Plays alarm/alarm.wav on a loop when the driver is
detected as drowsy or not visible, and stops automatically once things
return to normal. Also supports a single-shot "warning chime" used for
frame-quality alerts (poor lighting, head not frontal, etc.) that
shouldn't loop continuously.

Uses pygame's mixer. Every public method here is defensive: if the audio
backend fails for any reason (no device, driver issue, etc.), the failure
is logged but NEVER raised -- a broken speaker/driver must not silently
break the rest of the per-frame pipeline (rendering, CSV logging, etc.),
which is what happened when this class used to raise on failure.
"""

from __future__ import annotations

from typing import Optional

import config
from utils.logger import get_logger

logger = get_logger(__name__)


class AlarmManager:
    def __init__(self, sound_path: Optional[str] = None) -> None:
        self.sound_path = sound_path or str(config.ALARM_SOUND_PATH)
        self._playing = False
        self._mixer_ready = False
        self._sound = None
        self._init_mixer()

    def _init_mixer(self) -> None:
        try:
            import pygame

            try:
                pygame.mixer.init()
            except Exception:
                # Some Windows audio setups need explicit mixer parameters
                # instead of pygame's auto-detected defaults. Retry once
                # with a conservative, widely-compatible configuration
                # before giving up.
                pygame.mixer.quit()
                pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)

            if not config.ALARM_SOUND_PATH.exists():
                logger.warning(
                    "Alarm sound file not found at %s -- alerts will be "
                    "silent (visual alert still shown). Run "
                    "'python -m alarm.generate_alarm' or supply your own "
                    "alarm.wav.",
                    self.sound_path,
                )
                return

            self._sound = pygame.mixer.Sound(self.sound_path)
            self._mixer_ready = True
            logger.info("Audio mixer ready (%s).", self.sound_path)
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning(
                "Could not initialise audio mixer -- alerts will be silent "
                "(visual/pop-up alerts still work). Error: %s", exc
            )
            self._mixer_ready = False

    def start(self) -> None:
        """Starts the continuous looping alarm (Drowsy / Driver Not Visible)."""
        if self._playing or not self._mixer_ready:
            return
        try:
            self._sound.play(loops=-1)
            self._playing = True
            logger.info("Alarm started.")
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to play alarm (continuing without sound): %s", exc)
            self._mixer_ready = False

    def stop(self) -> None:
        if not self._playing or not self._mixer_ready:
            return
        try:
            self._sound.stop()
            self._playing = False
            logger.info("Alarm stopped.")
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to stop alarm: %s", exc)

    def play_warning_once(self) -> None:
        """Plays a single (non-looping) chime -- used for frame-quality
        alerts, which should get the driver's attention without sounding
        like a full drowsiness alarm."""
        if not self._mixer_ready:
            return
        try:
            self._sound.play(loops=0)
            logger.info("Warning chime played.")
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to play warning chime: %s", exc)
            self._mixer_ready = False

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def is_ready(self) -> bool:
        return self._mixer_ready

    def shutdown(self) -> None:
        if self._mixer_ready:
            try:
                import pygame

                self.stop()
                pygame.mixer.quit()
            except Exception:  # pragma: no cover
                pass