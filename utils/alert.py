"""
alert.py
Alarm playback manager. Plays alarm/alarm.wav on a loop when the driver is
detected as drowsy, and stops automatically once the driver becomes alert
again. Uses pygame's mixer, which works reliably across Windows/Mac/Linux
and supports non-blocking looped playback + explicit stop.
"""

from __future__ import annotations

from typing import Optional

import config
from utils.exceptions import AudioPlaybackError
from utils.logger import get_logger

logger = get_logger(__name__)


class AlarmManager:
    def __init__(self, sound_path: Optional[str] = None) -> None:
        self.sound_path = sound_path or str(config.ALARM_SOUND_PATH)
        self._playing = False
        self._mixer_ready = False
        self._init_mixer()

    def _init_mixer(self) -> None:
        try:
            import pygame

            pygame.mixer.init()
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
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning("Could not initialise audio mixer: %s", exc)
            self._mixer_ready = False

    def start(self) -> None:
        if self._playing or not self._mixer_ready:
            return
        try:
            self._sound.play(loops=-1)
            self._playing = True
            logger.info("Alarm started.")
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to play alarm: %s", exc)
            raise AudioPlaybackError(str(exc)) from exc

    def stop(self) -> None:
        if not self._playing or not self._mixer_ready:
            return
        try:
            self._sound.stop()
            self._playing = False
            logger.info("Alarm stopped.")
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to stop alarm: %s", exc)

    @property
    def is_playing(self) -> bool:
        return self._playing

    def shutdown(self) -> None:
        if self._mixer_ready:
            try:
                import pygame

                self.stop()
                pygame.mixer.quit()
            except Exception:  # pragma: no cover
                pass
