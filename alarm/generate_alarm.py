"""
generate_alarm.py
A ready-to-use synthesized alarm.wav ships with this project already
(alarm/alarm.wav), so you do not need to run this. It is included in case
you want to regenerate it (e.g. change the tone/duration) or the file
was accidentally deleted.

Usage:
    python -m alarm.generate_alarm
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

OUTPUT_PATH = Path(__file__).resolve().parent / "alarm.wav"


def generate_alarm_wav(
    output_path: Path = OUTPUT_PATH,
    duration_sec: float = 1.0,
    framerate: int = 44100,
    freq_low: int = 1000,
    freq_high: int = 1400,
    warble_hz: float = 4.0,
    amplitude: float = 0.6,
) -> None:
    n_samples = int(framerate * duration_sec)
    frames = bytearray()

    for i in range(n_samples):
        t = i / framerate
        segment = int(t * warble_hz) % 2
        freq = freq_low if segment == 0 else freq_high
        value = int(32767 * amplitude * math.sin(2 * math.pi * freq * t))
        frames += struct.pack("<h", value)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(framerate)
        wav_file.writeframes(bytes(frames))

    print(f"Alarm sound written to {output_path}")


if __name__ == "__main__":
    generate_alarm_wav()
