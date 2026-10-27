"""Short-window audio activity estimates without transcription or retention."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class AudioWindow:
    start_seconds: float
    end_seconds: float
    rms_dbfs: float
    activity: str
    speech_likelihood: float


def _mono_float(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float32)
    if values.ndim == 2:
        values = values.mean(axis=1)
    values = values.reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Audio must contain finite, non-empty samples.")
    return np.clip(values, -1.0, 1.0)


def _features(window: np.ndarray) -> tuple[float, str, float]:
    rms = float(np.sqrt(np.mean(np.square(window), dtype=np.float64)))
    dbfs = 20.0 * math.log10(max(rms, 1e-8))
    if dbfs < -48.0:
        return dbfs, "quiet", 0.0

    centered = window - float(np.mean(window))
    zcr = float(np.mean(np.diff(np.signbit(centered)))) if centered.size > 1 else 0.0
    spectrum = np.abs(np.fft.rfft(centered))
    power = np.square(spectrum) + 1e-12
    flatness = float(np.exp(np.mean(np.log(power))) / np.mean(power))
    # A signal-shape heuristic, not speech recognition or speaker classification.
    likelihood = float(np.clip(1.0 - abs(zcr - 0.12) * 3.0 - flatness * 0.35, 0.0, 1.0))
    activity = "speech-like" if likelihood >= 0.42 else "ambient sound"
    return dbfs, activity, likelihood


def analyze_audio_bytes(data: bytes, window_seconds: float = 1.0) -> list[AudioWindow]:
    """Decode a local WAV/FLAC/OGG clip into coarse, timestamped measurements."""
    if not data:
        raise ValueError("Choose a non-empty audio recording.")
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive.")
    try:
        samples, sample_rate = sf.read(BytesIO(data), dtype="float32", always_2d=False)
    except (RuntimeError, ValueError) as error:
        raise ValueError("Audio could not be decoded. Use WAV, FLAC, or OGG audio.") from error
    samples = _mono_float(samples)
    if sample_rate <= 0:
        raise ValueError("Audio has an invalid sample rate.")

    window_size = max(1, int(sample_rate * window_seconds))
    windows: list[AudioWindow] = []
    for start in range(0, samples.size, window_size):
        stop = min(samples.size, start + window_size)
        dbfs, activity, likelihood = _features(samples[start:stop])
        windows.append(
            AudioWindow(
                start_seconds=start / sample_rate,
                end_seconds=stop / sample_rate,
                rms_dbfs=round(dbfs, 2),
                activity=activity,
                speech_likelihood=round(likelihood, 3),
            )
        )
    return windows
