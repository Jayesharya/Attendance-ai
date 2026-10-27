"""Transparent fusion of session-level vision and room-audio signals."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import pandas as pd

from .audio import AudioWindow
from .vision import VideoResult


def _audio_for_time(windows: list[AudioWindow], elapsed: float) -> AudioWindow | None:
    for window in windows:
        if window.start_seconds <= elapsed < window.end_seconds:
            return window
    return None


def build_session_metrics(
    video: VideoResult,
    audio_windows: list[AudioWindow] | None = None,
    expected_class_size: int | None = None,
) -> pd.DataFrame:
    """Align coarse audio windows to video elapsed time and expose each component."""
    if expected_class_size is not None and expected_class_size < 1:
        raise ValueError("expected_class_size must be a positive integer.")
    windows = audio_windows or []
    started_at = datetime.now(timezone.utc)
    rows: list[dict[str, object]] = []

    for sample in video.timeline:
        elapsed = float(sample["elapsed_seconds"])
        audio = _audio_for_time(windows, elapsed)
        active_profiles = int(sample["active_profiles"])
        present_profiles = int(sample["present_profiles"])
        motion_score = float(sample["motion_score"])
        roster_coverage = (
            min(100.0, present_profiles * 100.0 / expected_class_size)
            if expected_class_size is not None
            else None
        )
        audio_score = audio.speech_likelihood * 100.0 if audio else None
        components = [motion_score]
        if audio_score is not None:
            components.append(audio_score)
        signal_index = sum(components) / len(components)
        sample_time = started_at.timestamp() + elapsed

        rows.append(
            {
                "timestamp": datetime.fromtimestamp(sample_time, timezone.utc).isoformat(timespec="seconds"),
                "elapsed_seconds": elapsed,
                "detected_faces": int(sample["detected_faces"]),
                "active_profiles": active_profiles,
                "present_profiles": present_profiles,
                "motion_signal": round(motion_score, 1),
                "audio_activity": audio.activity if audio else "not provided",
                "audio_dbfs": audio.rms_dbfs if audio else None,
                "speech_likelihood": audio.speech_likelihood if audio else None,
                "roster_coverage": round(roster_coverage, 1) if roster_coverage is not None else None,
                "activity_signal": round(signal_index, 1),
            }
        )
    return pd.DataFrame(rows)


def summarize_session(
    metrics: pd.DataFrame,
    attendance: Iterable[dict[str, object]],
    expected_class_size: int | None,
) -> dict[str, object]:
    """Calculate session averages without implying student-level judgments."""
    if expected_class_size is not None and expected_class_size < 1:
        raise ValueError("expected_class_size must be a positive integer.")
    tracks = list(attendance)
    present = sum(row.get("status") == "Present" for row in tracks)
    active = sum(bool(row.get("active")) for row in tracks)
    mean_activity = float(metrics["activity_signal"].mean()) if not metrics.empty else 0.0
    audio = metrics[metrics["audio_dbfs"].notna()] if not metrics.empty else metrics
    return {
        "sample_count": int(len(metrics)),
        "mean_activity_signal": round(mean_activity, 1),
        "present_profiles": int(present),
        "tracked_profiles": int(len(tracks)),
        "active_profiles_at_end": int(active),
        "expected_class_size": expected_class_size,
        "roster_coverage": round(present * 100.0 / expected_class_size, 1)
        if expected_class_size
        else None,
        "mean_audio_dbfs": round(float(audio["audio_dbfs"].mean()), 1)
        if not audio.empty
        else None,
        "audio_windows": int(len(audio)),
    }
