"""Application-level coordination for one local analysis session."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

import pandas as pd

from .analytics import build_session_metrics, summarize_session
from .audio import AudioWindow, analyze_audio_bytes
from .recommendations import teaching_suggestions
from .vision import VideoResult, analyze_video


@dataclass(frozen=True)
class SessionResult:
    started_at: str
    video: VideoResult
    metrics: pd.DataFrame
    summary: dict[str, object]
    suggestions: list[dict[str, str]]
    audio_windows: list[AudioWindow]


def analyze_session(
    source: int | str,
    max_seconds: float,
    audio_bytes: bytes | None,
    expected_class_size: int | None,
    progress: Callable[[float, str], None] | None = None,
) -> SessionResult:
    """Run local vision/audio analysis and combine only timestamped features."""
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    video = analyze_video(source, max_seconds=max_seconds, progress=progress)
    audio_windows = analyze_audio_bytes(audio_bytes) if audio_bytes else []
    metrics = build_session_metrics(video, audio_windows, expected_class_size)
    summary = summarize_session(metrics, video.attendance, expected_class_size)
    suggestions = teaching_suggestions(metrics, summary)
    return SessionResult(
        started_at=started_at,
        video=video,
        metrics=metrics,
        summary=summary,
        suggestions=suggestions,
        audio_windows=audio_windows,
    )
