"""Synthetic, media-free sessions for the public EduSense AI demo."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math

import pandas as pd

from .recommendations import teaching_suggestions


_SCENARIOS: dict[str, tuple[float, float, int, str]] = {
    "Quiet reading": (18.0, 0.14, 20, "A low-motion, low-speech sample for quiet independent work."),
    "Group discussion": (56.0, 0.77, 19, "A higher-motion, speech-like sample for collaborative work."),
    "Room transition": (76.0, 0.84, 16, "A busier sample that can prompt a clear next-step cue."),
}


def build_demo_session(
    scenario: str = "Group discussion",
    expected_class_size: int = 24,
    started_at: str | None = None,
) -> dict[str, object]:
    """Return deterministic, clearly synthetic metrics for a hosted walkthrough."""
    if scenario not in _SCENARIOS:
        raise ValueError(f"Unknown demo scenario: {scenario}")
    if expected_class_size < 1:
        raise ValueError("expected_class_size must be a positive integer.")

    base_motion, base_speech, present_count, description = _SCENARIOS[scenario]
    started = datetime.fromisoformat(started_at) if started_at else datetime.now(timezone.utc)
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)

    attendance = [
        {
            "profile_id": f"DEMO-{index:02d}",
            "status": "Present" if index <= present_count else "Checking",
            "active": True,
        }
        for index in range(1, present_count + 2)
    ]

    rows: list[dict[str, object]] = []
    for sample_index in range(13):
        elapsed = float(sample_index * 5)
        wave = math.sin(sample_index * math.pi / 4)
        motion = max(0.0, min(100.0, base_motion + 9.0 * wave))
        speech = max(0.0, min(1.0, base_speech + 0.08 * math.cos(sample_index * math.pi / 5)))
        detected = max(0, present_count + int(round(math.cos(sample_index * math.pi / 6))))
        audio_activity = "speech-like" if speech >= 0.35 else "quiet / ambient"
        activity = (motion + speech * 100.0) / 2.0
        timestamp = (started + timedelta(seconds=elapsed)).astimezone(timezone.utc)
        rows.append(
            {
                "timestamp": timestamp.isoformat(timespec="seconds"),
                "elapsed_seconds": elapsed,
                "detected_faces": detected,
                "active_profiles": present_count + 1,
                "present_profiles": present_count,
                "motion_signal": round(motion, 1),
                "audio_activity": audio_activity,
                "audio_dbfs": round(-54.0 + speech * 23.0, 1),
                "speech_likelihood": round(speech, 3),
                "roster_coverage": round(min(100.0, present_count * 100.0 / expected_class_size), 1),
                "activity_signal": round(activity, 1),
            }
        )

    metrics = pd.DataFrame(rows)
    summary: dict[str, object] = {
        "sample_count": int(len(metrics)),
        "mean_activity_signal": round(float(metrics["activity_signal"].mean()), 1),
        "present_profiles": present_count,
        "tracked_profiles": len(attendance),
        "active_profiles_at_end": len(attendance),
        "expected_class_size": expected_class_size,
        "roster_coverage": round(min(100.0, present_count * 100.0 / expected_class_size), 1),
        "mean_audio_dbfs": round(float(metrics["audio_dbfs"].mean()), 1),
        "audio_windows": int(len(metrics)),
    }
    return {
        "started_at": started.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "source_label": "Synthetic demo data",
        "scenario": scenario,
        "scenario_description": description,
        "summary": summary,
        "metrics": metrics,
        "attendance": attendance,
        "suggestions": teaching_suggestions(metrics, summary),
    }
