"""Contextual, non-evaluative suggestions from aggregate session signals."""

from __future__ import annotations

import pandas as pd


def teaching_suggestions(metrics: pd.DataFrame, summary: dict[str, object]) -> list[dict[str, str]]:
    """Return practical instructor prompts; outputs remain suggestions to review."""
    if metrics.empty:
        return [{
            "title": "Start with a quick check-in",
            "action": "Ask students to signal how ready they feel for the next activity.",
            "reason": "There is not enough session data to infer a useful pattern yet.",
        }]

    suggestions: list[dict[str, str]] = []
    audio_present = metrics[metrics["audio_dbfs"].notna()]
    mean_motion = float(metrics["motion_signal"].mean())
    mean_speech = (
        float(audio_present["speech_likelihood"].mean())
        if not audio_present.empty
        else None
    )
    coverage = summary.get("roster_coverage")

    if mean_speech is not None and mean_speech < 0.25:
        suggestions.append({
            "title": "Check understanding in a low-pressure way",
            "action": "Offer a one-minute written response, poll, or pair discussion.",
            "reason": "The audio clip had little speech-like activity; quiet work may still be focused.",
        })
    if mean_speech is not None and mean_speech > 0.72 and mean_motion > 55:
        suggestions.append({
            "title": "Make the next transition explicit",
            "action": "Pause, state the next step, and give a short time cue.",
            "reason": "Audio and movement signals were elevated during this sample.",
        })
    if coverage is not None and float(coverage) < 80:
        suggestions.append({
            "title": "Verify attendance manually",
            "action": "Use the class roster to confirm anyone the camera may have missed.",
            "reason": "Anonymous face detections are not reliable identity or attendance proof.",
        })
    if not suggestions:
        suggestions.append({
            "title": "Keep the current activity and gather a learner response",
            "action": "Ask one short question and compare the response with the classroom context.",
            "reason": "The aggregate signals do not point to a clear need for a change.",
        })
    return suggestions[:3]
