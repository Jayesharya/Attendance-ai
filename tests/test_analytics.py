from __future__ import annotations

import unittest

from edusense.analytics import build_session_metrics, summarize_session
from edusense.audio import AudioWindow
from edusense.vision import VideoResult


class AnalyticsTests(unittest.TestCase):
    def test_audio_and_visual_features_are_aligned_and_exposed_separately(self) -> None:
        video = VideoResult(
            timeline=[
                {"elapsed_seconds": 0.5, "detected_faces": 2, "active_profiles": 2, "present_profiles": 1, "motion_score": 20.0},
                {"elapsed_seconds": 1.5, "detected_faces": 1, "active_profiles": 1, "present_profiles": 0, "motion_score": 10.0},
            ],
            attendance=[{"profile_id": "P001", "status": "Present", "active": True}],
            preview_jpeg=None,
            duration_seconds=2.0,
            source_label="clip.mp4",
        )
        windows = [AudioWindow(0.0, 1.0, -20.0, "speech-like", 0.6)]
        metrics = build_session_metrics(video, windows, expected_class_size=2)
        self.assertAlmostEqual(metrics.iloc[0]["activity_signal"], 40.0)
        self.assertAlmostEqual(metrics.iloc[0]["roster_coverage"], 50.0)
        self.assertEqual(metrics.iloc[1]["audio_activity"], "not provided")
        self.assertAlmostEqual(metrics.iloc[1]["activity_signal"], 10.0)

    def test_summary_uses_confirmed_anonymous_profiles(self) -> None:
        metrics = build_session_metrics(
            VideoResult(
                timeline=[{"elapsed_seconds": 0.0, "detected_faces": 1, "active_profiles": 1, "present_profiles": 1, "motion_score": 30.0}],
                attendance=[],
                preview_jpeg=None,
                duration_seconds=0.1,
                source_label="clip.mp4",
            ),
            [],
            expected_class_size=4,
        )
        attendance = [
            {"profile_id": "P001", "status": "Present", "active": True},
            {"profile_id": "P002", "status": "Checking", "active": True},
        ]
        summary = summarize_session(metrics, attendance, expected_class_size=4)
        self.assertEqual(summary["present_profiles"], 1)
        self.assertEqual(summary["tracked_profiles"], 2)
        self.assertEqual(summary["roster_coverage"], 25.0)
        self.assertIsNone(summary["mean_audio_dbfs"])

    def test_expected_class_size_must_be_positive(self) -> None:
        video = VideoResult([], [], None, 0.0, "clip.mp4")
        with self.assertRaises(ValueError):
            build_session_metrics(video, expected_class_size=0)


if __name__ == "__main__":
    unittest.main()
