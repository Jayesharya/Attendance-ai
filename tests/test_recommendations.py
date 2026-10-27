from __future__ import annotations

import unittest

import pandas as pd

from edusense.recommendations import teaching_suggestions


class RecommendationTests(unittest.TestCase):
    def test_recommendations_are_available_without_audio(self) -> None:
        metrics = pd.DataFrame([{
            "motion_signal": 0.0,
            "audio_dbfs": None,
            "speech_likelihood": None,
        }])
        summary = {"roster_coverage": None}
        suggestions = teaching_suggestions(metrics, summary)
        self.assertEqual(len(suggestions), 1)
        self.assertIn("Keep the current activity", suggestions[0]["title"])

    def test_low_roster_coverage_prompts_manual_review(self) -> None:
        metrics = pd.DataFrame([{
            "motion_signal": 10.0,
            "audio_dbfs": -25.0,
            "speech_likelihood": 0.4,
        }])
        summary = {"roster_coverage": 40.0}
        suggestions = teaching_suggestions(metrics, summary)
        self.assertTrue(any("Verify attendance manually" == item["title"] for item in suggestions))


if __name__ == "__main__":
    unittest.main()
