from __future__ import annotations

import unittest

from edusense.demo import build_demo_session


class DemoSessionTests(unittest.TestCase):
    def test_demo_session_is_synthetic_and_internally_consistent(self) -> None:
        result = build_demo_session(
            "Group discussion",
            expected_class_size=24,
            started_at="2026-10-27T10:00:00+00:00",
        )
        metrics = result["metrics"]
        summary = result["summary"]
        self.assertEqual(result["source_label"], "Synthetic demo data")
        self.assertEqual(len(metrics), 13)
        self.assertAlmostEqual(float(metrics["activity_signal"].mean()), summary["mean_activity_signal"], places=1)
        self.assertEqual(summary["expected_class_size"], 24)
        self.assertGreater(summary["present_profiles"], 0)
        self.assertTrue(all(str(row["profile_id"]).startswith("DEMO-") for row in result["attendance"]))

    def test_demo_rejects_unknown_scenario_and_invalid_class_size(self) -> None:
        with self.assertRaises(ValueError):
            build_demo_session("Unknown")
        with self.assertRaises(ValueError):
            build_demo_session(expected_class_size=0)


if __name__ == "__main__":
    unittest.main()
