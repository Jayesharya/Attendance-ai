from __future__ import annotations

import unittest

from edusense.tracking import Detection, FaceTracker


class FaceTrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.face = Detection(10, 10, 80, 80)

    def test_presence_requires_sustained_observation(self) -> None:
        tracker = FaceTracker(presence_seconds=2.0)
        tracker.update([self.face], 0.0)
        tracker.update([self.face], 1.0)
        self.assertEqual(tracker.present_count(), 0)
        tracker.update([self.face], 2.0)
        self.assertEqual(tracker.present_count(), 1)
        self.assertEqual(tracker.snapshot()[0]["profile_id"], "P001")

    def test_short_occlusion_can_be_bridged_but_long_gap_resets_presence(self) -> None:
        tracker = FaceTracker(presence_seconds=2.0, max_missing_seconds=1.5)
        tracker.update([self.face], 0.0)
        tracker.update([self.face], 1.0)
        tracker.update([], 2.0)
        tracker.update([self.face], 3.0)
        self.assertEqual(tracker.snapshot()[0]["status"], "Checking")
        tracker.update([], 4.6)
        self.assertFalse(tracker.snapshot()[0]["active"])

    def test_elapsed_time_must_be_monotonic(self) -> None:
        tracker = FaceTracker()
        tracker.update([self.face], 2.0)
        with self.assertRaises(ValueError):
            tracker.update([self.face], 1.0)

    def test_distinct_detections_receive_session_only_labels(self) -> None:
        tracker = FaceTracker()
        tracker.update(
            [Detection(0, 0, 50, 50), Detection(400, 0, 50, 50)],
            0.0,
        )
        self.assertEqual([row["profile_id"] for row in tracker.snapshot()], ["P001", "P002"])


if __name__ == "__main__":
    unittest.main()
