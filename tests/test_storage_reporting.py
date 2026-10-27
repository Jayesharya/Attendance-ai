from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from edusense.reporting import make_pdf_report
from edusense.storage import SessionStore


class StorageAndReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = SessionStore(Path(self.temp_dir.name) / "history.sqlite3", keep_latest=2)
        self.metrics = pd.DataFrame([{
            "timestamp": "2026-10-26T10:00:00+00:00",
            "elapsed_seconds": 0.0,
            "detected_faces": 1,
            "present_profiles": 1,
            "motion_signal": 25.0,
            "audio_activity": "not provided",
            "audio_dbfs": None,
            "speech_likelihood": None,
            "roster_coverage": 50.0,
            "activity_signal": 25.0,
        }])
        self.attendance = [{"profile_id": "P001", "status": "Present", "active": True}]

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _summary(self) -> dict[str, object]:
        return {
            "sample_count": 1,
            "mean_activity_signal": 25.0,
            "present_profiles": 1,
            "tracked_profiles": 1,
            "active_profiles_at_end": 1,
            "expected_class_size": 2,
            "roster_coverage": 50.0,
            "mean_audio_dbfs": None,
            "audio_windows": 0,
        }

    def test_saved_sessions_round_trip_and_clear(self) -> None:
        session_id = self.store.save("sample.mp4", self._summary(), self.metrics, self.attendance)
        saved = self.store.load(session_id)
        self.assertIsNotNone(saved)
        self.assertEqual(saved["source_label"], "sample.mp4")
        self.assertEqual(saved["summary"]["present_profiles"], 1)
        self.assertAlmostEqual(saved["metrics"].iloc[0]["activity_signal"], 25.0)
        self.assertEqual(saved["attendance"][0]["profile_id"], "P001")
        self.assertEqual(len(self.store.list()), 1)
        self.assertEqual(self.store.clear(), 1)
        self.assertEqual(self.store.list(), [])

    def test_history_keeps_only_the_configured_number_of_rows(self) -> None:
        for index in range(3):
            self.store.save(
                f"sample-{index}.mp4",
                self._summary(),
                self.metrics,
                self.attendance,
                started_at=f"2026-10-26T10:00:0{index}+00:00",
            )
        self.assertEqual(len(self.store.list()), 2)
        self.assertEqual(self.store.list()[0]["source_label"], "sample-2.mp4")

    def test_pdf_is_generated_in_memory(self) -> None:
        pdf = make_pdf_report("2026-10-26T10:00:00+00:00", "sample.mp4", self._summary(), self.metrics)
        self.assertTrue(pdf.startswith(b"%PDF-"))


if __name__ == "__main__":
    unittest.main()
