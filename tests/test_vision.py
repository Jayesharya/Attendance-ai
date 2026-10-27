from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from edusense.vision import OpenCVFaceDetector, analyze_video


class VisionTests(unittest.TestCase):
    def test_face_detector_handles_empty_frame_without_identity_output(self) -> None:
        detector = OpenCVFaceDetector()
        detections = detector.detect(np.zeros((240, 320, 3), dtype=np.uint8))
        self.assertEqual(detections, [])

    def test_local_video_pipeline_returns_aggregate_rows_and_generic_source_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            video_path = Path(directory) / "sample.avi"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                8.0,
                (320, 240),
            )
            if not writer.isOpened():
                self.skipTest("The installed OpenCV build cannot write the MJPG test clip.")
            for _ in range(8):
                writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
            writer.release()

            result = analyze_video(str(video_path), max_seconds=2.0, sample_interval_seconds=0.25)

        self.assertGreaterEqual(len(result.timeline), 2)
        self.assertTrue(all(row["detected_faces"] == 0 for row in result.timeline))
        self.assertEqual(result.attendance, [])
        self.assertEqual(result.source_label, "Uploaded video")
        self.assertIsNotNone(result.preview_jpeg)


if __name__ == "__main__":
    unittest.main()
