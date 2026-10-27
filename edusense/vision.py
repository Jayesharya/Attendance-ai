"""Local video sampling and OpenCV face-presence detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Union

import cv2
import numpy as np

from .tracking import Detection, FaceTracker


@dataclass(frozen=True)
class VideoResult:
    """Aggregate outputs from a single local video or camera run."""

    timeline: list[dict[str, object]]
    attendance: list[dict[str, object]]
    preview_jpeg: bytes | None
    duration_seconds: float
    source_label: str


class OpenCVFaceDetector:
    """OpenCV's built-in frontal-face cascade; no identity recognition is performed."""

    def __init__(self) -> None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(str(cascade_path))
        if self._cascade.empty():
            raise RuntimeError("OpenCV could not load its built-in face detector.")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return face boxes only; callers discard the source frame after each sample."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.12,
            minNeighbors=5,
            minSize=(48, 48),
        )
        return [Detection(int(x), int(y), int(w), int(h)) for x, y, w, h in boxes]


def analyze_video(
    source: Union[int, str],
    max_seconds: float = 120.0,
    sample_interval_seconds: float = 1.0,
    progress: Callable[[float, str], None] | None = None,
) -> VideoResult:
    """Sample a local webcam or video file and retain only summary metrics.

    Passing an integer opens a local camera attached to the machine running the
    app. A string is interpreted as a local video path. Frames are discarded
    after each sample and the optional JPEG is only an in-memory preview.
    """
    if max_seconds <= 0 or sample_interval_seconds <= 0:
        raise ValueError("Video duration and sample interval must be positive.")
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("Could not open the selected camera or video file.")

    try:
        detector = OpenCVFaceDetector()
    except Exception:
        capture.release()
        raise
    tracker = FaceTracker()
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0
    is_camera = isinstance(source, int)
    rows: list[dict[str, object]] = []
    preview_jpeg: bytes | None = None
    frame_index = 0
    next_sample = 0.0
    clock_start = cv2.getTickCount() / cv2.getTickFrequency()
    duration = 0.0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            if is_camera:
                elapsed = cv2.getTickCount() / cv2.getTickFrequency() - clock_start
            else:
                raw_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                elapsed = raw_ms / 1000.0 if raw_ms > 0 else frame_index / fps
            duration = max(duration, elapsed)
            if elapsed > max_seconds:
                break
            if elapsed + 1e-6 < next_sample:
                continue

            detections = detector.detect(frame)
            tracks = tracker.update(detections, elapsed)
            active_tracks = [row for row in tracks if row["active"]]
            movement = tracker.movement_values()
            motion_score = min(100.0, (sum(movement) / max(1, len(movement))) * 2.5)
            rows.append(
                {
                    "elapsed_seconds": round(elapsed, 2),
                    "detected_faces": len(detections),
                    "active_profiles": len(active_tracks),
                    "present_profiles": sum(row["status"] == "Present" for row in active_tracks),
                    "motion_score": round(motion_score, 1),
                }
            )
            annotated = frame.copy()
            for detection in detections:
                cv2.rectangle(
                    annotated,
                    (detection.x, detection.y),
                    (detection.x + detection.width, detection.y + detection.height),
                    (80, 92, 232),
                    2,
                )
            encoded, buffer = cv2.imencode(".jpg", annotated)
            if encoded:
                preview_jpeg = buffer.tobytes()
            next_sample = elapsed + sample_interval_seconds
            if progress is not None:
                progress(min(1.0, elapsed / max_seconds), f"Sampled {elapsed:.0f}s · {len(detections)} faces detected")

    finally:
        capture.release()

    if not rows:
        raise RuntimeError("The selected video source did not provide any readable frames.")
    duration = max(duration, float(rows[-1]["elapsed_seconds"]))
    if progress is not None:
        progress(1.0, "Analysis complete")
    return VideoResult(
        timeline=rows,
        attendance=tracker.snapshot(),
        preview_jpeg=preview_jpeg,
        duration_seconds=duration,
        source_label="Local webcam" if is_camera else "Uploaded video",
    )
