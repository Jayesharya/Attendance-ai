"""Anonymous, session-scoped tracking for sustained visual presence."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class Detection:
    """A detected face rectangle in image pixels."""

    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2


@dataclass
class _Track:
    track_id: str
    center: tuple[float, float]
    first_seen_seconds: float
    last_seen_seconds: float
    presence_threshold_seconds: float
    observed_seconds: float = 0.0
    missing_seconds: float = 0.0
    movement_pixels: float = 0.0
    samples: int = 1
    active: bool = True

    @property
    def confirmed(self) -> bool:
        return self.observed_seconds >= self.presence_threshold_seconds

    def public_record(self) -> dict[str, object]:
        return {
            "profile_id": self.track_id,
            "status": "Present" if self.confirmed else "Checking",
            "observed_seconds": round(self.observed_seconds, 2),
            "samples": self.samples,
            "first_seen_seconds": round(self.first_seen_seconds, 2),
            "last_seen_seconds": round(self.last_seen_seconds, 2),
            "active": self.active,
        }


class FaceTracker:
    """Greedy centroid tracker; labels expire with the current analysis session."""

    def __init__(
        self,
        presence_seconds: float = 3.0,
        max_missing_seconds: float = 2.5,
        max_match_distance: float = 150.0,
    ) -> None:
        if presence_seconds <= 0 or max_missing_seconds < 0 or max_match_distance <= 0:
            raise ValueError("Tracking thresholds must be positive (missing time may be zero).")
        self.presence_seconds = presence_seconds
        self.max_missing_seconds = max_missing_seconds
        self.max_match_distance = max_match_distance
        self._tracks: dict[str, _Track] = {}
        self._next_id = 1
        self._last_time = -math.inf

    @staticmethod
    def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
        return math.dist(first, second)

    def update(self, detections: Iterable[Detection], elapsed_seconds: float) -> list[dict[str, object]]:
        """Update tracks from one sampled frame and return anonymous track rows."""
        if elapsed_seconds < 0 or elapsed_seconds < self._last_time:
            raise ValueError("elapsed_seconds must be non-negative and monotonic.")
        self._last_time = elapsed_seconds
        current = list(detections)
        live = [track for track in self._tracks.values() if track.active]
        remaining_tracks = {track.track_id for track in live}
        remaining_detections = set(range(len(current)))

        pairs = sorted(
            (self._distance(track.center, detection.center), track.track_id, index)
            for track in live
            for index, detection in enumerate(current)
            if self._distance(track.center, detection.center) <= self.max_match_distance
        )
        for distance, track_id, index in pairs:
            if track_id not in remaining_tracks or index not in remaining_detections:
                continue
            track = self._tracks[track_id]
            gap = max(0.0, elapsed_seconds - track.last_seen_seconds)
            if gap <= self.max_missing_seconds:
                track.observed_seconds += gap
            else:
                track.first_seen_seconds = elapsed_seconds
                track.observed_seconds = 0.0
            track.center = current[index].center
            track.last_seen_seconds = elapsed_seconds
            track.missing_seconds = 0.0
            track.movement_pixels = distance
            track.samples += 1
            remaining_tracks.remove(track_id)
            remaining_detections.remove(index)

        for index in sorted(remaining_detections):
            track_id = f"P{self._next_id:03d}"
            self._next_id += 1
            detection = current[index]
            self._tracks[track_id] = _Track(
                track_id=track_id,
                center=detection.center,
                first_seen_seconds=elapsed_seconds,
                last_seen_seconds=elapsed_seconds,
                presence_threshold_seconds=self.presence_seconds,
            )

        for track_id in remaining_tracks:
            track = self._tracks[track_id]
            track.missing_seconds = elapsed_seconds - track.last_seen_seconds
            track.movement_pixels = 0.0
            if track.missing_seconds > self.max_missing_seconds:
                track.active = False

        return self.snapshot()

    def snapshot(self) -> list[dict[str, object]]:
        """Return current track state without exposing images or embeddings."""
        return [track.public_record() for track in self._tracks.values()]

    def active_count(self) -> int:
        return sum(track.active for track in self._tracks.values())

    def present_count(self) -> int:
        return sum(track.active and track.confirmed for track in self._tracks.values())

    def movement_values(self) -> list[float]:
        return [track.movement_pixels for track in self._tracks.values() if track.active]
