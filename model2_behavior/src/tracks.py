"""Tracking primitives: per-frame observations and rolling track state.

The rule engine needs frame-rate-aware motion history, so each track keeps a
time-stamped deque of observations. Speed is normalized by bounding-box height
and expressed in BODY-LENGTHS PER SECOND (bl/s), which is scale-invariant.

LIMITATION: without homography / camera calibration there is no true metric
speed. Body-lengths/s is a relative proxy only; do not infer km/h from it.

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

Point = Tuple[float, float]
Box = Tuple[float, float, float, float]


@dataclass
class Observation:
    """A single detection of one track at one frame.

    Attributes:
        track_id: Tracker-assigned id for this observation.
        frame_index: Frame number.
        time_s: Timestamp in seconds from stream start.
        box: ``(x1, y1, x2, y2)`` pixel box.
        confidence: Detector confidence for this detection.
        class_id: COCO class id.
        class_name: Class name resolved from the model.
    """

    track_id: int
    frame_index: int
    time_s: float
    box: Box
    confidence: float
    class_id: int
    class_name: str

    @property
    def height(self) -> float:
        """Box height in pixels, clamped to a minimum of 1e-6."""
        return max(self.box[3] - self.box[1], 1e-6)

    @property
    def width(self) -> float:
        """Box width in pixels."""
        return max(self.box[2] - self.box[0], 1e-6)

    @property
    def center(self) -> Point:
        """Box center as ``(x, y)``."""
        return ((self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0)

    @property
    def bottom_center(self) -> Point:
        """Bottom-center point, the usual ground-contact proxy for a vehicle."""
        return ((self.box[0] + self.box[2]) / 2.0, self.box[3])


@dataclass
class TrackState:
    """Rolling state for a single track id.

    Attributes:
        track_id: Tracker-assigned id.
        observations: Time-ordered observations, capped at ``max_history``.
        first_seen_s: Time of the first observation.
        last_seen_s: Time of the most recent observation.
        missed_frames: Consecutive frames since the last observation.
    """

    track_id: int
    observations: Deque[Observation] = field(default_factory=lambda: deque(maxlen=180))
    first_seen_s: float = 0.0
    last_seen_s: float = 0.0
    missed_frames: int = 0

    def update(self, observation: Observation) -> None:
        """Append an observation and refresh timing fields.

        Args:
            observation: Latest observation for this track.
        """
        if not self.observations:
            self.first_seen_s = observation.time_s
        self.observations.append(observation)
        self.last_seen_s = observation.time_s
        self.missed_frames = 0

    def mark_missed(self) -> None:
        """Increment the missed-frame counter."""
        self.missed_frames += 1

    @property
    def latest(self) -> Optional[Observation]:
        """Most recent observation, or None."""
        return self.observations[-1] if self.observations else None

    @property
    def age_s(self) -> float:
        """Track lifetime in seconds."""
        return self.last_seen_s - self.first_seen_s

    @property
    def mean_confidence(self) -> float:
        """Mean detector confidence across retained observations."""
        if not self.observations:
            return 0.0
        return sum(obs.confidence for obs in self.observations) / len(self.observations)

    @property
    def class_id(self) -> Optional[int]:
        """Class id of the most recent observation."""
        latest = self.latest
        return latest.class_id if latest else None

    @property
    def class_name(self) -> Optional[str]:
        """Class name of the most recent observation."""
        latest = self.latest
        return latest.class_name if latest else None

    def history_spanning(self, window_s: float) -> List[Observation]:
        """Return observations within the last ``window_s`` seconds.

        Args:
            window_s: Look-back window in seconds.

        Returns:
            Time-ordered observations newer than ``last_seen_s - window_s``.
        """
        if not self.observations:
            return []
        cutoff = self.last_seen_s - window_s
        return [obs for obs in self.observations if obs.time_s >= cutoff]

    def speed_bl_s(self, window_s: float = 0.5) -> Optional[float]:
        """Return mean normalized speed over ``window_s`` in body-lengths/second.

        The displacement is measured between the oldest and newest observations
        inside the window and divided by the elapsed time and the mean box
        height, giving a resolution-invariant speed.

        Args:
            window_s: Look-back window in seconds.

        Returns:
            Speed in bl/s, or None when fewer than two observations fall in the
            window or no time elapsed.
        """
        window = self.history_spanning(window_s)
        if len(window) < 2:
            return None
        first, last = window[0], window[-1]
        elapsed = last.time_s - first.time_s
        if elapsed <= 1e-9:
            return None
        (x1, y1) = first.center
        (x2, y2) = last.center
        distance_px = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        mean_height = sum(obs.height for obs in window) / len(window)
        return (distance_px / elapsed) / mean_height

    def raw_speed_px_s(self, window_s: float = 0.5) -> Optional[float]:
        """Return unnormalized speed in pixels/second for diagnostic logging.

        Args:
            window_s: Look-back window in seconds.

        Returns:
            Speed in px/s, or None when not computable.
        """
        window = self.history_spanning(window_s)
        if len(window) < 2:
            return None
        first, last = window[0], window[-1]
        elapsed = last.time_s - first.time_s
        if elapsed <= 1e-9:
            return None
        (x1, y1) = first.center
        (x2, y2) = last.center
        return (((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5) / elapsed

    def displacement_vector(self, window_s: float = 0.5) -> Optional[Point]:
        """Return the net ``(dx, dy)`` center displacement over ``window_s``.

        Args:
            window_s: Look-back window in seconds.

        Returns:
            ``(dx, dy)`` in pixels (normalized by mean height is left to the
            caller), or None when not computable.
        """
        window = self.history_spanning(window_s)
        if len(window) < 2:
            return None
        (x1, y1) = window[0].center
        (x2, y2) = window[-1].center
        return (x2 - x1, y2 - y1)

    def normalized_displacement(self, window_s: float = 0.5) -> Optional[Point]:
        """Return displacement normalized by mean box height (body-lengths).

        Args:
            window_s: Look-back window in seconds.

        Returns:
            ``(dx_bl, dy_bl)`` or None.
        """
        window = self.history_spanning(window_s)
        if len(window) < 2:
            return None
        raw = self.displacement_vector(window_s)
        if raw is None:
            return None
        mean_height = sum(obs.height for obs in window) / len(window)
        return (raw[0] / mean_height, raw[1] / mean_height)

    def box_at_latest(self) -> Optional[Box]:
        """Return the most recent box, or None."""
        latest = self.latest
        return latest.box if latest else None


def boxes_iou(box_a: Box, box_b: Box) -> float:
    """Compute intersection-over-union for two boxes.

    Args:
        box_a: ``(x1, y1, x2, y2)``.
        box_b: ``(x1, y1, x2, y2)``.

    Returns:
        IoU in ``[0, 1]``.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 1e-9:
        return 0.0
    return inter_area / union


class TrackStore:
    """Holds :class:`TrackState` objects across frames."""

    def __init__(self, max_history: int = 180) -> None:
        """Initialise the store.

        Args:
            max_history: Observations retained per track.
        """
        self.max_history = max_history
        self._tracks: Dict[int, TrackState] = {}

    def update(self, observations: List[Observation]) -> Dict[int, TrackState]:
        """Ingest one frame of observations.

        Args:
            observations: Observations for the current frame.

        Returns:
            Mapping of track id to updated :class:`TrackState`.
        """
        seen: set[int] = set()
        for observation in observations:
            track_id = int(observation.track_id)
            seen.add(track_id)
            state = self._tracks.get(track_id)
            if state is None:
                state = TrackState(track_id=track_id, observations=deque(maxlen=self.max_history))
                self._tracks[track_id] = state
            state.update(observation)

        for track_id, state in self._tracks.items():
            if track_id not in seen:
                state.mark_missed()
        return dict(self._tracks)

    def get(self, track_id: int) -> Optional[TrackState]:
        """Return a track by id, or None."""
        return self._tracks.get(track_id)

    def all_tracks(self) -> List[TrackState]:
        """Return all tracks."""
        return list(self._tracks.values())

    def active_tracks(self, max_missed: int = 5) -> List[TrackState]:
        """Return tracks seen recently.

        Args:
            max_missed: Maximum consecutive missed frames to still count active.

        Returns:
            List of active tracks.
        """
        return [track for track in self._tracks.values() if track.missed_frames <= max_missed]

    def prune(self, max_missed: int = 60) -> int:
        """Drop tracks that have been missing for too long.

        Args:
            max_missed: Missed-frame count after which a track is removed.

        Returns:
            Number of tracks removed.
        """
        stale = [track_id for track_id, track in self._tracks.items() if track.missed_frames > max_missed]
        for track_id in stale:
            del self._tracks[track_id]
        return len(stale)
