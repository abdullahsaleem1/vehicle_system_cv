"""Shared pytest fixtures and synthetic-trajectory helpers.

Tests must never depend on real footage or a GPU. Everything here builds
deterministic frames, geometry, and track states so the rule engine and the
event contract can be validated in isolation.

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pytest

# Make the repository root importable when pytest is invoked from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model2_behavior.src.config import RuleConfig  # noqa: E402
from model2_behavior.src.geometry import CameraGeometry, Lane, Polygon  # noqa: E402
from model2_behavior.src.tracks import Observation, TrackState, TrackStore  # noqa: E402

# A 640x480 reference frame, so synthetic pixel values stay readable.
REFERENCE_WIDTH = 640
REFERENCE_HEIGHT = 480

# A large lane covering the lower half: anything below y=240 is "in lane".
LANE_POLYGON: List[List[float]] = [
    [40.0, 240.0],
    [600.0, 240.0],
    [600.0, 470.0],
    [40.0, 470.0],
]

# Left-to-right travel, used by most fixtures.
LANE_DIRECTION: List[float] = [100.0, 400.0, 540.0, 400.0]

# A second lane to the right, used by the multi-lane tests.
SECOND_LANE_POLYGON: List[List[float]] = [
    [610.0, 240.0],
    [639.0, 240.0],
    [639.0, 470.0],
    [610.0, 470.0],
]


def lane_from_points(
    name: str,
    polygon: Optional[List[List[float]]],
    direction: Optional[Sequence[float]] = None,
) -> Lane:
    """Build a :class:`Lane` from raw pixel lists.

    Args:
        name: Lane name.
        polygon: Polygon points, or None when no polygon is needed.
        direction: ``[x1, y1, x2, y2]``, or None for no direction.

    Returns:
        A populated lane.
    """
    if polygon is None:
        raise ValueError("lane_from_points requires a polygon")
    parsed_direction = None
    if direction is not None:
        parsed_direction = (
            (float(direction[0]), float(direction[1])),
            (float(direction[2]), float(direction[3])),
        )
    return Lane(
        name=name,
        polygon=Polygon([(float(x), float(y)) for x, y in polygon]),
        direction=parsed_direction,
    )


def make_geometry(
    reference_resolution: Optional[Tuple[int, int]] = (REFERENCE_WIDTH, REFERENCE_HEIGHT),
    lanes: Optional[List[Lane]] = None,
    driving_lane: Optional[List[List[float]]] = None,
    lane_direction: Optional[Sequence[float]] = None,
) -> CameraGeometry:
    """Build a :class:`CameraGeometry` for tests.

    Args:
        reference_resolution: Resolution the pixels were authored in.
        lanes: Explicit list of :class:`Lane` objects. Takes precedence.
        driving_lane: Convenience shorthand: builds a single lane named
            ``"driving_lane"`` from these points.
        lane_direction: Convenience shorthand: direction for the shorthand lane.

    Returns:
        A populated geometry object.
    """
    if lanes is None:
        lanes = []
        if driving_lane is not None:
            lanes.append(lane_from_points("driving_lane", driving_lane, lane_direction))
    return CameraGeometry(
        reference_resolution=reference_resolution,
        lanes=lanes,
    )


def make_rule_config(
    name: str,
    min_consecutive_s: float = 0.1,
    debounce_s: float = 0.0,
    severity: str = "Medium",
    enabled: bool = True,
    **params,
) -> RuleConfig:
    """Build a :class:`RuleConfig` with explicit, non-defaulted thresholds.

    Args:
        name: Rule name.
        min_consecutive_s: Persistence requirement in seconds.
        debounce_s: Repeat suppression in seconds.
        severity: Severity label.
        enabled: Whether the rule is enabled.
        **params: Rule-specific threshold values.

    Returns:
        A populated rule configuration.
    """
    return RuleConfig(
        name=name,
        enabled=enabled,
        severity=severity,
        debounce_s=debounce_s,
        min_consecutive_s=min_consecutive_s,
        params=dict(params),
    )


def make_observation(
    track_id: int,
    frame_index: int,
    time_s: float,
    box: Tuple[float, float, float, float],
    confidence: float = 0.8,
    class_id: int = 2,
    class_name: str = "car",
) -> Observation:
    """Build a single synthetic observation."""
    return Observation(
        track_id=track_id,
        frame_index=frame_index,
        time_s=time_s,
        box=box,
        confidence=confidence,
        class_id=class_id,
        class_name=class_name,
    )


def box_at(cx: float, cy: float, height: float = 60.0, aspect: float = 1.6) -> Tuple[float, float, float, float]:
    """Return an ``(x1, y1, x2, y2)`` box centred on ``(cx, cy)``.

    Args:
        cx: Centre x in pixels.
        cy: Centre y in pixels.
        height: Box height in pixels; speeds normalize by this.
        aspect: Width/height ratio.

    Returns:
        The box as a 4-tuple.
    """
    width = height * aspect
    return (cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)


def build_track(
    track_id: int,
    centers: Sequence[Tuple[float, float]],
    fps: float = 25.0,
    start_frame: int = 0,
    height: float = 60.0,
    confidence: float = 0.8,
    class_id: int = 2,
    class_name: str = "car",
) -> TrackState:
    """Build a :class:`TrackState` from a list of box centres.

    Each centre becomes one observation, spaced one frame apart at ``fps``.

    Args:
        track_id: Track id.
        centers: Ordered ``(cx, cy)`` positions in pixels.
        fps: Frame rate used to derive timestamps.
        start_frame: Frame index of the first observation.
        height: Box height in pixels.
        confidence: Detector confidence for every observation.
        class_id: COCO class id.
        class_name: Class name.

    Returns:
        A populated track state.
    """
    state = TrackState(track_id=track_id)
    for offset, (cx, cy) in enumerate(centers):
        frame_index = start_frame + offset
        state.update(
            make_observation(
                track_id=track_id,
                frame_index=frame_index,
                time_s=frame_index / fps,
                box=box_at(cx, cy, height=height),
                confidence=confidence,
                class_id=class_id,
                class_name=class_name,
            )
        )
    return state


def trajectory(
    start: Tuple[float, float],
    end: Tuple[float, float],
    count: int,
) -> List[Tuple[float, float]]:
    """Linearly interpolate ``count`` centres from ``start`` to ``end``.

    Args:
        start: First ``(cx, cy)``.
        end: Last ``(cx, cy)``.
        count: Number of points to generate (>= 2).

    Returns:
        List of interpolated centres, inclusive of both endpoints.
    """
    if count < 2:
        raise ValueError("count must be >= 2")
    points: List[Tuple[float, float]] = []
    for index in range(count):
        t = index / (count - 1)
        points.append((start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t))
    return points


@pytest.fixture
def fps() -> float:
    """Standard test frame rate."""
    return 25.0


@pytest.fixture
def lane_geometry() -> CameraGeometry:
    """Geometry with a driving lane but no lane direction."""
    return make_geometry(driving_lane=LANE_POLYGON)


@pytest.fixture
def directional_geometry() -> CameraGeometry:
    """Geometry with a driving lane and a left-to-right lane direction."""
    return make_geometry(driving_lane=LANE_POLYGON, lane_direction=[100.0, 400.0, 540.0, 400.0])


@pytest.fixture
def empty_geometry() -> CameraGeometry:
    """Geometry with neither lane nor direction configured."""
    return make_geometry(reference_resolution=None)
