"""Base class and shared machinery for behavior rules.

Every rule:
    * declares the camera geometry it requires and REFUSES TO CONSTRUCT when
      that geometry is absent (see :meth:`Rule.required_geometry`),
    * converts seconds-based thresholds to frames using the measured FPS,
    * enforces a minimum number of consecutive qualifying frames,
    * applies per-track debounce so one incident does not spam alerts.

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from model2_behavior.src.config import RuleConfig
from model2_behavior.src.geometry import CameraGeometry, GeometryError
from model2_behavior.src.logging_setup import get_logger
from model2_behavior.src.tracks import TrackState


class RuleGeometryError(GeometryError):
    """Raised when a rule cannot be constructed without required geometry."""


@dataclass
class FrameContext:
    """Everything a rule needs about the current frame.

    Attributes:
        frame_index: Current frame number.
        time_s: Seconds since stream start.
        fps: Measured frames per second.
        geometry: Camera geometry, already rescaled to the frame size.
        geometry_describe: Serialisable geometry snapshot for ``rule_params``.
    """

    frame_index: int
    time_s: float
    fps: float
    geometry: CameraGeometry
    geometry_describe: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleHit:
    """A rule's decision to raise an incident for the current frame.

    Attributes:
        track_ids: Involved track ids, sorted ascending.
        zone_id: Matched ROI zone, or None.
        confidence: DETECTION confidence (not incident probability).
        rule_params: Actual threshold values used for this decision.
        reason: Short human-readable explanation for logs.
    """

    track_ids: List[int]
    zone_id: Optional[str]
    confidence: float
    rule_params: Dict[str, Any]
    reason: str = ""


class Rule(ABC):
    """Abstract base for all behavior rules."""

    #: Rule name; must match the key in ``model2.yaml``.
    name: str = "base"
    #: Geometry keys this rule cannot operate without.
    required_geometry: tuple[str, ...] = ()

    def __init__(self, config: RuleConfig, fps: float) -> None:
        """Initialise the rule.

        Args:
            config: Resolved rule configuration.
            fps: Measured frames per second, used to convert seconds to frames.

        Raises:
            ValueError: If ``fps`` is not positive.
        """
        if fps <= 0:
            raise ValueError(f"{self.name}: fps must be positive, got {fps}")
        self.config = config
        self.fps = fps
        self.logger = get_logger(f"{__name__}.{self.name}")
        # Frame thresholds derived from the seconds-based config (never hardcoded).
        self.min_consecutive_frames = max(1, int(round(config.min_consecutive_s * fps)))
        self.debounce_frames = max(0, int(round(config.debounce_s * fps)))
        # track_id -> frames remaining before the rule may fire again
        self._debounce_until: Dict[int, int] = {}

    def seconds_to_frames(self, seconds: float) -> int:
        """Convert a duration in seconds to a frame count at the measured FPS.

        Args:
            seconds: Duration in seconds.

        Returns:
            Frame count, at least 1.
        """
        return max(1, int(round(seconds * self.fps)))

    def check_geometry(self, geometry: CameraGeometry) -> None:
        """Validate that required geometry is present.

        Args:
            geometry: Camera geometry for the stream.

        Raises:
            RuleGeometryError: If a required geometry element is missing.
        """
        for requirement in self.required_geometry:
            if requirement == "driving_lane" and not geometry.has_lanes():
                raise RuleGeometryError(
                    f"Rule '{self.name}' requires at least one lane under rois.lanes but none is "
                    "configured. Author lanes with scripts/define_camera_geometry.py."
                )
            if requirement == "lane_direction" and not geometry.lanes_with_direction():
                raise RuleGeometryError(
                    f"Rule '{self.name}' requires a lane direction, but no lane under rois.lanes "
                    "has a 'direction' set. Author it with scripts/define_camera_geometry.py."
                )

    def is_debounced(self, track_key: int, frame_index: int) -> bool:
        """Return True if the rule is currently suppressed for ``track_key``.

        Args:
            track_key: Track id (or a synthetic pair key for multi-track rules).
            frame_index: Current frame.

        Returns:
            True when suppressed.
        """
        until = self._debounce_until.get(track_key)
        return until is not None and frame_index < until

    def arm_debounce(self, track_key: int, frame_index: int) -> None:
        """Suppress the rule for ``track_key`` for the debounce duration.

        Args:
            track_key: Track id (or synthetic pair key).
            frame_index: Frame at which the incident fired.
        """
        if self.debounce_frames > 0:
            self._debounce_until[track_key] = frame_index + self.debounce_frames

    def base_rule_params(self) -> Dict[str, Any]:
        """Return the always-present rule parameters for ``rule_params``.

        Returns:
            Dict including thresholds actually used and frame conversions.
        """
        return {
            "min_consecutive_s": self.config.min_consecutive_s,
            "min_consecutive_frames": self.min_consecutive_frames,
            "debounce_s": self.config.debounce_s,
            "debounce_frames": self.debounce_frames,
            "fps": round(self.fps, 3),
        }

    @abstractmethod
    def update(self, context: FrameContext, tracks: List[TrackState]) -> Optional[RuleHit]:
        """Evaluate the rule for one frame.

        Args:
            context: Frame context, including geometry and timing.
            tracks: Active tracks for this frame.

        Returns:
            A :class:`RuleHit` when the rule fires, otherwise None.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Clear per-track debounce state (e.g. between videos)."""
        self._debounce_until.clear()
