"""Static-camera assumption check: detect global frame motion.

Why this exists
---------------
Every geometric and kinematic rule in this module assumes the camera is
**static**:

    * lane polygons are authored once, in reference-frame pixels, and never move;
    * ``wrong_way`` compares a vehicle's displacement against a fixed lane
      direction;
    * ``stopped_vehicle`` and ``sudden_decel`` measure motion *relative to the
      image*, which is only motion relative to the ground while the camera is
      fixed.

On a moving camera (a vehicle-mounted or panning unit, or a camera knocked by
wind) every one of those assumptions breaks, yet nothing in the pipeline would
notice: the output would still look well-formed, just wrong. This module makes
that failure mode visible. It estimates how much the scene moved between the
current frame and the **first frame of the run**, and reports a warning plus a
``camera_motion_suspected`` flag in ``run_summary.json`` when the shift exceeds a
configured fraction of the frame.

Method
------
Sparse optical flow (``cv2.goodFeaturesToTrack`` + ``cv2.calcOpticalFlowPyrLK``)
between a downscaled version of the current frame and a downscaled version of the
reference frame. The **median** displacement of the tracked corners is taken as
the global shift: a median is robust to the independent motion of vehicles, which
is exactly the signal that must NOT be mistaken for camera motion. A median near
zero means the scene is still; a median that grows means the world is moving in
the image while (by assumption) the vehicles are not the only thing that changed.

This is a cheap, self-contained sanity check on the static-camera precondition.
It is deliberately **not** a calibration or stabilisation routine: it flags a
suspicion for a human to review, and it never blocks a run or alters an event.

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2

from model2_behavior.src.logging_setup import get_logger

logger = get_logger(__name__)

#: Frame is downscaled so the check costs a few milliseconds rather than a few
#: tens of milliseconds. Feature tracking does not need full resolution: a global
#: shift of a handful of pixels is already far above the suspicion threshold.
DEFAULT_WORK_WIDTH = 480

#: Maximum corners to track. A few hundred is plenty for a global median and
#: keeps the check bounded on a large frame.
DEFAULT_MAX_CORNERS = 200

#: Minimum corner separation, in downscaled pixels, so the corners are spread
#: over the scene rather than clustered on one high-contrast object.
DEFAULT_MIN_DISTANCE = 8

#: Robustness guard: a frame is only judged when at least this many corners
#: survived tracking, otherwise the evidence is too thin to make a claim.
DEFAULT_MIN_TRACKED_CORNERS = 12


@dataclass
class MotionCheckConfig:
    """Tunable parameters for the static-camera check.

    Attributes:
        enabled: Whether the check runs at all. A disabled check never changes a
            run's summary, so it is reported as ``checked: false``.
        work_width: Width, in pixels, to downscale frames to before tracking.
            Keeps the cost independent of the source resolution.
        max_corners: Maximum corners to track per frame.
        min_distance: Minimum separation between corners, in downscaled pixels.
        threshold: Median global shift, as a fraction of the frame WIDTH, at or
            above which camera motion is suspected.
        min_tracked_corners: Minimum surviving corners required to make a
            judgement. Below this the frame is skipped, not guessed.
        check_every_n_frames: Run the estimate every N frames. A camera does not
            start moving for a single frame and stop, so sampling keeps the cost
            negligible.
    """

    enabled: bool = True
    work_width: int = DEFAULT_WORK_WIDTH
    max_corners: int = DEFAULT_MAX_CORNERS
    min_distance: int = DEFAULT_MIN_DISTANCE
    threshold: float = 0.01
    min_tracked_corners: int = DEFAULT_MIN_TRACKED_CORNERS
    check_every_n_frames: int = 15

    @classmethod
    def from_mapping(cls, raw: Optional[Dict[str, Any]]) -> "MotionCheckConfig":
        """Build a config from a YAML mapping, rejecting unknown keys.

        Unknown keys are rejected rather than ignored, so a typo in the config
        cannot silently leave a check at its default.

        Args:
            raw: The ``static_camera_check`` mapping, or None for defaults.

        Returns:
            A populated config.

        Raises:
            ValueError: If an unknown key is present or a value is out of range.
        """
        if raw is None:
            return cls()
        known = set(cls().to_dict())
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(
                f"unknown static_camera_check key(s) {unknown}; known keys are {sorted(known)}"
            )
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        """Check the parameters are usable.

        Raises:
            ValueError: If a parameter would make the check meaningless.
        """
        if self.work_width < 32:
            raise ValueError(f"static_camera_check.work_width must be >= 32, got {self.work_width}")
        if self.max_corners < self.min_tracked_corners:
            raise ValueError(
                "static_camera_check.max_corners must be >= min_tracked_corners "
                f"({self.max_corners} < {self.min_tracked_corners})"
            )
        if self.threshold <= 0.0:
            raise ValueError(f"static_camera_check.threshold must be > 0, got {self.threshold}")
        if self.check_every_n_frames < 1:
            raise ValueError(
                "static_camera_check.check_every_n_frames must be >= 1, got "
                f"{self.check_every_n_frames}"
            )
        if self.min_distance < 1:
            raise ValueError(
                f"static_camera_check.min_distance must be >= 1, got {self.min_distance}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable form."""
        return {
            "enabled": self.enabled,
            "work_width": self.work_width,
            "max_corners": self.max_corners,
            "min_distance": self.min_distance,
            "threshold": self.threshold,
            "min_tracked_corners": self.min_tracked_corners,
            "check_every_n_frames": self.check_every_n_frames,
        }


@dataclass
class MotionSample:
    """One estimate of global frame motion.

    Attributes:
        frame_index: Frame the estimate was taken on.
        median_shift_px: Median corner displacement, in downscaled pixels.
        median_shift_fraction: ``median_shift_px`` divided by the downscaled
            frame width, so it is resolution-independent.
        tracked_corners: Corners that survived tracking.
        suspected: Whether the shift reached the suspicion threshold.
    """

    frame_index: int
    median_shift_px: float
    median_shift_fraction: float
    tracked_corners: int
    suspected: bool


@dataclass
class MotionCheckResult:
    """Summary of the static-camera check for a whole run.

    Attributes:
        checked: Whether the check ran at all.
        camera_motion_suspected: True when any sample reached the threshold.
        threshold: Threshold used, recorded so the flag is auditable.
        max_shift_fraction: Largest global shift fraction observed.
        median_shift_fraction: Median of all observed shift fractions.
        samples_taken: Number of estimates performed.
        skipped_frames: Estimates skipped for too few trackable corners.
        first_suspected_frame: Frame of the first suspicious sample, if any.
        config: The parameters actually used.
    """

    checked: bool = False
    camera_motion_suspected: bool = False
    threshold: float = 0.0
    max_shift_fraction: float = 0.0
    median_shift_fraction: float = 0.0
    samples_taken: int = 0
    skipped_frames: int = 0
    first_suspected_frame: Optional[int] = None
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable form."""
        return {
            "checked": self.checked,
            "camera_motion_suspected": self.camera_motion_suspected,
            "threshold_fraction_of_width": self.threshold,
            "max_shift_fraction": round(self.max_shift_fraction, 6),
            "median_shift_fraction": round(self.median_shift_fraction, 6),
            "samples_taken": self.samples_taken,
            "skipped_frames": self.skipped_frames,
            "first_suspected_frame": self.first_suspected_frame,
            "config": self.config,
            "note": (
                "Median optical-flow shift of the scene against the run's first frame. "
                "A high value means the STATIC-CAMERA assumption may be violated, which "
                "invalidates fixed lane geometry and image-relative speed rules."
            ),
        }


class CameraMotionChecker:
    """Tracks the scene's global shift against the run's first frame.

    The first frame seen becomes the reference. Each estimate downscales both
    frames, finds trackable corners in the reference, tracks them forward into the
    current frame, and takes the median displacement of those that survive.

    Args:
        config: Check parameters. Defaults to :class:`MotionCheckConfig`.
    """

    def __init__(self, config: Optional[MotionCheckConfig] = None) -> None:
        self.config = config or MotionCheckConfig()
        self._reference_gray: Optional[Any] = None
        self._work_size: Optional[Tuple[int, int]] = None
        self._last_tracked = 0
        self._samples: List[float] = []
        self._result = MotionCheckResult(
            threshold=self.config.threshold, config=self.config.to_dict()
        )

    @property
    def result(self) -> MotionCheckResult:
        """The running summary, updated as frames are offered."""
        return self._result

    def _prepare(self, frame: Any) -> Any:
        """Downscale a frame and convert it to greyscale.

        Args:
            frame: BGR frame.

        Returns:
            A greyscale, downscaled frame.
        """
        height, width = frame.shape[:2]
        if width <= 0 or height <= 0:
            return frame
        scale = self.config.work_width / float(width)
        target = (self.config.work_width, max(1, int(round(height * scale))))
        self._work_size = target
        resized = cv2.resize(frame, target, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    def observe(self, frame: Any, frame_index: int) -> Optional[MotionSample]:
        """Offer a frame to the check.

        The first frame becomes the reference and no estimate is returned for it,
        because there is nothing to compare against yet.

        Args:
            frame: BGR frame as read from the source.
            frame_index: Index of the frame in the stream.

        Returns:
            The sample for this frame, or None when no estimate was taken
            (reference frame, disabled check, off-schedule frame, or too few
            trackable corners).
        """
        if not self.config.enabled:
            return None

        try:
            gray = self._prepare(frame)
        except cv2.error:  # pragma: no cover - malformed frame
            logger.exception("Camera-motion check could not prepare frame %s", frame_index)
            return None

        if self._reference_gray is None:
            self._reference_gray = gray
            self._result.checked = True
            return None

        if frame_index % self.config.check_every_n_frames != 0:
            return None

        shift_px = self._estimate_shift(self._reference_gray, gray)
        if shift_px is None:
            self._result.skipped_frames += 1
            return None

        width = float(self._work_size[0]) if self._work_size else float(self.config.work_width)
        fraction = shift_px / width if width > 0 else 0.0
        sample = MotionSample(
            frame_index=frame_index,
            median_shift_px=shift_px,
            median_shift_fraction=fraction,
            tracked_corners=self._last_tracked,
            suspected=fraction >= self.config.threshold,
        )
        self._absorb(sample)
        return sample

    def _estimate_shift(self, reference_gray: Any, current_gray: Any) -> Optional[float]:
        """Estimate the median global shift between two greyscale frames.

        Args:
            reference_gray: Downscaled greyscale reference frame.
            current_gray: Downscaled greyscale current frame.

        Returns:
            Median displacement magnitude, or None when too few corners survive.
        """
        corners = cv2.goodFeaturesToTrack(
            reference_gray,
            maxCorners=self.config.max_corners,
            qualityLevel=0.01,
            minDistance=self.config.min_distance,
            blockSize=7,
        )
        if corners is None or len(corners) == 0:
            self._last_tracked = 0
            return None

        tracked, status, _ = cv2.calcOpticalFlowPyrLK(
            reference_gray, current_gray, corners, None
        )
        if tracked is None or status is None:
            self._last_tracked = 0
            return None

        displacements: List[float] = []
        for index, (old_point, new_point) in enumerate(zip(corners, tracked)):
            if status[index][0] != 1:
                continue
            dx = float(new_point[0][0] - old_point[0][0])
            dy = float(new_point[0][1] - old_point[0][1])
            displacements.append((dx * dx + dy * dy) ** 0.5)

        self._last_tracked = len(displacements)
        if len(displacements) < self.config.min_tracked_corners:
            return None

        displacements.sort()
        return displacements[len(displacements) // 2]

    def _absorb(self, sample: MotionSample) -> None:
        """Fold a sample into the running summary.

        Args:
            sample: The sample to record.
        """
        self._samples.append(sample.median_shift_fraction)
        self._result.samples_taken += 1
        self._result.max_shift_fraction = max(
            self._result.max_shift_fraction, sample.median_shift_fraction
        )
        if sample.suspected and not self._result.camera_motion_suspected:
            self._result.camera_motion_suspected = True
            self._result.first_suspected_frame = sample.frame_index
            logger.warning(
                "camera_motion_suspected: global scene shift %.4f of frame width at frame %s "
                "(threshold %.4f). The STATIC-CAMERA assumption may be violated; fixed lane "
                "geometry and image-relative speed rules are unreliable for this footage.",
                sample.median_shift_fraction,
                sample.frame_index,
                self.config.threshold,
            )

    def finalize(self) -> MotionCheckResult:
        """Compute the median shift and return the final summary.

        Returns:
            The populated :class:`MotionCheckResult`.
        """
        if self._samples:
            ordered = sorted(self._samples)
            self._result.median_shift_fraction = ordered[len(ordered) // 2]
        return self._result


def build_motion_checker(raw: Optional[Dict[str, Any]]) -> CameraMotionChecker:
    """Build a checker from a config mapping.

    Args:
        raw: The ``static_camera_check`` mapping, or None for defaults.

    Returns:
        A ready :class:`CameraMotionChecker`.
    """
    return CameraMotionChecker(MotionCheckConfig.from_mapping(raw))
