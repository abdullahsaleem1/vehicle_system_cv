"""Frame timing: derive timestamps from the video clock, not frame indices.

Why this exists
---------------
Dividing ``frame_index`` by a nominal FPS assumes a perfectly regular stream. Real
sources are not: webcams deliver variable intervals, RTSP streams stall and
resume, files contain variable frame rate (VFR) segments, and frames get dropped
whenever processing cannot keep up. Deriving durations from ``frame_index / fps``
silently corrupts every time-based threshold in the rule engine.

This module produces a monotonic timeline in seconds from an actual clock:

    * For FILE sources, ``cv2.CAP_PROP_POS_MSEC`` exposes the container
      presentation time (PTS) of the frame just decoded. This is the
      authoritative video clock and already accounts for VFR and drops.
    * A FILE source NEVER falls back to wall-clock time. Wall-clock time on a
      file measures how fast the CPU happens to decode the file, not what is
      happening in the video, so using it would make every time-based threshold
      meaningless. Instead a file degrades to the *container* clock:
      ``frame_index / container_fps``. That is still a property of the file, so
      the timeline stays video-relative; it is merely a less accurate clock, and
      the clock source is reported as ``frame_count`` so this is never silent.
    * For LIVE sources (RTSP, webcam), there is no meaningful PTS, so wall-clock
      time is used instead. This is the ONLY case where wall-clock time is used,
      and only because a live stream has no file timeline to fall back to.

Degradation to the container clock happens on the FIRST unusable PTS reading,
not after a tolerance window, because a partially-PTS timeline is worse than a
consistently frame-counted one: mixing the two produces gaps that look like
dropped frames. The event is logged as a warning.

``FrameClock`` also detects discontinuities: a gap far larger than the expected
frame interval is recorded as dropped frames, so a rule never mistakes a
processing stall for a real increase in elapsed time without it being visible in
the logs.

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from model2_behavior.src.logging_setup import get_logger

logger = get_logger(__name__)


class ClockSource(str, Enum):
    """Where timestamps come from for this run."""

    #: Container presentation timestamps (files).
    PTS = "pts"
    #: Wall-clock time. LIVE sources only; never used for a file.
    WALL_CLOCK = "wall_clock"
    #: Frame counting against the container's reported FPS. The fallback for a
    #: file whose backend does not expose a usable PTS.
    FRAME_COUNT = "frame_count"


@dataclass
class FrameClock:
    """Produces a monotonic seconds timeline and reports timeline discontinuities.

        Attributes:
            source: Which clock this instance is using.
            fps: Container frames per second, used for the expected interval and
                the frame-count fallback.
            default_fps: FPS assumed when the container reports none. Only used
                by the frame-count fallback, and recorded in :meth:`summary`.
            max_gap_factor: A gap larger than ``max_gap_factor`` times the expected
                frame interval is reported as dropped frames.
            degrade_after_pts_failures: Retained for compatibility with callers
                that set an explicit tolerance. A value of 1 means "degrade on
                the first unusable PTS reading", which is the default, because
                interpolating through a broken PTS stream silently invents
                timestamps.

        Note:
            A file source can never become :attr:`ClockSource.WALL_CLOCK`. See
            the module docstring for why.
        """

    source: ClockSource
    fps: float
    default_fps: float = 25.0
    max_gap_factor: float = 3.0
    degrade_after_pts_failures: int = 1
    #: Kept as an alias so existing construction sites that pass
    #: ``max_pts_failures=`` keep working. Values above 1 are clamped to 1 with a
    #: warning, since interpolation through a broken PTS stream is unsafe.
    max_pts_failures: Optional[int] = None

    _frames_seen: int = field(default=0, init=False)
    _last_time_s: Optional[float] = field(default=None, init=False)
    _wall_start: Optional[float] = field(default=None, init=False)
    _dropped_total: int = field(default=0, init=False)
    _pts_failures: int = field(default=0, init=False)
    gaps: List[dict] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Reconcile the legacy ``max_pts_failures`` argument.

        Raises:
            ValueError: If a clock is constructed with no usable FPS.
        """
        if self.max_pts_failures is not None:
            if self.max_pts_failures > 1:
                logger.warning(
                    "max_pts_failures=%s requested, but interpolating through a broken PTS "
                    "stream invents timestamps; degrading on the first failure instead.",
                    self.max_pts_failures,
                )
            self.degrade_after_pts_failures = 1
        if self.source is not ClockSource.WALL_CLOCK and self.fps <= 0:
            # The frame-count fallback divides by fps, so a zero would be fatal.
            logger.warning(
                "FrameClock constructed with fps=%s; using default_fps=%s for the "
                "frame-count fallback.",
                self.fps,
                self.default_fps,
            )
            self.fps = self.default_fps

    @property
    def interval_fps(self) -> float:
        """FPS used for the expected interval and the frame-count fallback.

        Returns:
            ``fps`` when positive, otherwise ``default_fps`` (always positive).
        """
        return self.fps if self.fps > 0 else self.default_fps

    @property
    def expected_interval_s(self) -> float:
        """Nominal interval between frames in seconds."""
        return 1.0 / self.interval_fps

    @property
    def dropped_total(self) -> int:
        """Estimated number of dropped/missing frames so far."""
        return self._dropped_total

    def read_pts_seconds(self, capture) -> Optional[float]:
        """Read the container presentation timestamp for the just-decoded frame.

        A PTS of exactly 0.0 is VALID and common: the first frame of a file
        usually carries zero presentation time. It must not be treated as a
        failure, or every file source would immediately degrade to wall-clock
        timing. Only negative values and NaN are rejected.

        Args:
            capture: An opened ``cv2.VideoCapture``.

        Returns:
            Presentation time in seconds, or None when the backend reports an
            unusable value (negative or NaN).
        """
        import cv2

        try:
            milliseconds = float(capture.get(cv2.CAP_PROP_POS_MSEC))
        except Exception:  # pragma: no cover - backend dependent
            self._pts_failures += 1
            return None

        if milliseconds < 0.0 or milliseconds != milliseconds:  # negative or NaN
            self._pts_failures += 1
            return None
        return milliseconds / 1000.0

    def next_timestamp(self, pts_seconds: Optional[float]) -> float:
        """Return the monotonic timestamp in seconds for the current frame.

        Timestamps never go backwards: if the incoming clock value is behind the
        previous one (a backend quirk), the previous value plus the expected
        interval is used so durations stay positive.

        A FILE source that reports an unusable PTS degrades to the container
        frame clock (``frame_index / container_fps``) and never to wall-clock
        time, because on a file the wall clock measures decode speed, not the
        video. See the module docstring.

        Args:
            pts_seconds: Container PTS in seconds, or None if unavailable.

        Returns:
            Monotonic timestamp in seconds from the start of the stream.
        """
        if self.source is ClockSource.PTS:
            if pts_seconds is None:
                self._pts_failures += 1
                if self._pts_failures >= self.degrade_after_pts_failures:
                    logger.warning(
                        "Container PTS unavailable (%s failure(s)); falling back to the "
                        "container frame clock (frame_index / container_fps = %.4f) for this "
                        "file source. Wall-clock timing is NOT used for files.",
                        self._pts_failures,
                        self.interval_fps,
                    )
                    self.source = ClockSource.FRAME_COUNT
                    candidate = self._frames_seen / self.interval_fps
                else:
                    candidate = (
                        self._last_time_s + self.expected_interval_s
                        if self._last_time_s is not None
                        else 0.0
                    )
            else:
                candidate = pts_seconds
        elif self.source is ClockSource.WALL_CLOCK:
            if self._wall_start is None:
                self._wall_start = time.perf_counter()
            candidate = time.perf_counter() - self._wall_start
        else:
            candidate = self._frames_seen / self.interval_fps

        if self._last_time_s is not None:
            if candidate <= self._last_time_s:
                candidate = self._last_time_s + self.expected_interval_s
            else:
                gap = candidate - self._last_time_s
                if gap > self.max_gap_factor * self.expected_interval_s:
                    missing = int(round(gap / self.expected_interval_s)) - 1
                    if missing > 0:
                        self._dropped_total += missing
                        self.gaps.append(
                            {
                                "after_frame": self._frames_seen - 1,
                                "gap_s": round(gap, 4),
                                "estimated_missing_frames": missing,
                            }
                        )

        self._last_time_s = candidate
        self._frames_seen += 1
        return candidate

    def summary(self) -> dict:
        """Return a JSON-serialisable summary of clock behaviour.

        Returns:
            Dict with the clock source, frame count, drops, and notable gaps.
        """
        return {
            "clock_source": self.source.value,
            "frames_timed": self._frames_seen,
            "fps_nominal": round(self.fps, 4),
            "fps_used": round(self.interval_fps, 4),
            "expected_interval_s": round(self.expected_interval_s, 6),
            "estimated_dropped_frames": self._dropped_total,
            "pts_failures": self._pts_failures,
            "degraded_to_container_clock": self.source is ClockSource.FRAME_COUNT
            and self._pts_failures > 0,
            "discontinuities": self.gaps[:50],
            "discontinuity_count": len(self.gaps),
        }


def is_live_source(source: str) -> bool:
    """Return True if ``source`` looks like a live stream rather than a file.

    Args:
        source: Video path, camera index, or stream URL.

    Returns:
        True for RTSP/RTMP/HTTP(S) URLs, bare camera indices, and other
        non-file schemes.
    """
    stripped = source.strip()
    lowered = stripped.lower()
    if lowered.startswith(("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://")):
        return True
    # A bare integer (1, 0) or the special "0"/"webcam" tokens are capture
    # devices, which have no file timeline.
    if lowered in ("webcam", "camera"):
        return True
    if stripped.lstrip("-").isdigit():
        return True
    # Any other explicit URI scheme (file:\\, ftp://, ...) is not a plain path.
    if "://" in stripped:
        return True
    return False


def build_clock(
    source: str,
    fps: float,
    force_wall_clock: bool = False,
    default_fps: float = 25.0,
) -> FrameClock:
    """Choose the appropriate clock for a source.

    Wall-clock timing is used ONLY for live sources (or when explicitly forced
    by a caller who knows the source is live). A file source always starts on
    container PTS and, if PTS is unusable, degrades to the container frame clock
    rather than to wall-clock time.

    Args:
        source: Video path, camera index, or stream URL.
        fps: Resolved frames per second for the source.
        force_wall_clock: Use wall-clock timing even for files. Only for live
            capture that does not look like a live source string.
        default_fps: FPS assumed when ``fps`` is not positive.

    Returns:
        A ready :class:`FrameClock`.
    """
    if force_wall_clock or is_live_source(source):
        logger.info(
            "Timing source: WALL CLOCK (%s)",
            "forced" if force_wall_clock else "live source detected",
        )
        return FrameClock(source=ClockSource.WALL_CLOCK, fps=fps, default_fps=default_fps)

    logger.info("Timing source: CONTAINER PTS (file source); wall-clock fallback is disabled")
    return FrameClock(source=ClockSource.PTS, fps=fps, default_fps=default_fps)
