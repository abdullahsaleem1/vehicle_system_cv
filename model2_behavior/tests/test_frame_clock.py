"""Frame-clock and timing tests: timestamps must not come from frame_index / fps.

Real sources drop frames and vary their rate. Deriving durations by dividing a
frame counter by a nominal FPS silently corrupts every time-based threshold, so
these tests pin the behaviour of the clock that replaces it.

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

import pytest

from model2_behavior.src.frame_clock import (
    ClockSource,
    FrameClock,
    build_clock,
    is_live_source,
)


class FakeCapture:
    """Minimal stand-in for ``cv2.VideoCapture`` exposing only POS_MSEC."""

    def __init__(self, values):
        """Store the scripted POS_MSEC values.

        Args:
            values: Sequence of millisecond values to report, one per read.
        """
        self.values = list(values)
        self.calls = 0

    def get(self, prop):  # noqa: D102 - matches cv2's untyped API
        if self.calls < len(self.values):
            value = self.values[self.calls]
        else:
            value = self.values[-1]
        self.calls += 1
        return value


class TestClockSelection:
    """Clock source selection by source type."""

    def test_file_source_uses_pts(self) -> None:
        clock = build_clock("clip.mp4", 25.0)
        assert clock.source is ClockSource.PTS

    def test_rtsp_source_uses_wall_clock(self) -> None:
        clock = build_clock("rtsp://host/stream", 25.0)
        assert clock.source is ClockSource.WALL_CLOCK

    def test_force_wall_clock_overrides_file(self) -> None:
        clock = build_clock("clip.mp4", 25.0, force_wall_clock=True)
        assert clock.source is ClockSource.WALL_CLOCK

    def test_is_live_source_classification(self) -> None:
        assert is_live_source("rtsp://host/stream") is True
        assert is_live_source("http://host/mjpeg") is True
        assert is_live_source("0") is True
        assert is_live_source("1") is True
        assert is_live_source("webcam") is True
        assert is_live_source("clip.mp4") is False
        assert is_live_source("E:/clips/a.avi") is False
        assert is_live_source("clips/2026-10-20/drive-in.avi") is False


class TestPtsTimeline:
    """Timestamps taken from container PTS."""

    def test_pts_seconds_read(self) -> None:
        clock = FrameClock(ClockSource.PTS, 25.0)
        capture = FakeCapture([40.0])
        assert clock.read_pts_seconds(capture) == pytest.approx(0.04)

    def test_zero_pts_is_valid(self) -> None:
        # The first frame of a file legitimately has PTS 0.0.
        clock = FrameClock(ClockSource.PTS, 25.0)
        assert clock.read_pts_seconds(FakeCapture([0.0])) == pytest.approx(0.0)

    def test_negative_pts_is_reported_unusable(self) -> None:
        clock = FrameClock(ClockSource.PTS, 25.0)
        assert clock.read_pts_seconds(FakeCapture([-5.0])) is None

    def test_first_frame_at_zero_keeps_pts_clock(self) -> None:
        clock = FrameClock(ClockSource.PTS, 25.0)
        capture = FakeCapture([0.0, 40.0, 80.0])
        times = [clock.next_timestamp(clock.read_pts_seconds(capture)) for _ in range(3)]
        assert clock.source is ClockSource.PTS
        assert times[0] == pytest.approx(0.0)
        assert times[1] == pytest.approx(0.04)
        assert clock.dropped_total == 0

    def test_timestamps_are_monotonic(self) -> None:
        clock = FrameClock(ClockSource.PTS, 25.0)
        times = [clock.next_timestamp(pts) for pts in (0.0, 0.04, 0.08, 0.12)]
        assert times == sorted(times)
        assert times[0] == pytest.approx(0.0)

    def test_backwards_pts_does_not_go_backwards(self) -> None:
        clock = FrameClock(ClockSource.PTS, 25.0)
        clock.next_timestamp(1.0)
        # A misbehaving backend jumps backwards; the timeline must not.
        assert clock.next_timestamp(0.5) > 1.0

    def test_dropped_frames_detected_from_gap(self) -> None:
        clock = FrameClock(ClockSource.PTS, 25.0)  # 0.04 s expected interval
        clock.next_timestamp(0.0)
        # Jump 0.20 s -> a gap of 0.16 s -> four missing frames.
        clock.next_timestamp(0.20)
        assert clock.dropped_total == 4
        assert clock.summary()["discontinuity_count"] == 1

    def test_small_jitter_is_not_a_drop(self) -> None:
        clock = FrameClock(ClockSource.PTS, 25.0)
        clock.next_timestamp(0.0)
        clock.next_timestamp(0.041)  # within the 3x tolerance
        assert clock.dropped_total == 0

    def test_pts_failure_degrades_to_container_clock_not_wall_clock(self) -> None:
        clock = FrameClock(ClockSource.PTS, 25.0)
        clock.next_timestamp(0.0)
        clock.next_timestamp(None)  # first failure -> degrade immediately
        assert clock.source is ClockSource.FRAME_COUNT
        assert clock.source is not ClockSource.WALL_CLOCK

    def test_container_clock_timeline_is_frame_index_over_fps(self) -> None:
        clock = FrameClock(ClockSource.PTS, 25.0)
        clock.next_timestamp(0.0)  # frame 0
        times = [clock.next_timestamp(None) for _ in range(4)]
        assert clock.source is ClockSource.FRAME_COUNT
        # frame_index / fps for frames 1..4
        assert times == [pytest.approx(index / 25.0) for index in range(1, 5)]

    def test_container_clock_has_no_spurious_drops(self) -> None:
        # Degrading mid-stream must not manufacture a gap: the frame clock is
        # continuous with the expected interval.
        clock = FrameClock(ClockSource.PTS, 25.0)
        clock.next_timestamp(0.0)
        clock.next_timestamp(0.04)
        clock.next_timestamp(None)  # degrade at frame 2
        for _ in range(10):
            clock.next_timestamp(None)
        assert clock.dropped_total == 0
        assert clock.summary()["degraded_to_container_clock"] is True

    def test_file_clock_never_becomes_wall_clock(self) -> None:
        clock = build_clock("clip.mp4", 25.0)
        for _ in range(50):
            clock.next_timestamp(None)
        assert clock.source is ClockSource.FRAME_COUNT

    def test_summary_is_serialisable(self) -> None:
        clock = FrameClock(ClockSource.PTS, 25.0)
        clock.next_timestamp(0.0)
        import json

        json.dumps(clock.summary())  # must not raise


class TestWallClockTimeline:
    """Wall-clock timing for live sources."""

    def test_wall_clock_starts_near_zero(self) -> None:
        clock = FrameClock(ClockSource.WALL_CLOCK, 25.0)
        first = clock.next_timestamp(None)
        assert first >= 0.0
        assert first < 1.0

    def test_wall_clock_is_monotonic(self) -> None:
        clock = FrameClock(ClockSource.WALL_CLOCK, 25.0)
        times = [clock.next_timestamp(None) for _ in range(5)]
        assert times == sorted(times)

    def test_summary_reports_wall_clock_source(self) -> None:
        clock = FrameClock(ClockSource.WALL_CLOCK, 25.0)
        clock.next_timestamp(None)
        assert clock.summary()["clock_source"] == "wall_clock"

    def test_wall_clock_only_for_live_sources(self) -> None:
        assert build_clock("rtsp://host/stream", 25.0).source is ClockSource.WALL_CLOCK
        assert build_clock("0", 25.0).source is ClockSource.WALL_CLOCK
        assert build_clock("clip.mp4", 25.0).source is ClockSource.PTS
