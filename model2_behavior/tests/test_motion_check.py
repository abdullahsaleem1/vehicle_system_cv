"""Tests for the static-camera precondition check, on synthetic video only.

The check exists to catch footage shot with a moving camera, which silently
invalidates fixed lane geometry and every image-relative speed rule. These tests
build small synthetic frames with a known global shift and assert the check
notices exactly that, and stays quiet on a still scene.

No real footage, no GPU, no network.

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from model2_behavior.src.motion_check import (
    CameraMotionChecker,
    MotionCheckConfig,
    build_motion_checker,
)


def textured_frame(
    width: int = 320,
    height: int = 240,
    shift_x: int = 0,
    shift_y: int = 0,
    seed: int = 1234,
) -> np.ndarray:
    """Build a trackable BGR frame from a fixed structured texture, optionally moved.

    The texture is smoothed random noise: per-pixel noise has no structure at the
    coarse levels of an optical-flow pyramid, so tracking it is unreliable, while
    a periodic grid aliases under flow (a shift near the grid pitch maps every
    corner onto its neighbour). Smoothed noise gives each corner a unique,
    multi-scale match, so the measured median shift reflects the true
    translation.

    The pattern is generated from ``seed`` so the SAME texture is reused for the
    shifted frame; translating a different random image would measure noise
    matching, not translation.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.
        shift_x: Horizontal translation in pixels.
        shift_y: Vertical translation in pixels.
        seed: Texture seed. Keep it fixed across a reference/shifted pair.

    Returns:
        A ``(height, width, 3)`` uint8 BGR image.
    """
    import cv2

    generator = np.random.default_rng(seed)
    coarse = generator.integers(0, 256, size=(height // 16 + 2, width // 16 + 2), dtype=np.uint8)
    base = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_LINEAR)
    base = cv2.GaussianBlur(base, (5, 5), 0)
    base = np.stack([base, base, base], axis=-1)
    if shift_x == 0 and shift_y == 0:
        return base
    # np.roll translates the texture; the wrapped edge is a thin strip that LK
    # simply fails to match, which the median ignores.
    return np.roll(base, shift=(shift_y, shift_x), axis=(0, 1))


class TestConfig:
    """Config parsing and validation."""

    def test_defaults_are_usable(self) -> None:
        config = MotionCheckConfig()
        config.validate()
        assert config.enabled is True
        assert config.threshold > 0

    def test_unknown_key_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown static_camera_check key"):
            MotionCheckConfig.from_mapping({"threshhold": 0.02})

    def test_non_positive_threshold_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="threshold must be > 0"):
            MotionCheckConfig.from_mapping({"threshold": 0.0})

    def test_zero_cadence_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="check_every_n_frames"):
            MotionCheckConfig.from_mapping({"check_every_n_frames": 0})

    def test_too_small_work_width_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="work_width"):
            MotionCheckConfig.from_mapping({"work_width": 8})

    def test_from_mapping_round_trips(self) -> None:
        config = build_motion_checker({"threshold": 0.05, "check_every_n_frames": 1}).config
        assert config.threshold == 0.05
        assert config.check_every_n_frames == 1

    def test_build_with_no_mapping_uses_defaults(self) -> None:
        assert build_motion_checker(None).config.enabled is True


class TestStillScene:
    """A still camera must not be flagged."""

    def test_static_frames_are_not_flagged(self) -> None:
        checker = CameraMotionChecker(MotionCheckConfig(check_every_n_frames=1))
        for index in range(5):
            checker.observe(textured_frame(), index)
        result = checker.finalize()
        assert result.checked is True
        assert result.camera_motion_suspected is False
        assert result.max_shift_fraction < 0.01

    def test_first_frame_becomes_reference_and_returns_no_sample(self) -> None:
        checker = CameraMotionChecker(MotionCheckConfig(check_every_n_frames=1))
        assert checker.observe(textured_frame(), 0) is None

    def test_disabled_check_reports_not_checked(self) -> None:
        checker = CameraMotionChecker(MotionCheckConfig(enabled=False))
        for index in range(5):
            assert checker.observe(textured_frame(), index) is None
        result = checker.finalize()
        assert result.checked is False
        assert result.samples_taken == 0
        assert result.camera_motion_suspected is False


class TestMovingScene:
    """A shifting camera must be flagged."""

    def test_large_global_shift_is_flagged(self) -> None:
        # 40 px of shift on a 320 px frame is 12.5% of the width, far above the
        # default 1% threshold.
        checker = CameraMotionChecker(MotionCheckConfig(check_every_n_frames=1))
        checker.observe(textured_frame(), 0)
        for index in range(1, 4):
            checker.observe(textured_frame(shift_x=40), index)
        result = checker.finalize()
        assert result.camera_motion_suspected is True
        assert result.samples_taken >= 1
        assert result.max_shift_fraction > 0.05
        assert result.first_suspected_frame is not None

    def test_shift_fraction_tracks_the_real_shift(self) -> None:
        # A 32 px shift on a 320 px frame is ~0.10 of the width; the median
        # estimate should land close to that.
        checker = CameraMotionChecker(MotionCheckConfig(check_every_n_frames=1))
        checker.observe(textured_frame(), 0)
        sample = checker.observe(textured_frame(shift_x=32), 1)
        assert sample is not None
        assert sample.median_shift_fraction == pytest.approx(0.10, abs=0.02)

    def test_threshold_makes_the_flag_configurable(self) -> None:
        strict = CameraMotionChecker(MotionCheckConfig(check_every_n_frames=1, threshold=0.9))
        lenient = CameraMotionChecker(MotionCheckConfig(check_every_n_frames=1, threshold=0.001))
        for checker in (strict, lenient):
            checker.observe(textured_frame(), 0)
            checker.observe(textured_frame(shift_x=40), 1)
        assert strict.finalize().camera_motion_suspected is False
        assert lenient.finalize().camera_motion_suspected is True

    def test_cadence_skips_off_schedule_frames(self) -> None:
        checker = CameraMotionChecker(MotionCheckConfig(check_every_n_frames=10))
        checker.observe(textured_frame(), 0)
        # Frames 1-9 are off-schedule and yield no sample.
        for index in range(1, 10):
            assert checker.observe(textured_frame(shift_x=40), index) is None
        # Frame 10 is on schedule.
        assert checker.observe(textured_frame(shift_x=40), 10) is not None

    def test_flat_reference_with_no_corners_is_skipped_not_guessed(self) -> None:
        # A featureless reference frame has nothing to track, so the check must
        # decline to judge rather than invent a shift.
        checker = CameraMotionChecker(MotionCheckConfig(check_every_n_frames=1))
        flat = np.full((240, 320, 3), 128, dtype=np.uint8)
        assert checker.observe(flat, 0) is None
        sample = checker.observe(textured_frame(shift_x=40), 1)
        assert sample is None
        result = checker.finalize()
        assert result.skipped_frames >= 1
        # A skipped frame is not evidence of motion.
        assert result.camera_motion_suspected is False

    def test_result_is_json_serialisable_and_documented(self) -> None:
        checker = CameraMotionChecker(MotionCheckConfig(check_every_n_frames=1))
        checker.observe(textured_frame(), 0)
        checker.observe(textured_frame(shift_x=40), 1)
        payload = checker.finalize().to_dict()
        assert payload["camera_motion_suspected"] is True
        assert "STATIC-CAMERA" in payload["note"]
        assert payload["threshold_fraction_of_width"] == pytest.approx(0.01)


class TestResolutionIndependence:
    """A shift is judged as a FRACTION of the frame, not in raw pixels."""

    def test_same_fraction_at_different_resolutions(self) -> None:
        # The SAME scene at two resolutions: the 1280-wide frame is the 320-wide
        # texture upscaled 4x, so an identical *fraction* of the frame moves. The
        # check must judge the shift as a fraction of the width, not in pixels.
        import cv2

        small = CameraMotionChecker(MotionCheckConfig(check_every_n_frames=1, work_width=320))
        large = CameraMotionChecker(MotionCheckConfig(check_every_n_frames=1, work_width=320))

        base_small = textured_frame(width=320, height=240)
        base_large = cv2.resize(base_small, (1280, 960), interpolation=cv2.INTER_LINEAR)

        small.observe(base_small, 0)
        small_sample = small.observe(textured_frame(width=320, height=240, shift_x=32), 1)

        large.observe(base_large, 0)
        shifted_large = np.roll(base_large, shift=128, axis=1)
        large_sample = large.observe(shifted_large, 1)

        assert small_sample is not None and large_sample is not None
        assert small_sample.median_shift_fraction == pytest.approx(
            large_sample.median_shift_fraction, abs=0.02
        )
