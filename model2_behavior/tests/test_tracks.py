"""Track state: normalized speed, displacement, IoU, and store bookkeeping.

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

import pytest

from model2_behavior.src.tracks import (
    Observation,
    TrackState,
    TrackStore,
    boxes_iou,
)
from model2_behavior.tests.conftest import build_track, box_at, make_observation, trajectory


class TestObservationGeometry:
    """Derived geometry on a single observation."""

    def test_height_width_center(self) -> None:
        obs = make_observation(1, 0, 0.0, (10.0, 20.0, 50.0, 80.0))
        assert obs.height == pytest.approx(60.0)
        assert obs.width == pytest.approx(40.0)
        assert obs.center == pytest.approx((30.0, 50.0))

    def test_bottom_center_uses_bottom_edge(self) -> None:
        obs = make_observation(1, 0, 0.0, (10.0, 20.0, 50.0, 80.0))
        assert obs.bottom_center == pytest.approx((30.0, 80.0))

    def test_degenerate_box_height_clamped(self) -> None:
        obs = make_observation(1, 0, 0.0, (10.0, 20.0, 50.0, 20.0))
        assert obs.height == pytest.approx(1e-6)


class TestSpeedNormalization:
    """Speed must be scale-invariant: px/s divided by box height."""

    def test_moving_track_has_expected_normalized_speed(self) -> None:
        # 60 px tall box moving 60 px in 0.24 s -> 250 px/s -> / 60 = ~4.17 bl/s
        centers = trajectory((100.0, 300.0), (160.0, 300.0), 7)
        track = build_track(1, centers, fps=25.0, height=60.0)
        speed = track.speed_bl_s(window_s=1.0)
        assert speed is not None
        assert speed == pytest.approx(250.0 / 60.0, rel=0.05)

    def test_doubling_box_height_halves_normalized_speed(self) -> None:
        centers = trajectory((100.0, 300.0), (160.0, 300.0), 7)
        small = build_track(1, centers, fps=25.0, height=30.0)
        large = build_track(2, centers, fps=25.0, height=60.0)
        small_speed = small.speed_bl_s(window_s=1.0)
        large_speed = large.speed_bl_s(window_s=1.0)
        assert small_speed is not None and large_speed is not None
        assert small_speed == pytest.approx(2.0 * large_speed, rel=0.02)

    def test_stationary_track_speed_near_zero(self) -> None:
        track = build_track(1, [(300.0, 300.0)] * 10, fps=25.0)
        speed = track.speed_bl_s(window_s=1.0)
        assert speed is not None
        assert speed == pytest.approx(0.0, abs=1e-6)

    def test_single_observation_has_no_speed(self) -> None:
        track = build_track(1, [(300.0, 300.0)], fps=25.0)
        assert track.speed_bl_s(window_s=1.0) is None

    def test_window_excludes_old_observations(self) -> None:
        # Fast at the start, then stationary for a long time.
        centers = trajectory((100.0, 300.0), (400.0, 300.0), 10) + [(400.0, 300.0)] * 50
        track = build_track(1, centers, fps=25.0)
        speed = track.speed_bl_s(window_s=0.5)
        assert speed is not None
        assert speed == pytest.approx(0.0, abs=1e-6)

    def test_raw_speed_is_in_pixels(self) -> None:
        centers = trajectory((100.0, 300.0), (160.0, 300.0), 7)
        track = build_track(1, centers, fps=25.0, height=60.0)
        assert track.raw_speed_px_s(window_s=1.0) == pytest.approx(250.0, rel=0.05)


class TestDisplacement:
    """Net displacement and its normalization."""

    def test_displacement_vector_direction(self) -> None:
        centers = trajectory((100.0, 300.0), (200.0, 300.0), 6)
        track = build_track(1, centers, fps=25.0)
        displacement = track.displacement_vector(window_s=1.0)
        assert displacement is not None
        assert displacement[0] == pytest.approx(100.0, rel=0.01)
        assert displacement[1] == pytest.approx(0.0, abs=1e-6)

    def test_normalized_displacement_divides_by_height(self) -> None:
        centers = trajectory((100.0, 300.0), (220.0, 300.0), 6)
        track = build_track(1, centers, fps=25.0, height=60.0)
        normalized = track.normalized_displacement(window_s=1.0)
        assert normalized is not None
        assert normalized[0] == pytest.approx(2.0, rel=0.01)

    def test_reversing_track_has_negative_displacement(self) -> None:
        centers = trajectory((300.0, 300.0), (200.0, 300.0), 6)
        track = build_track(1, centers, fps=25.0)
        displacement = track.displacement_vector(window_s=1.0)
        assert displacement is not None
        assert displacement[0] < 0


class TestTrackStateBookkeeping:
    """Age, confidence, and missed-frame accounting."""

    def test_age_is_last_minus_first(self) -> None:
        track = build_track(1, [(100.0, 100.0)] * 11, fps=10.0)
        assert track.age_s == pytest.approx(1.0)

    def test_mean_confidence_averages_observations(self) -> None:
        track = TrackState(track_id=1)
        track.update(make_observation(1, 0, 0.0, box_at(100.0, 100.0), confidence=0.4))
        track.update(make_observation(1, 1, 0.04, box_at(100.0, 100.0), confidence=0.6))
        assert track.mean_confidence == pytest.approx(0.5)

    def test_mark_missed_increments(self) -> None:
        track = TrackState(track_id=1)
        track.update(make_observation(1, 0, 0.0, box_at(100.0, 100.0)))
        track.mark_missed()
        track.mark_missed()
        assert track.missed_frames == 2

    def test_update_resets_missed(self) -> None:
        track = TrackState(track_id=1)
        track.update(make_observation(1, 0, 0.0, box_at(100.0, 100.0)))
        track.mark_missed()
        track.update(make_observation(1, 1, 0.04, box_at(100.0, 100.0)))
        assert track.missed_frames == 0

    def test_latest_class_accessors(self) -> None:
        track = build_track(1, [(100.0, 100.0)], class_id=7, class_name="truck")
        assert track.class_id == 7
        assert track.class_name == "truck"


class TestBoxesIoU:
    """IoU computation."""

    def test_identical_boxes_have_iou_one(self) -> None:
        assert boxes_iou((0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 10.0)) == pytest.approx(1.0)

    def test_disjoint_boxes_have_iou_zero(self) -> None:
        assert boxes_iou((0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0)) == pytest.approx(0.0)

    def test_half_overlap_known_value(self) -> None:
        # Two 10x10 boxes overlapping by 5x10: inter=50, union=150 -> 1/3.
        assert boxes_iou((0.0, 0.0, 10.0, 10.0), (5.0, 0.0, 15.0, 10.0)) == pytest.approx(1.0 / 3.0)

    def test_zero_area_boxes_have_iou_zero(self) -> None:
        assert boxes_iou((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)) == pytest.approx(0.0)


class TestTrackStore:
    """Cross-frame track bookkeeping."""

    def test_new_tracks_are_created(self) -> None:
        store = TrackStore()
        store.update([make_observation(1, 0, 0.0, box_at(100.0, 100.0))])
        assert store.get(1) is not None

    def test_missing_track_is_marked_missed(self) -> None:
        store = TrackStore()
        store.update([make_observation(1, 0, 0.0, box_at(100.0, 100.0))])
        store.update([])
        track = store.get(1)
        assert track is not None
        assert track.missed_frames == 1

    def test_two_tracks_kept_separate(self) -> None:
        store = TrackStore()
        store.update(
            [
                make_observation(1, 0, 0.0, box_at(100.0, 100.0)),
                make_observation(2, 0, 0.0, box_at(300.0, 300.0)),
            ]
        )
        assert len(store.all_tracks()) == 2

    def test_prune_removes_stale_tracks(self) -> None:
        store = TrackStore()
        store.update([make_observation(1, 0, 0.0, box_at(100.0, 100.0))])
        for _ in range(5):
            store.update([])
        removed = store.prune(max_missed=3)
        assert removed == 1
        assert store.get(1) is None

    def test_active_tracks_filters_by_missed(self) -> None:
        store = TrackStore()
        store.update([make_observation(1, 0, 0.0, box_at(100.0, 100.0))])
        for _ in range(4):
            store.update([])
        assert store.active_tracks(max_missed=1) == []
        assert len(store.active_tracks(max_missed=10)) == 1

    def test_same_track_id_accumulates_history(self) -> None:
        store = TrackStore()
        for index in range(5):
            store.update([make_observation(1, index, index / 25.0, box_at(100.0 + index, 100.0))])
        track = store.get(1)
        assert track is not None
        assert len(track.observations) == 5

    def test_max_history_is_enforced(self) -> None:
        store = TrackStore(max_history=4)
        for index in range(10):
            store.update([make_observation(1, index, index / 25.0, box_at(100.0, 100.0))])
        track = store.get(1)
        assert track is not None
        assert len(track.observations) == 4
