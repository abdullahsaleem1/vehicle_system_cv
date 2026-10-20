"""Rule construction, geometry refusal, and per-rule synthetic trajectories.

Every rule must REFUSE TO CONSTRUCT (or refuse to run) when its required camera
geometry is absent. Tests supply their own synthetic ROI and lane direction.

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

import pytest

from model2_behavior.src.config import load_config
from model2_behavior.src.geometry import GeometryError
from model2_behavior.src.rules import build_rule, build_rules, get_rule
from model2_behavior.src.rules.base import FrameContext, RuleGeometryError
from model2_behavior.src.rules.collision_proxy import CollisionProxyRule
from model2_behavior.src.rules.stopped_vehicle import StoppedVehicleRule
from model2_behavior.src.rules.sudden_decel import SuddenDecelRule
from model2_behavior.src.rules.wrong_way import WrongWayRule
from model2_behavior.tests.conftest import (
    LANE_POLYGON,
    build_track,
    make_geometry,
    make_observation,
    make_rule_config,
    trajectory,
)

FPS = 25.0


def context(frame_index: int, geometry, fps: float = FPS) -> FrameContext:
    """Build a frame context for a synthetic frame.

    Args:
        frame_index: Frame number.
        geometry: Camera geometry.
        fps: Frame rate.

    Returns:
        A populated frame context.
    """
    return FrameContext(
        frame_index=frame_index,
        time_s=frame_index / fps,
        fps=fps,
        geometry=geometry,
        geometry_describe=geometry.describe(),
    )


class TestGeometryRefusal:
    """Rules must refuse without their required geometry."""

    def test_stopped_vehicle_requires_driving_lane(self) -> None:
        rule = StoppedVehicleRule(make_rule_config("stopped_vehicle"), FPS)
        with pytest.raises(RuleGeometryError, match="lane"):
            rule.check_geometry(make_geometry())

    def test_stopped_vehicle_rejects_full_frame_sentinel(self) -> None:
        rule = StoppedVehicleRule(make_rule_config("stopped_vehicle"), FPS)
        # A geometry with no lane at all is the only representation of
        # "full_frame" reaching the rule, and it must be refused.
        with pytest.raises(RuleGeometryError):
            rule.check_geometry(make_geometry(reference_resolution=None))

    def test_stopped_vehicle_accepts_configured_lane(self) -> None:
        rule = StoppedVehicleRule(make_rule_config("stopped_vehicle"), FPS)
        rule.check_geometry(make_geometry(driving_lane=LANE_POLYGON))

    def test_stopped_vehicle_update_raises_without_lane(self) -> None:
        rule = StoppedVehicleRule(make_rule_config("stopped_vehicle", min_consecutive_s=0.04), FPS)
        track = build_track(1, [(300.0, 400.0)] * 5, fps=FPS)
        with pytest.raises(RuleGeometryError):
            rule.update(context(4, make_geometry()), [track])

    def test_wrong_way_requires_lane_direction(self) -> None:
        rule = WrongWayRule(make_rule_config("wrong_way"), FPS)
        with pytest.raises(RuleGeometryError, match="direction"):
            rule.check_geometry(make_geometry(driving_lane=LANE_POLYGON))

    def test_wrong_way_accepts_configured_direction(self) -> None:
        rule = WrongWayRule(make_rule_config("wrong_way"), FPS)
        rule.check_geometry(make_geometry(driving_lane=LANE_POLYGON, lane_direction=[0.0, 0.0, 100.0, 0.0]))

    def test_sudden_decel_needs_no_geometry(self) -> None:
        rule = SuddenDecelRule(make_rule_config("sudden_decel"), FPS)
        rule.check_geometry(make_geometry())  # must not raise

    def test_collision_proxy_needs_no_geometry(self) -> None:
        rule = CollisionProxyRule(make_rule_config("collision_proxy"), FPS)
        rule.check_geometry(make_geometry())  # must not raise

    def test_build_rules_strict_raises_for_missing_lane(self) -> None:
        config = load_config(validate=False)
        with pytest.raises(RuleGeometryError):
            build_rules(config, FPS, make_geometry(), strict=True)

    def test_build_rules_non_strict_skips_missing_lane(self) -> None:
        config = load_config(validate=False)
        rules = build_rules(config, FPS, make_geometry(), strict=False)
        names = {rule.name for rule in rules}
        assert "stopped_vehicle" not in names
        assert "wrong_way" not in names
        # Geometry-free rules must still be constructed.
        assert "sudden_decel" in names
        assert "collision_proxy" in names

    def test_build_rule_rejects_unknown_name(self) -> None:
        with pytest.raises(KeyError):
            build_rule("teleportation", make_rule_config("teleportation"), FPS)

    def test_fps_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="fps"):
            StoppedVehicleRule(make_rule_config("stopped_vehicle"), 0.0)


class TestFpsAwareConversion:
    """Seconds must convert to frames using the measured FPS."""

    def test_frame_thresholds_scale_with_fps(self) -> None:
        config = make_rule_config("stopped_vehicle", min_consecutive_s=2.0, debounce_s=4.0)
        at_10 = StoppedVehicleRule(config, 10.0)
        at_30 = StoppedVehicleRule(config, 30.0)
        assert at_10.min_consecutive_frames == 20
        assert at_30.min_consecutive_frames == 60
        assert at_10.debounce_frames == 40
        assert at_30.debounce_frames == 120

    def test_sub_frame_duration_rounds_up_to_one(self) -> None:
        rule = StoppedVehicleRule(make_rule_config("stopped_vehicle", min_consecutive_s=0.001), 10.0)
        assert rule.min_consecutive_frames == 1


class TestStoppedVehicleRule:
    """Stationary vehicle inside the driving lane."""

    def test_fires_for_stationary_track_in_lane(self) -> None:
        geometry = make_geometry(driving_lane=LANE_POLYGON)
        rule = StoppedVehicleRule(
            make_rule_config("stopped_vehicle", min_consecutive_s=0.2, debounce_s=0.0), FPS
        )
        centers = [(300.0, 400.0)] * 10
        hit = None
        for frame_index in range(10):
            track = build_track(1, centers[: frame_index + 1], fps=FPS)
            hit = rule.update(context(frame_index, geometry), [track])
            if hit is not None:
                break
        assert hit is not None
        assert hit.track_ids == [1]
        assert hit.zone_id == "driving_lane"

    def test_does_not_fire_for_moving_track(self) -> None:
        geometry = make_geometry(driving_lane=LANE_POLYGON)
        rule = StoppedVehicleRule(
            make_rule_config("stopped_vehicle", min_consecutive_s=0.2, debounce_s=0.0), FPS
        )
        centers = trajectory((100.0, 400.0), (500.0, 400.0), 12)
        hit = None
        for frame_index in range(11):
            track = build_track(1, centers[: frame_index + 1], fps=FPS)
            hit = rule.update(context(frame_index, geometry), [track])
        assert hit is None

    def test_does_not_fire_outside_lane(self) -> None:
        geometry = make_geometry(driving_lane=LANE_POLYGON)
        rule = StoppedVehicleRule(
            make_rule_config("stopped_vehicle", min_consecutive_s=0.2, debounce_s=0.0), FPS
        )
        # y=100 is above the lane (which starts at y=240).
        centers = [(300.0, 100.0)] * 12
        hit = None
        for frame_index in range(12):
            track = build_track(1, centers[: frame_index + 1], fps=FPS)
            hit = rule.update(context(frame_index, geometry), [track])
        assert hit is None

    def test_requires_min_consecutive_duration(self) -> None:
        geometry = make_geometry(driving_lane=LANE_POLYGON)
        rule = StoppedVehicleRule(
            make_rule_config("stopped_vehicle", min_consecutive_s=1.0, debounce_s=0.0), FPS
        )
        centers = [(300.0, 400.0)] * 30
        # At 25 fps, 1.0 s needs 25 consecutive frames: frame 10 must not fire.
        track = build_track(1, centers[:11], fps=FPS)
        assert rule.update(context(10, geometry), [track]) is None

    def test_debounce_suppresses_repeat_events(self) -> None:
        geometry = make_geometry(driving_lane=LANE_POLYGON)
        rule = StoppedVehicleRule(
            make_rule_config("stopped_vehicle", min_consecutive_s=0.04, debounce_s=2.0), FPS
        )
        centers = [(300.0, 400.0)] * 20
        hits = 0
        for frame_index in range(20):
            track = build_track(1, centers[: frame_index + 1], fps=FPS)
            if rule.update(context(frame_index, geometry), [track]) is not None:
                hits += 1
        # Debounce is 50 frames, so only the first event may fire.
        assert hits == 1

    def test_rule_params_capture_actual_thresholds(self) -> None:
        geometry = make_geometry(driving_lane=LANE_POLYGON)
        rule = StoppedVehicleRule(
            make_rule_config(
                "stopped_vehicle",
                min_consecutive_s=0.04,
                debounce_s=0.0,
                speed_thresh_bl_s=0.05,
            ),
            FPS,
        )
        centers = [(300.0, 400.0)] * 5
        track = build_track(1, centers, fps=FPS)
        hit = rule.update(context(4, geometry), [track])
        assert hit is not None
        assert hit.rule_params["speed_thresh_bl_s"] == pytest.approx(0.05)
        assert hit.rule_params["fps"] == pytest.approx(FPS)
        assert hit.rule_params["lane"] is not None
        assert "PLACEHOLDER" in hit.rule_params["threshold_status"]


class TestWrongWayRule:
    """Movement opposing the configured lane direction."""

    # Lane direction is left-to-right: (1, 0).
    GEOMETRY = make_geometry(driving_lane=LANE_POLYGON, lane_direction=[100.0, 400.0, 540.0, 400.0])

    def test_fires_for_reverse_movement(self) -> None:
        rule = WrongWayRule(
            make_rule_config(
                "wrong_way", min_consecutive_s=0.2, debounce_s=0.0, min_displacement_bl=0.05
            ),
            FPS,
        )
        # Moving right-to-left is exactly opposite to the lane direction.
        centers = trajectory((500.0, 400.0), (200.0, 400.0), 12)
        hit = None
        for frame_index in range(11):
            track = build_track(1, centers[: frame_index + 1], fps=FPS)
            hit = rule.update(context(frame_index, self.GEOMETRY), [track])
            if hit is not None:
                break
        assert hit is not None
        assert hit.track_ids == [1]

    def test_does_not_fire_for_correct_direction(self) -> None:
        rule = WrongWayRule(
            make_rule_config(
                "wrong_way", min_consecutive_s=0.2, debounce_s=0.0, min_displacement_bl=0.05
            ),
            FPS,
        )
        centers = trajectory((200.0, 400.0), (500.0, 400.0), 12)
        hit = None
        for frame_index in range(11):
            track = build_track(1, centers[: frame_index + 1], fps=FPS)
            hit = rule.update(context(frame_index, self.GEOMETRY), [track])
        assert hit is None

    def test_does_not_fire_for_slow_jitter(self) -> None:
        rule = WrongWayRule(
            make_rule_config(
                "wrong_way", min_consecutive_s=0.04, debounce_s=0.0, min_displacement_bl=5.0
            ),
            FPS,
        )
        centers = trajectory((300.0, 400.0), (295.0, 400.0), 12)
        hit = None
        for frame_index in range(11):
            track = build_track(1, centers[: frame_index + 1], fps=FPS)
            hit = rule.update(context(frame_index, self.GEOMETRY), [track])
        assert hit is None

    def test_fires_for_perpendicular_movement(self) -> None:
        # Moving straight down is 90 deg from (1, 0); with a 45 deg tolerance
        # that is not "opposing", so the rule must not fire.
        rule = WrongWayRule(
            make_rule_config(
                "wrong_way", min_consecutive_s=0.04, debounce_s=0.0, min_displacement_bl=0.05
            ),
            FPS,
        )
        centers = trajectory((300.0, 260.0), (300.0, 460.0), 12)
        hit = None
        for frame_index in range(11):
            track = build_track(1, centers[: frame_index + 1], fps=FPS)
            hit = rule.update(context(frame_index, self.GEOMETRY), [track])
        assert hit is None

    def test_update_is_noop_without_direction(self) -> None:
        # Defensive path: geometry without direction returns None rather than raising
        # inside update (check_geometry is what enforces the requirement).
        rule = WrongWayRule(make_rule_config("wrong_way"), FPS)
        geometry = make_geometry(driving_lane=LANE_POLYGON)
        track = build_track(1, trajectory((500.0, 400.0), (200.0, 400.0), 8), fps=FPS)
        with pytest.raises(GeometryError):
            rule.update(context(7, geometry), [track])

    def test_zone_id_is_the_lane_name(self) -> None:
        rule = WrongWayRule(
            make_rule_config(
                "wrong_way", min_consecutive_s=0.2, debounce_s=0.0, min_displacement_bl=0.05
            ),
            FPS,
        )
        centers = trajectory((500.0, 400.0), (200.0, 400.0), 12)
        hit = None
        for frame_index in range(11):
            track = build_track(1, centers[: frame_index + 1], fps=FPS)
            hit = rule.update(context(frame_index, self.GEOMETRY), [track])
            if hit is not None:
                break
        assert hit is not None
        assert hit.zone_id == "driving_lane"
        assert hit.rule_params["lane_name"] == "driving_lane"

    def test_evaluated_per_lane_using_the_containing_lane(self) -> None:
        """Two lanes with opposite directions: only the containing lane applies."""
        from model2_behavior.src.geometry import Lane, Polygon

        left = Lane(
            "left",
            Polygon([(40.0, 240.0), (300.0, 240.0), (300.0, 470.0), (40.0, 470.0)]),
            direction=((100.0, 400.0), (280.0, 400.0)),  # left-to-right
        )
        right = Lane(
            "right",
            Polygon([(320.0, 240.0), (600.0, 240.0), (600.0, 470.0), (320.0, 470.0)]),
            direction=((560.0, 400.0), (340.0, 400.0)),  # right-to-left
        )
        geometry = make_geometry(lanes=[left, right])
        rule = WrongWayRule(
            make_rule_config(
                "wrong_way", min_consecutive_s=0.1, debounce_s=0.0, min_displacement_bl=0.02
            ),
            FPS,
        )

        # A vehicle inside `right` moving leftwards follows that lane: no fire.
        centers = trajectory((560.0, 400.0), (360.0, 400.0), 10)
        hit = None
        for frame_index in range(9):
            track = build_track(1, centers[: frame_index + 1], fps=FPS)
            hit = rule.update(context(frame_index, geometry), [track])
        assert hit is None

        # The same leftward motion inside `left` is wrong-way: fires, zone `left`.
        rule2 = WrongWayRule(
            make_rule_config(
                "wrong_way", min_consecutive_s=0.1, debounce_s=0.0, min_displacement_bl=0.02
            ),
            FPS,
        )
        centers = trajectory((280.0, 400.0), (80.0, 400.0), 10)
        hit = None
        for frame_index in range(9):
            track = build_track(2, centers[: frame_index + 1], fps=FPS)
            hit = rule2.update(context(frame_index, geometry), [track])
            if hit is not None:
                break
        assert hit is not None
        assert hit.zone_id == "left"

    def test_ignored_outside_every_lane(self) -> None:
        """A vehicle between lanes cannot be judged wrong-way."""
        from model2_behavior.src.geometry import Lane, Polygon

        left = Lane(
            "left",
            Polygon([(40.0, 240.0), (200.0, 240.0), (200.0, 470.0), (40.0, 470.0)]),
            direction=((60.0, 400.0), (180.0, 400.0)),
        )
        right = Lane(
            "right",
            Polygon([(400.0, 240.0), (600.0, 240.0), (600.0, 470.0), (400.0, 470.0)]),
            direction=((60.0, 400.0), (180.0, 400.0)),
        )
        geometry = make_geometry(lanes=[left, right])
        rule = WrongWayRule(
            make_rule_config(
                "wrong_way", min_consecutive_s=0.1, debounce_s=0.0, min_displacement_bl=0.02
            ),
            FPS,
        )
        # x=300 is the gap between the lanes.
        centers = trajectory((300.0, 400.0), (250.0, 400.0), 10)
        hit = None
        for frame_index in range(9):
            track = build_track(1, centers[: frame_index + 1], fps=FPS)
            hit = rule.update(context(frame_index, geometry), [track])
        assert hit is None

    def test_vehicle_outside_every_lane_yields_no_wrong_way_event(self) -> None:
        """Gate 3 requirement: a vehicle outside all lanes is never wrong-way.

        Two lanes with OPPOSITE directions are used, so if lane assignment were
        skipped and the first lane's direction were applied globally, the
        reversing vehicle here would fire. It must not: no lane contains it.
        """
        from model2_behavior.src.geometry import Lane, Polygon

        left = Lane(
            "left",
            Polygon([(40.0, 240.0), (200.0, 240.0), (200.0, 470.0), (40.0, 470.0)]),
            direction=((60.0, 400.0), (180.0, 400.0)),  # left -> right
        )
        right = Lane(
            "right",
            Polygon([(400.0, 240.0), (600.0, 240.0), (600.0, 470.0), (400.0, 470.0)]),
            direction=((580.0, 400.0), (420.0, 400.0)),  # right -> left
        )
        geometry = make_geometry(lanes=[left, right])
        rule = WrongWayRule(
            make_rule_config(
                "wrong_way", min_consecutive_s=0.04, debounce_s=0.0, min_displacement_bl=0.02
            ),
            FPS,
        )

        # Stationary in the gap between the lanes for a long time: never fires.
        for frame_index in range(30):
            track = build_track(
                1, [(300.0, 400.0)] * (frame_index + 1), fps=FPS
            )
            assert rule.update(context(frame_index, geometry), [track]) is None

        # Fast leftward motion ENTIRELY inside the gap (x 200..400): still never
        # fires, regardless of direction, because no lane contains the
        # ground-contact point. The segment must never cross a lane, or the test
        # would be measuring lane assignment rather than the gap.
        reversing = trajectory((390.0, 400.0), (215.0, 400.0), 20)
        for frame_index in range(19):
            track = build_track(2, reversing[: frame_index + 1], fps=FPS)
            assert geometry.lane_containing((reversing[frame_index][0], reversing[frame_index][1])) is None
            assert rule.update(context(frame_index, geometry), [track]) is None

        # Sanity: the same motion inside `left` DOES fire, so the test cannot
        # pass merely because the rule is inert.
        inside = trajectory((180.0, 400.0), (60.0, 400.0), 20)
        fired = None
        for frame_index in range(19):
            track = build_track(3, inside[: frame_index + 1], fps=FPS)
            fired = rule.update(context(frame_index, geometry), [track])
            if fired is not None:
                break
        assert fired is not None
        assert fired.zone_id == "left"


class TestStoppedVehicleMultiLane:
    """stopped_vehicle must match against any configured lane."""

    def test_zone_id_is_the_matched_lane(self) -> None:
        from model2_behavior.src.geometry import Lane, Polygon

        left = Lane(
            "left", Polygon([(40.0, 240.0), (300.0, 240.0), (300.0, 470.0), (40.0, 470.0)])
        )
        right = Lane(
            "right", Polygon([(320.0, 240.0), (600.0, 240.0), (600.0, 470.0), (320.0, 470.0)])
        )
        geometry = make_geometry(lanes=[left, right])
        rule = StoppedVehicleRule(
            make_rule_config(
                "stopped_vehicle", min_consecutive_s=0.04, debounce_s=0.0, speed_thresh_bl_s=0.05
            ),
            FPS,
        )
        # A stationary vehicle in the `right` lane.
        centers = [(500.0, 400.0)] * 5
        track = build_track(1, centers, fps=FPS)
        hit = rule.update(context(4, geometry), [track])
        assert hit is not None
        assert hit.zone_id == "right"


class TestSuddenDecelRule:
    """Sharp speed drop within a short window."""

    def test_fires_when_track_stops_abruptly(self) -> None:
        rule = SuddenDecelRule(
            make_rule_config(
                "sudden_decel",
                min_consecutive_s=0.04,
                debounce_s=0.0,
                decel_thresh_bl_s2=1.0,
                window_s=1.0,
                min_speed_before_bl_s=0.2,
            ),
            FPS,
        )
        # Fast for 15 frames, then frozen for 15 frames.
        fast = trajectory((100.0, 300.0), (400.0, 300.0), 15)
        centers = fast + [(400.0, 300.0)] * 15
        hit = None
        for frame_index in range(len(centers)):
            track = build_track(1, centers[: frame_index + 1], fps=FPS)
            hit = rule.update(context(frame_index, make_geometry()), [track])
            if hit is not None:
                break
        assert hit is not None
        assert hit.track_ids == [1]
        assert hit.rule_params["observed_decel_bl_s2"] >= 1.0

    def test_does_not_fire_for_constant_speed(self) -> None:
        rule = SuddenDecelRule(
            make_rule_config(
                "sudden_decel",
                min_consecutive_s=0.04,
                debounce_s=0.0,
                decel_thresh_bl_s2=1.0,
                window_s=1.0,
                min_speed_before_bl_s=0.2,
            ),
            FPS,
        )
        centers = trajectory((100.0, 300.0), (500.0, 300.0), 30)
        hit = None
        for frame_index in range(29):
            track = build_track(1, centers[: frame_index + 1], fps=FPS)
            hit = rule.update(context(frame_index, make_geometry()), [track])
        assert hit is None

    def test_does_not_fire_when_already_slow(self) -> None:
        rule = SuddenDecelRule(
            make_rule_config(
                "sudden_decel",
                min_consecutive_s=0.04,
                debounce_s=0.0,
                decel_thresh_bl_s2=0.5,
                window_s=1.0,
                min_speed_before_bl_s=5.0,
            ),
            FPS,
        )
        fast = trajectory((100.0, 300.0), (200.0, 300.0), 15)
        centers = fast + [(200.0, 300.0)] * 15
        hit = None
        for frame_index in range(len(centers)):
            track = build_track(1, centers[: frame_index + 1], fps=FPS)
            hit = rule.update(context(frame_index, make_geometry()), [track])
        assert hit is None

    def test_does_not_fire_with_insufficient_history(self) -> None:
        rule = SuddenDecelRule(
            make_rule_config("sudden_decel", min_consecutive_s=0.04, window_s=1.0), FPS
        )
        track = build_track(1, [(100.0, 300.0), (110.0, 300.0)], fps=FPS)
        assert rule.update(context(1, make_geometry()), [track]) is None


class TestCollisionProxyRule:
    """Sustained overlap with both tracks moving, then a speed drop."""

    def _config(self, **overrides):
        """Build a collision-proxy config, with overrides applied."""
        params = {
            "iou_thresh": 0.30,
            "min_overlap_s": 0.2,
            "min_consecutive_s": 0.04,
            "both_moving_before": True,
            "min_pre_contact_speed_bl_s": 0.1,
            "post_contact_speed_drop_bl_s": 0.2,
            "window_s": 1.5,
            "speed_window_s": 0.5,
            "debounce_s": 0.0,
        }
        params.update(overrides)
        return make_rule_config("collision_proxy", **params)

    def _rule(self, **overrides) -> CollisionProxyRule:
        """Build a collision-proxy rule at the test frame rate."""
        return CollisionProxyRule(self._config(**overrides), FPS)

    def _run(self, rule, tracks_by_frame):
        """Feed frames in order and return the first hit, if any."""
        for frame_index, tracks in enumerate(tracks_by_frame):
            hit = rule.update(context(frame_index, make_geometry()), tracks)
            if hit is not None:
                return hit
        return None

    def test_fires_for_approach_then_stop(self) -> None:
        """Two vehicles converge, collide, and one stops."""
        # Track 1 moves right then halts at x=300; track 2 moves left then halts.
        first = trajectory((100.0, 300.0), (295.0, 300.0), 20) + [(300.0, 300.0)] * 20
        second = trajectory((500.0, 300.0), (305.0, 300.0), 20) + [(300.0, 300.0)] * 20

        rule = self._rule()
        hit = None
        for frame_index in range(len(first)):
            track_a = build_track(1, first[: frame_index + 1], fps=FPS)
            track_b = build_track(2, second[: frame_index + 1], fps=FPS)
            hit = rule.update(context(frame_index, make_geometry()), [track_a, track_b])
            if hit is not None:
                break
        assert hit is not None
        assert sorted(hit.track_ids) == [1, 2]
        assert hit.rule_params["both_moving_before"] is True

    def test_does_not_fire_for_stationary_overlap(self) -> None:
        """Two parked, overlapping vehicles are not a collision."""
        rule = self._rule()
        centers = [(300.0, 300.0)] * 40
        hit = None
        for frame_index in range(len(centers)):
            track_a = build_track(1, centers[: frame_index + 1], fps=FPS)
            track_b = build_track(2, centers[: frame_index + 1], fps=FPS)
            hit = rule.update(context(frame_index, make_geometry()), [track_a, track_b])
        assert hit is None

    def test_does_not_fire_without_sustained_overlap(self) -> None:
        """A single overlapping frame must not be enough."""
        rule = self._rule(min_overlap_s=1.0)
        # Tracks cross for only a few frames, far fewer than 1.0 s at 25 fps.
        first = trajectory((100.0, 300.0), (500.0, 300.0), 40)
        second = trajectory((500.0, 300.0), (100.0, 300.0), 40)
        hit = None
        for frame_index in range(len(first)):
            track_a = build_track(1, first[: frame_index + 1], fps=FPS)
            track_b = build_track(2, second[: frame_index + 1], fps=FPS)
            hit = rule.update(context(frame_index, make_geometry()), [track_a, track_b])
        assert hit is None

    def test_does_not_fire_without_speed_drop(self) -> None:
        """Overlap without any deceleration is not a collision."""
        rule = self._rule(post_contact_speed_drop_bl_s=0.1)
        # Same starting point, both moving together at a constant speed.
        first = trajectory((100.0, 300.0), (500.0, 300.0), 40)
        second = trajectory((105.0, 300.0), (505.0, 300.0), 40)
        hit = None
        for frame_index in range(len(first)):
            track_a = build_track(1, first[: frame_index + 1], fps=FPS)
            track_b = build_track(2, second[: frame_index + 1], fps=FPS)
            hit = rule.update(context(frame_index, make_geometry()), [track_a, track_b])
        assert hit is None

    def test_rule_params_record_pair_details(self) -> None:
        rule = self._rule()
        first = trajectory((100.0, 300.0), (295.0, 300.0), 20) + [(300.0, 300.0)] * 20
        second = trajectory((500.0, 300.0), (305.0, 300.0), 20) + [(300.0, 300.0)] * 20
        hit = None
        for frame_index in range(len(first)):
            track_a = build_track(1, first[: frame_index + 1], fps=FPS)
            track_b = build_track(2, second[: frame_index + 1], fps=FPS)
            hit = rule.update(context(frame_index, make_geometry()), [track_a, track_b])
            if hit is not None:
                break
        assert hit is not None
        assert hit.rule_params["iou_thresh"] == pytest.approx(0.30)
        assert hit.rule_params["observed_iou"] >= 0.30
        assert hit.rule_params["dropping_track_ids"]


class TestGetRule:
    """Lookup helper."""

    def test_get_rule_returns_match(self) -> None:
        config = load_config(validate=False)
        rules = build_rules(config, FPS, make_geometry(), strict=False)
        assert get_rule(rules, "sudden_decel") is not None
        assert get_rule(rules, "stopped_vehicle") is None
