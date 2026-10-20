"""Pipeline end-to-end tests using a scripted detector (no GPU, no footage).

Exercises the full path: observations -> track store -> rule-layer precision
filter -> rule evaluation -> event emission -> JSON + snapshot artifacts.

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from model2_behavior.src.config import load_config
from model2_behavior.src.pipeline import FakeDetectorTracker, BehaviorPipeline, draw_annotations
from model2_behavior.src.tracks import Observation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_CONFIG = PROJECT_ROOT / "configs" / "model2.yaml"


def make_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Return a blank BGR frame of the given size."""
    return np.zeros((height, width, 3), dtype=np.uint8)


class TestDrawAnnotations:
    """Annotation drawing must use plain OpenCV only."""

    def test_draws_without_geometry(self) -> None:
        from model2_behavior.src.tracks import TrackState

        track = TrackState(track_id=1)
        track.update(Observation(1, 0, 0.0, (10.0, 10.0, 60.0, 70.0), 0.9, 2, "car"))
        frame = make_frame()
        annotated = draw_annotations(frame, [track])
        assert annotated.shape == frame.shape
        # Something must have been drawn.
        assert annotated.sum() > 0

    def test_draws_with_geometry(self) -> None:
        from model2_behavior.src.tracks import TrackState

        track = TrackState(track_id=1)
        track.update(Observation(1, 0, 0.0, (10.0, 10.0, 60.0, 70.0), 0.9, 2, "car"))
        frame = make_frame()
        annotated = draw_annotations(
            frame,
            [track],
            lanes=[
                {
                    "name": "lane_a",
                    "polygon": [[40.0, 240.0], [600.0, 240.0], [600.0, 470.0], [40.0, 470.0]],
                    "direction": [[100.0, 300.0], [540.0, 300.0]],
                }
            ],
            highlight_track_ids=[1],
        )
        assert annotated.sum() > 0

    def test_track_without_observations_is_skipped(self) -> None:
        from model2_behavior.src.tracks import TrackState

        frame = make_frame()
        empty = TrackState(track_id=99)
        annotated = draw_annotations(frame, [empty])
        assert annotated.sum() == 0


class TestFakeDetectorTracker:
    """The scripted detector used to drive pipeline tests."""

    def test_returns_scripted_frame(self) -> None:
        scripted = {0: [Observation(1, 0, 0.0, (0.0, 0.0, 10.0, 10.0), 0.9, 2, "car")]}
        detector = FakeDetectorTracker(scripted)
        assert len(detector.track_frame(None)) == 1
        assert detector.track_frame(None) == []

    def test_exposes_class_names(self) -> None:
        detector = FakeDetectorTracker({})
        assert detector.class_names[2] == "car"
        assert detector.class_names[7] == "truck"


class TestGeometryResolution:
    """Geometry must rescale to the actual stream resolution."""

    def test_scaled_to_stream_resolution(self) -> None:
        frame = make_frame(1280, 960)
        from model2_behavior.src.geometry import Lane, Polygon, CameraGeometry

        geometry = CameraGeometry(
            reference_resolution=(640, 480),
            lanes=[
                Lane("lane_a", Polygon([(10.0, 10.0), (100.0, 10.0), (100.0, 100.0)]))
            ],
        )
        scaled = geometry.scaled_to((frame.shape[1], frame.shape[0]))
        assert scaled.lanes[0].polygon.points[0] == (20.0, 20.0)


class TestSnapshotPathNaming:
    """Snapshot filenames must be stable and collision-resistant."""

    def test_snapshot_name_encodes_frame_and_type(self, tmp_path: Path) -> None:
        from model2_behavior.src.events import EventWriter

        writer = EventWriter(tmp_path / "run", camera_id="cam_01")
        event = writer.new_event(
            incident_type="wrong_way",
            severity="High",
            frame_index=1234,
            track_ids=[5],
            vehicle_classes=["car"],
            bounding_boxes=[[0.0, 0.0, 1.0, 1.0]],
            confidence=0.8,
        )
        path = writer.snapshot_path_for(event)
        assert path.name.startswith("00001234_wrong_way")
        assert path.suffix == ".jpg"


class TestPipelineConfigSnapshot:
    """Config snapshot must record the exact effective values in use."""

    def test_snapshot_contains_fps_and_rules(self, tmp_path: Path) -> None:
        config = load_config(REAL_CONFIG, validate=False)
        pipeline = BehaviorPipeline(
            config,
            tmp_path / "run",
            fps=25.0,
            write_video=False,
            strict_rules=False,
            draw=False,
        )
        pipeline.fps = 25.0
        pipeline.rules = []
        pipeline._write_snapshot()
        snapshot = json.loads(pipeline.config_snapshot_path.read_text(encoding="utf-8"))
        assert snapshot["fps_effective"] == pytest.approx(25.0)
        assert snapshot["fps_source"] == "override"
        assert snapshot["detector"]["conf"] == pytest.approx(0.10)
        assert "rules" in snapshot

    def test_effective_tracker_written_to_run_dir(self, tmp_path: Path) -> None:
        config = load_config(REAL_CONFIG, validate=False)
        run_dir = tmp_path / "run"
        BehaviorPipeline(
            config,
            run_dir,
            fps=25.0,
            write_video=False,
            strict_rules=False,
            draw=False,
        )
        assert (run_dir / "effective_tracker.yaml").is_file()
        assert (run_dir / "snapshots").is_dir()


class TestPipelineEventEmission:
    """A rule hit must produce a contract-valid event on disk."""

    def test_emitting_event_writes_json_and_snapshot(self, tmp_path: Path) -> None:
        from model2_behavior.src.events import validate_event_dict
        from model2_behavior.src.geometry import build_camera_geometry
        from model2_behavior.src.logging_setup import get_logger
        from model2_behavior.src.rules.base import FrameContext, RuleHit
        from model2_behavior.src.tracks import TrackState

        config = load_config(REAL_CONFIG, validate=False)
        run_dir = tmp_path / "run"
        pipeline = BehaviorPipeline(
            config,
            run_dir,
            fps=25.0,
            write_video=False,
            strict_rules=False,
            draw=False,
        )
        pipeline.fps = 25.0
        pipeline.geometry_scaled = build_camera_geometry({})

        track = TrackState(track_id=3)
        track.update(Observation(3, 5, 0.2, (100.0, 200.0, 160.0, 260.0), 0.9, 2, "car"))

        class DummyRule:
            name = "sudden_decel"
            config = type("C", (), {"severity": "Medium"})()

        hit = RuleHit(
            track_ids=[3],
            zone_id=None,
            confidence=0.9,
            rule_params={"fps": 25.0},
            reason="synthetic",
        )
        context = FrameContext(
            frame_index=5,
            time_s=0.2,
            fps=25.0,
            geometry=pipeline.geometry_scaled,
            geometry_describe={},
        )
        pipeline._emit_event(DummyRule(), hit, context, make_frame(), [track])

        events = json.loads(pipeline.event_writer.events_path.read_text(encoding="utf-8"))["events"]
        assert len(events) == 1
        payload = events[0]
        assert validate_event_dict(payload) == []
        assert payload["incident_type"] == "sudden_decel"
        assert payload["track_ids"] == [3]
        assert payload["vehicle_classes"] == ["car"]
        assert payload["evidence"]["snapshot_path"] is not None
        # The snapshot file must actually exist on disk.
        assert Path(payload["evidence"]["snapshot_path"]).is_file()


class TestRuleLayerPrecisionFilter:
    """Low-confidence tracks must be excluded before rules run."""

    def test_low_confidence_track_is_filtered(self) -> None:
        config = load_config(REAL_CONFIG, validate=False)
        min_conf = config.rule_layer["min_track_mean_conf"]
        from model2_behavior.src.tracks import TrackState

        track = TrackState(track_id=1)
        track.update(Observation(1, 0, 0.0, (0.0, 0.0, 10.0, 10.0), 0.1, 2, "car"))
        assert track.mean_confidence < min_conf

    def test_high_confidence_track_passes(self) -> None:
        config = load_config(REAL_CONFIG, validate=False)
        min_conf = config.rule_layer["min_track_mean_conf"]
        from model2_behavior.src.tracks import TrackState

        track = TrackState(track_id=1)
        track.update(Observation(1, 0, 0.0, (0.0, 0.0, 10.0, 10.0), 0.9, 2, "car"))
        assert track.mean_confidence >= min_conf


class TestEventTimestampFromClock:
    """Event timestamps come from the frame clock, not frame_index / fps."""

    def test_timestamp_is_absolute_iso(self) -> None:
        from model2_behavior.src.pipeline import _iso_from_stream_time

        value = _iso_from_stream_time(12.5)
        assert value.endswith("Z")
        # ISO-8601 with milliseconds.
        assert "T" in value
        assert len(value.split(".")[-1]) == 4  # "123Z"

    def test_zero_stream_time_is_anchor(self) -> None:
        from model2_behavior.src.pipeline import _iso_from_stream_time

        assert _iso_from_stream_time(0.0).endswith("Z")

    def test_event_carries_its_stream_timestamp(self, tmp_path: Path) -> None:
        """_emit_event must overwrite the event timestamp from context.time_s."""
        from model2_behavior.src.geometry import build_camera_geometry
        from model2_behavior.src.rules.base import FrameContext, RuleHit
        from model2_behavior.src.tracks import TrackState

        config = load_config(REAL_CONFIG, validate=False)
        pipeline = BehaviorPipeline(
            config, tmp_path / "run", fps=25.0, write_video=False, strict_rules=False, draw=False
        )
        pipeline.fps = 25.0
        pipeline.geometry_scaled = build_camera_geometry({})

        track = TrackState(track_id=1)
        track.update(Observation(1, 3, 0.12, (0.0, 0.0, 10.0, 10.0), 0.9, 2, "car"))

        class DummyRule:
            name = "sudden_decel"
            config = type("C", (), {"severity": "Medium"})()

        hit = RuleHit(track_ids=[1], zone_id=None, confidence=0.9, rule_params={}, reason="t")
        # Frame index and time_s disagree on purpose: 999 frames at 25 fps would
        # be ~40 s, but the clock says 2.0 s. The clock must win.
        context = FrameContext(
            frame_index=999, time_s=2.0, fps=25.0, geometry=pipeline.geometry_scaled,
            geometry_describe={},
        )
        pipeline._emit_event(DummyRule(), hit, context, make_frame(), [track])
        events = json.loads(pipeline.event_writer.events_path.read_text(encoding="utf-8"))["events"]
        assert len(events) == 1
        # The recorded timestamp anchors to now - time_s; verify by round-tripping.
        from datetime import datetime, timezone

        stamp = datetime.strptime(events[0]["timestamp"], "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=timezone.utc
        )
        age = (datetime.now(timezone.utc) - stamp).total_seconds()
        assert 1.5 < age < 3.0  # ~2.0 s of stream time, not 40 s.


class TestTrackingOnlyMode:
    """Tracking-only runs must need no geometry and emit no events."""

    def test_tracking_only_builds_without_geometry(self, tmp_path: Path) -> None:
        config = load_config(REAL_CONFIG, validate=False)
        pipeline = BehaviorPipeline(
            config,
            tmp_path / "run",
            fps=25.0,
            write_video=False,
            strict_rules=True,  # must be ignored in tracking-only mode
            draw=False,
            tracking_only=True,
        )
        assert pipeline.rules == []
        assert pipeline.tracking_only is True

    def test_tracking_only_snapshot_declares_no_rules(self, tmp_path: Path) -> None:
        config = load_config(REAL_CONFIG, validate=False)
        pipeline = BehaviorPipeline(
            config, tmp_path / "run", fps=25.0, write_video=False, draw=False, tracking_only=True
        )
        pipeline.fps = 25.0
        pipeline._write_snapshot()
        snapshot = json.loads(pipeline.config_snapshot_path.read_text(encoding="utf-8"))
        assert snapshot["tracking_only"] is True
        assert snapshot["rules_constructed"] == []
        assert "tracking_only mode" in snapshot["note"]

    def test_id_switch_heuristic_flags_fragmentation(self) -> None:
        from model2_behavior.src.pipeline import TrackSummary

        # Track 1 dies early after 2 observations; track 2 of the same class
        # appears much later: a plausible id switch.
        summaries = [
            TrackSummary(track_id=1, first_frame=0, last_frame=1, observations=2, class_name="car"),
            TrackSummary(
                track_id=2, first_frame=90, last_frame=99, observations=10, class_name="car"
            ),
        ]
        assert BehaviorPipeline._count_id_switch_indicators(summaries) == 1

    def test_id_switch_heuristic_ignores_healthy_tracks(self) -> None:
        from model2_behavior.src.pipeline import TrackSummary

        summaries = [
            TrackSummary(
                track_id=1, first_frame=0, last_frame=95, observations=96, class_name="car"
            ),
        ]
        assert BehaviorPipeline._count_id_switch_indicators(summaries) == 0


class TestFrameTimePercentile:
    """Percentile helper backing the reported p95 frame time."""

    def test_empty_input_is_zero(self) -> None:
        from model2_behavior.src.pipeline import _percentile

        assert _percentile([], 95) == 0.0

    def test_single_value_is_returned(self) -> None:
        from model2_behavior.src.pipeline import _percentile

        assert _percentile([12.5], 95) == pytest.approx(12.5)

    def test_median_of_odd_count_is_the_middle_value(self) -> None:
        from model2_behavior.src.pipeline import _percentile

        assert _percentile([1.0, 2.0, 3.0], 50) == pytest.approx(2.0)

    def test_median_of_even_count_interpolates(self) -> None:
        from model2_behavior.src.pipeline import _percentile

        assert _percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)

    def test_p95_of_twenty_values(self) -> None:
        from model2_behavior.src.pipeline import _percentile

        # 1..20: position = 19 * 0.95 = 18.05 -> between 19 and 20.
        assert _percentile(list(range(1, 21)), 95) == pytest.approx(19.05)

    def test_p95_is_not_the_max(self) -> None:
        from model2_behavior.src.pipeline import _percentile

        values = [10.0] * 99 + [1000.0]
        assert _percentile(values, 95) == pytest.approx(10.0)


class TestTrackStats:
    """Lifetime statistics: how long tracks live, and how many are very short."""

    def summary(self, track_id: int, first_s: float, last_s: float, observations: int = 10):
        """Build a TrackSummary with the given stream times.

        Args:
            track_id: Track id.
            first_s: First observation time.
            last_s: Last observation time.
            observations: Observation count.

        Returns:
            The summary.
        """
        from model2_behavior.src.pipeline import TrackSummary

        return TrackSummary(
            track_id=track_id,
            first_frame=0,
            last_frame=observations,
            first_time_s=first_s,
            last_time_s=last_s,
            observations=observations,
            mean_confidence=0.8,
            class_name="car",
        )

    def compute(self, pipeline, summaries):
        """Run ``_compute_track_stats`` against a fake result.

        Args:
            pipeline: A BehaviorPipeline (only ``result`` is used).
            summaries: Track summaries to score.

        Returns:
            The pipeline result.
        """
        pipeline.result.tracks = list(summaries)
        pipeline._compute_track_stats()
        return pipeline.result

    def test_median_lifetime_of_odd_count(self, tmp_path: Path) -> None:
        config = load_config(REAL_CONFIG, validate=False)
        pipeline = BehaviorPipeline(
            config, tmp_path / "run", fps=25.0, write_video=False, draw=False, tracking_only=True
        )
        result = self.compute(
            pipeline,
            [self.summary(1, 0.0, 1.0), self.summary(2, 0.0, 5.0), self.summary(3, 0.0, 9.0)],
        )
        assert result.median_track_lifetime_s == pytest.approx(5.0)

    def test_median_lifetime_of_even_count_interpolates(self, tmp_path: Path) -> None:
        config = load_config(REAL_CONFIG, validate=False)
        pipeline = BehaviorPipeline(
            config, tmp_path / "run", fps=25.0, write_video=False, draw=False, tracking_only=True
        )
        result = self.compute(
            pipeline, [self.summary(1, 0.0, 2.0), self.summary(2, 0.0, 4.0)]
        )
        assert result.median_track_lifetime_s == pytest.approx(3.0)

    def test_counts_tracks_shorter_than_one_second(self, tmp_path: Path) -> None:
        config = load_config(REAL_CONFIG, validate=False)
        pipeline = BehaviorPipeline(
            config, tmp_path / "run", fps=25.0, write_video=False, draw=False, tracking_only=True
        )
        result = self.compute(
            pipeline,
            [
                self.summary(1, 0.0, 0.5),  # short
                self.summary(2, 0.0, 0.99),  # short (just under)
                self.summary(3, 0.0, 1.0),  # exactly 1 s: NOT short
                self.summary(4, 0.0, 4.0),
            ],
        )
        assert result.tracks_shorter_than_1s == 2

    def test_no_tracks_yields_zeros(self, tmp_path: Path) -> None:
        config = load_config(REAL_CONFIG, validate=False)
        pipeline = BehaviorPipeline(
            config, tmp_path / "run", fps=25.0, write_video=False, draw=False, tracking_only=True
        )
        result = self.compute(pipeline, [])
        assert result.median_track_lifetime_s == 0.0
        assert result.tracks_shorter_than_1s == 0
        assert result.suspected_id_switches == 0

    def test_summary_exposes_renamed_metric_and_note(self, tmp_path: Path) -> None:
        config = load_config(REAL_CONFIG, validate=False)
        pipeline = BehaviorPipeline(
            config, tmp_path / "run", fps=25.0, write_video=False, draw=False, tracking_only=True
        )
        self.compute(pipeline, [self.summary(1, 0.0, 3.0)])
        payload = pipeline.result.to_dict()
        assert "suspected_id_switches" in payload
        assert "NOT a tracking-accuracy metric" in payload["suspected_id_switches_note"]
        # The deprecated alias is still present so an older reader keeps working.
        assert payload["id_switch_indicators"] == payload["suspected_id_switches"]

    def test_summary_reports_frame_time_percentiles(self, tmp_path: Path) -> None:
        config = load_config(REAL_CONFIG, validate=False)
        pipeline = BehaviorPipeline(
            config, tmp_path / "run", fps=25.0, write_video=False, draw=False, tracking_only=True
        )
        self.compute(pipeline, [self.summary(1, 0.0, 1.0)])
        payload = pipeline.result.to_dict()
        assert set(payload["frame_time_percentiles_ms"]) == {"p50", "p95", "max"}

    def test_summary_is_json_serialisable(self, tmp_path: Path) -> None:
        config = load_config(REAL_CONFIG, validate=False)
        pipeline = BehaviorPipeline(
            config, tmp_path / "run", fps=25.0, write_video=False, draw=False, tracking_only=True
        )
        self.compute(pipeline, [self.summary(1, 0.0, 1.0)])
        json.dumps(pipeline.result.to_dict())
