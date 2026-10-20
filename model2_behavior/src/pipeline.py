"""End-to-end pipeline: video/RTSP -> tracked vehicles -> rules -> events.

Flow per frame:
    1. Read the frame and obtain its timestamp from the frame clock.
    2. Detect + track with YOLO (local weights, effective tracker YAML).
    3. Convert detections to observations and update the track store.
    4. Apply the rule-layer precision filter (mean detection confidence).
    5. Rescale camera geometry if the stream resolution differs from reference.
    6. Evaluate every rule and emit events for any hits.
    7. Optionally write an annotated frame.

Two run modes:
    * Full mode: detection + tracking + rules. Requires lane geometry for the
      rules that need it.
    * ``tracking_only``: detection + tracking + annotated output, with NO rules
      and NO geometry required. Used to validate track-id stability on real
      footage before any geometry is authored.

Timing comes from :mod:`model2_behavior.src.frame_clock`, never from
``frame_index / fps``, so dropped frames and variable frame rates do not corrupt
time-based thresholds.

The pipeline is deliberately independent of the plate-crop (Model 1) pipeline and
uses full-vehicle boxes only.

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import cv2

from model2_behavior.src.config import Model2Config, build_effective_tracker_yaml, resolve_device
from model2_behavior.src.detector_tracker import DetectorSettings, DetectorTracker
from model2_behavior.src.events import EventWriter, IncidentEvent
from model2_behavior.src.frame_clock import FrameClock, build_clock
from model2_behavior.src.geometry import CameraGeometry, build_camera_geometry
from model2_behavior.src.logging_setup import get_logger
from model2_behavior.src.rules import build_rules, describe_rules
from model2_behavior.src.rules.base import FrameContext, Rule, RuleHit
from model2_behavior.src.tracks import Observation, TrackStore

logger = get_logger(__name__)

CLASS_COLORS = {
    "car": (60, 200, 60),
    "motorcycle": (200, 160, 40),
    "bus": (40, 160, 240),
    "truck": (200, 60, 200),
}
DEFAULT_COLOR = (180, 180, 180)
INCIDENT_COLOR = (40, 40, 240)
LANE_COLOR = (255, 200, 0)


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Return a percentile using linear interpolation between order statistics.

    Deliberately avoids a NumPy dependency here so the metric can be computed in
    any context the pipeline runs in, and so the exact method is explicit rather
    than whatever a library's default happens to be.

    Args:
        values: Measured values.
        percentile: Percentile to compute, 0-100.

    Returns:
        The interpolated percentile, or 0.0 for an empty input.
    """
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * (percentile / 100.0)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return float(ordered[lower_index])
    weight = position - lower_index
    return float(ordered[lower_index]) * (1.0 - weight) + float(ordered[upper_index]) * weight


@dataclass
class TrackSummary:
    """Per-track lifetime statistics, used for id-stability reporting.

    Attributes:
        track_id: Tracker-assigned id.
        first_frame: First frame index observed.
        last_frame: Last frame index observed.
        first_time_s: Timestamp of the first observation.
        last_time_s: Timestamp of the last observation.
        observations: Number of observed frames.
        mean_confidence: Mean detector confidence.
        class_name: Most common class name seen.
    """

    track_id: int
    first_frame: int = 0
    last_frame: int = 0
    first_time_s: float = 0.0
    last_time_s: float = 0.0
    observations: int = 0
    mean_confidence: float = 0.0
    class_name: str = "unknown"

    @property
    def duration_s(self) -> float:
        """Track lifetime in stream seconds.

        Returns:
            ``last_time_s - first_time_s``, clamped at zero so a non-monotonic
            input cannot report a negative lifetime.
        """
        return max(0.0, self.last_time_s - self.first_time_s)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable form."""
        return {
            "track_id": self.track_id,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "first_time_s": round(self.first_time_s, 4),
            "last_time_s": round(self.last_time_s, 4),
            "duration_s": round(self.duration_s, 4),
            "observations": self.observations,
            "mean_confidence": round(self.mean_confidence, 4),
            "class_name": self.class_name,
        }


@dataclass
class PipelineResult:
    """Summary of a pipeline run.

    Attributes:
        mode: ``"full"`` or ``"tracking_only"``.
        frames_processed: Number of frames read.
        frames_with_detections: Frames in which at least one track was present.
        total_tracks: Distinct track ids observed.
        events: Emitted events.
        counts_by_type: Event counts keyed by incident type.
        elapsed_s: Wall-clock duration.
        avg_fps: Throughput including warm-up.
        avg_fps_excl_warmup: Throughput with warm-up frames excluded.
        warmup_frames: Frames excluded from ``avg_fps_excl_warmup``.
        frame_times_ms_excl_warmup: Per-frame wall-clock times (ms) with warm-up
            excluded. Retained so a p95 can be reported; a mean alone hides stalls.
        p95_frame_time_ms: 95th percentile frame time (ms) excluding warm-up.
        clip_duration_s: Stream duration processed, from the frame clock.
        clock: Frame-clock behaviour summary.
        tracks: Per-track summaries (id-stability evidence).
        median_track_lifetime_s: Median track lifetime in seconds.
        tracks_shorter_than_1s: Count of tracks lasting under one second.
        suspected_id_switches: HEURISTIC count of possible track fragmentation.
        id_switch_indicators: Deprecated alias of ``suspected_id_switches``, kept
            in the JSON so an older reader does not break.
    """

    mode: str = "full"
    frames_processed: int = 0
    frames_with_detections: int = 0
    total_tracks: int = 0
    events: List[IncidentEvent] = field(default_factory=list)
    counts_by_type: Dict[str, int] = field(default_factory=dict)
    elapsed_s: float = 0.0
    avg_fps: float = 0.0
    avg_fps_excl_warmup: float = 0.0
    warmup_frames: int = 0
    frame_times_ms_excl_warmup: List[float] = field(default_factory=list)
    p95_frame_time_ms: float = 0.0
    clip_duration_s: float = 0.0
    clock: Dict[str, Any] = field(default_factory=dict)
    tracks: List[TrackSummary] = field(default_factory=list)
    median_track_lifetime_s: float = 0.0
    tracks_shorter_than_1s: int = 0
    suspected_id_switches: int = 0
    id_switch_indicators: int = 0
    output_dir: Optional[str] = None
    events_path: Optional[str] = None
    annotated_video_path: Optional[str] = None
    effective_tracker_path: Optional[str] = None
    config_snapshot_path: Optional[str] = None
    source_fps: float = 0.0
    used_fps_source: str = "container"
    source_resolution: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "mode": self.mode,
            "frames_processed": self.frames_processed,
            "frames_with_detections": self.frames_with_detections,
            "total_tracks": self.total_tracks,
            "event_count": len(self.events),
            "counts_by_type": self.counts_by_type,
            "elapsed_s": round(self.elapsed_s, 3),
            "clip_duration_s": round(self.clip_duration_s, 3),
            "avg_fps": round(self.avg_fps, 3),
            "avg_fps_excl_warmup": round(self.avg_fps_excl_warmup, 3),
            "warmup_frames": self.warmup_frames,
            "p95_frame_time_ms": round(self.p95_frame_time_ms, 3),
            "frame_time_percentiles_ms": {
                "p50": round(_percentile(self.frame_times_ms_excl_warmup, 50), 3),
                "p95": round(self.p95_frame_time_ms, 3),
                "max": round(max(self.frame_times_ms_excl_warmup), 3)
                if self.frame_times_ms_excl_warmup
                else 0.0,
            },
            "source_fps": round(self.source_fps, 3),
            "used_fps_source": self.used_fps_source,
            "source_resolution": self.source_resolution,
            "clock": self.clock,
            "median_track_lifetime_s": round(self.median_track_lifetime_s, 3),
            "tracks_shorter_than_1s": self.tracks_shorter_than_1s,
            "suspected_id_switches": self.suspected_id_switches,
            "suspected_id_switches_note": (
                "HEURISTIC: short-lived tracks that end early and share a class with a "
                "later track. This is NOT a tracking-accuracy metric."
            ),
            "id_switch_indicators": self.id_switch_indicators,
            "tracks": [track.to_dict() for track in self.tracks],
            "output_dir": self.output_dir,
            "events_path": self.events_path,
            "annotated_video_path": self.annotated_video_path,
            "effective_tracker_path": self.effective_tracker_path,
            "config_snapshot_path": self.config_snapshot_path,
        }


def draw_annotations(
    frame: Any,
    tracks: Iterable[Any],
    lanes: Optional[List[Dict[str, Any]]] = None,
    highlight_track_ids: Optional[Iterable[int]] = None,
) -> Any:
    """Draw boxes, ids, and lane geometry onto a frame using plain OpenCV.

    Deliberately avoids Ultralytics plotting helpers so no fonts or assets are
    fetched at runtime; only ``cv2`` primitives are used.

    Args:
        frame: BGR image (modified in place and returned).
        tracks: Track states to draw.
        lanes: Lane descriptions from ``CameraGeometry.describe()["lanes"]``,
            each with ``name``, ``polygon``, and ``direction``.
        highlight_track_ids: Track ids to draw in the incident colour.

    Returns:
        The annotated frame.
    """
    import numpy as np

    highlighted = set(highlight_track_ids or [])

    for lane in lanes or []:
        polygon = lane.get("polygon")
        if polygon and len(polygon) >= 3:
            points = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=True, color=LANE_COLOR, thickness=2)
            centroid = points.reshape(-1, 2).mean(axis=0)
            cv2.putText(
                frame,
                str(lane.get("name", "lane")),
                (int(centroid[0]), int(centroid[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                LANE_COLOR,
                2,
                cv2.LINE_AA,
            )
        direction = lane.get("direction")
        if direction and len(direction) == 2:
            start = tuple(int(v) for v in direction[0])
            end = tuple(int(v) for v in direction[1])
            cv2.arrowedLine(frame, start, end, (0, 255, 255), 2, tipLength=0.15)

    for track in tracks:
        latest = track.latest
        if latest is None:
            continue
        x1, y1, x2, y2 = (int(v) for v in latest.box)
        color = (
            INCIDENT_COLOR
            if track.track_id in highlighted
            else CLASS_COLORS.get(latest.class_name, DEFAULT_COLOR)
        )
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"#{track.track_id} {latest.class_name} {latest.confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(0, y1 - text_h - baseline - 2)
        cv2.rectangle(frame, (x1, top), (x1 + text_w + 4, y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - baseline - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return frame


class BehaviorPipeline:
    """Runs detection, tracking, rules, and event emission over a source."""

    def __init__(
        self,
        config: Model2Config,
        output_dir: Path,
        fps: Optional[float] = None,
        write_video: bool = True,
        strict_rules: bool = True,
        draw: bool = True,
        tracking_only: bool = False,
        warmup_frames: int = 10,
        force_wall_clock: bool = False,
    ) -> None:
        """Prepare the pipeline.

        Args:
            config: Resolved Model 2 configuration.
            output_dir: Directory for all run artifacts.
            fps: FPS override. When None, the container FPS is used (with a
                documented fallback if the container reports none).
            write_video: Whether to write an annotated MP4.
            strict_rules: Whether missing required geometry should raise. Ignored
                in ``tracking_only`` mode, where no rules are built.
            draw: Whether to render annotations at all.
            tracking_only: Skip rules and geometry entirely; just detect, track,
                and annotate. Enables id-stability validation on raw footage.
            warmup_frames: Frames excluded from the reported throughput, since
                the first frames include CUDA/tracker initialisation.
            force_wall_clock: Use wall-clock timing even for file sources.
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.write_video = write_video
        self.strict_rules = strict_rules and not tracking_only
        self.draw = draw
        self.tracking_only = tracking_only
        self.warmup_frames = max(0, warmup_frames)
        self.force_wall_clock = force_wall_clock

        self.device = resolve_device(config.device)
        self.fps_override = fps
        self.fps = fps if fps else 0.0
        self.used_fps_source = "override" if fps else "container"

        # The effective tracker YAML needs the measured FPS, so it is written in
        # run() once the source is open. The path is fixed here.
        self.effective_tracker_path = self.output_dir / "effective_tracker.yaml"

        self.geometry = build_camera_geometry(config.rois)
        self.geometry_scaled: Optional[CameraGeometry] = None

        self.settings = DetectorSettings(
            imgsz=int(config.detector.get("imgsz", 640)),
            conf=float(config.detector.get("conf", 0.10)),
            iou=float(config.detector.get("iou", 0.45)),
            classes=tuple(config.detector.get("classes", [2, 3, 5, 7])),
            max_det=int(config.detector.get("max_det", 100)),
            device=self.device,
            vid_stride=int(config.detector.get("vid_stride", 1)),
        )
        self.detector = DetectorTracker(
            weights_path=config.weights_path,
            tracker_yaml=self._ensure_bootstrap_tracker_yaml(),
            settings=self.settings,
        )

        self.rules: List[Rule] = []

        # Snapshot every effective value so nothing is silently defaulted.
        self._snapshot: Dict[str, Any] = config.effective_snapshot()
        self._snapshot["resolved_device"] = self.device
        self._snapshot["fps_override"] = fps
        self._snapshot["strict_rules"] = self.strict_rules
        self._snapshot["tracking_only"] = tracking_only
        self._snapshot["warmup_frames"] = self.warmup_frames
        self._snapshot["class_names"] = self.detector.class_names
        self.config_snapshot_path = self.output_dir / "effective_config.json"

        self.event_writer = EventWriter(self.output_dir, camera_id=config.camera_id)
        self.track_store = TrackStore()
        self.result = PipelineResult(mode="tracking_only" if tracking_only else "full")

    def _ensure_bootstrap_tracker_yaml(self) -> Path:
        """Write a tracker YAML before the FPS is known, then refine it later.

        The final effective YAML is rewritten in :meth:`run` with the measured
        FPS. This bootstrap exists only so the detector can be constructed; the
        file handed to ``model.track()`` is always the one written in ``run``.

        Returns:
            Path to the placeholder tracker YAML.
        """
        fps_for_buffer = self.fps_override or 25.0
        self._bootstrap_fps = fps_for_buffer
        return build_effective_tracker_yaml(self.config, self.output_dir, fps_for_buffer)

    # ------------------------------------------------------------------ helpers
    def _resolve_fps(self, cap: Any) -> float:
        """Resolve FPS from override or container, with a safe fallback.

        Args:
            cap: An opened ``cv2.VideoCapture``.

        Returns:
            Positive FPS value.
        """
        if self.fps_override:
            self.used_fps_source = "override"
            return float(self.fps_override)

        reported = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if reported and reported > 0:
            self.used_fps_source = "container"
            return reported

        self.used_fps_source = "fallback_25"
        logger.warning(
            "Container did not report FPS; falling back to 25.0 fps. "
            "Pass --fps explicitly to avoid silent defaulting."
        )
        return 25.0

    def _write_snapshot(self) -> None:
        """Write the effective config snapshot, including derived frame counts."""
        self._snapshot["fps_effective"] = round(self.fps, 4)
        self._snapshot["fps_source"] = self.used_fps_source
        self._snapshot["rules_constructed"] = describe_rules(self.rules) if self.rules else []
        self._snapshot["geometry"] = (
            self.geometry_scaled.describe() if self.geometry_scaled else self.geometry.describe()
        )
        if self.tracking_only:
            self._snapshot["note"] = (
                "tracking_only mode: no rules evaluated, no geometry required, "
                "events.json will contain zero events."
            )
        self.config_snapshot_path.write_text(
            json.dumps(self._snapshot, indent=2, default=str), encoding="utf-8"
        )

    def _lane_descriptions(self) -> Optional[List[Dict[str, Any]]]:
        """Return lane descriptions for drawing, if geometry is available."""
        geometry = self.geometry_scaled or self.geometry
        if not geometry.has_lanes():
            return None
        return geometry.describe()["lanes"]

    def _snapshot_frame(self, frame: Any, event: IncidentEvent, tracks: List[Any]) -> None:
        """Write a JPEG snapshot for an event and record its path.

        Args:
            frame: Frame to annotate and save.
            event: Event being documented.
            tracks: Tracks to draw.
        """
        annotated = draw_annotations(
            frame.copy(),
            tracks,
            lanes=self._lane_descriptions(),
            highlight_track_ids=event.track_ids,
        )
        destination = self.event_writer.snapshot_path_for(event)
        try:
            cv2.imwrite(str(destination), annotated)
            event.evidence.snapshot_path = str(destination)
        except Exception:  # pragma: no cover - disk issues
            logger.exception("Failed writing snapshot to %s", destination)

    def _build_track_summaries(self) -> List[TrackSummary]:
        """Return per-track lifetime summaries for id-stability reporting.

        Returns:
            List of track summaries, ordered by track id.
        """
        summaries: List[TrackSummary] = []
        for track in sorted(self.track_store.all_tracks(), key=lambda item: item.track_id):
            observations = list(track.observations)
            if not observations:
                continue
            summaries.append(
                TrackSummary(
                    track_id=track.track_id,
                    first_frame=observations[0].frame_index,
                    last_frame=observations[-1].frame_index,
                    first_time_s=observations[0].time_s,
                    last_time_s=observations[-1].time_s,
                    observations=len(observations),
                    mean_confidence=track.mean_confidence,
                    class_name=track.class_name or "unknown",
                )
            )
        return summaries

    def _compute_track_stats(self) -> None:
        """Populate median lifetime, short-track count, and the id-switch heuristic.

        Track lifetime is ``last_time_s - first_time_s``, i.e. stream time, not
        frame counts, so it is comparable across frame rates. A track shorter
        than one second is reported separately because a pile of sub-second
        tracks is the most visible symptom of tracker fragmentation.
        """
        summaries = self.result.tracks
        if not summaries:
            self.result.median_track_lifetime_s = 0.0
            self.result.tracks_shorter_than_1s = 0
            self.result.suspected_id_switches = 0
            return

        lifetimes = sorted(summary.duration_s for summary in summaries)
        midpoint = len(lifetimes) // 2
        if len(lifetimes) % 2 == 1:
            median = lifetimes[midpoint]
        else:
            median = (lifetimes[midpoint - 1] + lifetimes[midpoint]) / 2.0
        self.result.median_track_lifetime_s = median
        self.result.tracks_shorter_than_1s = sum(1 for value in lifetimes if value < 1.0)
        self.result.suspected_id_switches = self._count_id_switch_indicators(summaries)

    @staticmethod
    def _count_id_switch_indicators(summaries: List[TrackSummary]) -> int:
        """Count heuristic indications of track fragmentation.

        A track that has only a couple of observations, ended long before the run
        finished, and shares its class with a later track is a plausible id
        switch rather than a genuine short appearance. This is a HEURISTIC to
        direct human review; it is NOT a tracking-accuracy metric and cannot be
        one without labelled identity ground truth.

        Args:
            summaries: Per-track summaries.

        Returns:
            Number of plausible fragmentation events.
        """
        if not summaries:
            return 0
        last_frame_overall = max(summary.last_frame for summary in summaries)
        span = max(1, last_frame_overall)
        indicators = 0
        ordered = sorted(summaries, key=lambda item: item.first_frame)
        for index, summary in enumerate(ordered):
            short_lived = summary.observations <= 3
            ended_early = summary.last_frame < span * 0.8
            if short_lived and ended_early:
                for later in ordered[index + 1 :]:
                    if later.class_name == summary.class_name:
                        indicators += 1
                        break
        return indicators

    # ------------------------------------------------------------------- public
    def run(self, source: str | Path, max_frames: Optional[int] = None) -> PipelineResult:
        """Process a video source end to end.

        Args:
            source: Video file path, camera index, or RTSP URL string.
            max_frames: Optional cap on processed frames (useful for smoke runs).

        Returns:
            A populated :class:`PipelineResult`.

        Raises:
            RuntimeError: If the source cannot be opened.
        """
        source_str = str(source)
        cap = cv2.VideoCapture(source_str)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source_str}")

        self.fps = self._resolve_fps(cap)
        self.result.source_fps = self.fps
        self.result.used_fps_source = self.used_fps_source
        logger.info("FPS resolved | %.4f (source: %s)", self.fps, self.used_fps_source)

        # Rewrite the tracker YAML now that the real FPS is known, and point the
        # detector at it. This is the file actually handed to model.track().
        self.effective_tracker_path = build_effective_tracker_yaml(self.config, self.output_dir, self.fps)
        self.detector.tracker_yaml = self.effective_tracker_path
        logger.info("Effective tracker YAML in use | %s", self.effective_tracker_path)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        reported_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.result.source_resolution = [frame_width, frame_height]
        logger.info(
            "Source opened | resolution=%sx%s reported_frames=%s", frame_width, frame_height, reported_total
        )

        # Geometry must be rescaled to the actual stream resolution.
        if frame_width > 0 and frame_height > 0:
            self.geometry_scaled = self.geometry.scaled_to((frame_width, frame_height))
            if self.geometry.reference_resolution and self.geometry.reference_resolution != (
                frame_width,
                frame_height,
            ):
                logger.warning(
                    "Stream resolution %sx%s differs from reference %s; geometry rescaled.",
                    frame_width,
                    frame_height,
                    tuple(self.geometry.reference_resolution),
                )
        else:
            self.geometry_scaled = self.geometry

        if self.tracking_only:
            self.rules = []
            logger.info("tracking_only mode | rules skipped, geometry not required")
        else:
            self.rules = build_rules(self.config, self.fps, self.geometry_scaled, strict=self.strict_rules)
        self._write_snapshot()

        clock: FrameClock = build_clock(
            source_str, self.fps, force_wall_clock=self.force_wall_clock
        )

        writer = None
        video_path: Optional[Path] = None
        if self.write_video and frame_width > 0 and frame_height > 0:
            video_path = self.output_dir / "annotated.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, self.fps, (frame_width, frame_height))
            if not writer.isOpened():
                logger.warning("Annotated video writer could not be opened; continuing without video output.")
                writer = None
                video_path = None

        min_conf = float(self.config.rule_layer.get("min_track_mean_conf", 0.35))
        min_age_s = float(self.config.rule_layer.get("min_track_age_s", 0.5))
        log_raw_pixels = bool(self.config.rule_layer.get("log_raw_pixels", True))
        if not self.tracking_only:
            logger.info(
                "Rule-layer filter | min_track_mean_conf=%.3f min_track_age_s=%.3f log_raw_pixels=%s",
                min_conf,
                min_age_s,
                log_raw_pixels,
            )
        else:
            # In tracking-only mode we still want track history, but no rule
            # gating, so every tracked vehicle is reported.
            min_conf = 0.0
            min_age_s = 0.0

        frame_index = 0
        started = time.perf_counter()
        warmup_end_time: Optional[float] = None
        seen_track_ids: set[int] = set()
        frame_times_ms: List[float] = []
        last_frame_time: Optional[float] = None
        last_time_s = 0.0

        try:
            while True:
                grabbed, frame = cap.read()
                if not grabbed:
                    break
                if max_frames is not None and frame_index >= max_frames:
                    break

                # Timestamp comes from the clock, never frame_index / fps.
                pts_seconds = clock.read_pts_seconds(cap) if clock.source.value == "pts" else None
                time_s = clock.next_timestamp(pts_seconds)
                last_time_s = time_s

                if frame_index == self.warmup_frames:
                    warmup_end_time = time.perf_counter()

                # Measure only the per-frame work, and only past warm-up, so the
                # reported p95 is not polluted by CUDA/tracker initialisation.
                now = time.perf_counter()
                if warmup_end_time is not None and last_frame_time is not None:
                    frame_times_ms.append((now - last_frame_time) * 1000.0)
                if frame_index >= self.warmup_frames:
                    last_frame_time = now

                raw_observations = self.detector.track_frame(frame)
                observations = [
                    Observation(
                        track_id=obs.track_id,
                        frame_index=frame_index,
                        time_s=time_s,
                        box=obs.box,
                        confidence=obs.confidence,
                        class_id=obs.class_id,
                        class_name=obs.class_name,
                    )
                    for obs in raw_observations
                ]
                if observations:
                    self.result.frames_with_detections += 1
                    seen_track_ids.update(obs.track_id for obs in observations)

                self.track_store.update(observations)
                self.track_store.prune()

                active = [
                    track
                    for track in self.track_store.active_tracks()
                    if track.mean_confidence >= min_conf and track.age_s >= min_age_s
                ]

                if log_raw_pixels and not self.tracking_only:
                    for track in active:
                        raw_speed = track.raw_speed_px_s()
                        if raw_speed is not None:
                            logger.debug(
                                "track=%s raw_speed=%.2f px/s normalized=%.4f bl/s",
                                track.track_id,
                                raw_speed,
                                track.speed_bl_s() or 0.0,
                            )

                highlight: List[int] = []
                if self.rules:
                    context = FrameContext(
                        frame_index=frame_index,
                        time_s=time_s,
                        fps=self.fps,
                        geometry=self.geometry_scaled if self.geometry_scaled else self.geometry,
                        geometry_describe=self.geometry_scaled.describe()
                        if self.geometry_scaled
                        else self.geometry.describe(),
                    )
                    for rule in self.rules:
                        try:
                            hit = rule.update(context, active)
                        except Exception:
                            logger.exception(
                                "Rule %s raised; skipping this frame for that rule", rule.name
                            )
                            continue
                        if hit is not None:
                            highlight.extend(hit.track_ids)
                            self._emit_event(rule, hit, context, frame, active)

                if writer is not None and self.draw:
                    annotated = draw_annotations(
                        frame.copy(),
                        active,
                        lanes=self._lane_descriptions(),
                        highlight_track_ids=highlight,
                    )
                    writer.write(annotated)

                frame_index += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        elapsed = time.perf_counter() - started
        self.result.frames_processed = frame_index
        self.result.total_tracks = len(seen_track_ids)
        self.result.elapsed_s = elapsed
        self.result.avg_fps = frame_index / elapsed if elapsed > 0 else 0.0
        self.result.warmup_frames = min(self.warmup_frames, frame_index)
        if warmup_end_time is not None and frame_index > self.warmup_frames:
            measured = time.perf_counter() - warmup_end_time
            measured_frames = frame_index - self.warmup_frames
            self.result.avg_fps_excl_warmup = measured_frames / measured if measured > 0 else 0.0
        else:
            self.result.avg_fps_excl_warmup = self.result.avg_fps
        self.result.clock = clock.summary()
        self.result.tracks = self._build_track_summaries()
        self.result.frame_times_ms_excl_warmup = frame_times_ms
        self.result.p95_frame_time_ms = _percentile(frame_times_ms, 95)
        self.result.clip_duration_s = last_time_s
        self._compute_track_stats()
        self.result.id_switch_indicators = self.result.suspected_id_switches
        self.result.events = self.event_writer.events
        self.result.counts_by_type = self.event_writer.counts_by_type()
        self.result.events_path = str(self.event_writer.events_path)
        self.result.annotated_video_path = str(video_path) if video_path else None
        self.result.effective_tracker_path = str(self.effective_tracker_path)
        self.result.config_snapshot_path = str(self.config_snapshot_path)

        self._write_snapshot()
        logger.info(
            "Run complete | mode=%s frames=%s tracks=%s events=%s elapsed=%.2fs "
            "avg_fps=%.2f avg_fps_excl_warmup=%.2f",
            self.result.mode,
            self.result.frames_processed,
            self.result.total_tracks,
            len(self.result.events),
            elapsed,
            self.result.avg_fps,
            self.result.avg_fps_excl_warmup,
        )
        logger.info("Event counts | %s", self.result.counts_by_type)
        logger.info("Clock | %s", self.result.clock)
        if self.result.suspected_id_switches:
            logger.warning(
                "suspected_id_switches (heuristic) flagged %s possible fragmentation(s); "
                "this is NOT a tracking-accuracy metric. Inspect tracks in run_summary.json.",
                self.result.suspected_id_switches,
            )
        return self.result

    def _emit_event(
        self,
        rule: Rule,
        hit: RuleHit,
        context: FrameContext,
        frame: Any,
        active_tracks: List[Any],
    ) -> None:
        """Build, snapshot, and record an event for a rule hit.

        Args:
            rule: Rule that fired.
            hit: The rule's decision.
            context: Frame context.
            frame: Current frame.
            active_tracks: Tracks for drawing.
        """
        by_id = {track.track_id: track for track in active_tracks}
        boxes: List[List[float]] = []
        classes: List[str] = []
        for track_id in hit.track_ids:
            track = by_id.get(track_id)
            box = track.box_at_latest() if track else None
            boxes.append([round(float(v), 2) for v in box] if box else [0.0, 0.0, 0.0, 0.0])
            classes.append(track.class_name if track and track.class_name else "unknown")

        event = self.event_writer.new_event(
            incident_type=rule.name,
            severity=rule.config.severity,
            frame_index=context.frame_index,
            track_ids=sorted(hit.track_ids),
            vehicle_classes=classes,
            bounding_boxes=boxes,
            confidence=float(hit.confidence),
            rule_params=dict(hit.rule_params),
            zone_id=hit.zone_id,
        )
        # The event timestamp comes from the frame clock, so it survives drops.
        event.timestamp = _iso_from_stream_time(context.time_s)
        self._snapshot_frame(frame, event, active_tracks)
        self.event_writer.add(event)
        logger.info(
            "EVENT %s | severity=%s tracks=%s conf=%.3f | %s",
            event.incident_type,
            event.severity,
            event.track_ids,
            event.confidence,
            hit.reason,
        )


def _iso_from_stream_time(time_s: float) -> str:
    """Convert a stream-relative time to an ISO-8601 UTC string.

    The stream origin is treated as the wall-clock moment the run started, so
    ``time_s`` is mapped onto a real timestamp. This keeps event timestamps
    anchored to the video timeline while remaining absolute.

    Args:
        time_s: Seconds from stream start, from the frame clock.

    Returns:
        ISO-8601 UTC string ending in ``Z``.
    """
    from datetime import datetime, timedelta, timezone

    # Anchor to the current UTC time minus the elapsed stream time.
    anchor = datetime.now(timezone.utc) - timedelta(seconds=time_s)
    return anchor.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class FakeDetectorTracker:
    """A detector stand-in for pipeline tests without a GPU.

    Yields a scripted sequence of frames' observations so the full pipeline can
    be exercised deterministically.

    Args:
        scripted: Mapping of frame index to a list of ``Observation`` objects.
    """

    def __init__(self, scripted: Dict[int, List[Observation]]) -> None:
        self.scripted = scripted
        self.calls = 0
        self._class_names = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    @property
    def class_names(self) -> Dict[int, str]:
        """Class names exposed for parity with the real detector."""
        return dict(self._class_names)

    def track_frame(self, frame: Any, persist: bool = True) -> List[Observation]:
        """Return the scripted observations for the current call index.

        Args:
            frame: Ignored.
            persist: Ignored.

        Returns:
            Observations for this step.
        """
        result = self.scripted.get(self.calls, [])
        self.calls += 1
        return result

    def reset_tracker(self) -> None:
        """No-op for the fake detector."""
        return None
