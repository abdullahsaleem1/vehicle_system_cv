"""Gate 3 evidence report for Model 2, Tier A (post-correction).

Consolidates the evidence for the corrections applied after conditional Gate 2
approval:

    1. Dependency discipline: `lap` is pinned in requirements-model2.txt, and
       Model 1 still loads and runs in the shared environment (proves the pin
       did not disturb Model 1).
    2. Multiprocess event timestamps come from the frame clock (video PTS /
       wall-clock), not frame_index / fps. Timestamps are monotonic and gaps
       are reported as dropped frames.
    3. `track_buffer_s` is converted to a frame count using the measured FPS.
    4. Multi-lane geometry: each lane carries its own polygon and direction, and
       `wrong_way` is evaluated per lane.
    5. Tracking-only mode constructs without geometry and emits zero events.
    6. The event contract still validates (detection-confidence semantics,
       snapshot evidence paths).
    7. Gate 3 review corrections: the file-source clock never falls back to
       wall-clock (it degrades to the container frame clock), and lane geometry is
       validated at load time (point count, self-intersection, in-frame,
       non-zero direction, overlap detection with a deterministic tie-break).

Runs real pipeline objects with synthetic data. No GPU, no footage, no network.

Usage:
    python -m model2_behavior.scripts.gate3_report

Exit codes:
    0  report generated, all checks passed
    1  a check failed

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model2_behavior.src.config import load_config  # noqa: E402
from model2_behavior.src.geometry import build_camera_geometry  # noqa: E402
from model2_behavior.src.frame_clock import ClockSource, FrameClock, build_clock  # noqa: E402
from model2_behavior.src.logging_setup import get_logger, setup_logging  # noqa: E402
from model2_behavior.src.rules.base import FrameContext  # noqa: E402

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "model2.yaml"
REPORT_DIR = PROJECT_ROOT / "outputs" / "gate3"
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements-model2.txt"

FPS_CASES = (10.0, 25.0, 30.0)


def check_requirements_file() -> Dict[str, Any]:
    """Confirm the dependency pin file exists and names `lap` with a version.

    Returns:
        Mapping describing the pin file and the packages it pins.
    """
    exists = REQUIREMENTS_PATH.is_file()
    pinned: List[str] = []
    if exists:
        for line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                pinned.append(stripped)
    lap_pinned = any(entry.startswith("lap==") for entry in pinned)
    return {
        "path": str(REQUIREMENTS_PATH),
        "exists": exists,
        "pinned": pinned,
        "lap_pinned_with_exact_version": lap_pinned,
    }


def check_model1_still_runs() -> Dict[str, Any]:
    """Load and predict with the Model 1 plate detector in this environment.

    This proves the `lap` pin did not disturb Model 1, which shares the same
    `yolo_env`.

    Returns:
        Mapping describing the Model 1 load result.
    """
    result: Dict[str, Any] = {"attempted": True, "loaded": False, "predicted": False}
    try:
        import numpy as np
        from ultralytics import YOLO

        detector = (
            PROJECT_ROOT.parent
            / "plate_detection"
            / "pakistani_finetune"
            / "weights"
            / "pk_plate_finetune_v2"
            / "weights"
            / "best.pt"
        )
        if not detector.is_file():
            result["error"] = f"Model 1 weights not found at {detector}"
            return result
        result["weights_path"] = str(detector)
        model = YOLO(str(detector))
        result["loaded"] = True
        result["class_names"] = {int(k): str(v) for k, v in model.names.items()}
        predictions = model.predict(
            np.zeros((640, 640, 3), dtype=np.uint8), verbose=False, conf=0.25
        )
        result["predicted"] = len(predictions) == 1
    except Exception as exc:  # noqa: BLE001 - report, do not crash the report
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def check_frame_clock() -> Dict[str, Any]:
    """Exercise PTS and wall-clock timing and capture the results.

    Returns:
        Mapping of clock-name to observed behaviour.
    """

    class FakeCapture:
        """Reports scripted POS_MSEC values."""

        def __init__(self, values: List[float]) -> None:
            self.values = list(values)
            self.calls = 0

        def get(self, prop):  # noqa: D102 - cv2-compatible
            value = self.values[min(self.calls, len(self.values) - 1)]
            self.calls += 1
            return value

    # PTS case: first frame at 0.0 (valid), then a real gap from 0.08 to 0.24.
    # The gap is 0.16 s = 4 intervals; the frame at 0.24 is present, so the
    # three frames at 0.12, 0.16, and 0.20 were dropped.
    pts_clock = FrameClock(ClockSource.PTS, 25.0)
    capture = FakeCapture([0.0, 40.0, 80.0, 240.0])
    pts_times = [
        pts_clock.next_timestamp(pts_clock.read_pts_seconds(capture)) for _ in range(4)
    ]

    # Wall-clock case: live source.
    wall_clock = build_clock("rtsp://example/stream", 25.0)
    wall_times = [wall_clock.next_timestamp(None) for _ in range(3)]

    # Degradation case: a FILE source whose PTS is unusable must land on the
    # container frame clock, never on wall-clock time.
    degraded = build_clock("clip.mp4", 25.0)
    degraded_times = [degraded.next_timestamp(None) for _ in range(3)]

    # A file source must still be on PTS after a healthy first frame.
    healthy_file = build_clock("clip.mp4", 25.0)
    healthy_file.next_timestamp(0.0)
    healthy_file.next_timestamp(0.04)

    return {
        "file_source_clock": build_clock("clip.mp4", 25.0).source.value,
        "live_source_clock": build_clock("rtsp://example/stream", 25.0).source.value,
        "webcam_index_clock": build_clock("0", 25.0).source.value,
        "pts_zero_is_valid": pts_times[0] == 0.0,
        "pts_times_s": [round(value, 4) for value in pts_times],
        "pts_monotonic": pts_times == sorted(pts_times),
        "pts_dropped_frames_detected": pts_clock.dropped_total,
        "pts_dropped_frames_expected": 3,
        "wall_clock_times_s": [round(value, 4) for value in wall_times],
        "wall_clock_monotonic": wall_times == sorted(wall_times),
        "file_with_bad_pts_clock": degraded.source.value,
        "file_with_bad_pts_never_wall_clock": degraded.source is not ClockSource.WALL_CLOCK,
        "file_with_bad_pts_times_s": [round(value, 4) for value in degraded_times],
        "file_with_bad_pts_is_frame_over_fps": all(
            abs(value - index / 25.0) < 1e-9 for index, value in enumerate(degraded_times)
        ),
        "file_with_bad_pts_no_spurious_drops": degraded.dropped_total == 0,
        "degradation_is_reported": degraded.summary()["degraded_to_container_clock"] is True,
        "healthy_file_stays_on_pts": healthy_file.source is ClockSource.PTS,
    }


def check_track_buffer_conversion() -> Dict[str, Any]:
    """Convert track_buffer_s to frames at several frame rates.

    Returns:
        Mapping keyed by fps label with the seconds and frame values.
    """
    import yaml

    from model2_behavior.src.config import build_effective_tracker_yaml

    config = load_config(CONFIG_PATH, validate=False)
    buffer_s = float(config.tracker_params["track_buffer_s"])
    results: Dict[str, Any] = {
        "track_buffer_s": buffer_s,
        "by_fps": {},
    }
    for fps in FPS_CASES:
        out_dir = REPORT_DIR / f"tracker_{fps:g}fps"
        destination = build_effective_tracker_yaml(config, out_dir, fps=fps)
        merged = yaml.safe_load(destination.read_text(encoding="utf-8"))
        results["by_fps"][f"{fps:g}fps"] = {
            "track_buffer_frames": merged["track_buffer"],
            "expected": max(1, int(round(buffer_s * fps))),
            "has_seconds_key": "track_buffer_s" in merged,
        }
    return results


def check_multi_lane() -> Dict[str, Any]:
    """Build a two-lane geometry and confirm per-lane containment and direction.

    Returns:
        Mapping describing the multi-lane geometry behaviour.
    """
    geometry = build_camera_geometry(
        {
            "lanes": [
                {
                    "name": "northbound",
                    "polygon": [[100, 240], [500, 240], [500, 470], [100, 470]],
                    "direction": [300, 460, 300, 260],
                },
                {
                    "name": "southbound",
                    "polygon": [[560, 240], [960, 240], [960, 470], [560, 470]],
                    "direction": [760, 260, 760, 460],
                },
            ],
            "reference_resolution": [1280, 720],
        }
    )
    described = geometry.describe()
    return {
        "lane_count": len(geometry.lanes),
        "lane_names": [lane.name for lane in geometry.lanes],
        "lanes_with_direction": len(geometry.lanes_with_direction()),
        "left_point_matches": getattr(geometry.lane_containing((300.0, 400.0)), "name", None),
        "right_point_matches": getattr(geometry.lane_containing((760.0, 400.0)), "name", None),
        "gap_point_matches": geometry.lane_containing((530.0, 400.0)),
        "directions_are_per_lane": all(lane["direction"] for lane in described["lanes"]),
        "legacy_keys_rejected": _legacy_keys_rejected(),
        "validation": check_geometry_validation(),
    }


def check_geometry_validation() -> Dict[str, Any]:
    """Exercise load-time geometry validation and the overlap tie-break.

    Returns:
        Mapping of each validation case to whether it behaved as required.
    """
    from model2_behavior.src.geometry import GeometryError, build_camera_geometry

    def rejects(payload: Dict[str, Any]) -> bool:
        """Return True when the loader rejects ``payload``.

        Args:
            payload: ``rois`` mapping to attempt.

        Returns:
            True when a :class:`GeometryError` is raised.
        """
        try:
            build_camera_geometry(payload)
        except GeometryError:
            return True
        return False

    two_point = {"lanes": [{"name": "a", "polygon": [[0, 0], [10, 0]]}]}
    bowtie = {"lanes": [{"name": "a", "polygon": [[0, 0], [100, 100], [100, 0], [0, 100]]}]}
    off_frame = {
        "lanes": [{"name": "a", "polygon": [[0, 0], [700, 0], [700, 100], [0, 100]]}],
        "reference_resolution": [640, 480],
    }
    degenerate = {"lanes": [{"name": "a", "polygon": [[0, 0], [10, 0], [20, 0]]}]}
    zero_direction = {
        "lanes": [{"name": "a", "polygon": [[0, 0], [10, 0], [10, 10]], "direction": [5, 5, 5, 5]}]
    }
    overlap = {
        "lanes": [
            {"name": "first", "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]]},
            {"name": "second", "polygon": [[50, 0], [150, 0], [150, 100], [50, 100]]},
        ]
    }

    # Overlapping lanes are resolvable, so the loader warns instead of failing,
    # and the FIRST lane in config order must deterministically win.
    overlap_geometry = build_camera_geometry(overlap)
    overlap_wins = getattr(overlap_geometry.lane_containing((75.0, 50.0)), "name", None)

    return {
        "two_point_polygon_rejected": rejects(two_point),
        "self_intersecting_rejected": rejects(bowtie),
        "outside_frame_rejected": rejects(off_frame),
        "degenerate_rejected": rejects(degenerate),
        "zero_length_direction_rejected": rejects(zero_direction),
        "overlap_reported": overlap_geometry.overlap_problems() != [],
        "overlap_tie_break_first_lane": overlap_wins == "first",
    }


def _legacy_keys_rejected() -> bool:
    """Confirm the loader refuses the old single-lane config keys.

    Returns:
        True if both legacy keys raise.
    """
    from model2_behavior.src.geometry import GeometryError

    for key in ("driving_lane", "lane_direction"):
        try:
            build_camera_geometry({key: [[0, 0], [10, 0], [10, 10]]})
        except GeometryError:
            continue
        return False
    return True


def check_tracking_only() -> Dict[str, Any]:
    """Confirm tracking-only mode constructs without geometry and has no rules.

    Returns:
        Mapping describing the tracking-only pipeline state.
    """
    from model2_behavior.src.pipeline import BehaviorPipeline

    config = load_config(CONFIG_PATH, validate=False)
    pipeline = BehaviorPipeline(
        config,
        REPORT_DIR / "tracking_only",
        fps=25.0,
        write_video=False,
        strict_rules=True,
        draw=False,
        tracking_only=True,
    )
    pipeline.fps = 25.0
    pipeline._write_snapshot()
    snapshot = json.loads(pipeline.config_snapshot_path.read_text(encoding="utf-8"))
    return {
        "rules_constructed": len(pipeline.rules),
        "tracking_only_flag": pipeline.tracking_only,
        "snapshot_declares_tracking_only": snapshot.get("tracking_only"),
        "snapshot_note": snapshot.get("note"),
    }


def check_contract() -> Dict[str, Any]:
    """Validate the last emitted mock-contract document.

    Returns:
        Mapping describing the contract check.
    """
    from model2_behavior.src.events import validate_event_dict

    document = PROJECT_ROOT / "outputs" / "mock_contract" / "events.json"
    result: Dict[str, Any] = {
        "document": str(document),
        "document_present": document.is_file(),
        "event_count": 0,
        "all_valid": False,
        "types_covered": [],
        "problems": [],
    }
    if not document.is_file():
        result["problems"].append("mock contract document missing; run mock_test_contract first")
        return result

    payload = json.loads(document.read_text(encoding="utf-8"))
    events = payload.get("events", [])
    result["event_count"] = len(events)
    types = sorted({event.get("incident_type") for event in events})
    result["types_covered"] = types
    for index, event in enumerate(events):
        for problem in validate_event_dict(event):
            result["problems"].append(f"event[{index}]: {problem}")
    result["all_valid"] = not result["problems"] and len(events) > 0
    # Explicit confirmation of the semantics the backend depends on.
    result["confidence_is_detection_confidence"] = all(
        0.0 <= float(event.get("confidence", -1)) <= 1.0 for event in events
    )
    result["evidence_paths_recorded"] = all(
        event.get("evidence", {}).get("snapshot_path") for event in events
    )
    return result


def main() -> int:
    """Generate the Gate 3 report.

    Returns:
        Process exit code.
    """
    setup_logging(name="gate3_report")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    requirements = check_requirements_file()
    model1 = check_model1_still_runs()
    clock = check_frame_clock()
    buffer_conversion = check_track_buffer_conversion()
    multi_lane = check_multi_lane()
    tracking_only = check_tracking_only()
    contract = check_contract()

    failed: List[str] = []
    if not requirements["lap_pinned_with_exact_version"]:
        failed.append("lap is not pinned with an exact version in requirements-model2.txt")
    if not (model1["loaded"] and model1["predicted"]):
        failed.append(f"Model 1 did not load/predict: {model1.get('error', 'unknown')}")
    if not clock["pts_zero_is_valid"]:
        failed.append("frame clock rejects a valid PTS of 0.0")
    if not clock["pts_monotonic"]:
        failed.append("PTS timeline is not monotonic")
    if clock["pts_dropped_frames_detected"] != clock["pts_dropped_frames_expected"]:
        failed.append(
            f"dropped-frame detection off: got {clock['pts_dropped_frames_detected']}, "
            f"expected {clock['pts_dropped_frames_expected']}"
        )
    if not clock["wall_clock_monotonic"]:
        failed.append("wall-clock timeline is not monotonic")
    if clock["file_source_clock"] != "pts":
        failed.append(f"file source did not start on container PTS (got {clock['file_source_clock']})")
    if clock["live_source_clock"] != "wall_clock":
        failed.append(f"live source did not use wall-clock (got {clock['live_source_clock']})")
    if not clock["healthy_file_stays_on_pts"]:
        failed.append("a healthy file source left the PTS clock")
    if not clock["file_with_bad_pts_never_wall_clock"]:
        failed.append("a FILE source fell back to WALL-CLOCK timing (must never happen)")
    if clock["file_with_bad_pts_clock"] != "frame_count":
        failed.append("a FILE source did not degrade to the container frame clock")
    if not clock["file_with_bad_pts_is_frame_over_fps"]:
        failed.append("container frame clock timeline is not frame_index / container_fps")
    if not clock["file_with_bad_pts_no_spurious_drops"]:
        failed.append("degrading to the container clock manufactured spurious dropped frames")
    if not clock["degradation_is_reported"]:
        failed.append("clock degradation is not reported in the clock summary")
    for label, entry in buffer_conversion["by_fps"].items():
        if entry["track_buffer_frames"] != entry["expected"]:
            failed.append(f"track_buffer conversion wrong at {label}")
        if entry["has_seconds_key"]:
            failed.append(f"effective tracker YAML at {label} still contains track_buffer_s")
    if multi_lane["lane_count"] != 2:
        failed.append("multi-lane geometry did not parse two lanes")
    if not multi_lane["directions_are_per_lane"]:
        failed.append("lane directions are not stored per lane")
    if not multi_lane["legacy_keys_rejected"]:
        failed.append("legacy single-lane keys were not rejected")
    validation = multi_lane["validation"]
    if not validation["two_point_polygon_rejected"]:
        failed.append("a 2-point lane polygon was accepted")
    if not validation["self_intersecting_rejected"]:
        failed.append("a self-intersecting lane polygon was accepted")
    if not validation["outside_frame_rejected"]:
        failed.append("a lane polygon outside the reference frame was accepted")
    if not validation["degenerate_rejected"]:
        failed.append("a degenerate (zero-area) lane polygon was accepted")
    if not validation["zero_length_direction_rejected"]:
        failed.append("a zero-length lane direction was accepted")
    if not validation["overlap_reported"]:
        failed.append("overlapping lanes were not reported")
    if not validation["overlap_tie_break_first_lane"]:
        failed.append("overlap tie-break did not pick the first lane in config order")
    if tracking_only["rules_constructed"] != 0:
        failed.append("tracking-only mode constructed rules")
    if not tracking_only["snapshot_declares_tracking_only"]:
        failed.append("tracking-only snapshot does not declare the mode")
    if not contract["all_valid"]:
        failed.append("output contract validation failed")

    report: Dict[str, Any] = {
        "requirement": "FR-20 (pending supervisor sign-off)",
        "stage": "Tier A, Gate 3 (post-Gate-2 corrections + Gate 3 review corrections)",
        "validated_on_real_footage": False,
        "requirements": requirements,
        "model1_still_runs": model1,
        "frame_clock": clock,
        "track_buffer_conversion": buffer_conversion,
        "multi_lane": multi_lane,
        "tracking_only": tracking_only,
        "contract": contract,
        "failed_checks": failed,
        "blockers": [
            "Step 7 (real clip run) is pending a sample clip or RTSP URL.",
            "Step 8 (evaluation on labelled clips) is pending labelled data.",
        ],
    }
    report_path = REPORT_DIR / "gate3_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # ---- console summary ----------------------------------------------------
    print("=" * 74)
    print("MODEL 2, TIER A - GATE 3 REPORT (post-correction)")
    print("FR-20 (pending supervisor sign-off)")
    print("=" * 74)
    print(f"Config        : {CONFIG_PATH}")
    print()
    print("Dependency discipline:")
    print(f"  requirements file          : {requirements['path']}")
    print(f"  exists                     : {requirements['exists']}")
    print(f"  pinned                     : {requirements['pinned']}")
    print(f"  lap pinned exactly         : {requirements['lap_pinned_with_exact_version']}")
    print(f"  Model 1 loads + predicts   : {model1['loaded'] and model1['predicted']}")
    if model1.get("class_names"):
        print(f"  Model 1 classes            : {model1['class_names']}")
    print()
    print("Frame clock (timestamps not from frame_index/fps):")
    print(f"  file source clock          : {clock['file_source_clock']}")
    print(f"  live source clock          : {clock['live_source_clock']}")
    print(f"  webcam index clock         : {clock['webcam_index_clock']}")
    print(f"  PTS=0.0 accepted           : {clock['pts_zero_is_valid']}")
    print(f"  PTS timeline (s)           : {clock['pts_times_s']}")
    print(f"  PTS monotonic              : {clock['pts_monotonic']}")
    print(
        f"  dropped frames detected    : {clock['pts_dropped_frames_detected']} "
        f"(expected {clock['pts_dropped_frames_expected']})"
    )
    print(f"  wall-clock monotonic       : {clock['wall_clock_monotonic']}")
    print(f"  healthy file stays on PTS  : {clock['healthy_file_stays_on_pts']}")
    print(
        f"  file w/ bad PTS -> clock   : {clock['file_with_bad_pts_clock']} "
        f"(never wall_clock: {clock['file_with_bad_pts_never_wall_clock']})"
    )
    print(f"  file w/ bad PTS timeline   : {clock['file_with_bad_pts_times_s']} s")
    print(f"  file w/ bad PTS = f/n_fps  : {clock['file_with_bad_pts_is_frame_over_fps']}")
    print(f"  file w/ bad PTS drops      : {clock['file_with_bad_pts_no_spurious_drops']}")
    print(f"  degradation is reported    : {clock['degradation_is_reported']}")
    print()
    print(f"track_buffer_s conversion (default {buffer_conversion['track_buffer_s']} s):")
    for label, entry in buffer_conversion["by_fps"].items():
        print(
            f"  {label:>6}: {entry['track_buffer_frames']:>4} frames "
            f"(expected {entry['expected']})"
        )
    print()
    print("Multi-lane geometry:")
    print(f"  lanes                      : {multi_lane['lane_names']}")
    print(f"  lanes with direction       : {multi_lane['lanes_with_direction']}")
    print(f"  point (300,400) -> lane    : {multi_lane['left_point_matches']}")
    print(f"  point (760,400) -> lane    : {multi_lane['right_point_matches']}")
    print(f"  point in gap   -> lane     : {multi_lane['gap_point_matches']}")
    print(f"  directions per lane        : {multi_lane['directions_are_per_lane']}")
    print(f"  legacy keys rejected       : {multi_lane['legacy_keys_rejected']}")
    print("Geometry validation at load time:")
    validation = multi_lane["validation"]
    for label in (
        "two_point_polygon_rejected",
        "self_intersecting_rejected",
        "outside_frame_rejected",
        "degenerate_rejected",
        "zero_length_direction_rejected",
        "overlap_reported",
        "overlap_tie_break_first_lane",
    ):
        print(f"  {label:30s}: {validation[label]}")
    print(f"  lanes with direction       : {multi_lane['lanes_with_direction']}")
    print(f"  point (300,400) -> lane    : {multi_lane['left_point_matches']}")
    print(f"  point (760,400) -> lane    : {multi_lane['right_point_matches']}")
    print(f"  point in gap   -> lane     : {multi_lane['gap_point_matches']}")
    print(f"  directions per lane        : {multi_lane['directions_are_per_lane']}")
    print(f"  legacy keys rejected       : {multi_lane['legacy_keys_rejected']}")
    print()
    print("Tracking-only mode:")
    print(f"  rules constructed          : {tracking_only['rules_constructed']}")
    print(f"  snapshot declares mode     : {tracking_only['snapshot_declares_tracking_only']}")
    print()
    print("Output contract:")
    print(f"  document                   : {contract['document']}")
    print(f"  event count                : {contract['event_count']}")
    print(f"  types covered              : {contract['types_covered']}")
    print(f"  all valid                  : {contract['all_valid']}")
    print(f"  confidence is detection    : {contract['confidence_is_detection_confidence']}")
    print(f"  snapshot evidence recorded : {contract['evidence_paths_recorded']}")
    for problem in contract["problems"]:
        print(f"    PROBLEM: {problem}")
    print()
    print("Blockers:")
    for blocker in report["blockers"]:
        print(f"  - {blocker}")
    print()
    print(f"Report written to: {report_path}")
    print("=" * 74)
    if failed:
        print("GATE 3 REPORT: CHECKS FAILED")
        for item in failed:
            print(f"  FAIL: {item}")
        return 1
    print("GATE 3 REPORT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
