"""Mock contract test: validate the incident-event contract without inference.

This is the project-convention gate that must pass BEFORE any live-inference run.
It drives the real pipeline objects (event writer, rules, geometry, drawing) with
synthetic data only, then asserts that everything written to disk satisfies the
contract the Node.js/Express backend consumes.

No GPU, no weights, no video, no network.

Usage:
    python -m model2_behavior.scripts.mock_test_contract
    python -m model2_behavior.scripts.mock_test_contract --output-dir outputs/mock_contract

Exit codes:
    0  all checks passed
    1  one or more checks failed

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from model2_behavior.src.events import (  # noqa: E402
    INCIDENT_TYPES,
    REQUIRED_EVENT_FIELDS,
    EventWriter,
    validate_event_dict,
)
from model2_behavior.src.geometry import build_camera_geometry  # noqa: E402
from model2_behavior.src.logging_setup import get_logger, setup_logging  # noqa: E402
from model2_behavior.src.pipeline import draw_annotations  # noqa: E402
from model2_behavior.src.rules import build_rules, describe_rules  # noqa: E402
from model2_behavior.src.rules.base import FrameContext, RuleHit  # noqa: E402
from model2_behavior.src.tracks import Observation, TrackState, TrackStore  # noqa: E402

logger = get_logger(__name__)

FPS = 25.0
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Synthetic geometry used only inside this script.
LANE_POLYGON = [[40.0, 240.0], [600.0, 240.0], [600.0, 470.0], [40.0, 470.0]]
LANE_DIRECTION = [100.0, 400.0, 540.0, 400.0]
LANE_NAME = "lane_1"


def mock_geometry() -> Any:
    """Build the mock camera geometry: one named lane with a direction.

    Returns:
        A :class:`CameraGeometry` with a single directional lane.
    """
    return build_camera_geometry(
        {
            "lanes": [
                {"name": LANE_NAME, "polygon": LANE_POLYGON, "direction": LANE_DIRECTION}
            ],
            "reference_resolution": [640, 480],
        }
    )


class CheckFailure(AssertionError):
    """Raised when a contract check fails."""


def check(condition: bool, message: str) -> None:
    """Assert ``condition``, raising :class:`CheckFailure` with ``message``.

    Args:
        condition: Must be True.
        message: Failure description.

    Raises:
        CheckFailure: If ``condition`` is False.
    """
    if not condition:
        raise CheckFailure(message)


def frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Return a blank BGR frame."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def synthetic_tracks() -> List[TrackState]:
    """Build a set of tracks exercising every rule's input shape.

    Returns:
        Four tracks: a stopped one, a wrong-way one, a decelerating one, and a
        pair that converge.
    """
    tracks: List[TrackState] = []

    # Track 1: stationary inside the lane.
    stopped = TrackState(track_id=1)
    for index in range(60):
        stopped.update(Observation(1, index, index / FPS, (280.0, 370.0, 340.0, 430.0), 0.85, 2, "car"))
    tracks.append(stopped)

    # Track 2: moving right-to-left (wrong way for a left-to-right lane).
    reversing = TrackState(track_id=2)
    for index in range(40):
        x = 500.0 - index * 6.0
        reversing.update(Observation(2, index, index / FPS, (x - 30, 370.0, x + 30, 430.0), 0.9, 3, "motorcycle"))
    tracks.append(reversing)

    # Track 3: fast then stopped (sudden deceleration).
    decelerating = TrackState(track_id=3)
    for index in range(40):
        x = 100.0 + min(index, 20) * 15.0
        decelerating.update(Observation(3, index, index / FPS, (x - 30, 270.0, x + 30, 330.0), 0.88, 7, "truck"))
    tracks.append(decelerating)

    return tracks


class SyntheticRule:
    """A stand-in rule object carrying the attributes the pipeline reads."""

    def __init__(self, name: str, severity: str) -> None:
        """Store the rule identity.

        Args:
            name: Rule name, used as ``incident_type``.
            severity: Severity label.
        """
        self.name = name
        self.config = type("RuleConfigLike", (), {"severity": severity})()


def check_event_contract(output_dir: Path) -> Tuple[int, List[Dict[str, Any]]]:
    """Emit one event per incident type and validate every contract rule.

    Args:
        output_dir: Directory to write artifacts into.

    Returns:
        ``(event_count, events)``.

    Raises:
        CheckFailure: If any contract rule is violated.
    """
    writer = EventWriter(output_dir, camera_id="mock_cam")
    geometry = mock_geometry()
    context = FrameContext(
        frame_index=100,
        time_s=4.0,
        fps=FPS,
        geometry=geometry,
        geometry_describe=geometry.describe(),
    )
    tracks = synthetic_tracks()
    canvas = frame()
    annotated = draw_annotations(
        canvas,
        tracks,
        lanes=geometry.describe()["lanes"],
        highlight_track_ids=[1],
    )
    check(annotated.shape == canvas.shape, "annotated frame shape changed")
    check(int(annotated.sum()) > 0, "annotated frame is empty; drawing produced nothing")

    severities = {
        "stopped_vehicle": "Medium",
        "wrong_way": "High",
        "sudden_decel": "Medium",
        "collision_proxy": "High",
    }

    for incident_type in INCIDENT_TYPES:
        hit = RuleHit(
            track_ids=[1],
            zone_id=LANE_NAME if incident_type in ("stopped_vehicle", "wrong_way") else None,
            confidence=0.85,
            rule_params={
                "fps": FPS,
                "min_consecutive_s": 3.0,
                "min_consecutive_frames": 75,
                "debounce_s": 10.0,
                "debounce_frames": 250,
                "threshold_status": "PLACEHOLDER, unvalidated until Step 8 evaluation",
            },
            reason="mock contract check",
        )
        rule = SyntheticRule(incident_type, severities[incident_type])
        by_id = {track.track_id: track for track in tracks}
        track = by_id[1]
        event = writer.new_event(
            incident_type=incident_type,
            severity=rule.config.severity,
            frame_index=context.frame_index,
            track_ids=[1],
            vehicle_classes=[track.class_name or "unknown"],
            bounding_boxes=[[round(v, 2) for v in (track.box_at_latest() or (0, 0, 0, 0))]],
            confidence=0.85,
            rule_params=dict(hit.rule_params),
            zone_id=hit.zone_id,
        )
        destination = writer.snapshot_path_for(event)
        import cv2

        cv2.imwrite(str(destination), annotated)
        event.evidence.snapshot_path = str(destination)
        writer.add(event)

    payload = json.loads(writer.events_path.read_text(encoding="utf-8"))
    events = payload.get("events", [])

    check(isinstance(events, list) and len(events) == len(INCIDENT_TYPES), "unexpected event count")
    check(payload.get("camera_id") == "mock_cam", "document camera_id mismatch")
    check(payload.get("total_events") == len(events), "total_events mismatch")

    for event in events:
        problems = validate_event_dict(event)
        check(not problems, f"event {event.get('incident_type')}: {problems}")
        check(
            set(event) == set(REQUIRED_EVENT_FIELDS),
            f"event has unexpected top-level keys: {set(event) ^ set(REQUIRED_EVENT_FIELDS)}",
        )
        snapshot_path = event["evidence"]["snapshot_path"]
        check(snapshot_path is not None, "snapshot_path must be set for Tier A")
        check(Path(snapshot_path).is_file(), f"snapshot missing on disk: {snapshot_path}")
        check(event["evidence"]["clip_path"] is None, "clip_path must be null in Tier A")
        check(event["confidence"] <= 1.0 and event["confidence"] >= 0.0, "confidence out of range")

    return len(events), events


def check_geometry_refusal() -> None:
    """Confirm rules refuse to construct without required geometry.

    Raises:
        CheckFailure: If a rule constructs without its geometry.
    """
    from model2_behavior.src.config import load_config
    from model2_behavior.src.geometry import CameraGeometry
    from model2_behavior.src.rules.base import RuleGeometryError

    config = load_config(PROJECT_ROOT / "configs" / "model2.yaml", validate=False)
    empty = CameraGeometry()

    try:
        build_rules(config, FPS, empty, strict=True)
    except RuleGeometryError as exc:
        logger.info("Refusal confirmed for strict build | %s", exc)
    else:
        raise CheckFailure("build_rules(strict=True) did not refuse missing geometry")

    partial = build_rules(config, FPS, empty, strict=False)
    names = {rule.name for rule in partial}
    check("stopped_vehicle" not in names, "stopped_vehicle must not construct without a lane")
    check("wrong_way" not in names, "wrong_way must not construct without a lane direction")
    check("sudden_decel" in names, "sudden_decel should construct (needs no geometry)")
    check("collision_proxy" in names, "collision_proxy should construct (needs no geometry)")


def check_rule_outputs() -> None:
    """Drive the four rules with synthetic tracks and validate their hits.

    Raises:
        CheckFailure: If a rule with satisfied conditions fails to fire, or a
            rule with unsatisfied conditions fires.
    """
    from model2_behavior.src.config import load_config

    config = load_config(PROJECT_ROOT / "configs" / "model2.yaml", validate=False)
    geometry = mock_geometry()
    # Override durations so the mock fires quickly; production values stay in config.
    for name, rule_config in config.rules.items():
        rule_config.min_consecutive_s = 0.1
        rule_config.debounce_s = 0.0
    config.rules["stopped_vehicle"].params["speed_thresh_bl_s"] = 0.1
    config.rules["wrong_way"].params["min_displacement_bl"] = 0.05
    config.rules["sudden_decel"].params["decel_thresh_bl_s2"] = 0.5
    config.rules["sudden_decel"].params["min_speed_before_bl_s"] = 0.1

    rules = build_rules(config, FPS, geometry, strict=False)
    descriptions = describe_rules(rules)
    check(len(descriptions) >= 1, "no rules constructed")
    for description in descriptions:
        check("min_consecutive_frames" in description, f"{description['name']} missing frame conversion")
        check("min_consecutive_s" in description, f"{description['name']} missing seconds threshold")
        check("debounce_frames" in description, f"{description['name']} missing debounce conversion")

    stopped_hit = None
    stopped_rule = next(rule for rule in rules if rule.name == "stopped_vehicle")
    for index in range(5, 40):
        track = TrackState(track_id=1)
        for step in range(index + 1):
            track.update(Observation(1, step, step / FPS, (280.0, 370.0, 340.0, 430.0), 0.85, 2, "car"))
        context = FrameContext(index, index / FPS, FPS, geometry, geometry.describe())
        stopped_hit = stopped_rule.update(context, [track])
        if stopped_hit is not None:
            break
    check(stopped_hit is not None, "stopped_vehicle did not fire for a stationary in-lane vehicle")
    check(stopped_hit.zone_id == LANE_NAME, f"stopped_vehicle zone_id must be {LANE_NAME}")

    wrong_rule = next(rule for rule in rules if rule.name == "wrong_way")
    wrong_hit = None
    for index in range(5, 40):
        track = TrackState(track_id=2)
        for step in range(index + 1):
            x = 500.0 - step * 6.0
            track.update(Observation(2, step, step / FPS, (x - 30, 370.0, x + 30, 430.0), 0.9, 3, "motorcycle"))
        context = FrameContext(index, index / FPS, FPS, geometry, geometry.describe())
        wrong_hit = wrong_rule.update(context, [track])
        if wrong_hit is not None:
            break
    check(wrong_hit is not None, "wrong_way did not fire for opposing movement")
    check(wrong_hit.zone_id == LANE_NAME, f"wrong_way zone_id must be the lane name {LANE_NAME}")

    decel_rule = next(rule for rule in rules if rule.name == "sudden_decel")
    decel_hit = None
    centers = [(100.0 + min(step, 20) * 15.0, 300.0) for step in range(40)]
    for index in range(25, 40):
        track = TrackState(track_id=3)
        for step in range(index + 1):
            cx, cy = centers[step]
            track.update(Observation(3, step, step / FPS, box(cx, cy), 0.88, 7, "truck"))
        context = FrameContext(index, index / FPS, FPS, geometry, geometry.describe())
        decel_hit = decel_rule.update(context, [track])
        if decel_hit is not None:
            break
    check(decel_hit is not None, "sudden_decel did not fire for a sharp stop")

    collision_rule = next(rule for rule in rules if rule.name == "collision_proxy")
    collision_rule.min_overlap_s = 0.2
    collision_rule.min_pre_contact_speed_bl_s = 0.1
    collision_rule.post_contact_speed_drop_bl_s = 0.2
    collision_rule.min_overlap_frames = max(1, int(round(0.2 * FPS)))
    collision_hit = None
    first = [(100.0 + step * 9.75, 300.0) for step in range(20)] + [(295.0, 300.0)] * 20
    second = [(500.0 - step * 9.75, 300.0) for step in range(20)] + [(305.0, 300.0)] * 20
    for index in range(20, 40):
        track_a = TrackState(track_id=4)
        track_b = TrackState(track_id=5)
        for step in range(index + 1):
            ax, ay = first[step]
            bx, by = second[step]
            track_a.update(Observation(4, step, step / FPS, box(ax, ay), 0.9, 2, "car"))
            track_b.update(Observation(5, step, step / FPS, box(bx, by), 0.9, 2, "car"))
        context = FrameContext(index, index / FPS, FPS, geometry, geometry.describe())
        collision_hit = collision_rule.update(context, [track_a, track_b])
        if collision_hit is not None:
            break
    check(collision_hit is not None, "collision_proxy did not fire for converge-then-stop")
    check(sorted(collision_hit.track_ids) == [4, 5], "collision_proxy must report both tracks")


def box(cx: float, cy: float, height: float = 60.0, aspect: float = 1.6) -> Tuple[float, float, float, float]:
    """Return a box centred on ``(cx, cy)``."""
    width = height * aspect
    return (cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)


def check_track_store_contract() -> None:
    """Confirm the track store maintains frame-rate-aware history.

    Raises:
        CheckFailure: If bookkeeping is wrong.
    """
    store = TrackStore()
    for index in range(10):
        store.update([Observation(1, index, index / FPS, box(100.0 + index * 5.0, 300.0), 0.9, 2, "car")])
    track = store.get(1)
    check(track is not None, "track 1 was not stored")
    check(track.speed_bl_s() is not None, "speed should be computable after 10 frames")
    check(track.age_s > 0, "track age should be positive")
    check(len(track.observations) == 10, "track history length mismatch")
    store.update([])
    check(store.get(1).missed_frames == 1, "missed frame was not recorded")  # type: ignore[union-attr]


def main(argv: List[str] | None = None) -> int:
    """Run every contract check and report results.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Validate the Model 2 incident-event contract with mock data.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "mock_contract",
        help="Directory for mock artifacts.",
    )
    args = parser.parse_args(argv)

    setup_logging(name="mock_test_contract")
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    checks: List[Tuple[str, Callable[[], Any]]] = [
        ("geometry refusal", check_geometry_refusal),
        ("track store contract", check_track_store_contract),
        ("rule outputs", check_rule_outputs),
    ]

    failures: List[Tuple[str, str]] = []
    for name, function in checks:
        try:
            function()
            print(f"PASS  {name}")
        except CheckFailure as exc:
            failures.append((name, str(exc)))
            print(f"FAIL  {name}: {exc}")
        except Exception as exc:  # unexpected errors are failures too
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")

    try:
        count, events = check_event_contract(output_dir)
        print(f"PASS  event contract ({count} events, all types)")
    except CheckFailure as exc:
        failures.append(("event contract", str(exc)))
        print(f"FAIL  event contract: {exc}")
    except Exception as exc:
        failures.append(("event contract", f"{type(exc).__name__}: {exc}"))
        print(f"ERROR event contract: {type(exc).__name__}: {exc}")

    print("-" * 70)
    if failures:
        print(f"MOCK CONTRACT TEST FAILED ({len(failures)} problem(s))")
        for name, message in failures:
            print(f"  - {name}: {message}")
        return 1

    print("MOCK CONTRACT TEST PASSED")
    print(f"Artifacts: {output_dir}")
    print(f"Events   : {output_dir / 'events.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
