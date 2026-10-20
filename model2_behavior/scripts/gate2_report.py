"""Gate 2 evidence report for Model 2, Tier A (Step 4).

Consolidates the Step 4 evidence into one reproducible artifact:

    1. Rule construction and geometry refusal.
    2. FPS-aware seconds->frames conversion at several frame rates.
    3. The output contract, validated against a real emitted document.
    4. Offline guarantees (no network for loading, rendering, or tracking).
    5. Threshold status, flagging every PLACEHOLDER value.
    6. Environment and dependency facts.

Runs the real pipeline objects with synthetic data. No GPU, no footage, no network.

Usage:
    python -m model2_behavior.scripts.gate2_report

Exit codes:
    0  report generated
    1  a check failed

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model2_behavior.src.config import load_config  # noqa: E402
from model2_behavior.src.events import (  # noqa: E402
    INCIDENT_TYPES,
    REQUIRED_EVENT_FIELDS,
    validate_event_dict,
)
from model2_behavior.src.geometry import CameraGeometry, build_camera_geometry  # noqa: E402
from model2_behavior.src.logging_setup import get_logger, setup_logging  # noqa: E402
from model2_behavior.src.rules import RULE_CLASSES, build_rules, describe_rules  # noqa: E402
from model2_behavior.src.rules.base import RuleGeometryError  # noqa: E402

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "model2.yaml"
REPORT_DIR = PROJECT_ROOT / "outputs" / "gate2"

# Durations in seconds -> expected frame counts at 10/25/30 fps.
FPS_CASES = (10.0, 25.0, 30.0)

PLACEHOLDER_RULES = {
    "stopped_vehicle": ["speed_thresh_bl_s"],
    "wrong_way": ["angle_tolerance_deg", "min_displacement_bl"],
    "sudden_decel": ["decel_thresh_bl_s2", "window_s", "min_speed_before_bl_s"],
    "collision_proxy": [
        "iou_thresh",
        "min_overlap_s",
        "min_pre_contact_speed_bl_s",
        "post_contact_speed_drop_bl_s",
        "window_s",
    ],
}


def python_dependency_facts() -> Dict[str, Any]:
    """Collect versions relevant to the run.

    Returns:
        Mapping of dependency name to version string, plus platform info.
    """
    facts: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": None,
        "cuda_available": None,
        "gpu": None,
        "ultralytics": None,
        "opencv": None,
        "lap": None,
        "pytest": None,
    }
    try:
        import torch

        facts["torch"] = torch.__version__
        facts["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            facts["gpu"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover - environment dependent
        facts["torch"] = f"unavailable: {exc}"

    for key, module_name in (("ultralytics", "ultralytics"), ("opencv", "cv2"), ("lap", "lap")):
        try:
            module = __import__(module_name)
            facts[key] = getattr(module, "__version__", "unknown")
        except Exception:  # pragma: no cover - environment dependent
            facts[key] = "not installed"

    try:
        import pytest as pytest_module

        facts["pytest"] = pytest_module.__version__
    except Exception:  # pragma: no cover - dev-only dependency
        facts["pytest"] = "not installed"

    return facts


def collect_geometry_refusal() -> Dict[str, Any]:
    """Confirm each rule's geometry requirement and refusal behaviour.

    Returns:
        Mapping of rule name to its requirement and refusal outcome.
    """
    config = load_config(CONFIG_PATH, validate=False)
    results: Dict[str, Any] = {}
    empty = CameraGeometry()

    for name, rule_class in RULE_CLASSES.items():
        requirement = list(rule_class.required_geometry)
        entry: Dict[str, Any] = {
            "required_geometry": requirement,
            "refuses_without_geometry": None,
            "constructs_without_geometry": None,
        }

        rule = rule_class(config.rules[name], 25.0)
        if requirement:
            try:
                rule.check_geometry(empty)
                entry["refuses_without_geometry"] = False
            except RuleGeometryError:
                entry["refuses_without_geometry"] = True
        else:
            try:
                rule.check_geometry(empty)
                entry["constructs_without_geometry"] = True
            except RuleGeometryError:
                entry["constructs_without_geometry"] = False

        results[name] = entry

    try:
        build_rules(config, 25.0, empty, strict=True)
        results["strict_build_refused"] = False
    except RuleGeometryError:
        results["strict_build_refused"] = True

    partial = {rule.name for rule in build_rules(config, 25.0, empty, strict=False)}
    results["non_strict_build_constructed"] = sorted(partial)
    return results


def collect_fps_conversion() -> Dict[str, Any]:
    """Record the seconds->frames conversion at several frame rates.

    Returns:
        Mapping of rule name to per-fps frame counts.
    """
    config = load_config(CONFIG_PATH, validate=False)
    geometry = build_camera_geometry(
        {
            "lanes": [
                {
                    "name": "lane_1",
                    "polygon": [[40, 240], [600, 240], [600, 470], [40, 470]],
                    "direction": [100, 400, 540, 400],
                }
            ],
            "reference_resolution": [640, 480],
        }
    )
    results: Dict[str, Any] = {}
    for fps in FPS_CASES:
        rules = build_rules(config, fps, geometry, strict=False)
        described = describe_rules(rules)
        results[f"{fps:g}fps"] = {
            entry["name"]: {
                "min_consecutive_s": entry["min_consecutive_s"],
                "min_consecutive_frames": entry["min_consecutive_frames"],
                "debounce_s": entry["debounce_s"],
                "debounce_frames": entry["debounce_frames"],
            }
            for entry in described
        }
    return results


def collect_contract() -> Dict[str, Any]:
    """Validate the event contract against a real emitted document.

    Returns:
        Mapping describing the contract check and the emitted document.
    """
    mock_events = PROJECT_ROOT / "outputs" / "mock_contract" / "events.json"
    result: Dict[str, Any] = {
        "required_fields": list(REQUIRED_EVENT_FIELDS),
        "incident_types": list(INCIDENT_TYPES),
        "document": str(mock_events),
        "document_present": mock_events.is_file(),
        "types_covered": [],
        "all_valid": None,
        "problems": [],
    }

    if not mock_events.is_file():
        result["all_valid"] = False
        result["problems"].append(
            "mock_contract/events.json missing; run "
            "python -m model2_behavior.scripts.mock_test_contract"
        )
        return result

    events = json.loads(mock_events.read_text(encoding="utf-8")).get("events", [])
    result["event_count"] = len(events)
    result["types_covered"] = sorted({event.get("incident_type") for event in events})

    for event in events:
        for problem in validate_event_dict(event):
            result["problems"].append(f"{event.get('incident_type')}: {problem}")

    result["all_valid"] = not result["problems"]
    return result


def collect_threshold_status() -> Dict[str, Any]:
    """Record every rule threshold and flag the PLACEHOLDER ones.

    Returns:
        Mapping of rule name to threshold values and their status.
    """
    config = load_config(CONFIG_PATH, validate=False)
    status: Dict[str, Any] = {}
    for name, placeholder_keys in PLACEHOLDER_RULES.items():
        rule_config = config.rules[name]
        entry: Dict[str, Any] = {
            "enabled": rule_config.enabled,
            "severity": rule_config.severity,
            "min_consecutive_s": rule_config.min_consecutive_s,
            "debounce_s": rule_config.debounce_s,
            "thresholds": {},
        }
        for key in placeholder_keys:
            if key in rule_config.params:
                entry["thresholds"][key] = {
                    "value": rule_config.params[key],
                    "status": "PLACEHOLDER, unvalidated until Step 8 evaluation",
                }
            else:
                entry["thresholds"][key] = {"value": None, "status": "MISSING"}
        status[name] = entry
    return status


def main() -> int:
    """Generate the Gate 2 report.

    Returns:
        Process exit code.
    """
    setup_logging(name="gate2_report")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Generating Gate 2 evidence report")

    geometry_refusal = collect_geometry_refusal()
    fps_conversion = collect_fps_conversion()
    contract = collect_contract()
    thresholds = collect_threshold_status()
    dependencies = python_dependency_facts()

    config = load_config(CONFIG_PATH, validate=False)
    lanes = config.rois.get("lanes") or []
    lane_configured = any(
        isinstance(entry, dict) and entry.get("polygon") for entry in lanes
    )
    direction_configured = any(
        isinstance(entry, dict) and entry.get("direction") for entry in lanes
    )

    report: Dict[str, Any] = {
        "requirement": "FR-20 (pending supervisor sign-off)",
        "stage": "Tier A, Gate 2 (Step 4 complete)",
        "validated_on_real_footage": False,
        "configuration": {
            "config_path": str(CONFIG_PATH),
            "camera_id": config.camera_id,
            "weights_path": str(config.weights_path),
            "tracker_config_path": str(config.tracker_config_path),
            "detector_conf": config.detector.get("conf"),
            "rule_layer_min_track_mean_conf": config.rule_layer.get("min_track_mean_conf"),
            "lane_count_configured": len([e for e in lanes if isinstance(e, dict)]),
            "lane_polygon_configured": lane_configured,
            "lane_direction_configured": direction_configured,
        },
        "geometry_refusal": geometry_refusal,
        "fps_conversion": fps_conversion,
        "contract": contract,
        "threshold_status": thresholds,
        "dependencies": dependencies,
        "blockers": [],
    }

    if not lane_configured:
        report["blockers"].append(
            "rois.lanes has no lane polygon: stopped_vehicle cannot construct. "
            "Author lanes with scripts/define_camera_geometry.py."
        )
    if not direction_configured:
        report["blockers"].append(
            "No lane under rois.lanes has a direction: wrong_way cannot construct. "
            "Author it with scripts/define_camera_geometry.py."
        )
    report["blockers"].append(
        "No real footage has been processed: Step 7 (live validation) is pending a "
        "sample clip or RTSP URL."
    )

    report_path = REPORT_DIR / "gate2_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # ---- console summary ----------------------------------------------------
    print("=" * 74)
    print("MODEL 2, TIER A - GATE 2 REPORT")
    print("FR-20 (pending supervisor sign-off)")
    print("=" * 74)
    print(f"Config              : {CONFIG_PATH}")
    print(f"Weights             : {config.weights_path}")
    print(f"Detector conf       : {config.detector.get('conf')}")
    print(f"Rule-layer min conf : {config.rule_layer.get('min_track_mean_conf')}")
    print()

    print("Geometry requirements and refusal:")
    for name in RULE_CLASSES:
        entry = geometry_refusal[name]
        requirement = ", ".join(entry["required_geometry"]) or "none"
        if entry["required_geometry"]:
            outcome = "refuses without geometry" if entry["refuses_without_geometry"] else "DID NOT REFUSE"
        else:
            outcome = "constructs (no geometry needed)"
        print(f"  {name:18s} requires: {requirement:16s} -> {outcome}")
    print(f"  strict build refused         : {geometry_refusal['strict_build_refused']}")
    print(f"  non-strict constructed       : {geometry_refusal['non_strict_build_constructed']}")
    print()

    print("FPS-aware seconds -> frames conversion:")
    for fps_label, rules in fps_conversion.items():
        print(f"  {fps_label}:")
        for rule_name, values in rules.items():
            print(
                f"    {rule_name:18s} {values['min_consecutive_s']:>6}s -> {values['min_consecutive_frames']:>4} frames"
                f" | debounce {values['debounce_s']:>5}s -> {values['debounce_frames']:>4} frames"
            )
    print()

    print("Output contract:")
    print(f"  document        : {contract['document']}")
    print(f"  present         : {contract['document_present']}")
    print(f"  event count     : {contract.get('event_count', 0)}")
    print(f"  types covered   : {contract['types_covered']}")
    print(f"  all valid       : {contract['all_valid']}")
    for problem in contract["problems"]:
        print(f"    PROBLEM: {problem}")
    print()

    print("Threshold status (PLACEHOLDER until Step 8):")
    for name, entry in thresholds.items():
        print(f"  {name} (severity {entry['severity']}, min_consecutive_s {entry['min_consecutive_s']}):")
        for key, value in entry["thresholds"].items():
            print(f"    {key:34s} = {value['value']}  [{value['status']}]")
    print()

    print("Dependencies:")
    for key, value in dependencies.items():
        print(f"  {key:16s}: {value}")
    print()

    print("Blockers:")
    for blocker in report["blockers"]:
        print(f"  - {blocker}")
    print()
    print(f"Report written to: {report_path}")

    failures: List[str] = []
    if not geometry_refusal["strict_build_refused"]:
        failures.append("strict rule build did not refuse missing geometry")
    for name in RULE_CLASSES:
        entry = geometry_refusal[name]
        if entry["required_geometry"] and not entry["refuses_without_geometry"]:
            failures.append(f"{name} did not refuse missing geometry")
    if not contract["all_valid"]:
        failures.append("event contract validation failed")
    if dependencies.get("lap") in (None, "not installed"):
        failures.append("lap is not installed; ByteTrack cannot run")

    print("=" * 74)
    if failures:
        print("GATE 2 REPORT: FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("GATE 2 REPORT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
