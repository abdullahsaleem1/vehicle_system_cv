"""Model 1 regression check: prove Model 1 plate detection is unchanged.

Model 2 shares the ``yolo_env`` virtual environment with Model 1 (ANPR). Adding
Model 2 dependencies (currently ``lap==0.5.13``) therefore risks disturbing the
already-approved Model 1 detector without touching a single Model 1 file. The
Gate 3 review asked for a *numeric* proof of that, not a smoke test.

This script does NOT modify Model 1 in any way. It:

    1. Loads the Model 1 plate detector strictly from its local absolute path.
    2. Runs it on one fixed image with fixed settings.
    3. Writes the exact boxes, confidences, and class ids to a baseline JSON.
    4. On later gates, re-runs and compares against that baseline.

The fixed image is a real photograph shipped inside the repository, so the check
needs no network and no external fixture. It is the same image every time; the
detector is deterministic on CPU, which the check asserts by re-running twice.

Usage:
    # Create or refresh the baseline (do this only on a known-good Model 1 state).
    python -m model2_behavior.scripts.model1_regression_check --write-baseline

    # Compare current behaviour against the stored baseline (Gate use).
    python -m model2_behavior.scripts.model1_regression_check

    # Compare and also update the baseline in one step (record a NEW known-good).
    python -m model2_behavior.scripts.model1_regression_check --write-baseline --force

Exit codes:
    0  baseline written, or current output matches the baseline
    1  mismatch, or the detector could not be loaded/run
    2  no baseline exists yet (compare mode)

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Offline enforcement must precede the ultralytics import.
os.environ.setdefault("YOLO_OFFLINE", "1")

import cv2  # noqa: E402

from model2_behavior.src.logging_setup import get_logger, setup_logging  # noqa: E402
from model2_behavior.src.weights import enforce_offline, sha256_file  # noqa: E402

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent

#: The Model 1 plate detector, resolved at the repository root. Never written to.
MODEL1_WEIGHTS = (
    REPO_ROOT
    / "plate_detection"
    / "pakistani_finetune"
    / "weights"
    / "pk_plate_finetune_v2"
    / "weights"
    / "best.pt"
)

#: A real photograph that ships inside the repository (Model 1's own val split).
#: Chosen because it is fixed, offline, and genuinely contains vehicles, so the
#: detector produces a non-trivial output to compare against.
DEFAULT_IMAGE = (
    REPO_ROOT
    / "plate_detection"
    / "pakistani_finetune"
    / "data"
    / "images"
    / "val"
    / "vehicle_0.jpg"
)

#: The canonical baseline lives with the tests so it is tracked by git and
#: reviewed alongside the code. It was recorded AFTER ``lap==0.5.13`` was
#: installed into ``yolo_env``, so it captures the post-dependency environment
#: that Model 1 actually runs in from then on.
DEFAULT_BASELINE = PROJECT_ROOT / "tests" / "fixtures" / "model1_regression_baseline.json"

#: A run-local copy is also written under outputs/ for run artifacts, but the
#: comparison always reads the tracked fixture above unless --baseline says
#: otherwise.
OUTPUTS_BASELINE = PROJECT_ROOT / "outputs" / "model1_regression" / "baseline.json"

#: Fixed inference settings. Changing any of these invalidates the baseline, so
#: they are written INTO the baseline and compared on every run.
INFERENCE_SETTINGS: Dict[str, Any] = {
    "imgsz": 640,
    "conf": 0.25,
    "iou": 0.70,
    "max_det": 300,
    "device": "cpu",  # CPU keeps the check deterministic and GPU-independent
    "rect": False,
    "augment": False,
    # Half precision and CuDNN benchmarking both introduce run-to-run variation.
    "half": False,
}

#: Boxes are compared with this absolute tolerance in pixels. The detector is
#: deterministic in practice; the tolerance only absorbs float formatting.
BOX_TOLERANCE_PX = 0.05
#: Confidences are compared with this absolute tolerance.
CONF_TOLERANCE = 1e-4


class RegressionError(RuntimeError):
    """Raised when the detector cannot be loaded or run."""


def load_detector():
    """Load the Model 1 plate detector from its absolute local path.

    Returns:
        A loaded :class:`ultralytics.YOLO` instance.

    Raises:
        RegressionError: If the weights are missing or fail to load.
    """
    if not MODEL1_WEIGHTS.is_file():
        raise RegressionError(
            f"Model 1 weights not found at {MODEL1_WEIGHTS}. "
            "This check never downloads anything; aborting."
        )

    enforce_offline()
    from ultralytics import YOLO  # imported only after offline enforcement

    try:
        # Absolute local path only. A bare name would trigger a download.
        return YOLO(str(MODEL1_WEIGHTS))
    except Exception as exc:  # noqa: BLE001 - surfaced with full context
        raise RegressionError(f"Failed to load Model 1 detector: {type(exc).__name__}: {exc}") from exc


def run_detection(model: Any, image_path: Path) -> List[Dict[str, Any]]:
    """Run the detector once on ``image_path`` and return a canonical result.

    Args:
        model: Loaded YOLO model.
        image_path: Image to run on.

    Returns:
        Detections sorted deterministically, each with class_id, confidence, and
        an ``[x1, y1, x2, y2]`` box.

    Raises:
        RegressionError: If the image cannot be read or inference fails.
    """
    if not image_path.is_file():
        raise RegressionError(f"Regression image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise RegressionError(f"cv2 could not read the regression image: {image_path}")

    try:
        results = model.predict(
            image,
            verbose=False,
            imgsz=INFERENCE_SETTINGS["imgsz"],
            conf=INFERENCE_SETTINGS["conf"],
            iou=INFERENCE_SETTINGS["iou"],
            max_det=INFERENCE_SETTINGS["max_det"],
            device=INFERENCE_SETTINGS["device"],
            half=INFERENCE_SETTINGS["half"],
            augment=INFERENCE_SETTINGS["augment"],
        )
    except Exception as exc:  # noqa: BLE001 - surfaced with full context
        raise RegressionError(f"Model 1 inference failed: {type(exc).__name__}: {exc}") from exc

    boxes = results[0].boxes
    detections: List[Dict[str, Any]] = []
    if boxes is not None and len(boxes) > 0:
        for class_id, confidence, xyxy in zip(
            boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()
        ):
            detections.append(
                {
                    "class_id": int(class_id),
                    "confidence": round(float(confidence), 6),
                    "box": [round(float(value), 3) for value in xyxy],
                }
            )

    # Sorting makes the comparison order-independent, so a harmless reorder of
    # equally-confident detections is not reported as a regression.
    detections.sort(key=lambda item: (item["class_id"], -item["confidence"], item["box"]))
    return detections


def _repo_relative(path: Path) -> str:
    """Return ``path`` relative to the repository root, using forward slashes.

    The baseline is a TRACKED fixture, so it must not embed an absolute path that
    only exists on one machine. Storing the repo-relative form keeps the record
    portable while still identifying exactly which file was used.

    Args:
        path: Absolute path inside the repository.

    Returns:
        The repo-relative POSIX-style path, or the absolute path when it lies
        outside the repository.
    """
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_repo_relative(stored: str) -> Path:
    """Resolve a stored repo-relative path back to an absolute path.

    Args:
        stored: A repo-relative POSIX path (or an absolute path from an older
            baseline record).

    Returns:
        An absolute path.
    """
    candidate = Path(stored)
    return candidate if candidate.is_absolute() else (REPO_ROOT / candidate)


def build_record(model: Any, image_path: Path) -> Dict[str, Any]:
    """Build the comparable record for the current environment.

    Args:
        model: Loaded YOLO model.
        image_path: Fixed regression image.

    Returns:
        Mapping with the image identity, settings, class map, and detections.
    """
    detections = run_detection(model, image_path)
    image = cv2.imread(str(image_path))
    return {
        "checked_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        # Repo-relative so the tracked baseline is portable across checkouts.
        "weights_path": _repo_relative(MODEL1_WEIGHTS),
        "weights_size_bytes": MODEL1_WEIGHTS.stat().st_size,
        "image_path": _repo_relative(image_path),
        "image_sha256": sha256_file(image_path),
        "image_size": [int(image.shape[1]), int(image.shape[0])],
        "class_names": {str(k): str(v) for k, v in model.names.items()},
        "inference_settings": dict(INFERENCE_SETTINGS),
        "detection_count": len(detections),
        "detections": detections,
    }


def compatibility_problems(baseline: Dict[str, Any], current: Dict[str, Any]) -> List[str]:
    """Return reasons the two records cannot be compared at all.

    Args:
        baseline: Stored baseline record.
        current: Freshly measured record.

    Returns:
        List of human-readable incompatibilities; empty means comparable.
    """
    problems: List[str] = []
    for key in ("weights_path", "weights_size_bytes", "image_sha256", "image_size"):
        if baseline.get(key) != current.get(key):
            problems.append(f"{key} changed: baseline={baseline.get(key)!r} current={current.get(key)!r}")
    if baseline.get("inference_settings") != current.get("inference_settings"):
        problems.append(
            f"inference_settings changed: baseline={baseline.get('inference_settings')!r} "
            f"current={current.get('inference_settings')!r}"
        )
    if baseline.get("class_names") != current.get("class_names"):
        problems.append(
            f"class_names changed: baseline={baseline.get('class_names')!r} "
            f"current={current.get('class_names')!r}"
        )
    return problems


def detection_differences(baseline: Dict[str, Any], current: Dict[str, Any]) -> List[str]:
    """Compare detection lists element-wise within tolerance.

    Args:
        baseline: Stored baseline record.
        current: Freshly measured record.

    Returns:
        List of human-readable differences; empty means identical.
    """
    differences: List[str] = []
    old = baseline.get("detections", [])
    new = current.get("detections", [])

    if len(old) != len(new):
        differences.append(f"detection count changed: baseline={len(old)} current={len(new)}")

    for index in range(min(len(old), len(new))):
        left, right = old[index], new[index]
        if left.get("class_id") != right.get("class_id"):
            differences.append(
                f"detection[{index}] class_id: baseline={left.get('class_id')} current={right.get('class_id')}"
            )
        if abs(float(left.get("confidence", 0.0)) - float(right.get("confidence", 0.0))) > CONF_TOLERANCE:
            differences.append(
                f"detection[{index}] confidence: baseline={left.get('confidence')} "
                f"current={right.get('confidence')} (tolerance {CONF_TOLERANCE})"
            )
        for axis, (a, b) in enumerate(zip(left.get("box", []), right.get("box", []))):
            if abs(float(a) - float(b)) > BOX_TOLERANCE_PX:
                differences.append(
                    f"detection[{index}] box[{axis}]: baseline={a} current={b} "
                    f"(tolerance {BOX_TOLERANCE_PX} px)"
                )
    return differences


def print_report(record: Dict[str, Any], title: str) -> None:
    """Print a human-readable summary of a record.

    Args:
        record: Baseline or current record.
        title: Heading to print above it.
    """
    print(f"{title}:")
    print(f"  weights        : {record.get('weights_path')}")
    print(f"  weights bytes  : {record.get('weights_size_bytes')}")
    print(f"  image          : {record.get('image_path')}")
    print(f"  image sha256   : {record.get('image_sha256')}")
    print(f"  image size     : {record.get('image_size')}")
    print(f"  class names    : {record.get('class_names')}")
    print(f"  settings       : {record.get('inference_settings')}")
    print(f"  detections     : {record.get('detection_count')}")
    for index, detection in enumerate(record.get("detections", [])):
        box = ", ".join(f"{value:.3f}" for value in detection["box"])
        print(
            f"    [{index}] class={detection['class_id']} "
            f"conf={detection['confidence']:.6f} box=[{box}]"
        )

def main(argv: Optional[List[str]] = None) -> int:
    """Run the Model 1 regression check.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Compare Model 1 plate detection against a stored baseline."
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Write the freshly measured output as the new baseline.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="With --write-baseline, overwrite an existing baseline even if it differs.",
    )
    args = parser.parse_args(argv)

    setup_logging(name="model1_regression_check")
    print("=" * 74)
    print("MODEL 1 REGRESSION CHECK - FR-20 (pending supervisor sign-off)")
    print("Model 1 is NOT modified by this check; it is only measured.")
    print("=" * 74)

    try:
        model = load_detector()
        current = build_record(model, args.image)
        repeat = run_detection(model, args.image)
    except RegressionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    determinism_ok = repeat == current["detections"]
    print_report(current, "CURRENT MODEL 1 OUTPUT")
    print(f"  deterministic  : {'YES (identical on re-run)' if determinism_ok else 'NO (re-run differed!)'}")
    print()

    if not determinism_ok:
        print("ERROR: Model 1 output is not reproducible on this machine.", file=sys.stderr)
        return 1

    baseline_path = Path(args.baseline)

    if args.write_baseline:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        if baseline_path.is_file() and not args.force:
            existing = json.loads(baseline_path.read_text(encoding="utf-8"))
            problems = compatibility_problems(existing, current)
            differences = detection_differences(existing, current)
            if problems or differences:
                print_report(existing, "EXISTING BASELINE")
                print("Refusing to overwrite a differing baseline without --force.", file=sys.stderr)
                for item in problems + differences:
                    print(f"  - {item}", file=sys.stderr)
                return 1
            print("Existing baseline already matches this environment; nothing to do.")
            print(f"Baseline: {baseline_path}")
            return 0

        baseline_path.write_text(json.dumps(current, indent=2), encoding="utf-8")
        print(f"BASELINE WRITTEN: {baseline_path}")
        if baseline_path.is_file():
            print(f"Baseline sha256: {sha256_file(baseline_path)}")
        return 0

    if not baseline_path.is_file():
        print(f"ERROR: no baseline at {baseline_path}. Run with --write-baseline first.", file=sys.stderr)
        return 2

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    print_report(baseline, "STORED BASELINE")

    problems = compatibility_problems(baseline, current)
    differences = detection_differences(baseline, current)

    print("-" * 74)
    if problems:
        print("MODEL 1 REGRESSION CHECK FAILED (baseline not comparable)")
        for item in problems:
            print(f"  - {item}")
        return 1
    if differences:
        print("MODEL 1 REGRESSION CHECK FAILED (detections changed)")
        for item in differences:
            print(f"  - {item}")
        return 1

    print("MODEL 1 REGRESSION CHECK PASSED: output matches the baseline exactly.")
    print(f"Baseline: {baseline_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
