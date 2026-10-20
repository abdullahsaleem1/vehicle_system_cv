"""Gate 1 report for Model 2 (Tier A).

Produces the evidence required to review before implementation continues:
    1. Environment summary (Python, CUDA, packages, OpenCV video smoke test).
    2. Weight verification (path, size vs official, SHA256, class map).
    3. The merged effective configuration, including the effective tracker YAML
       that will be handed to ``model.track()``.

Usage:
    python -m model2_behavior.scripts.gate1_report

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os  # noqa: E402

os.environ.setdefault("YOLO_OFFLINE", "1")

from model2_behavior.scripts.verify_environment import (  # noqa: E402
    collect_environment,
    missing_packages,
)
from model2_behavior.src.config import (  # noqa: E402
    ConfigError,
    build_effective_tracker_yaml,
    load_config,
    resolve_device,
)
from model2_behavior.src.logging_setup import setup_logging  # noqa: E402
from model2_behavior.src.weights import (  # noqa: E402
    TARGET_CLASS_IDS,
    enforce_offline,
    resolve_weights_path,
    sha256_file,
    verify_zip_integrity,
)

OFFICIAL_ASSET_SIZE_BYTES = 40_684_120


def _section(title: str) -> None:
    """Print a section banner.

    Args:
        title: Section heading.
    """
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def main() -> int:
    """Print the Gate 1 report. Returns a process exit code."""
    enforce_offline()
    log_path = setup_logging(name="gate1", run_tag="gate1")

    exit_code = 0

    # ---------------- 1. Environment ----------------
    _section("1. ENVIRONMENT")
    env = collect_environment()
    missing = missing_packages(env)

    print(f"python                : {env['python']['version']} ({'64-bit' if env['python']['is_64bit'] else '32-bit'})")
    print(f"executable            : {env['python']['executable']}")
    print(f"platform              : {env['python']['platform']}")
    print()
    print("packages:")
    for name, version in env["packages"].items():
        if name == "optional":
            continue
        print(f"  {name:<14}: {version}")
    print("  optional:")
    for name, version in env["packages"]["optional"].items():
        print(f"    {name:<12}: {version if version else 'NOT INSTALLED'}")
    print()
    cuda = env["cuda"]
    print(f"cuda available        : {cuda.get('available')}")
    print(f"torch cuda version    : {cuda.get('torch_cuda_version')}")
    for device in cuda.get("devices", []):
        print(
            f"  gpu {device['index']}               : {device['name']} "
            f"({device['total_vram_gb']} GB, cc {device['compute_capability']})"
        )
    video = env["opencv_video"]
    print()
    print(f"opencv video smoke    : {'PASS' if video['ok'] else 'FAIL'}")
    print(f"  wrote {video['frames_written']} / read {video['frames_read']} frames ({video['details']})")
    if missing:
        print(f"\nMISSING REQUIRED PACKAGES: {', '.join(missing)}")
        exit_code = 1

    # ---------------- 2. Weights ----------------
    _section("2. WEIGHT VERIFICATION (local disk, no download)")
    try:
        resolved = resolve_weights_path()
        size = resolved.stat().st_size
        digest = sha256_file(resolved)
        crc_ok = verify_zip_integrity(resolved)

        print(f"absolute path         : {resolved}")
        print(f"size_bytes            : {size}")
        print(f"official v8.3.0 size  : {OFFICIAL_ASSET_SIZE_BYTES}")
        print(f"size match            : {'YES' if size == OFFICIAL_ASSET_SIZE_BYTES else 'NO'}")
        print(f"sha256 (local)        : {digest}")
        print("sha256 (official)     : NOT PUBLISHED for .pt assets on the official release")
        print(f"container CRC         : {'PASS' if crc_ok else 'FAIL'}")

        from ultralytics import YOLO
        from ultralytics.utils import is_online

        online = is_online()
        print(f"is_online()           : {online} (expected False)")

        model = YOLO(str(resolved))
        print()
        print("class map (COCO):")
        for class_id, expected in TARGET_CLASS_IDS.items():
            actual = model.names.get(class_id)
            mark = "OK" if actual == expected else "MISMATCH"
            print(f"  id {class_id} -> {actual!r} (expected {expected!r}) {mark}")
        if online:
            exit_code = 2
    except Exception as exc:
        print(f"WEIGHT VERIFICATION FAILED: {exc!r}")
        exit_code = 2

    # ---------------- 3. Configuration ----------------
    _section("3. EFFECTIVE CONFIGURATION")
    roi_pending = False
    try:
        # validate=False so a pending required input (lane ROI) is displayed
        # rather than raised; any structural error still surfaces below.
        config = load_config(validate=False)
        snapshot = config.effective_snapshot()
        device = resolve_device(config.device)

        lane_entries = config.rois.get("lanes") or []
        lane = next(
            (entry.get("polygon") for entry in lane_entries if isinstance(entry, dict)), None
        )
        roi_pending = config.rules["stopped_vehicle"].enabled and lane in (None, "full_frame")

        print(f"config file           : {config.config_path}")
        print(f"camera_id             : {config.camera_id}")
        print(f"device (resolved)     : {device}")
        print()
        print("detector:")
        for key, value in config.detector.items():
            print(f"  {key:<16}: {value}")
        print()
        print("rule_layer:")
        for key, value in config.rule_layer.items():
            print(f"  {key:<24}: {value}")
        print()
        print("rules:")
        for name, rule in config.rules.items():
            print(f"  {name}  (enabled={rule.enabled}, severity={rule.severity})")
            print(f"    min_consecutive_s      : {rule.min_consecutive_s}")
            print(f"    debounce_s             : {rule.debounce_s}")
            for key, value in rule.params.items():
                print(f"    {key:<22}: {value}")
        print()
        print("rois:")
        for key, value in config.rois.items():
            print(f"  {key:<16}: {value}")

        # Effective tracker YAML, written to the run output folder.
        out_dir = Path(__file__).resolve().parents[1] / "outputs" / "gate1"
        tracker_path = build_effective_tracker_yaml(config, out_dir, fps=25.0)
        print()
        print(f"effective tracker yaml: {tracker_path}")
        with tracker_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                print(f"    {line.rstrip()}")
        print()
        print("full snapshot (JSON):")
        print(json.dumps(snapshot, indent=2, default=str))

        if roi_pending:
            print()
            print("!! REQUIRED INPUT PENDING: rules.stopped_vehicle is enabled but")
            print("   rois.lanes has no lane polygon. The rule will NOT be constructed")
            print("   (and thus cannot fire) until a lane polygon is supplied.")
            print("   This is intended: stopped_vehicle is meaningless without a lane ROI.")
    except (ConfigError, FileNotFoundError) as exc:
        print(f"CONFIGURATION FAILED: {exc}")
        exit_code = 3

    _section(f"GATE 1 REPORT {'COMPLETE' if exit_code == 0 else 'HAS ERRORS'}")
    print(f"log file: {log_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
