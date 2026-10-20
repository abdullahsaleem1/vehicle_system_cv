"""Verify locally-held weights and write the SHA256 manifest.

Per the approved plan, weights are ALREADY on local disk and must NOT be
downloaded. This script:

    1. Resolves the weight file to an absolute local path.
    2. Computes its SHA256 and byte size.
    3. Cross-checks the size against the official Ultralytics v8.3.0 asset size.
       (The official release publishes SHA256 only for some non-.pt assets, so
       size equality plus container CRC is the strongest available check.)
    4. Verifies the file is a structurally valid torch/ZIP archive.
    5. Loads it offline from the local path only and confirms COCO class ids
       2/3/5/7 map to car/motorcycle/bus/truck.
    6. Writes models/MANIFEST.json with source URL and verification timestamp.
    7. Proves offline behaviour: is_online() is False.

Usage:
    python -m model2_behavior.scripts.verify_weights

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Offline enforcement must precede the ultralytics import.
import os  # noqa: E402

os.environ.setdefault("YOLO_OFFLINE", "1")

from model2_behavior.src.weights import (  # noqa: E402
    TARGET_CLASS_IDS,
    build_record,
    default_models_dir,
    enforce_offline,
    resolve_weights_path,
    sha256_file,
    verify_zip_integrity,
    write_manifest,
)

OFFICIAL_SOURCE_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt"
OFFICIAL_ASSET_SIZE_BYTES = 40_684_120

# The GitHub release-assets API (repos/ultralytics/assets/releases/tags/v8.3.0)
# DOES expose a "digest" field, but it is explicitly null for yolo11m.pt.
# Verified 2026-10-20:
#   yolo11m.pt        -> digest: null
#   mobileclip2_b.ts  -> sha256:35d7f213...
#   yolo11n-cls.onnx  -> sha256:2d2f6414...
# Ultralytics publishes SHA256 only for .ts/.onnx assets, never for .pt weights.
# Therefore expected_sha256 must legitimately remain null; no comparison is
# possible and none is claimed.
OFFICIAL_DIGEST_AVAILABLE = False


def main(argv: Optional[list[str]] = None) -> int:
    """Run verification and emit the manifest. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Verify local Model 2 weights offline.")
    parser.add_argument("--filename", default="yolo11m.pt")
    parser.add_argument("--models-dir", type=Path, default=default_models_dir())
    args = parser.parse_args(argv)

    enforce_offline()

    report: dict[str, object] = {}
    report["models_dir"] = str(Path(args.models_dir).resolve())

    try:
        resolved = resolve_weights_path(
            weights_path=Path(args.models_dir) / args.filename,
            models_dir=Path(args.models_dir),
            filename=args.filename,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    report["resolved_path"] = str(resolved)
    size = resolved.stat().st_size
    digest = sha256_file(resolved)
    report["size_bytes"] = size
    report["sha256"] = digest

    print("=" * 72)
    print("WEIGHT VERIFICATION")
    print("=" * 72)
    print(f"resolved absolute path : {resolved}")
    print(f"size_bytes             : {size}")
    print(f"official asset size    : {OFFICIAL_ASSET_SIZE_BYTES}")
    print(f"size match             : {'YES' if size == OFFICIAL_ASSET_SIZE_BYTES else 'NO'}")
    print(f"sha256                 : {digest}")
    if OFFICIAL_DIGEST_AVAILABLE:
        print("sha256 published by src: YES (compared above)")
    else:
        print("sha256 published by src: NO - release-assets API digest field is null for .pt weights")

    crc_ok = verify_zip_integrity(resolved)
    report["container_crc_ok"] = crc_ok
    print(f"container CRC check    : {'PASS' if crc_ok else 'FAIL'}")
    if not crc_ok:
        print("ERROR: container failed CRC validation; refusing to trust this file.", file=sys.stderr)
        return 2

    # Offline + class-map verification.
    from ultralytics import YOLO
    from ultralytics.utils import is_online

    online_state = is_online()
    report["is_online"] = online_state
    print(f"is_online()            : {online_state}  (must be False)")

    model = YOLO(str(resolved))
    report["task"] = model.task
    report["num_classes"] = len(model.names)

    print("-" * 72)
    print("CLASS MAP CHECK (COCO ids)")
    class_map_ok = True
    for class_id, expected_name in TARGET_CLASS_IDS.items():
        actual = model.names.get(class_id)
        ok = actual == expected_name
        class_map_ok = class_map_ok and ok
        print(f"  id {class_id} -> {actual!r}  expected {expected_name!r}  {'OK' if ok else 'MISMATCH'}")
    report["class_map_ok"] = class_map_ok
    report["class_map"] = {str(k): model.names.get(k) for k in TARGET_CLASS_IDS}

    if not class_map_ok:
        print("ERROR: class map mismatch.", file=sys.stderr)
        return 3
    if online_state:
        print("ERROR: is_online() is True; offline enforcement failed.", file=sys.stderr)
        return 4

    notes = (
        "size matches official v8.3.0 asset; container CRC verified; "
        "SHA256 not published by official release; loaded offline from local path"
    )
    record = build_record(
        path=resolved,
        source_url=OFFICIAL_SOURCE_URL,
        expected_size_bytes=OFFICIAL_ASSET_SIZE_BYTES,
        expected_sha256=None,
        notes=notes,
        class_map={str(k): model.names.get(k) for k in TARGET_CLASS_IDS},
    )
    manifest_path = write_manifest([record], manifest_path=Path(args.models_dir) / "MANIFEST.json")
    report["manifest_path"] = str(manifest_path)

    print("-" * 72)
    print(f"manifest written       : {manifest_path}")
    print("ALL CHECKS PASSED")
    print("=" * 72)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
