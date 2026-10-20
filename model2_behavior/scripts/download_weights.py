"""Fallback weight downloader for Model 2 (Tier A).

NOT USED BY THE PIPELINE. Weights are expected to be present on local disk.
This script exists only so the weights can be re-acquired from the official
Ultralytics release assets if the local copy is lost.

Policy:
    * Official source only: https://github.com/ultralytics/assets/releases
    * No mirrors, hubs, model zoos, or cloud inference APIs.
    * If the official download fails, this script exits non-zero and does not
      substitute any other source.

Usage:
    python -m model2_behavior.scripts.download_weights --filename yolo11m.pt

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model2_behavior.src.weights import (  # noqa: E402
    TARGET_CLASS_IDS,
    build_record,
    default_models_dir,
    sha256_file,
    verify_zip_integrity,
    write_manifest,
)

OFFICIAL_RELEASE_TAG = "v8.3.0"
OFFICIAL_BASE_URL = "https://github.com/ultralytics/assets/releases/download"

# Published asset sizes (bytes) from the official release, used as a sanity check.
# NOTE: the official release does NOT publish SHA256 digests for the .pt weights;
# only size is available for cross-checking.
KNOWN_ASSET_SIZES = {
    "yolo11m.pt": 40_684_120,
    "yolo11s.pt": 19_320_000,
    "yolo11n.pt": 5_610_000,
}

DEFAULT_TIMEOUT_S = 60
DEFAULT_RETRIES = 3


def build_url(filename: str, tag: str = OFFICIAL_RELEASE_TAG) -> str:
    """Return the official download URL for a weight file.

    Args:
        filename: Asset filename, e.g. ``"yolo11m.pt"``.
        tag: Release tag.

    Returns:
        Fully qualified asset URL.
    """
    return f"{OFFICIAL_BASE_URL}/{tag}/{filename}"


def download_file(url: str, destination: Path, retries: int = DEFAULT_RETRIES, timeout: int = DEFAULT_TIMEOUT_S) -> None:
    """Stream ``url`` to ``destination`` with retries.

    Args:
        url: Official asset URL.
        destination: Output path.
        retries: Attempt count on network failure.
        timeout: Per-attempt timeout in seconds.

    Raises:
        RuntimeError: If every attempt fails.
    """
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")

    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            print(f"[{attempt}/{retries}] GET {url}")
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                total = int(response.headers.get("Content-Length", 0))
                written = 0
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if chunk:
                            handle.write(chunk)
                            written += len(chunk)
                if total and written != total:
                    raise RuntimeError(f"incomplete download: {written}/{total} bytes")
            temporary.replace(destination)
            print(f"Downloaded {destination} ({destination.stat().st_size} bytes)")
            return
        except Exception as exc:
            last_error = exc
            print(f"  attempt {attempt} failed: {exc}", file=sys.stderr)
            if temporary.exists():
                temporary.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(2 * attempt)

    raise RuntimeError(
        f"Official download failed after {retries} attempts: {last_error}. "
        "No alternative source will be substituted."
    )


def main(argv: Optional[list[str]] = None) -> int:
    """Download, structurally verify, and manifest a weight file."""
    parser = argparse.ArgumentParser(description="Fallback downloader for Model 2 weights (official assets only).")
    parser.add_argument("--filename", default="yolo11m.pt", help="Weight filename to fetch.")
    parser.add_argument("--tag", default=OFFICIAL_RELEASE_TAG, help="Official release tag.")
    parser.add_argument("--dest-dir", type=Path, default=default_models_dir(), help="Destination directory.")
    parser.add_argument("--force", action="store_true", help="Re-download even if the file exists.")
    args = parser.parse_args(argv)

    destination = Path(args.dest_dir) / args.filename
    url = build_url(args.filename, args.tag)

    if destination.is_file() and not args.force:
        print(f"Already present: {destination} ({destination.stat().st_size} bytes)")
    else:
        try:
            download_file(url, destination)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    if not verify_zip_integrity(destination):
        print(f"ERROR: {destination} is not a valid archive; refusing to trust it.", file=sys.stderr)
        return 1

    expected_size = KNOWN_ASSET_SIZES.get(args.filename)
    digest = sha256_file(destination)
    size = destination.stat().st_size
    size_note = "size matches official asset" if expected_size == size else f"size {size} vs official {expected_size}"

    record = build_record(
        path=destination,
        source_url=url,
        expected_size_bytes=expected_size,
        expected_sha256=None,
        notes=f"{size_note}; zip CRC verified; SHA256 not published by official release",
        class_map={str(k): v for k, v in TARGET_CLASS_IDS.items()},
    )
    manifest = write_manifest([record])
    print(f"sha256  : {digest}")
    print(f"manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
