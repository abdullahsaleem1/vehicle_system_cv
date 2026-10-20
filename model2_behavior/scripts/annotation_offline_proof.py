"""Prove annotated output rendering works with network access blocked.

Condition 1 for Step 4 requires that annotated output works offline. This script:

    1. Blocks DNS and socket connections in-process.
    2. Draws the same annotations the pipeline draws (plain ``cv2`` primitives).
    3. Writes a JPEG and an MP4, then reads them back to confirm validity.

Ultralytics plotting helpers are deliberately NOT used, so no font or asset can be
fetched at render time. ``Arial.ttf`` is already cached locally, but relying on it
is unnecessary because we only use ``cv2``.

Usage:
    python -m model2_behavior.scripts.annotation_offline_proof

Exit codes:
    0  annotation rendering passed with the network blocked
    1  a check failed

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from model2_behavior.src.logging_setup import get_logger, setup_logging  # noqa: E402
from model2_behavior.src.pipeline import draw_annotations  # noqa: E402
from model2_behavior.src.tracks import Observation, TrackState  # noqa: E402

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LANE_POLYGON = [[40.0, 240.0], [600.0, 240.0], [600.0, 470.0], [40.0, 470.0]]


def build_tracks() -> List[TrackState]:
    """Return a small set of tracks covering each vehicle class.

    Returns:
        List of populated track states.
    """
    track_templates = [
        (1, 2, "car", (120.0, 300.0, 200.0, 360.0)),
        (2, 3, "motorcycle", (280.0, 320.0, 330.0, 375.0)),
        (3, 5, "bus", (360.0, 250.0, 520.0, 350.0)),
        (4, 7, "truck", (540.0, 260.0, 620.0, 340.0)),
    ]
    tracks: List[TrackState] = []
    for track_id, class_id, class_name, box in track_templates:
        track = TrackState(track_id=track_id)
        for step in range(5):
            track.update(
                Observation(
                    track_id,
                    step,
                    step / 25.0,
                    (box[0] + step, box[1], box[2] + step, box[3]),
                    0.85,
                    class_id,
                    class_name,
                )
            )
        tracks.append(track)
    return tracks


def main() -> int:
    """Run the offline annotation proof.

    Returns:
        Process exit code.
    """
    setup_logging(name="annotation_offline_proof")
    output_dir = PROJECT_ROOT / "outputs" / "annotation_offline_proof"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- block the network in-process ---------------------------------------
    original_create_connection = socket.create_connection
    original_getaddrinfo = socket.getaddrinfo
    blocked: List[str] = []

    def blocked_create_connection(address, *args, **kwargs):
        blocked.append(f"create_connection({address})")
        raise OSError("network blocked by annotation_offline_proof")

    def blocked_getaddrinfo(host, *args, **kwargs):
        blocked.append(f"getaddrinfo({host})")
        raise OSError("network blocked by annotation_offline_proof")

    socket.create_connection = blocked_create_connection
    socket.getaddrinfo = blocked_getaddrinfo

    failures: List[str] = []
    try:
        tracks = build_tracks()
        frame = np.full((480, 640, 3), 40, dtype=np.uint8)

        annotated = draw_annotations(
            frame.copy(),
            tracks,
            lanes=[
                {
                    "name": "lane_1",
                    "polygon": LANE_POLYGON,
                    "direction": [[100.0, 400.0], [540.0, 400.0]],
                }
            ],
            highlight_track_ids=[3],
        )

        if annotated.shape != frame.shape:
            failures.append(f"annotated shape changed: {annotated.shape} != {frame.shape}")
        if int(annotated.sum()) <= int(frame.sum()):
            failures.append("annotations did not change the frame; nothing was drawn")

        jpeg_path = output_dir / "annotated_proof.jpg"
        if not cv2.imwrite(str(jpeg_path), annotated):
            failures.append(f"cv2.imwrite failed for {jpeg_path}")
        else:
            reloaded = cv2.imread(str(jpeg_path))
            if reloaded is None:
                failures.append("written JPEG could not be read back")
            else:
                logger.info("JPEG written and re-read | %s | shape=%s", jpeg_path, reloaded.shape)

        video_path = output_dir / "annotated_proof.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 25.0, (640, 480))
        if not writer.isOpened():
            failures.append("cv2.VideoWriter could not be opened")
        else:
            for step in range(15):
                shifted = draw_annotations(
                    frame.copy(),
                    tracks,
                    lanes=[
                        {
                            "name": "lane_1",
                            "polygon": [[x, y + step] for x, y in LANE_POLYGON],
                            "direction": None,
                        }
                    ],
                    highlight_track_ids=[3],
                )
                writer.write(shifted)
            writer.release()

            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                failures.append("written MP4 could not be re-opened")
            else:
                read_back = 0
                while True:
                    grabbed, _ = capture.read()
                    if not grabbed:
                        break
                    read_back += 1
                capture.release()
                if read_back != 15:
                    failures.append(f"MP4 frame count mismatch: {read_back} != 15")
                else:
                    logger.info("MP4 written and re-read | %s | frames=%s", video_path, read_back)
    finally:
        socket.create_connection = original_create_connection
        socket.getaddrinfo = original_getaddrinfo

    print("-" * 70)
    print(f"Blocked network attempts: {len(blocked)}")
    for event in blocked:
        print(f"  blocked: {event}")
    print(f"Output directory: {output_dir}")
    if failures:
        print("ANNOTATION OFFLINE PROOF FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    if blocked:
        print("ANNOTATION OFFLINE PROOF FAILED: annotations attempted network access")
        return 1
    print("ANNOTATION OFFLINE PROOF PASSED (zero network attempts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
