"""Interactive camera geometry authoring.

Opens a reference frame (or the first frame of a video), lets the operator click:

    1. A lane polygon            - left click to add points, right click / Enter to finish.
    2. Two points for its direction (tail -> head) - two left clicks.

Then offers to author ANOTHER lane, so a multi-lane camera can be described in one
session. Every lane gets its own name, polygon, and direction.

The clicked pixel coordinates are written into the config under ``rois.lanes``
together with the reference resolution, so the pipeline can rescale geometry at
runtime when the live stream resolution differs. ``rois.lanes[].direction`` is the
SINGLE SOURCE OF TRUTH for lane direction; nothing is written under ``rules``.

Usage:
    python -m model2_behavior.scripts.define_camera_geometry --source image.jpg
    python -m model2_behavior.scripts.define_camera_geometry --source clip.mp4
    python -m model2_behavior.scripts.define_camera_geometry --source clip.mp4 --config configs/model2.yaml
    python -m model2_behavior.scripts.define_camera_geometry --source clip.mp4 --replace

Controls:
    left click        add polygon point / set direction point
    right click       close the polygon (needs >= 3 points)
    Enter / Space     close the polygon
    r                 restart the current step
    u                 undo the last polygon point
    q / Esc           abort without writing
    s                 (direction step) finish and save
    a                 (after a lane) author another lane

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2  # noqa: E402

from model2_behavior.src.geometry import GeometryError, parse_lane_direction, parse_polygon  # noqa: E402
from model2_behavior.src.logging_setup import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)

WINDOW = "Model2 Camera Geometry"
POLYGON_COLOR = (255, 200, 0)
DIRECTION_COLOR = (0, 255, 255)
POINT_COLOR = (0, 140, 255)
MIN_POLYGON_POINTS = 3

# Set from --skip-direction in main(). Dialogues cannot easily pass it through the
# call chain, so it lives at module scope.
_SKIP_DIRECTION = False


def read_reference_frame(source: str) -> Tuple[Any, Tuple[int, int]]:
    """Read a reference frame from an image or the first frame of a video.

    Args:
        source: Path to an image or video, or an RTSP URL.

    Returns:
        ``(frame, (width, height))`` where ``frame`` is a BGR array.

    Raises:
        RuntimeError: If the source cannot be read.
    """
    path = Path(source)
    if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Could not decode image: {source}")
        height, width = frame.shape[:2]
        return frame, (width, height)

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source: {source}")
    try:
        grabbed, frame = capture.read()
        if not grabbed or frame is None:
            raise RuntimeError(f"Could not read first frame from: {source}")
        height, width = frame.shape[:2]
        return frame, (width, height)
    finally:
        capture.release()


def _draw_overlay(
    frame: Any,
    polygon: List[Tuple[int, int]],
    direction: List[Tuple[int, int]],
    step: str,
    status: str,
    existing_lanes: Optional[List[Dict[str, Any]]] = None,
) -> Any:
    """Render the current selection state onto a copy of the frame.

    Args:
        frame: Base frame.
        polygon: Points clicked so far for the polygon.
        direction: Points clicked so far for the direction.
        step: Current step name, for the instructions banner.
        status: Extra status text.
        existing_lanes: Lanes already captured this session, drawn for context.

    Returns:
        Annotated copy of the frame.
    """
    import numpy as np

    canvas = frame.copy()

    # Already-captured lanes, drawn dimmed so the new selection stands out.
    for lane in existing_lanes or []:
        lane_polygon = lane.get("polygon") or []
        if len(lane_polygon) >= MIN_POLYGON_POINTS:
            existing_points = np.array(lane_polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [existing_points], isClosed=True, color=(110, 110, 110), thickness=1)

    if len(polygon) >= 2:
        points = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        closed = len(polygon) >= MIN_POLYGON_POINTS
        cv2.polylines(canvas, [points], isClosed=closed, color=POLYGON_COLOR, thickness=2)
    for point in polygon:
        cv2.circle(canvas, point, 5, POINT_COLOR, -1)

    if len(direction) >= 2:
        cv2.arrowedLine(canvas, direction[0], direction[1], DIRECTION_COLOR, 2, tipLength=0.15)
    for point in direction:
        cv2.circle(canvas, point, 5, DIRECTION_COLOR, -1)

    instructions = {
        "polygon": "LEFT click: add point | RIGHT click / Enter: finish | u: undo | r: restart | q: abort",
        "direction": "LEFT click: tail then head | r: restart | s / Enter: save | q: abort",
    }[step]

    banner = canvas.copy()
    cv2.rectangle(banner, (0, 0), (canvas.shape[1], 56), (0, 0, 0), -1)
    cv2.addWeighted(banner, 0.55, canvas, 0.45, 0, canvas)
    cv2.putText(canvas, instructions, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, status, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    return canvas


def collect_polygon(
    frame: Any, existing_lanes: Optional[List[Dict[str, Any]]] = None
) -> Optional[List[Tuple[int, int]]]:
    """Interactively collect a lane polygon.

    Previously authored lanes are drawn in the background so the operator can see
    what has already been captured and avoid overlapping.

    Args:
        frame: Reference frame to display.
        existing_lanes: Lanes already captured this session.

    Returns:
        List of clicked points, or None if the operator aborted.
    """
    points: List[Tuple[int, int]] = []
    window = WINDOW
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((int(x), int(y)))
        elif event == cv2.EVENT_RBUTTONDOWN and len(points) >= MIN_POLYGON_POINTS:
            points.append((-1, -1))  # sentinel meaning "finish"

    cv2.setMouseCallback(window, on_mouse)

    while True:
        has_sentinel = bool(points) and points[-1] == (-1, -1)
        if has_sentinel:
            points.pop()
            break

        lane_index = len(existing_lanes or []) + 1
        status = (
            f"STEP 1/2 lane #{lane_index} polygon | points: {len(points)} "
            f"(need >= {MIN_POLYGON_POINTS})"
        )
        cv2.imshow(
            window,
            _draw_overlay(
                frame,
                [p for p in points if p != (-1, -1)],
                [],
                "polygon",
                status,
                existing_lanes=existing_lanes,
            ),
        )
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 32):  # Enter / Space
            if len(points) >= MIN_POLYGON_POINTS:
                break
        elif key == ord("u"):
            if points:
                points.pop()
        elif key == ord("r"):
            points.clear()
        elif key in (ord("q"), 27):  # q / Esc
            cv2.destroyWindow(window)
            return None

    cv2.destroyWindow(window)
    return points if len(points) >= MIN_POLYGON_POINTS else None


def collect_direction(
    frame: Any,
    polygon: List[Tuple[int, int]],
    existing_lanes: Optional[List[Dict[str, Any]]] = None,
) -> Optional[List[Tuple[int, int]]]:
    """Interactively collect two lane-direction points.

    Args:
        frame: Reference frame to display.
        polygon: Already-collected polygon, drawn for context.
        existing_lanes: Prior lanes, drawn dimmed for context.

    Returns:
        ``[(x1, y1), (x2, y2)]``, or None if the operator aborted.
    """
    points: List[Tuple[int, int]] = []
    window = WINDOW
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((int(x), int(y)))

    cv2.setMouseCallback(window, on_mouse)

    while True:
        status = f"STEP 2/2 lane direction | points: {len(points)}/2 (tail then head)"
        cv2.imshow(
            window,
            _draw_overlay(frame, polygon, points, "direction", status, existing_lanes=existing_lanes),
        )
        key = cv2.waitKey(20) & 0xFF

        if key in (13, ord("s")) and len(points) == 2:
            break
        if key == ord("r"):
            points.clear()
        elif key in (ord("q"), 27):
            cv2.destroyWindow(window)
            return None

    cv2.destroyWindow(window)
    return points if len(points) == 2 else None


def update_config_geometry(
    config_path: Path,
    lanes: List[Dict[str, Any]],
    resolution: Tuple[int, int],
    backup: bool = True,
    replace: bool = True,
) -> Path:
    """Write authored lanes into the config's ``rois.lanes`` list.

    Preserves all other config content by loading and re-dumping YAML.

    Args:
        config_path: Target config file.
        lanes: List of lane mappings, each ``{"name", "polygon", "direction"}``.
            ``polygon`` is a list of ``(x, y)`` tuples; ``direction`` is a list of
            two ``(x, y)`` tuples or None.
        resolution: Reference resolution ``(width, height)``.
        backup: Whether to write a ``.bak`` copy first.
        replace: Replace the existing ``rois.lanes`` list. When False, the new
            lanes are appended.

    Returns:
        Path to the written config.

    Raises:
        FileNotFoundError: If the config does not exist.
        GeometryError: If the geometry fails validation before writing, or lane
            names are missing/duplicated.
    """
    import yaml

    from model2_behavior.src.geometry import parse_lanes

    target = Path(config_path)
    if not target.is_file():
        raise FileNotFoundError(f"Config not found: {target}")

    with target.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    serialised: List[Dict[str, Any]] = []
    for lane in lanes:
        name = lane.get("name")
        polygon = lane.get("polygon")
        if not name:
            raise GeometryError("Each lane needs a non-empty 'name'")
        if polygon is None:
            raise GeometryError(f"Lane {name!r} needs a polygon")

        polygon_list = [[float(x), float(y)] for x, y in polygon]
        parse_polygon(polygon_list, f"rois.lanes[{name}].polygon")

        direction_value: Optional[List[float]] = None
        raw_direction = lane.get("direction")
        if raw_direction is not None:
            flat = [
                float(raw_direction[0][0]),
                float(raw_direction[0][1]),
                float(raw_direction[1][0]),
                float(raw_direction[1][1]),
            ]
            parsed = parse_lane_direction(flat, f"rois.lanes[{name}].direction")
            if parsed is None:
                raise GeometryError(f"Lane {name!r} direction could not be parsed")
            direction_value = [round(value, 4) for value in flat]

        serialised.append(
            {"name": str(name), "polygon": polygon_list, "direction": direction_value}
        )

    # Validate the whole set at once so duplicate names are caught before writing.
    parse_lanes(serialised, "rois.lanes")

    rois = raw.setdefault("rois", {})
    if replace:
        rois["lanes"] = serialised
    else:
        rois["lanes"] = list(rois.get("lanes") or []) + serialised
    rois["reference_resolution"] = [int(resolution[0]), int(resolution[1])]

    # Legacy keys must not linger: the loader rejects them outright.
    rois.pop("driving_lane", None)
    rois.pop("lane_direction", None)

    if backup:
        backup_path = target.with_suffix(target.suffix + ".bak")
        backup_path.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")

    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False, default_flow_style=False)
    return target


def _default_lane_name(existing: List[Dict[str, Any]]) -> str:
    """Return the next unused ``lane_<n>`` name.

    Args:
        existing: Lane mappings already collected this session.

    Returns:
        A name not present in ``existing``.
    """
    used = {str(lane.get("name")) for lane in existing}
    index = 1
    while f"lane_{index}" in used:
        index += 1
    return f"lane_{index}"


def collect_lanes(frame: Any) -> Optional[List[Dict[str, Any]]]:
    """Interactively collect one or more lanes from the operator.

    Each lane is a polygon plus an optional direction. After a lane is accepted
    the operator may press ``a`` to author another, or Enter to stop.

    Args:
        frame: Reference frame to display.

    Returns:
        List of lane mappings, or None if the operator aborted.
    """
    lanes: List[Dict[str, Any]] = []

    while True:
        polygon = collect_polygon(frame, existing_lanes=lanes)
        if polygon is None:
            return None if not lanes else lanes

        direction = None
        if not _SKIP_DIRECTION:
            direction = collect_direction(frame, polygon, existing_lanes=lanes)
            if direction is None:
                return None if not lanes else lanes

        default_name = _default_lane_name(lanes)
        name = input(
            f"Lane name [default {default_name}] (Enter to accept, 'x' to discard this lane): "
        ).strip()
        if name.lower() == "x":
            continue
        if not name:
            name = default_name

        lanes.append({"name": name, "polygon": polygon, "direction": direction})
        print(f"Lane {name!r} captured ({len(lanes)} total).")

        answer = input("Author another lane? [y/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            break

    return lanes


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point. Returns a process exit code."""
    global _SKIP_DIRECTION

    parser = argparse.ArgumentParser(description="Author camera geometry for Model 2 (Tier A).")
    parser.add_argument("--source", required=True, help="Reference image or video path.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "model2.yaml",
        help="Config file to update.",
    )
    parser.add_argument("--skip-direction", action="store_true", help="Only author lane polygons.")
    parser.add_argument("--no-backup", action="store_true", help="Do not write a .bak copy.")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append lanes to the existing rois.lanes instead of replacing them.",
    )
    parser.add_argument("--dump", type=Path, default=None, help="Also write raw geometry JSON here.")
    args = parser.parse_args(argv)

    setup_logging(name="define_camera_geometry")
    _SKIP_DIRECTION = args.skip_direction

    try:
        frame, resolution = read_reference_frame(args.source)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Reference frame: {resolution[0]}x{resolution[1]} px from {args.source}")

    lanes = collect_lanes(frame)
    if not lanes:
        print("Aborted; nothing written.", file=sys.stderr)
        return 2

    try:
        written = update_config_geometry(
            args.config, lanes, resolution, backup=not args.no_backup, replace=not args.append
        )
    except (FileNotFoundError, GeometryError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 4

    geometry = {
        "reference_resolution": [resolution[0], resolution[1]],
        "lanes": [
            {
                "name": lane["name"],
                "polygon": [[float(x), float(y)] for x, y in lane["polygon"]],
                "direction": (
                    [float(v) for point in lane["direction"] for v in point]
                    if lane["direction"]
                    else None
                ),
            }
            for lane in lanes
        ],
    }
    if args.dump:
        args.dump.write_text(json.dumps(geometry, indent=2), encoding="utf-8")
        print(f"Raw geometry written to {args.dump}")

    print(f"Updated config: {written}")
    print(json.dumps(geometry, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
