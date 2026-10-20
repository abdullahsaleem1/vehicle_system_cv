"""Run the Model 2 behaviour/incident pipeline over a video source or webcam.

This is the Tier A entry point. It loads everything from local config, writes the
effective tracker YAML and config snapshot into the run directory, processes the
source, and writes ``events.json`` plus annotated output.

Note: Steps 7 (live validation) requires a real clip or stream. This script is
functional but has NOT been validated against footage yet.

Usage:
    python -m model2_behavior.scripts.run_pipeline --source path/to/clip.mp4
    python -m model2_behavior.scripts.run_pipeline --source rtsp://host/stream --fps 25
    python -m model2_behavior.scripts.run_pipeline --source clip.mp4 --max-frames 300 --no-video

    # Validate track-id stability BEFORE authoring any lane geometry:
    python -m model2_behavior.scripts.run_pipeline --source clip.mp4 --tracking-only

Exit codes:
    0  run completed
    1  run failed

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model2_behavior.src.config import ConfigError, load_config  # noqa: E402
from model2_behavior.src.geometry import GeometryError  # noqa: E402
from model2_behavior.src.logging_setup import get_logger, setup_logging  # noqa: E402
from model2_behavior.src.pipeline import BehaviorPipeline  # noqa: E402
from model2_behavior.src.rules.base import RuleGeometryError  # noqa: E402

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv: Optional[List[str]] = None) -> int:
    """Run the pipeline.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Run the Model 2 behavior/incident pipeline (Tier A).")
    parser.add_argument("--source", required=True, help="Video file path, camera index, or RTSP URL.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "model2.yaml",
        help="Config file to use.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Run output directory. Defaults to outputs/run_<UTC timestamp>.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override FPS. Strongly recommended; without it the container FPS is used "
        "and a 25.0 fallback is applied if the container reports none.",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after this many frames.")
    parser.add_argument("--no-video", action="store_true", help="Skip the annotated video output.")
    parser.add_argument("--no-draw", action="store_true", help="Skip drawing entirely (faster).")
    parser.add_argument(
        "--tracking-only",
        action="store_true",
        help="Detector + tracker + annotated output ONLY. No rules are evaluated and no "
        "lane geometry is required. Use this to validate track-id stability on real "
        "footage before configuring lanes.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=10,
        help="Frames excluded from the reported throughput, since early frames include "
        "CUDA/tracker initialisation (default: 10).",
    )
    parser.add_argument(
        "--allow-missing-geometry",
        action="store_true",
        help="Run the rules that can run, instead of failing when a rule lacks its ROI.",
    )
    args = parser.parse_args(argv)

    setup_logging(name="run_pipeline")
    logger.info("Model 2 Tier A pipeline starting | FR-20 (pending supervisor sign-off)")

    if args.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = PROJECT_ROOT / "outputs" / f"run_{stamp}"
    else:
        output_dir = args.output_dir

    # Tracking-only mode requires no geometry, so strict validation is relaxed
    # for it: the point is to measure tracker behaviour on raw footage first.
    validate = not (args.allow_missing_geometry or args.tracking_only)

    try:
        config = load_config(args.config, validate=validate)
    except (ConfigError, FileNotFoundError, GeometryError) as exc:
        logger.error("Configuration rejected: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        if isinstance(exc, ConfigError) and "lanes" in str(exc):
            print(
                "Hint: author lanes with "
                "python -m model2_behavior.scripts.define_camera_geometry --source <frame-or-video>",
                file=sys.stderr,
            )
            print(
                "Or, to validate tracking only (no geometry needed), re-run with --tracking-only.",
                file=sys.stderr,
            )
        return 1

    try:
        pipeline = BehaviorPipeline(
            config,
            output_dir,
            fps=args.fps,
            write_video=not args.no_video,
            strict_rules=not (args.allow_missing_geometry or args.tracking_only),
            draw=not args.no_draw,
            tracking_only=args.tracking_only,
            warmup_frames=args.warmup_frames,
        )
        result = pipeline.run(args.source, max_frames=args.max_frames)
    except RuleGeometryError as exc:
        logger.error("Rule geometry requirement not satisfied: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        logger.error("Pipeline failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    summary_path = Path(result.output_dir or output_dir) / "run_summary.json"
    summary_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    print("-" * 70)
    print(f"Mode             : {result.mode}")
    print(f"Frames processed : {result.frames_processed}")
    print(f"Distinct tracks  : {result.total_tracks}")
    print(f"Events           : {len(result.events)}")
    for name, count in result.counts_by_type.items():
        print(f"  {name:18s}: {count}")
    print(f"Throughput       : {result.avg_fps:.2f} fps (incl. warm-up)")
    print(
        f"                   {result.avg_fps_excl_warmup:.2f} fps "
        f"(excl. first {result.warmup_frames} frames)"
    )
    print(
        f"Frame time       : p50={result.to_dict()['frame_time_percentiles_ms']['p50']:.2f} ms "
        f"p95={result.p95_frame_time_ms:.2f} ms "
        f"(excl. warm-up; a mean alone hides stalls)"
    )
    print(f"FPS source       : {result.used_fps_source} ({result.source_fps:.3f})")
    print(f"Clip resolution  : {result.source_resolution[0]}x{result.source_resolution[1]}")
    print(
        f"Timing clock     : {result.clock.get('clock_source')} "
        f"(dropped≈{result.clock.get('estimated_dropped_frames')}, "
        f"discontinuities={result.clock.get('discontinuity_count')})"
    )
    print(f"Clip duration    : {result.clip_duration_s:.2f} s of stream time")
    if result.tracks:
        print(
            f"Track lifetimes  : n={len(result.tracks)} "
            f"median={result.median_track_lifetime_s:.2f}s "
            f"shorter_than_1s={result.tracks_shorter_than_1s}"
        )
    print(
        f"suspected_id_switches (heuristic) : {result.suspected_id_switches} "
        "(NOT a tracking-accuracy metric)"
    )
    print(f"Events file      : {result.events_path}")
    print(f"Annotated video  : {result.annotated_video_path}")
    print(f"Config snapshot  : {result.config_snapshot_path}")
    print(f"Tracker YAML     : {result.effective_tracker_path}")
    print(f"Run summary      : {summary_path}")
    if args.tracking_only:
        print("NOTE: tracking-only mode wrote no events by design.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
