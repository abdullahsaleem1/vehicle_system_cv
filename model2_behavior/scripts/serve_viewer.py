"""Local check viewer: watch the Model 2 pipeline run in a browser, offline.

Why this exists
---------------
Reviewing a run currently means opening the annotated MP4 after the fact. That is
fine for a finished clip, but it hides *how* the run got there: which frames the
tracker held an id on, when a rule fired, and whether the whole thing is keeping
up in real time. This script streams the run live to a local browser page so a
supervisor can watch the tracking behaviour — most importantly, track-id
stability on raw footage *before* any lane geometry is authored.

The viewer is deliberately minimal and self-contained:

    * Python standard library ``http.server`` only. No Flask, no Jinja, no CDN.
    * The page is ONE inline HTML string with inline CSS/JS. It makes **no
      external requests** — no fonts, no scripts, no images from the network — so
      it works on a machine with the network disabled, which is the whole point.
    * Frames are drawn with plain ``cv2`` primitives (the same
      :func:`~model2_behavior.src.pipeline.draw_annotations` used by the run
      itself), then JPEG-encoded and pushed as MJPEG.

Security posture
----------------
The server binds to ``127.0.0.1`` **only**. It is a local review tool, not a
service: it has no authentication, no TLS, and it exposes video frames. Binding
to loopback means it is unreachable from the network; ``--host`` exists but
defaults to loopback and warns loudly when pointed at anything else. The static
file handler is restricted to the run's output directory, so it can never serve
files from elsewhere on disk.

Privacy
-------
Frames may contain licence plates and faces. **Local use only.** Do not expose
this server beyond the machine, do not screen-share the page outside the team,
and do not commit recorded frames.

Two run modes
-------------
``tracking-only``
    Detector + tracker only, no rules and no geometry required. This is the mode
    for validating track-id stability on new footage.

``full``
    Rules are evaluated. This mode **requires lane geometry**; if ``rois.lanes``
    is empty the rules that need it cannot be built. Rather than crash, the
    viewer reports that plainly in the page and refuses to start the full run.

Usage:
    python -m model2_behavior.scripts.serve_viewer --source path/to/clip.mp4 --mode tracking-only
    python -m model2_behavior.scripts.serve_viewer --source rtsp://host/stream --mode full
    python -m model2_behavior.scripts.serve_viewer --source clip.mp4 --loop --port 9000

Then open http://127.0.0.1:8765/ in a browser. Ctrl+C stops the server.

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2  # noqa: E402

from model2_behavior.src.config import ConfigError, build_effective_tracker_yaml, load_config  # noqa: E402
from model2_behavior.src.events import IncidentEvent  # noqa: E402
from model2_behavior.src.frame_clock import build_clock  # noqa: E402
from model2_behavior.src.geometry import GeometryError  # noqa: E402
from model2_behavior.src.logging_setup import get_logger, setup_logging  # noqa: E402
from model2_behavior.src.pipeline import (  # noqa: E402
    BehaviorPipeline,
    draw_annotations,
)
from model2_behavior.src.rules import build_rules  # noqa: E402
from model2_behavior.src.rules.base import FrameContext  # noqa: E402
from model2_behavior.src.tracks import Observation  # noqa: E402

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Default loopback port. 8765 is uncommon enough to avoid clashing with the
#: Model 1 server on 8000.
DEFAULT_PORT = 8765

#: Loopback by default. ``--host`` can override it, but every non-loopback bind
#: prints a warning because this server has no authentication.
DEFAULT_HOST = "127.0.0.1"

#: JPEG quality for the MJPEG stream. 75 keeps bandwidth and CPU sane while
#: staying legible; the point is to judge tracking, not to archive.
STREAM_JPEG_QUALITY = 75

#: How long ``/stream`` waits for a frame before re-checking shutdown. Keeps the
#: generator responsive to Ctrl+C instead of blocking forever on an empty queue.
STREAM_POLL_S = 0.05

#: The multipart boundary for the MJPEG stream. Fixed, not random: a fixed token
#: is easier to assert on in tests and carries no security meaning here.
MJPEG_BOUNDARY = "model2frame"

VALID_MODES = ("tracking-only", "full")


@dataclass
class ViewerState:
    """Mutable run state shared between the pipeline thread and the HTTP server.

    The pipeline thread writes; request handlers read. A single :class:`threading.Lock`
    guards every field, because HTTP handlers run on their own threads and a
    half-updated stats dict served mid-write is exactly the kind of flake that
    wastes a reviewer's afternoon.

    Attributes:
        mode: ``tracking-only`` or ``full``.
        source: The source string being processed.
        status: Human-readable run status for the page (``starting``, ``running``,
            ``finished``, ``error``, ``refused``).
        message: A longer explanation shown prominently in the page. Used for the
            full-mode-without-geometry refusal and for errors.
        frame_index: Frames processed so far.
        active_tracks: Track count on the most recent frame.
        fps: Recent throughput, measured over a short window.
        clock_source: Which clock the run is using.
        total_tracks: Distinct track ids seen.
        events: Emitted events, as contract dicts, oldest first.
        latest_jpeg: The most recent annotated frame, JPEG-encoded.
        run_summary: Populated when the run finishes.
    """

    mode: str
    source: str
    status: str = "starting"
    message: str = ""
    frame_index: int = 0
    active_tracks: int = 0
    fps: float = 0.0
    clock_source: str = ""
    total_tracks: int = 0
    events: List[Dict[str, Any]] = field(default_factory=list)
    latest_jpeg: Optional[bytes] = None
    run_summary: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Attach the lock and per-frame bookkeeping."""
        self._lock = threading.Lock()
        self._frame_event = threading.Event()
        self._fps_samples: List[float] = []
        self._last_frame_time: Optional[float] = None
        self._seen_track_ids: set[int] = set()
        self._stop = threading.Event()

    # -- lifecycle ---------------------------------------------------------

    @property
    def stopped(self) -> bool:
        """Whether a shutdown has been requested."""
        return self._stop.is_set()

    def request_stop(self) -> None:
        """Ask the pipeline thread to finish its current frame and stop."""
        self._stop.set()

    # -- pipeline thread writes -------------------------------------------

    def set_status(self, status: str, message: str = "") -> None:
        """Set the run status and message.

        Args:
            status: New status string.
            message: Optional longer explanation.
        """
        with self._lock:
            self.status = status
            if message:
                self.message = message
        if status in ("finished", "error", "refused"):
            self._frame_event.set()

    def publish_frame(self, jpeg: bytes, frame_index: int, active_tracks: int) -> None:
        """Record a newly annotated frame and the timing that goes with it.

        Args:
            jpeg: JPEG-encoded annotated frame.
            frame_index: Index of the frame just processed.
            active_tracks: Number of active tracks on this frame.
        """
        now = time.perf_counter()
        with self._lock:
            self.frame_index = frame_index
            self.active_tracks = active_tracks
            self.latest_jpeg = jpeg
            if self._last_frame_time is not None:
                delta = now - self._last_frame_time
                if delta > 0:
                    self._fps_samples.append(1.0 / delta)
                    # A short window tracks a changing throughput; a lifetime mean
                    # would report a number that describes no part of the run.
                    if len(self._fps_samples) > 30:
                        self._fps_samples.pop(0)
                    self.fps = sum(self._fps_samples) / len(self._fps_samples)
            self._last_frame_time = now
        self._frame_event.set()

    def note_tracks(self, track_ids: List[int]) -> None:
        """Record track ids seen, for the distinct-track count.

        Args:
            track_ids: Track ids observed on the current frame.
        """
        with self._lock:
            self._seen_track_ids.update(track_ids)
            self.total_tracks = len(self._seen_track_ids)

    def add_event(self, event: IncidentEvent) -> None:
        """Append an emitted event to the visible list.

        Args:
            event: The event that fired.
        """
        with self._lock:
            self.events.append(event.to_dict())

    def set_run_summary(self, summary: Dict[str, Any]) -> None:
        """Store the final run summary.

        Args:
            summary: ``run_summary.json`` payload.
        """
        with self._lock:
            self.run_summary = summary

    # -- HTTP handlers read ------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Return a consistent copy of the current state.

        Returns:
            A JSON-serialisable mapping. The frame bytes are excluded; use
            :meth:`latest_frame` for those.
        """
        with self._lock:
            return {
                "mode": self.mode,
                "source": self.source,
                "status": self.status,
                "message": self.message,
                "frame_index": self.frame_index,
                "active_tracks": self.active_tracks,
                "total_tracks": self.total_tracks,
                "fps": round(self.fps, 2),
                "clock_source": self.clock_source,
                "event_count": len(self.events),
                "run_summary": self.run_summary,
            }

    def events_snapshot(self) -> List[Dict[str, Any]]:
        """Return a copy of the emitted events.

        Returns:
            List of event contract dicts.
        """
        with self._lock:
            return list(self.events)

    def latest_frame(self) -> Optional[bytes]:
        """Return the most recent JPEG, or None before the first frame.

        Returns:
            JPEG bytes or None.
        """
        with self._lock:
            return self.latest_jpeg

    def wait_for_frame(self, timeout: float) -> bool:
        """Block briefly until a new frame arrives or the run ends.

        Args:
            timeout: Maximum seconds to wait.

        Returns:
            True if a frame (or a terminal status) may have arrived.
        """
        return self._frame_event.wait(timeout)


class ViewerRunner:
    """Runs the pipeline in a background thread and publishes frames to state.

    This deliberately does NOT call :meth:`BehaviorPipeline.run`, because that
    method owns the whole clip start to finish and offers no hook per frame. The
    viewer instead drives the same sequence of steps itself — clock, detect,
    track, rule, draw — so it can publish each annotated frame as it is produced.

    Args:
        pipeline: A prepared :class:`BehaviorPipeline` (detector already built).
        state: Shared state to publish into.
        source: Source string to open.
        fps_override: Optional FPS override.
        loop: Whether to restart the clip when it ends.
        max_frames: Optional cap on total processed frames.
    """

    def __init__(
        self,
        pipeline: BehaviorPipeline,
        state: ViewerState,
        source: str,
        fps_override: Optional[float] = None,
        loop: bool = False,
        max_frames: Optional[int] = None,
    ) -> None:
        self.pipeline = pipeline
        self.state = state
        self.source = source
        self.fps_override = fps_override
        self.loop = loop
        self.max_frames = max_frames

    def _open(self) -> Any:
        """Open the capture, raising a clear error when it cannot be opened.

        Returns:
            An opened ``cv2.VideoCapture``.

        Raises:
            RuntimeError: If the source cannot be opened.
        """
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")
        return cap

    def run(self) -> None:
        """Process the source until it ends or a stop is requested.

        Catches everything: a background thread that dies silently would leave the
        page frozen with no explanation, which is worse than an error banner.
        """
        try:
            self._run_inner()
        except Exception as exc:  # noqa: BLE001 - reported to the page, not swallowed
            logger.exception("Viewer run failed")
            self.state.set_status("error", f"Run failed: {exc}")

    def _run_inner(self) -> None:
        """The actual frame loop.

        Raises:
            RuntimeError: If the source cannot be opened.
        """
        pipeline = self.pipeline
        cap = self._open()
        try:
            fps = pipeline._resolve_fps(cap)
            pipeline.fps = fps

            # ``run`` normally does this once the FPS is known; the viewer drives
            # its own loop, so it must do the same or the tracker silently uses the
            # bootstrap YAML with a wrong-derived track buffer.
            pipeline.effective_tracker_path = build_effective_tracker_yaml(
                pipeline.config, pipeline.output_dir, fps
            )
            pipeline.detector.tracker_yaml = pipeline.effective_tracker_path

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if frame_width > 0 and frame_height > 0:
                pipeline.geometry_scaled = pipeline.geometry.scaled_to((frame_width, frame_height))
            else:
                pipeline.geometry_scaled = pipeline.geometry

            if pipeline.tracking_only:
                pipeline.rules = []
            else:
                pipeline.rules = build_rules(
                    pipeline.config, fps, pipeline.geometry_scaled, strict=pipeline.strict_rules
                )

            clock = build_clock(str(self.source), fps, force_wall_clock=pipeline.force_wall_clock)
            self.state.clock_source = clock.source.value

            frame_index = 0
            self.state.set_status("running")

            while not self.state.stopped:
                ok, frame = cap.read()
                if not ok:
                    if self.loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        # A looped clip is a new view of the same scene; the clock
                        # would otherwise keep climbing and the timeline would
                        # stop matching the footage.
                        clock = build_clock(
                            str(self.source), fps, force_wall_clock=pipeline.force_wall_clock
                        )
                        continue
                    break
                if self.max_frames is not None and frame_index >= self.max_frames:
                    break

                pts_seconds = clock.read_pts_seconds(cap) if clock.source.value == "pts" else None
                time_s = clock.next_timestamp(pts_seconds)

                pipeline.motion_checker.observe(frame, frame_index)

                raw = pipeline.detector.track_frame(frame)
                observations = [
                    Observation(
                        track_id=obs.track_id,
                        frame_index=frame_index,
                        time_s=time_s,
                        box=obs.box,
                        confidence=obs.confidence,
                        class_id=obs.class_id,
                        class_name=obs.class_name,
                    )
                    for obs in raw
                ]
                pipeline.track_store.update(observations)
                pipeline.track_store.prune()
                self.state.note_tracks([obs.track_id for obs in observations])

                # The viewer shows every tracked vehicle. In tracking-only mode
                # that is the whole point; in full mode the rule layer applies its
                # own precision filter, so the same records are enough.
                active = pipeline.track_store.active_tracks()

                highlight: List[int] = []
                if pipeline.rules:
                    context = FrameContext(
                        frame_index=frame_index,
                        time_s=time_s,
                        fps=fps,
                        geometry=pipeline.geometry_scaled or pipeline.geometry,
                        geometry_describe=(pipeline.geometry_scaled or pipeline.geometry).describe(),
                    )
                    for rule in pipeline.rules:
                        try:
                            hit = rule.update(context, active)
                        except Exception:
                            logger.exception("Rule %s raised; skipping this frame", rule.name)
                            continue
                        if hit is not None:
                            highlight.extend(hit.track_ids)
                            event = self._build_event(rule, hit, context, active)
                            pipeline.event_writer.add(event)
                            self.state.add_event(event)

                annotated = draw_annotations(
                    frame.copy(),
                    active,
                    lanes=pipeline._lane_descriptions(),
                    highlight_track_ids=highlight,
                )
                ok_encode, buffer = cv2.imencode(
                    ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_JPEG_QUALITY]
                )
                if ok_encode:
                    self.state.publish_frame(buffer.tobytes(), frame_index, len(active))

                frame_index += 1

            if not self.state.stopped:
                self.state.set_status("finished", "Clip finished.")
        finally:
            cap.release()

    def _build_event(self, rule: Any, hit: Any, context: FrameContext, active: List[Any]) -> IncidentEvent:
        """Build an event for a rule hit, without writing a snapshot to disk.

        The run's own pipeline snapshots events to ``outputs/``; the viewer keeps
        everything in memory so a review session leaves no frame artifacts behind.

        Args:
            rule: Rule that fired.
            hit: Its decision.
            context: Frame context.
            active: Active tracks.

        Returns:
            The event.
        """
        by_id = {track.track_id: track for track in active}
        boxes: List[List[float]] = []
        classes: List[str] = []
        for track_id in hit.track_ids:
            track = by_id.get(track_id)
            box = track.box_at_latest() if track else None
            boxes.append([round(float(v), 2) for v in box] if box else [0.0, 0.0, 0.0, 0.0])
            classes.append(track.class_name if track and track.class_name else "unknown")

        event = self.pipeline.event_writer.new_event(
            incident_type=rule.name,
            severity=rule.config.severity,
            frame_index=context.frame_index,
            track_ids=sorted(hit.track_ids),
            vehicle_classes=classes,
            bounding_boxes=boxes,
            confidence=float(hit.confidence),
            rule_params=dict(hit.rule_params),
            zone_id=hit.zone_id,
        )
        event.timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return event


# ---------------------------------------------------------------------------
# The page. One string, inline CSS/JS, zero external requests.
# ---------------------------------------------------------------------------

PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model 2 - local check viewer</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: #14171c; color: #e6e9ef;
    font: 14px/1.45 -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  }
  header { padding: 14px 20px; border-bottom: 1px solid #262b33; background: #191d24; }
  header h1 { margin: 0; font-size: 16px; font-weight: 600; letter-spacing: .2px; }
  header .sub { color: #98a2b3; font-size: 12px; margin-top: 3px; }
  .warn { color: #ffcf70; }
  main { display: grid; grid-template-columns: minmax(0, 2fr) minmax(320px, 1fr); gap: 16px; padding: 16px 20px 40px; }
  @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  .panel { background: #191d24; border: 1px solid #262b33; border-radius: 8px; overflow: hidden; }
  .panel h2 { margin: 0; padding: 10px 14px; font-size: 13px; font-weight: 600;
              border-bottom: 1px solid #262b33; color: #b8c0cc; text-transform: uppercase; letter-spacing: .5px; }
  .video-wrap { background: #000; display: flex; align-items: center; justify-content: center; min-height: 240px; }
  .video-wrap img { max-width: 100%; height: auto; display: block; }
  .placeholder { color: #6b7480; padding: 48px 16px; text-align: center; }
  .stats { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1px; background: #262b33; }
  .stat { background: #191d24; padding: 10px 14px; }
  .stat .k { color: #98a2b3; font-size: 11px; text-transform: uppercase; letter-spacing: .5px; }
  .stat .v { font-size: 20px; font-variant-numeric: tabular-nums; margin-top: 2px; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; font-weight: 600; }
  .badge.running { background: #17381f; color: #66d98a; }
  .badge.finished { background: #1d2733; color: #7fb2ff; }
  .badge.error, .badge.refused { background: #3a1d1d; color: #ff8a8a; }
  .badge.starting { background: #37311a; color: #ffcf70; }
  table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid #23282f; vertical-align: top; }
  th { color: #98a2b3; font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: .4px; }
  td.mono { font-family: ui-monospace, Consolas, monospace; }
  .empty { color: #6b7480; padding: 20px 14px; }
  .note { padding: 10px 14px; color: #98a2b3; font-size: 12px; border-top: 1px solid #262b33; }
  .sev-High { color: #ff8a8a; } .sev-Medium { color: #ffcf70; } .sev-Low { color: #7fb2ff; }
</style>
</head>
<body>
<header>
  <h1>Model 2 &mdash; local check viewer <span class="warn">(FR-20, pending supervisor sign-off)</span></h1>
  <div class="sub" id="subtitle">%MODE_LABEL% &middot; <span class="mono">%SOURCE_LABEL%</span></div>
</header>
<main>
  <section class="panel">
    <h2>Live annotated view</h2>
    <div class="video-wrap" id="videoWrap">
      <div class="placeholder" id="placeholder">Waiting for the first frame&hellip;</div>
      <img id="stream" alt="annotated video stream" style="display:none">
    </div>
    <div class="note">
      Confidence shown in events is <strong>detection confidence</strong> &mdash; the
      detector's certainty that the box is a vehicle, <strong>not</strong> the
      probability that the incident occurred. Behaviour rules are kinematic
      heuristics, not calibrated classifiers.
    </div>
    <div class="note">
      <strong>Privacy:</strong> frames may show licence plates and faces. Local use only.
    </div>
  </section>
  <aside>
    <section class="panel" style="margin-bottom:16px">
      <h2>Run stats</h2>
      <div class="stats">
        <div class="stat"><div class="k">Status</div><div class="v"><span class="badge starting" id="statusBadge">starting</span></div></div>
        <div class="stat"><div class="k">Mode</div><div class="v" id="statMode">%MODE_LABEL%</div></div>
        <div class="stat"><div class="k">FPS (recent)</div><div class="v" id="statFps">0.0</div></div>
        <div class="stat"><div class="k">Frame index</div><div class="v" id="statFrame">0</div></div>
        <div class="stat"><div class="k">Active tracks</div><div class="v" id="statActive">0</div></div>
        <div class="stat"><div class="k">Distinct tracks</div><div class="v" id="statTotal">0</div></div>
      </div>
      <div class="note" id="messageBox" style="display:none"></div>
    </section>
    <section class="panel">
      <h2>Events <span id="eventCount" style="color:#6b7480">(0)</span></h2>
      <div id="eventsWrap">
        <div class="empty">No events yet.</div>
      </div>
    </section>
  </aside>
</main>
<script>
(function () {
  "use strict";
  // Point the stream at MJPEG. The browser keeps the connection open and simply
  // replaces each JPEG part as it arrives; nothing here polls the network.
  var img = document.getElementById("stream");
  var placeholder = document.getElementById("placeholder");
  img.onload = function () {
    img.style.display = "block";
    if (placeholder) { placeholder.style.display = "none"; }
  };
  img.src = "/stream?t=" + Date.now();

  var badge = document.getElementById("statusBadge");
  var messageBox = document.getElementById("messageBox");

  function setText(id, value) {
    var el = document.getElementById(id);
    if (el) { el.textContent = value; }
  }

  function renderEvents(events) {
    var wrap = document.getElementById("eventsWrap");
    setText("eventCount", "(" + events.length + ")");
    if (!events.length) {
      wrap.innerHTML = '<div class="empty">No events yet.</div>';
      return;
    }
    var rows = events.slice().reverse().map(function (e) {
      var sev = "sev-" + (e.severity || "Low");
      var tracks = (e.track_ids || []).join(", ");
      var conf = (typeof e.confidence === "number") ? e.confidence.toFixed(3) : "-";
      return "<tr>" +
        "<td class='mono'>" + (e.incident_type || "") + "</td>" +
        "<td class='" + sev + "'>" + (e.severity || "") + "</td>" +
        "<td class='mono'>" + tracks + "</td>" +
        "<td class='mono'>" + (e.timestamp || "") + "</td>" +
        "<td class='mono'>" + conf + "</td>" +
        "</tr>";
    }).join("");
    wrap.innerHTML =
      "<table><thead><tr><th>Incident</th><th>Severity</th><th>Tracks</th>" +
      "<th>Timestamp</th><th>Conf.</th></tr></thead><tbody>" + rows + "</tbody></table>";
  }

  function poll() {
    fetch("/stats", { cache: "no-store" })
      .then(function (r) { return r.json(); })
      .then(function (s) {
        badge.textContent = s.status;
        badge.className = "badge " + s.status;
        setText("statFps", (s.fps || 0).toFixed(1));
        setText("statFrame", String(s.frame_index));
        setText("statActive", String(s.active_tracks));
        setText("statTotal", String(s.total_tracks));
        if (s.message) {
          messageBox.textContent = s.message;
          messageBox.style.display = "block";
        } else {
          messageBox.style.display = "none";
        }
      })
      .catch(function () { /* a transient fetch failure is not worth a banner */ });

    fetch("/events", { cache: "no-store" })
      .then(function (r) { return r.json(); })
      .then(function (payload) { renderEvents(payload.events || []); })
      .catch(function () {});
  }

  poll();
  setInterval(poll, 1000);  // page polls every 1 s, per the contract
})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


def build_page(state: ViewerState) -> bytes:
    """Render the single self-contained HTML page for a run.

    Args:
        state: The run state, for the header labels.

    Returns:
        The page as UTF-8 bytes.
    """
    mode_label = "Tracking only" if state.mode == "tracking-only" else "Full (rules)"
    html = (
        PAGE_TEMPLATE
        .replace("%MODE_LABEL%", mode_label)
        .replace("%SOURCE_LABEL%", _escape(state.source))
    )
    return html.encode("utf-8")


def _escape(text: str) -> str:
    """Escape the handful of characters that matter in this page's context.

    A source path is user input and ends up inside an HTML document, so the
    angle brackets and ampersand are escaped rather than trusted.

    Args:
        text: Raw text.

    Returns:
        Escaped text.
    """
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


class ViewerRequestHandler(BaseHTTPRequestHandler):
    """Serves the viewer page, the MJPEG stream, events, and stats.

    Every response is local to this run; no file outside the run's output
    directory can be reached through this handler.
    """

    #: Injected by :func:`serve`.
    state: ViewerState
    output_dir: Path
    server_version = "Model2Viewer/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        """Route the default access log to our logger instead of stderr.

        Args:
            format: Log format string.
            *args: Format arguments.
        """
        logger.debug("viewer %s - %s", self.address_string(), format % args)

    # -- helpers -----------------------------------------------------------

    def _send_json(self, payload: Any, status: int = 200) -> None:
        """Send a JSON response.

        Args:
            payload: JSON-serialisable object.
            status: HTTP status code.
        """
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, body: bytes, status: int = 200) -> None:
        """Send an HTML response.

        Args:
            body: Page bytes.
            status: HTTP status code.
        """
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # -- routes ------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        """Dispatch a GET to the right handler."""
        path = urlparse(self.path).path
        if path == "/":
            self._send_html(build_page(self.state))
        elif path == "/stream":
            self._stream()
        elif path == "/events":
            self._send_json({"events": self.state.events_snapshot()})
        elif path == "/stats":
            self._send_json(self.state.snapshot())
        else:
            self._send_json({"error": f"unknown path {path}"}, status=404)

    def _stream(self) -> None:
        """Serve the annotated frames as a ``multipart/x-mixed-replace`` MJPEG stream.

        The connection stays open. Each part is one JPEG frame. When no frame is
        available yet the loop waits briefly rather than busy-spinning, so an idle
        viewer costs almost nothing.
        """
        self.send_response(200)
        self.send_header(
            "Content-Type", f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}"
        )
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        last_sent: Optional[bytes] = None
        while not self.state.stopped:
            frame = self.state.latest_frame()
            if frame is None or frame is last_sent:
                if self.state.status in ("finished", "error", "refused"):
                    # Nothing more will arrive; end the stream so the browser can
                    # fall back to the placeholder rather than hang.
                    break
                self.state.wait_for_frame(STREAM_POLL_S)
                continue
            try:
                self.wfile.write(b"--" + MJPEG_BOUNDARY.encode() + b"\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n")
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                # The browser navigated away; that is normal, not an error.
                return
            last_sent = frame
        try:
            self.wfile.write(b"--" + MJPEG_BOUNDARY.encode() + b"--\r\n")
        except (BrokenPipeError, ConnectionResetError):
            return


def serve(
    state: ViewerState,
    output_dir: Path,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> ThreadingHTTPServer:
    """Create (but do not start) the viewer HTTP server.

    Args:
        state: Shared run state.
        output_dir: The run's output directory. Retained so the handler can never
            be pointed at anything outside it.
        host: Interface to bind.
        port: Port to bind. Pass 0 to let the OS choose (used by tests).

    Returns:
        A configured :class:`ThreadingHTTPServer`.

    Raises:
        ValueError: If the host is not loopback and no explicit override was
            intended — this server has no authentication, so binding it wider is
            a decision that should never happen by accident.
    """
    handler = type(
        "BoundViewerHandler",
        (ViewerRequestHandler,),
        {"state": state, "output_dir": output_dir},
    )
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def _is_loopback(host: str) -> bool:
    """Return whether a host string refers to loopback only.

    Args:
        host: Bind host.

    Returns:
        True for ``127.0.0.1``, ``localhost``, or ``::1``.
    """
    return host in ("127.0.0.1", "localhost", "::1")


def main(argv: Optional[List[str]] = None) -> int:
    """Run the viewer.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Local Model 2 check viewer (stdlib HTTP server, offline, loopback only)."
    )
    parser.add_argument("--source", required=True, help="Video file path, camera index, or RTSP URL.")
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default="tracking-only",
        help="tracking-only (no geometry needed) or full (requires lane geometry). Default: tracking-only.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "model2.yaml",
        help="Config file to use.",
    )
    parser.add_argument("--fps", type=float, default=None, help="FPS override.")
    parser.add_argument("--loop", action="store_true", help="Restart the clip when it ends.")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after this many frames.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port (default {DEFAULT_PORT}).")
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Bind interface (default {DEFAULT_HOST}). Non-loopback binds warn loudly.",
    )
    args = parser.parse_args(argv)

    setup_logging(name="serve_viewer")
    logger.info("Model 2 local check viewer | FR-20 (pending supervisor sign-off)")

    if not _is_loopback(args.host):
        print(
            "WARNING: binding to a non-loopback address exposes live frames with NO "
            "authentication. Anyone who can reach this host can watch the stream. "
            "Use a trusted network only.",
            file=sys.stderr,
        )
        logger.warning("Viewer bound to non-loopback host %s (no authentication).", args.host)

    if not Path(args.source).exists() and not args.source.lower().startswith(
        ("rtsp://", "rtmp://", "http://", "https://")
    ) and not args.source.isdigit():
        print(f"ERROR: source not found: {args.source}", file=sys.stderr)
        return 1

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = PROJECT_ROOT / "outputs" / f"viewer_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = args.mode
    state = ViewerState(mode=mode, source=args.source)

    # Full mode needs the rules, which need lane geometry. Refuse clearly rather
    # than starting a run that cannot produce events.
    try:
        config = load_config(args.config, validate=(mode == "full"))
    except (ConfigError, FileNotFoundError, GeometryError) as exc:
        message = (
            f"Configuration rejected: {exc}. "
            "Full mode needs lane geometry in rois.lanes. Author it with "
            "define_camera_geometry.py, or run with --mode tracking-only."
        )
        logger.error("Viewer refusing to start: %s", exc)
        print(f"ERROR: {message}", file=sys.stderr)
        state.set_status("refused", message)
        # Serve the refusal so the operator can read it in the page.
        return _serve_refusal(state, output_dir, args)

    if mode == "full":
        lanes = (config.rois or {}).get("lanes") or []
        if not lanes:
            message = (
                "Full mode requires lane geometry, but rois.lanes is empty. "
                "Author lanes from a frame of THIS camera with "
                "define_camera_geometry.py, or run with --mode tracking-only."
            )
            logger.error("Viewer refusing to start: %s", message)
            print(f"ERROR: {message}", file=sys.stderr)
            state.set_status("refused", message)
            return _serve_refusal(state, output_dir, args)

    pipeline = BehaviorPipeline(
        config,
        output_dir,
        fps=args.fps,
        write_video=False,
        strict_rules=(mode == "full"),
        draw=True,
        tracking_only=(mode == "tracking-only"),
        warmup_frames=10,
    )

    runner = ViewerRunner(
        pipeline,
        state,
        args.source,
        fps_override=args.fps,
        loop=args.loop,
        max_frames=args.max_frames,
    )

    server = serve(state, output_dir, host=args.host, port=args.port)
    bound_host, bound_port = server.server_address[0], server.server_address[1]
    thread = threading.Thread(target=runner.run, name="viewer-pipeline", daemon=True)
    thread.start()

    print("=" * 74)
    print("MODEL 2 LOCAL CHECK VIEWER - FR-20 (pending supervisor sign-off)")
    print("=" * 74)
    print(f"Open in a browser : http://{args.host}:{bound_port}/")
    print(f"Mode              : {mode}")
    print(f"Source            : {args.source}")
    print(f"Output dir        : {output_dir}")
    print("Privacy           : frames may show plates/faces. Local use only.")
    print("Press Ctrl+C to stop.")
    print("=" * 74)

    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\nStopping viewer...")
    finally:
        state.request_stop()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)
    logger.info("Viewer stopped")
    return 0


def _serve_refusal(state: ViewerState, output_dir: Path, args: argparse.Namespace) -> int:
    """Serve only the page, showing why the run was refused, then wait for Ctrl+C.

    The requirement is explicit: refuse with a clear message *in the page*, do not
    crash. So the server still starts and still renders ``/`` and ``/stats``; the
    stream simply never produces frames.

    Args:
        state: State already carrying the refusal message.
        output_dir: Run output directory.
        args: Parsed arguments (for host/port).

    Returns:
        Process exit code.
    """
    server = serve(state, output_dir, host=args.host, port=args.port)
    bound_port = server.server_address[1]
    print("=" * 74)
    print("MODEL 2 LOCAL CHECK VIEWER - RUN REFUSED")
    print("=" * 74)
    print(f"The reason is shown in the page: http://{args.host}:{bound_port}/")
    print(f"Reason: {state.message}")
    print("Press Ctrl+C to stop.")
    print("=" * 74)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\nStopping viewer...")
    finally:
        state.request_stop()
        server.shutdown()
        server.server_close()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
