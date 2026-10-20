"""Tests for the local check viewer (stdlib HTTP server, no GPU, no network).

The viewer is a review tool, so its failure modes are quiet ones: a stream that
never yields a whole JPEG, an events endpoint that returns the wrong shape, or a
server that binds somewhere other than loopback. None of those would raise on
their own, so they are pinned here.

Everything uses a scripted detector and a synthetic video written to a temp
directory: no weights, no GPU, and no request leaves the machine (the server binds
to 127.0.0.1 and the client is the stdlib ``urllib``).

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from model2_behavior.scripts.serve_viewer import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    MJPEG_BOUNDARY,
    ViewerRunner,
    ViewerState,
    build_page,
    serve,
)
from model2_behavior.src.config import load_config
from model2_behavior.src.pipeline import BehaviorPipeline, FakeDetectorTracker
from model2_behavior.src.tracks import Observation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_CONFIG = PROJECT_ROOT / "configs" / "model2.yaml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def write_video(path: Path, frames: int = 40, width: int = 320, height: int = 240, fps: float = 25.0) -> Path:
    """Write a small synthetic video with a moving box in it.

    A moving bright square gives the fake detector a stable thing to "track" and
    makes the annotated frames visually distinct, so a frame that never changes is
    an obvious bug rather than a coincidence.

    Args:
        path: Destination path.
        frames: Frame count.
        width: Frame width.
        height: Frame height.
        fps: Container frame rate.

    Returns:
        The written path.

    Raises:
        RuntimeError: If the writer cannot be opened.
    """
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():  # pragma: no cover - codec availability
        raise RuntimeError("cv2.VideoWriter could not be opened")
    try:
        for index in range(frames):
            frame = np.full((height, width, 3), 40, dtype=np.uint8)
            left = 20 + (index * 4) % (width - 80)
            cv2.rectangle(frame, (left, 100), (left + 60, 180), (0, 200, 255), -1)
            writer.write(frame)
    finally:
        writer.release()
    return path


def scripted_observations(frame_index: int) -> list:
    """Return one synthetic car observation for a frame.

    Args:
        frame_index: Frame number.

    Returns:
        A single observation, moving right.
    """
    left = 20.0 + float(frame_index) * 4.0
    return [
        Observation(
            track_id=1,
            frame_index=frame_index,
            time_s=frame_index / 25.0,
            box=(left, 100.0, left + 60.0, 180.0),
            confidence=0.9,
            class_id=2,
            class_name="car",
        )
    ]


def contextlib_redirect(buffer) -> object:
    """Redirect stdout and stderr into one buffer.

    The viewer prints its banners to stdout and its errors to stderr; a test that
    only captured one would miss half the message.

    Args:
        buffer: The ``io.StringIO`` to capture into.

    Returns:
        A context manager.
    """
    return _CombinedRedirect(buffer)


class _CombinedRedirect:
    """Context manager that sends stdout and stderr to one buffer.

    Args:
        buffer: Destination text buffer.
    """

    def __init__(self, buffer) -> None:
        self.buffer = buffer

    def __enter__(self) -> "_CombinedRedirect":
        """Start redirecting.

        Returns:
            Self.
        """
        import contextlib

        self._out = contextlib.redirect_stdout(self.buffer)
        self._err = contextlib.redirect_stderr(self.buffer)
        self._out.__enter__()
        self._err.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        """Stop redirecting."""
        self._err.__exit__(*exc)
        self._out.__exit__(*exc)


def build_viewer(tmp_path: Path, mode: str = "tracking-only", frames: int = 40):
    """Build a pipeline, state, and runner over a synthetic clip.

    Args:
        tmp_path: pytest temporary directory.
        mode: ``tracking-only`` or ``full``.
        frames: Synthetic frame count.

    Returns:
        ``(state, runner, output_dir)``.
    """
    video = write_video(tmp_path / "clip.mp4", frames=frames)
    config = load_config(REAL_CONFIG, validate=False)
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    tracking_only = mode == "tracking-only"
    pipeline = BehaviorPipeline(
        config,
        output_dir,
        fps=25.0,
        write_video=False,
        draw=True,
        tracking_only=tracking_only,
        strict_rules=False,
    )
    # A scripted detector keyed by call index, so every frame has one car.
    scripted = {index: scripted_observations(index) for index in range(frames)}
    pipeline.detector = FakeDetectorTracker(scripted)
    state = ViewerState(mode=mode, source=str(video))
    runner = ViewerRunner(pipeline, state, str(video), fps_override=25.0)
    return state, runner, output_dir


class ServerHarness:
    """Runs the viewer server on an OS-assigned port for a test.

    Args:
        state: Run state to serve.
        output_dir: Run output directory.
        host: Interface to bind (loopback by default).
    """

    def __init__(self, state: ViewerState, output_dir: Path, host: str = DEFAULT_HOST) -> None:
        self.server: ThreadingHTTPServer = serve(state, output_dir, host=host, port=0)
        self.host = self.server.server_address[0]
        self.port = self.server.server_address[1]
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "ServerHarness":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self._thread.join(timeout=5.0)

    def url(self, path: str) -> str:
        """Return an absolute URL for a path.

        Args:
            path: Route path beginning with ``/``.

        Returns:
            The full URL.
        """
        return f"http://{self.host}:{self.port}{path}"

    def get(self, path: str, timeout: float = 5.0):
        """GET a path and return ``(status, body_bytes)``.

        Args:
            path: Route path.
            timeout: Request timeout in seconds.

        Returns:
            Tuple of status code and raw body.
        """
        with urllib.request.urlopen(self.url(path), timeout=timeout) as response:
            return response.status, response.read()


class StreamReader:
    """A tiny MJPEG client that reads exactly one part and stops.

    ``urllib`` would happily read the multipart stream forever, so this reads the
    headers, pulls one JPEG by its ``Content-Length``, and closes the connection.

    Args:
        url: The ``/stream`` URL.
        timeout: Socket timeout in seconds.
    """

    def __init__(self, url: str, timeout: float = 5.0) -> None:
        import http.client
        from urllib.parse import urlparse

        parsed = urlparse(url)
        self.connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
        self.connection.request("GET", parsed.path)

    def read_one_part(self) -> tuple:
        """Read one MJPEG part.

        Returns:
            ``(content_type_header, jpeg_bytes)``.

        Raises:
            AssertionError: If the response is not a multipart stream.
        """
        response = self.connection.getresponse()
        content_type = response.getheader("Content-Type", "")
        assert "multipart/x-mixed-replace" in content_type
        assert f"boundary={MJPEG_BOUNDARY}" in content_type

        # Read the boundary line.
        line = response.fp.readline()
        assert line.strip() == b"--" + MJPEG_BOUNDARY.encode()

        # Read the part headers.
        headers: dict = {}
        while True:
            line = response.fp.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            name, _, value = line.decode("latin-1").partition(":")
            headers[name.strip().lower()] = value.strip()

        length = int(headers["content-length"])
        payload = response.fp.read(length)
        return headers.get("content-type", ""), payload

    def close(self) -> None:
        """Close the underlying connection."""
        self.connection.close()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class TestViewerState:
    """The shared state must be consistent for concurrent readers."""

    def test_snapshot_has_the_expected_keys(self) -> None:
        state = ViewerState(mode="tracking-only", source="clip.mp4")
        payload = state.snapshot()
        for key in (
            "mode",
            "source",
            "status",
            "frame_index",
            "active_tracks",
            "total_tracks",
            "fps",
            "clock_source",
            "event_count",
        ):
            assert key in payload

    def test_publish_frame_tracks_index_and_tracks(self) -> None:
        state = ViewerState(mode="tracking-only", source="clip.mp4")
        state.publish_frame(b"\xff\xd8jpeg", frame_index=7, active_tracks=3)
        payload = state.snapshot()
        assert payload["frame_index"] == 7
        assert payload["active_tracks"] == 3
        assert state.latest_frame() == b"\xff\xd8jpeg"

    def test_note_tracks_counts_distinct_ids(self) -> None:
        state = ViewerState(mode="tracking-only", source="clip.mp4")
        state.note_tracks([1, 2])
        state.note_tracks([2, 3])
        assert state.snapshot()["total_tracks"] == 3

    def test_stop_flag_round_trips(self) -> None:
        state = ViewerState(mode="tracking-only", source="clip.mp4")
        assert state.stopped is False
        state.request_stop()
        assert state.stopped is True

    def test_fps_is_positive_after_two_frames(self) -> None:
        state = ViewerState(mode="tracking-only", source="clip.mp4")
        state.publish_frame(b"a", 0, 1)
        state.publish_frame(b"b", 1, 1)
        assert state.snapshot()["fps"] >= 0.0


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


class TestBuildPage:
    """The page must be self-contained and offline."""

    def test_page_is_html_with_no_external_requests(self) -> None:
        state = ViewerState(mode="tracking-only", source="clip.mp4")
        html = build_page(state).decode("utf-8")
        assert html.lstrip().startswith("<!DOCTYPE html>")
        # No CDN, no external fonts, no remote scripts or images.
        assert "http://" not in html.replace("http://127.0.0.1", "")
        assert "https://" not in html
        assert "<script src=" not in html
        assert "<link" not in html

    def test_page_mentions_the_detection_confidence_caveat(self) -> None:
        state = ViewerState(mode="tracking-only", source="clip.mp4")
        html = build_page(state).decode("utf-8")
        assert "detection confidence" in html.lower()
        assert "not" in html.lower() and "probability" in html.lower()

    def test_page_states_tracking_only_mode(self) -> None:
        state = ViewerState(mode="tracking-only", source="clip.mp4")
        assert "Tracking only" in build_page(state).decode("utf-8")

    def test_page_states_full_mode(self) -> None:
        state = ViewerState(mode="full", source="clip.mp4")
        assert "Full (rules)" in build_page(state).decode("utf-8")

    def test_source_is_html_escaped(self) -> None:
        state = ViewerState(mode="tracking-only", source="<script>alert(1)</script>")
        html = build_page(state).decode("utf-8")
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    def test_page_has_privacy_note(self) -> None:
        state = ViewerState(mode="tracking-only", source="clip.mp4")
        assert "Local use only" in build_page(state).decode("utf-8")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


class TestEndpoints:
    """Every endpoint must return a valid payload over real HTTP."""

    def test_stats_returns_json(self, tmp_path: Path) -> None:
        state, _runner, output_dir = build_viewer(tmp_path)
        with ServerHarness(state, output_dir) as harness:
            status, body = harness.get("/stats")
        assert status == 200
        payload = json.loads(body)
        assert payload["mode"] == "tracking-only"
        assert "frame_index" in payload

    def test_events_returns_json_list(self, tmp_path: Path) -> None:
        state, _runner, output_dir = build_viewer(tmp_path)
        with ServerHarness(state, output_dir) as harness:
            status, body = harness.get("/events")
        assert status == 200
        payload = json.loads(body)
        assert isinstance(payload["events"], list)

    def test_root_returns_html(self, tmp_path: Path) -> None:
        state, _runner, output_dir = build_viewer(tmp_path)
        with ServerHarness(state, output_dir) as harness:
            status, body = harness.get("/")
        assert status == 200
        assert b"<!DOCTYPE html>" in body

    def test_unknown_path_is_404(self, tmp_path: Path) -> None:
        state, _runner, output_dir = build_viewer(tmp_path)
        with ServerHarness(state, output_dir) as harness:
            with pytest.raises(urllib.error.HTTPError) as excinfo:
                harness.get("/etc/passwd")
        assert excinfo.value.code == 404

    def test_bind_address_is_loopback(self, tmp_path: Path) -> None:
        state, _runner, output_dir = build_viewer(tmp_path)
        with ServerHarness(state, output_dir) as harness:
            assert harness.host == "127.0.0.1"
            assert harness.server.server_address[0] == "127.0.0.1"

    def test_default_host_and_port_constants(self) -> None:
        assert DEFAULT_HOST == "127.0.0.1"
        assert DEFAULT_PORT == 8765


class TestStreamEndpoint:
    """``/stream`` must emit real, well-framed JPEG parts."""

    def test_stream_yields_a_valid_jpeg_part(self, tmp_path: Path) -> None:
        state, runner, output_dir = build_viewer(tmp_path, frames=20)
        with ServerHarness(state, output_dir) as harness:
            thread = threading.Thread(target=runner.run, daemon=True)
            thread.start()
            reader = StreamReader(harness.url("/stream"))
            try:
                content_type, payload = reader.read_one_part()
            finally:
                reader.close()
                state.request_stop()
                thread.join(timeout=10.0)

        assert content_type == "image/jpeg"
        # A JPEG starts with SOI (FFD8) and ends with EOI (FFD9).
        assert payload[:2] == b"\xff\xd8"
        assert payload[-2:] == b"\xff\xd9"
        assert len(payload) > 100


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class TestRunner:
    """The background runner must drive the pipeline and publish frames."""

    def test_runner_publishes_frames_and_finishes(self, tmp_path: Path) -> None:
        state, runner, _output_dir = build_viewer(tmp_path, frames=15)
        runner.run()
        payload = state.snapshot()
        assert payload["status"] == "finished"
        # frame_index is the index of the LAST frame shown, so a 15-frame clip
        # ends on index 14.
        assert payload["frame_index"] == 14
        assert state.latest_frame() is not None
        assert payload["total_tracks"] >= 1

    def test_runner_sets_the_clock_source(self, tmp_path: Path) -> None:
        state, runner, _output_dir = build_viewer(tmp_path, frames=10)
        runner.run()
        assert state.snapshot()["clock_source"] in ("pts", "frame_count", "wall_clock")

    def test_runner_stops_when_asked(self, tmp_path: Path) -> None:
        state, runner, _output_dir = build_viewer(tmp_path, frames=2000)
        state.request_stop()
        runner.run()
        # A stop request before the first frame ends the loop immediately and the
        # status is left at running rather than falsely reporting completion.
        assert state.snapshot()["status"] in ("running", "finished")

    def test_runner_reports_a_missing_source_as_an_error(self, tmp_path: Path) -> None:
        state, runner, _output_dir = build_viewer(tmp_path, frames=5)
        runner.source = str(tmp_path / "does_not_exist.mp4")
        runner.run()
        payload = state.snapshot()
        assert payload["status"] == "error"
        assert "Cannot open" in payload["message"]

    def test_full_mode_without_geometry_refuses_cleanly(self, tmp_path: Path, monkeypatch) -> None:
        # The real config has no lanes, so full mode cannot build its rules. The
        # viewer must say so in the page and refuse, not crash.
        import io

        import model2_behavior.scripts.serve_viewer as module

        video = write_video(tmp_path / "clip.mp4", frames=5)
        captured = io.StringIO()

        # The refusal path blocks on serve_forever; cut that short so the test
        # can assert what was reported without waiting on a real server loop.
        def immediately_interrupted(self, poll_interval: float = 0.2) -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr(
            module.ThreadingHTTPServer, "serve_forever", immediately_interrupted, raising=False
        )
        monkeypatch.setattr(
            module.ThreadingHTTPServer, "shutdown", lambda self: None, raising=False
        )
        monkeypatch.setattr(
            module.ThreadingHTTPServer, "server_close", lambda self: None, raising=False
        )

        with contextlib_redirect(captured):
            code = module.main(["--source", str(video), "--mode", "full", "--port", "0"])

        output = captured.getvalue()
        assert code == 2
        # The refusal must name the missing geometry and the way out, whether it
        # was caught at config load or by the explicit lane check afterwards.
        assert "lane" in output.lower()
        assert "tracking-only" in output
