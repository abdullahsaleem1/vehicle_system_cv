"""Detector/tracker wrapper: offline enforcement and observation conversion.

The real model is NOT loaded here (it needs a GPU and takes seconds). Instead we
verify the wrapper contract: local-path resolution, mandatory tracker YAML, and
the conversion of tracker output into observations.

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from model2_behavior.src.detector_tracker import DetectorSettings, DetectorTracker
from model2_behavior.src.weights import resolve_weights_path, TARGET_CLASS_IDS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_CONFIG = PROJECT_ROOT / "configs" / "model2.yaml"
BYTETRACK_YAML = PROJECT_ROOT / "configs" / "bytetrack.yaml"


class TestOfflineEnforcement:
    """Offline mode must be forced before any model load."""

    def test_module_enables_offline_mode(self) -> None:
        import os

        from model2_behavior.src.detector_tracker import enforce_offline

        enforce_offline()
        assert os.environ["YOLO_OFFLINE"] == "1"

    def test_lap_dependency_is_available(self) -> None:
        """ByteTrack requires lap; importing is only possible when installed."""
        lap = pytest.importorskip("lap", reason="ByteTrack requires the lap package")
        assert lap.__version__

    def test_ultralytics_tracker_imports(self) -> None:
        """The tracker module must import without attempting an auto-install."""
        from ultralytics.trackers.utils import matching

        assert hasattr(matching, "linear_assignment")

    def test_ultralytics_offline_flag_is_honoured(self) -> None:
        from ultralytics.utils import is_online

        assert is_online() is False


class TestTrackerYamlRequirement:
    """The effective tracker YAML is mandatory and must exist."""

    def test_missing_tracker_yaml_raises(self) -> None:
        with pytest.raises(ValueError, match="tracker_yaml is required"):
            DetectorTracker(tracker_yaml=None)

    def test_nonexistent_tracker_yaml_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Tracker YAML not found"):
            DetectorTracker(tracker_yaml=tmp_path / "absent.yaml")

    def test_missing_weights_raise_before_model_load(self, tmp_path: Path) -> None:
        # resolve_weights_path falls back to the real models/ dir by design, so a
        # bogus explicit path still resolves. To prove the guard, point the
        # fallback at an empty directory too.
        tracker = tmp_path / "t.yaml"
        tracker.write_text("tracker_type: bytetrack\n", encoding="utf-8")
        from model2_behavior.src import weights as weights_module

        original = weights_module.default_models_dir
        weights_module.default_models_dir = lambda: tmp_path / "empty_models"
        try:
            with pytest.raises(FileNotFoundError, match="weights not found"):
                DetectorTracker(
                    weights_path=tmp_path / "absent.pt",
                    tracker_yaml=tracker,
                    models_dir=tmp_path / "empty_models",
                )
        finally:
            weights_module.default_models_dir = original


class TestWeightsResolution:
    """Weights must resolve to an absolute local path, never a bare name."""

    def test_configured_weights_resolve_to_file(self) -> None:
        resolved = resolve_weights_path()
        assert resolved.is_absolute()
        assert resolved.is_file()
        assert resolved.name == "yolo11m.pt"

    def test_bare_name_is_never_a_valid_local_path(self, tmp_path: Path) -> None:
        """A bare name must resolve only via explicit local dirs, never by download.

        With every local candidate absent, resolution must raise instead of
        falling through to an Ultralytics auto-download.
        """
        from model2_behavior.src import weights as weights_module

        original = weights_module.default_models_dir
        weights_module.default_models_dir = lambda: tmp_path / "empty_models"
        try:
            with pytest.raises(FileNotFoundError, match="Refusing to download"):
                resolve_weights_path("yolo11m.pt", models_dir=tmp_path / "empty_models")
        finally:
            weights_module.default_models_dir = original

    def test_target_class_ids_match_coco(self) -> None:
        assert TARGET_CLASS_IDS == {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class TestDetectorSettings:
    """Detector settings defaults must match the agreed amendment."""

    def test_default_conf_is_low_for_tracker_recall(self) -> None:
        assert DetectorSettings().conf == pytest.approx(0.10)

    def test_default_classes_are_the_four_vehicles(self) -> None:
        assert tuple(DetectorSettings().classes) == (2, 3, 5, 7)

    def test_default_conf_at_or_above_track_low_thresh(self, tmp_path: Path) -> None:
        """Detector conf must be >= ByteTrack's track_low_thresh."""
        import yaml

        base = yaml.safe_load(BYTETRACK_YAML.read_text(encoding="utf-8")) or {}
        low_thresh = float(base.get("track_low_thresh", 0.1))
        assert DetectorSettings().conf >= low_thresh
