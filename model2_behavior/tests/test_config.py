"""Config loading, validation, and effective-tracker materialisation.

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from model2_behavior.src.config import (
    ConfigError,
    build_effective_tracker_yaml,
    load_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_CONFIG = PROJECT_ROOT / "configs" / "model2.yaml"


def write_config(tmp_path: Path, overrides: dict) -> Path:
    """Write a config based on the real file with shallow overrides applied.

    Args:
        tmp_path: pytest temporary directory.
        overrides: Top-level keys to replace.

    Returns:
        Path to the written config.
    """
    raw = yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))
    raw.update(overrides)
    target = tmp_path / "model2.yaml"
    target.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return target


def valid_rois() -> dict:
    """Return a complete ``rois`` block, so validation can pass."""
    return {
        "lanes": [
            {
                "name": "lane_a",
                "polygon": [[40, 240], [600, 240], [600, 470], [40, 470]],
                "direction": [100, 400, 540, 400],
            }
        ],
        "zones": {},
        "reference_resolution": [640, 480],
    }


class TestRuleConfig:
    """Per-rule threshold resolution."""

    def test_seconds_are_kept_as_seconds(self) -> None:
        config = load_config(REAL_CONFIG, validate=False)
        assert config.rules["stopped_vehicle"].min_consecutive_s == pytest.approx(10.0)

    def test_min_consecutive_frames_is_rejected(self, tmp_path: Path) -> None:
        raw = yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))
        raw["rules"]["sudden_decel"]["min_consecutive_frames"] = 5
        target = tmp_path / "bad.yaml"
        target.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        with pytest.raises(ConfigError, match="min_consecutive_frames"):
            load_config(target, validate=False)

    def test_invalid_severity_rejected(self, tmp_path: Path) -> None:
        raw = yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))
        raw["rules"]["wrong_way"]["severity"] = "Catastrophic"
        target = tmp_path / "bad.yaml"
        target.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        with pytest.raises(ConfigError, match="severity"):
            load_config(target, validate=False)

    def test_negative_debounce_rejected(self, tmp_path: Path) -> None:
        raw = yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))
        raw["rules"]["sudden_decel"]["debounce_s"] = -1.0
        target = tmp_path / "bad.yaml"
        target.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        with pytest.raises(ConfigError, match="debounce_s"):
            load_config(target, validate=False)


class TestValidation:
    """Cross-field validation."""

    def test_real_config_needs_lane_for_stopped_vehicle(self) -> None:
        # The shipped config has no lane yet, so strict loading must refuse.
        with pytest.raises(ConfigError, match="lanes"):
            load_config(REAL_CONFIG, validate=True)

    def test_validate_false_allows_pending_geometry(self) -> None:
        config = load_config(REAL_CONFIG, validate=False)
        assert config.rois.get("lanes") == []

    def test_full_frame_lane_is_rejected_by_validation(self, tmp_path: Path) -> None:
        target = write_config(
            tmp_path,
            {
                "rois": {
                    **valid_rois(),
                    "lanes": [{"name": "lane_a", "polygon": "full_frame", "direction": None}],
                }
            },
        )
        with pytest.raises(ConfigError, match="lanes"):
            load_config(target, validate=True)

    def test_lane_present_passes_validation(self, tmp_path: Path) -> None:
        target = write_config(tmp_path, {"rois": valid_rois()})
        config = load_config(target, validate=True)
        assert len(config.rois["lanes"]) == 1

    def test_missing_required_top_level_key_rejected(self, tmp_path: Path) -> None:
        raw = yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))
        raw.pop("camera_id")
        target = tmp_path / "bad.yaml"
        target.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        with pytest.raises(ConfigError, match="camera_id"):
            load_config(target, validate=False)

    def test_missing_rule_rejected(self, tmp_path: Path) -> None:
        raw = yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))
        raw["rules"].pop("collision_proxy")
        target = tmp_path / "bad.yaml"
        target.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        with pytest.raises(ConfigError, match="collision_proxy"):
            load_config(target, validate=False)

    def test_missing_config_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nope.yaml")


class TestPathResolution:
    """Weights and tracker paths must resolve to absolute local paths."""

    def test_weights_path_is_absolute_and_local(self) -> None:
        config = load_config(REAL_CONFIG, validate=False)
        assert config.weights_path.is_absolute()
        assert config.weights_path.is_file()
        # Never a bare name: the resolved path must contain a directory.
        assert config.weights_path.parent != Path(".")

    def test_tracker_config_path_exists(self) -> None:
        config = load_config(REAL_CONFIG, validate=False)
        assert config.tracker_config_path.is_absolute()
        assert config.tracker_config_path.is_file()


class TestEffectiveTracker:
    """Merging tracker params into a run-local tracker YAML."""

    def test_effective_yaml_written_to_output_dir(self, tmp_path: Path) -> None:
        config = load_config(REAL_CONFIG, validate=False)
        destination = build_effective_tracker_yaml(config, tmp_path, fps=25.0)
        assert destination.is_file()
        assert destination.parent == tmp_path
        assert destination.name == "effective_tracker.yaml"

    def test_params_override_base_file(self, tmp_path: Path) -> None:
        config = load_config(REAL_CONFIG, validate=False)
        destination = build_effective_tracker_yaml(config, tmp_path, fps=25.0)
        merged = yaml.safe_load(destination.read_text(encoding="utf-8"))
        for key, value in config.tracker_params.items():
            if key == "track_buffer_s":
                continue
            assert merged[key] == value

    def test_tracker_type_preserved(self, tmp_path: Path) -> None:
        config = load_config(REAL_CONFIG, validate=False)
        destination = build_effective_tracker_yaml(config, tmp_path, fps=25.0)
        merged = yaml.safe_load(destination.read_text(encoding="utf-8"))
        assert merged["tracker_type"] == "bytetrack"

    def test_track_buffer_s_is_converted_to_frames(self, tmp_path: Path) -> None:
        config = load_config(REAL_CONFIG, validate=False)
        # 3.0 s at 10 fps -> 30 frames; at 30 fps -> 90 frames.
        for fps, expected in ((10.0, 30), (30.0, 90)):
            destination = build_effective_tracker_yaml(config, tmp_path / f"f{fps}", fps=fps)
            merged = yaml.safe_load(destination.read_text(encoding="utf-8"))
            assert merged["track_buffer"] == expected
            assert "track_buffer_s" not in merged

    def test_non_positive_fps_rejected(self, tmp_path: Path) -> None:
        config = load_config(REAL_CONFIG, validate=False)
        with pytest.raises(ConfigError):
            build_effective_tracker_yaml(config, tmp_path, fps=0.0)

    def test_track_buffer_frames_key_is_rejected(self, tmp_path: Path) -> None:
        raw = yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))
        raw["tracker_params"].pop("track_buffer_s", None)
        raw["tracker_params"]["track_buffer"] = 30
        target = tmp_path / "bad.yaml"
        target.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        with pytest.raises(ConfigError, match="track_buffer"):
            load_config(target, validate=False)


class TestEffectiveSnapshot:
    """Every effective value must be recorded, not silently defaulted."""

    def test_snapshot_records_all_sections(self) -> None:
        config = load_config(REAL_CONFIG, validate=False)
        snapshot = config.effective_snapshot()
        for key in ("camera_id", "device", "weights_path", "tracker_config_path", "detector", "rules", "rois"):
            assert key in snapshot

    def test_snapshot_records_rule_thresholds(self) -> None:
        config = load_config(REAL_CONFIG, validate=False)
        snapshot = config.effective_snapshot()
        stopped = snapshot["rules"]["stopped_vehicle"]
        assert stopped["min_consecutive_s"] == pytest.approx(10.0)
        assert stopped["speed_thresh_bl_s"] == pytest.approx(0.05)

    def test_detector_conf_is_low_for_tracker_recall(self) -> None:
        config = load_config(REAL_CONFIG, validate=False)
        assert config.detector["conf"] == pytest.approx(0.10)

    def test_rule_layer_precision_filter_present(self) -> None:
        config = load_config(REAL_CONFIG, validate=False)
        assert config.rule_layer["min_track_mean_conf"] == pytest.approx(0.35)


class TestStaticCameraCheck:
    """The static-camera precondition check is configured and validated."""

    def test_real_config_declares_the_check(self) -> None:
        config = load_config(REAL_CONFIG, validate=False)
        assert config.static_camera_check["enabled"] is True
        assert config.static_camera_check["threshold"] > 0

    def test_snapshot_records_the_check(self) -> None:
        config = load_config(REAL_CONFIG, validate=False)
        snapshot = config.effective_snapshot()
        assert snapshot["static_camera_check"]["enabled"] is True

    def test_missing_block_defaults_to_enabled(self, tmp_path: Path) -> None:
        raw = yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))
        raw.pop("static_camera_check", None)
        target = tmp_path / "model2.yaml"
        target.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        config = load_config(target, validate=False)
        # An absent block means "check with defaults", not "silently skip".
        assert config.static_camera_check == {}
        assert config.effective_snapshot()["static_camera_check"] == {}

    def test_invalid_threshold_is_rejected_at_load(self, tmp_path: Path) -> None:
        path = write_config(tmp_path, {"static_camera_check": {"threshold": 0.0}})
        with pytest.raises(ConfigError, match="static_camera_check is invalid"):
            load_config(path, validate=False)

    def test_unknown_key_is_rejected_at_load(self, tmp_path: Path) -> None:
        path = write_config(tmp_path, {"static_camera_check": {"threshhold": 0.02}})
        with pytest.raises(ConfigError, match="static_camera_check is invalid"):
            load_config(path, validate=False)
