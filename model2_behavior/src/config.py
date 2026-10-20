"""Configuration loading, validation, and effective-tracker materialisation.

Responsibilities:
    * Load ``configs/model2.yaml`` and validate required keys and value ranges.
    * Resolve all paths to absolute local paths (pathlib, Windows-safe).
    * Merge ``tracker_config_path`` with ``tracker_params`` into an effective
      tracker YAML written into the run output folder, which is the file handed
      to ``model.track()``.
    * Expose the fully resolved configuration so nothing is silently defaulted.

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from model2_behavior.src.logging_setup import get_logger
from model2_behavior.src.weights import project_root

logger = get_logger(__name__)

REQUIRED_TOP_LEVEL = ("camera_id", "device", "weights_path", "tracker_config_path", "detector", "rules")
VALID_SEVERITIES = ("High", "Medium", "Low")
RULE_NAMES = ("stopped_vehicle", "wrong_way", "sudden_decel", "collision_proxy")


class ConfigError(ValueError):
    """Raised when the configuration is missing keys or holds invalid values."""


@dataclass
class RuleConfig:
    """Resolved configuration for a single rule."""

    name: str
    enabled: bool
    severity: str
    debounce_s: float
    min_consecutive_s: float
    params: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable view used in event ``rule_params``."""
        payload = {
            "enabled": self.enabled,
            "severity": self.severity,
            "debounce_s": self.debounce_s,
            "min_consecutive_s": self.min_consecutive_s,
        }
        payload.update(self.params)
        return payload


@dataclass
class Model2Config:
    """Fully resolved Model 2 configuration."""

    raw: Dict[str, Any]
    config_path: Path
    camera_id: str
    device: Any
    weights_path: Path
    tracker_config_path: Path
    detector: Dict[str, Any]
    tracker_params: Dict[str, Any]
    rule_layer: Dict[str, Any]
    rules: Dict[str, RuleConfig]
    rois: Dict[str, Any]

    def effective_snapshot(self) -> Dict[str, Any]:
        """Return a serialisable snapshot of every effective value.

        Written alongside each run so the logs never rely on silent defaults.
        """
        return {
            "config_path": str(self.config_path),
            "camera_id": self.camera_id,
            "device": self.device,
            "weights_path": str(self.weights_path),
            "tracker_config_path": str(self.tracker_config_path),
            "detector": dict(self.detector),
            "tracker_params": dict(self.tracker_params),
            "rule_layer": dict(self.rule_layer),
            "rules": {name: cfg.as_dict() for name, cfg in self.rules.items()},
            "rois": self.rois,
        }


def _require(mapping: Dict[str, Any], key: str, where: str) -> Any:
    """Return ``mapping[key]`` or raise :class:`ConfigError`.

    Args:
        mapping: Mapping to read.
        key: Required key.
        where: Human-readable location used in the error message.

    Returns:
        The value at ``key``.

    Raises:
        ConfigError: If the key is absent.
    """
    if key not in mapping:
        raise ConfigError(f"Missing required key '{key}' in {where}")
    return mapping[key]


def _as_positive_float(value: Any, key: str) -> float:
    """Coerce ``value`` to a strictly positive float.

    Args:
        value: Raw value.
        key: Key name for error messages.

    Returns:
        Positive float.

    Raises:
        ConfigError: If the value is not numeric or not positive.
    """
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"'{key}' must be numeric, got {value!r}") from exc
    if number <= 0:
        raise ConfigError(f"'{key}' must be > 0, got {number}")
    return number


def load_config(path: Optional[str | Path] = None, validate: bool = True) -> Model2Config:
    """Load and validate the Model 2 configuration.

    Args:
        path: Config file path. Defaults to ``configs/model2.yaml``.
        validate: When True (default), enforce cross-field requirements such as
            the driving-lane ROI needed by ``stopped_vehicle``. Set False only
            for inspection/reporting so a pending required input can be shown
            rather than raised.

    Returns:
        A validated :class:`Model2Config`.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ConfigError: If required keys are missing or values are invalid.
    """
    config_path = Path(path) if path is not None else project_root() / "configs" / "model2.yaml"
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Config root must be a mapping, got {type(raw).__name__}")

    for key in REQUIRED_TOP_LEVEL:
        _require(raw, key, "config root")

    detector = raw["detector"]
    if not isinstance(detector, dict):
        raise ConfigError("'detector' must be a mapping")
    _require(detector, "imgsz", "detector")
    _require(detector, "conf", "detector")

    tracker_params = raw.get("tracker_params") or {}
    if not isinstance(tracker_params, dict):
        raise ConfigError("'tracker_params' must be a mapping")

    # track_buffer is a frame count in Ultralytics' tracker YAML. The project
    # specifies durations in SECONDS, converted at runtime using the measured FPS,
    # so a frames value here would silently depend on the stream rate.
    if "track_buffer" in tracker_params:
        raise ConfigError(
            "tracker_params.track_buffer is a frame count; use tracker_params.track_buffer_s "
            "instead (seconds, converted using the measured FPS)."
        )
    has_buffer_s = "track_buffer_s" in tracker_params
    if has_buffer_s:
        track_buffer_s = _as_positive_float(tracker_params["track_buffer_s"], "tracker_params.track_buffer_s")
    else:
        track_buffer_s = 3.0  # documented default; materialised into the effective YAML
    tracker_params["track_buffer_s"] = track_buffer_s

    # ---- path resolution (absolute, Windows-safe) ---------------------------
    weights_raw = Path(str(raw["weights_path"]))
    weights_path = weights_raw if weights_raw.is_absolute() else (project_root() / weights_raw)
    weights_path = weights_path.resolve()

    tracker_raw = Path(str(raw["tracker_config_path"]))
    tracker_config_path = tracker_raw if tracker_raw.is_absolute() else (project_root() / tracker_raw)
    tracker_config_path = tracker_config_path.resolve()
    if not tracker_config_path.is_file():
        raise FileNotFoundError(f"Tracker config not found: {tracker_config_path}")

    # ---- rule layer ---------------------------------------------------------
    rule_layer = dict(raw.get("rule_layer") or {})
    rule_layer.setdefault("min_track_mean_conf", 0.35)
    rule_layer.setdefault("min_track_age_s", 0.5)
    rule_layer.setdefault("stationary_speed_thresh", 0.05)
    rule_layer.setdefault("log_raw_pixels", True)

    # ---- rules --------------------------------------------------------------
    rules_raw = raw["rules"]
    if not isinstance(rules_raw, dict):
        raise ConfigError("'rules' must be a mapping")
    for missing in (name for name in RULE_NAMES if name not in rules_raw):
        raise ConfigError(f"Missing rule definition: '{missing}'")

    rules: Dict[str, RuleConfig] = {}
    for name in RULE_NAMES:
        entry = rules_raw[name]
        if not isinstance(entry, dict):
            raise ConfigError(f"Rule '{name}' must be a mapping")

        severity = str(entry.get("severity", "Medium"))
        if severity not in VALID_SEVERITIES:
            raise ConfigError(
                f"Rule '{name}' severity must be one of {VALID_SEVERITIES}, got {severity!r}"
            )

        if "min_consecutive_frames" in entry:
            raise ConfigError(
                f"Rule '{name}' uses min_consecutive_frames; use min_consecutive_s instead "
                "(frame counts are derived from measured FPS at runtime)."
            )

        debounce_s = float(entry.get("debounce_s", 0.0))
        if debounce_s < 0:
            raise ConfigError(f"Rule '{name}' debounce_s must be >= 0")

        min_consecutive_s = _as_positive_float(
            entry.get("min_consecutive_s", 0.1), f"rules.{name}.min_consecutive_s"
        )

        params = {
            k: v
            for k, v in entry.items()
            if k not in ("enabled", "severity", "debounce_s", "min_consecutive_s")
        }
        rules[name] = RuleConfig(
            name=name,
            enabled=bool(entry.get("enabled", True)),
            severity=severity,
            debounce_s=debounce_s,
            min_consecutive_s=min_consecutive_s,
            params=params,
        )

    # ---- ROI sanity ---------------------------------------------------------
    rois = dict(raw.get("rois") or {})
    if validate and rules["stopped_vehicle"].enabled:
        lanes = rois.get("lanes")
        has_lane = isinstance(lanes, (list, tuple)) and any(
            isinstance(entry, dict) and entry.get("polygon") not in (None, "full_frame")
            for entry in lanes
        )
        if not has_lane:
            raise ConfigError(
                "rules.stopped_vehicle is enabled but rois.lanes has no lane polygon. "
                "At least one lane with a polygon is required (full_frame is not accepted). "
                "Author lanes with scripts/define_camera_geometry.py."
            )

    return Model2Config(
        raw=raw,
        config_path=config_path,
        camera_id=str(raw["camera_id"]),
        device=raw["device"],
        weights_path=weights_path,
        tracker_config_path=tracker_config_path,
        detector=detector,
        tracker_params=tracker_params,
        rule_layer=rule_layer,
        rules=rules,
        rois=rois,
    )


def build_effective_tracker_yaml(config: Model2Config, output_dir: Path, fps: float) -> Path:
    """Merge tracker params into the tracker config and write it to ``output_dir``.

    The tracker YAML is a flat top-level mapping, so a shallow update of the
    base file with ``tracker_params`` is the correct merge.

    ``tracker_params.track_buffer_s`` is a DURATION in seconds and is converted
    here to the integer ``track_buffer`` frame count that Ultralytics expects,
    using the measured FPS. The conversion is recorded in the run config
    snapshot so it is never a silent default.

    Args:
        config: Resolved configuration.
        output_dir: Run output directory (created if absent).
        fps: Measured frames per second, used to convert ``track_buffer_s``.

    Returns:
        Path to the effective tracker YAML that must be passed to ``model.track()``.

    Raises:
        ConfigError: If the base tracker file is not a mapping, or ``fps`` is not
            positive.
    """
    if fps <= 0:
        raise ConfigError(f"fps must be positive to convert track_buffer_s, got {fps}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with config.tracker_config_path.open("r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle) or {}
    if not isinstance(base, dict):
        raise ConfigError(f"Tracker config must be a mapping: {config.tracker_config_path}")

    merged = dict(base)
    merged.update(config.tracker_params)

    # Convert the seconds-based buffer duration into the frame count Ultralytics uses.
    track_buffer_s = float(merged.pop("track_buffer_s"))
    track_buffer_frames = max(1, int(round(track_buffer_s * fps)))
    merged["track_buffer"] = track_buffer_frames

    # tracker_type must stay consistent with the chosen file.
    merged.setdefault(
        "tracker_type",
        "bytetrack" if "bytetrack" in config.tracker_config_path.name.lower() else "botsort",
    )

    destination = output_path / "effective_tracker.yaml"
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(merged, handle, sort_keys=True)

    logger.info(
        "track_buffer_s %.3f s -> track_buffer %s frames at %.3f fps (written to %s)",
        track_buffer_s,
        track_buffer_frames,
        fps,
        destination,
    )
    return destination


def resolve_device(device: Any) -> str | int:
    """Resolve a configured device to a usable torch device.

    Falls back to CPU when CUDA is requested but unavailable, so the pipeline
    never crashes on a CPU-only machine.

    Args:
        device: Configured device (``0``, ``"0"``, ``"cpu"``, ...).

    Returns:
        ``"cpu"`` or a valid CUDA index.
    """
    import torch

    if isinstance(device, str) and device.strip().lower() == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    try:
        index = int(device)
    except (TypeError, ValueError):
        return 0
    if index < 0 or index >= torch.cuda.device_count():
        return 0
    return index


def list_effective_paths(config: Model2Config) -> List[str]:
    """Return the resolved local paths, for logging and Gate review.

    Args:
        config: Resolved configuration.

    Returns:
        List of ``key=path`` strings.
    """
    return [
        f"config={config.config_path}",
        f"weights={config.weights_path}",
        f"tracker={config.tracker_config_path}",
    ]
