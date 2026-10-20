"""Incident event schema and JSON writer.

The event contract is consumed by the Node.js/Express backend to create Alerts,
so field names and types are stable. This module is the single source of truth.

IMPORTANT SEMANTICS
-------------------
``confidence`` is the DETECTION confidence of the tracked vehicle(s) that
triggered the rule. It is NOT the probability that the incident actually
occurred. Behavior rules are heuristics, not calibrated classifiers, so a high
``confidence`` means "the detector was sure this was a vehicle", not "the
incident is certain".

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

INCIDENT_TYPES = ("stopped_vehicle", "wrong_way", "sudden_decel", "collision_proxy")
SEVERITIES = ("High", "Medium", "Low")


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with ``Z`` suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


@dataclass
class Evidence:
    """Paths to supporting artifacts for an event.

    Attributes:
        snapshot_path: JPEG frame with the incident annotated. Always present.
        clip_path: Short video clip around the incident, or None. Clip extraction
            is a Tier B item; Tier A writes snapshots only.
    """

    snapshot_path: Optional[str] = None
    clip_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return the JSON-serialisable form."""
        return {"snapshot_path": self.snapshot_path, "clip_path": self.clip_path}


@dataclass
class IncidentEvent:
    """A single detected incident.

    Field order mirrors the agreed contract; ``to_dict`` emits this order.

    Attributes:
        incident_type: One of :data:`INCIDENT_TYPES`.
        severity: One of :data:`SEVERITIES`.
        camera_id: Camera identifier from config.
        timestamp: ISO-8601 UTC time at which the incident fired.
        frame_index: Frame index at which the incident fired.
        track_ids: Track ids involved, sorted.
        vehicle_classes: Class names for the involved tracks, parallel to
            ``track_ids``.
        bounding_boxes: Boxes at trigger time, ``[[x1, y1, x2, y2], ...]``,
            parallel to ``track_ids``.
        confidence: DETECTION confidence (not incident probability).
        evidence: Snapshot (and optional clip) paths.
        rule_params: The threshold values actually used for this decision.
        zone_id: Matched ROI zone name, or None.
        event_id: UUID4 string.
    """

    incident_type: str
    severity: str
    camera_id: str
    timestamp: str
    frame_index: int
    track_ids: List[int]
    vehicle_classes: List[str]
    bounding_boxes: List[List[float]]
    confidence: float
    evidence: Evidence = field(default_factory=Evidence)
    rule_params: Dict[str, Any] = field(default_factory=dict)
    zone_id: Optional[str] = None
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        if self.incident_type not in INCIDENT_TYPES:
            raise ValueError(f"Unknown incident_type {self.incident_type!r}; expected {INCIDENT_TYPES}")
        if self.severity not in SEVERITIES:
            raise ValueError(f"Unknown severity {self.severity!r}; expected {SEVERITIES}")
        if len(self.track_ids) != len(self.bounding_boxes):
            raise ValueError("track_ids and bounding_boxes must have equal length")
        if len(self.track_ids) != len(self.vehicle_classes):
            raise ValueError("track_ids and vehicle_classes must have equal length")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    def to_dict(self) -> Dict[str, Any]:
        """Return the event as an ordered, JSON-serialisable dict."""
        return {
            "event_id": self.event_id,
            "incident_type": self.incident_type,
            "severity": self.severity,
            "camera_id": self.camera_id,
            "zone_id": self.zone_id,
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "track_ids": list(self.track_ids),
            "vehicle_classes": list(self.vehicle_classes),
            "bounding_boxes": [list(box) for box in self.bounding_boxes],
            "confidence": round(float(self.confidence), 4),
            "evidence": self.evidence.to_dict(),
            "rule_params": dict(self.rule_params),
        }


class EventWriter:
    """Appends incident events to a JSON document and writes snapshots.

    Events are held in memory and flushed to disk after every event, so a crash
    mid-run still leaves a valid JSON file containing everything so far.
    """

    def __init__(self, output_dir: Path, camera_id: str, snapshot_dir: Optional[Path] = None) -> None:
        """Initialise the writer.

        Args:
            output_dir: Run output directory; ``events.json`` is written here.
            camera_id: Camera identifier recorded in each event.
            snapshot_dir: Directory for snapshots. Defaults to
                ``output_dir / snapshots``.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.camera_id = camera_id
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else self.output_dir / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / "events.json"
        self._events: List[IncidentEvent] = []

    @property
    def events(self) -> List[IncidentEvent]:
        """The events recorded so far."""
        return list(self._events)

    def new_event(self, **kwargs: Any) -> IncidentEvent:
        """Construct an event bound to this writer's camera.

        Args:
            **kwargs: Fields forwarded to :class:`IncidentEvent`. ``camera_id``
                is filled from the writer unless explicitly overridden.

        Returns:
            The constructed (not yet recorded) event.
        """
        kwargs.setdefault("camera_id", self.camera_id)
        kwargs.setdefault("timestamp", utc_now_iso())
        return IncidentEvent(**kwargs)

    def add(self, event: IncidentEvent) -> None:
        """Record an event and flush immediately.

        Args:
            event: Event to record.
        """
        self._events.append(event)
        self.flush()

    def snapshot_path_for(self, event: IncidentEvent) -> Path:
        """Return the canonical snapshot path for an event.

        Args:
            event: Event being snapshotted.

        Returns:
            Path under the snapshot directory.
        """
        return self.snapshot_dir / f"{event.frame_index:08d}_{event.incident_type}_{event.event_id[:8]}.jpg"

    def flush(self) -> Path:
        """Write all recorded events to ``events.json``.

        Returns:
            Path to the written JSON file.

        Raises:
            OSError: If the file cannot be written.
        """
        payload = {
            "camera_id": self.camera_id,
            "generated_utc": utc_now_iso(),
            "schema_version": 1,
            "total_events": len(self._events),
            "events": [event.to_dict() for event in self._events],
        }
        self.events_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return self.events_path

    def counts_by_type(self) -> Dict[str, int]:
        """Return a mapping of incident type to count."""
        counts = {name: 0 for name in INCIDENT_TYPES}
        for event in self._events:
            counts[event.incident_type] = counts.get(event.incident_type, 0) + 1
        return counts


REQUIRED_EVENT_FIELDS: Sequence[str] = (
    "event_id",
    "incident_type",
    "severity",
    "camera_id",
    "zone_id",
    "timestamp",
    "frame_index",
    "track_ids",
    "vehicle_classes",
    "bounding_boxes",
    "confidence",
    "evidence",
    "rule_params",
)


def validate_event_dict(payload: Dict[str, Any]) -> List[str]:
    """Validate an event dict against the contract.

    Args:
        payload: Event dictionary, typically from JSON.

    Returns:
        List of human-readable problems. Empty means valid.
    """
    problems: List[str] = []

    for name in REQUIRED_EVENT_FIELDS:
        if name not in payload:
            problems.append(f"missing field: {name}")

    if payload.get("incident_type") not in INCIDENT_TYPES:
        problems.append(f"incident_type must be one of {INCIDENT_TYPES}, got {payload.get('incident_type')!r}")
    if payload.get("severity") not in SEVERITIES:
        problems.append(f"severity must be one of {SEVERITIES}, got {payload.get('severity')!r}")

    try:
        uuid.UUID(str(payload.get("event_id")))
    except (ValueError, TypeError, AttributeError):
        problems.append(f"event_id must be a UUID, got {payload.get('event_id')!r}")

    timestamp = payload.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
        problems.append(f"timestamp must be an ISO-8601 UTC string ending in Z, got {timestamp!r}")

    frame_index = payload.get("frame_index")
    if not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index < 0:
        problems.append(f"frame_index must be a non-negative int, got {frame_index!r}")

    track_ids = payload.get("track_ids")
    boxes = payload.get("bounding_boxes")
    classes = payload.get("vehicle_classes")
    if not isinstance(track_ids, list) or not track_ids:
        problems.append("track_ids must be a non-empty list")
    if not isinstance(boxes, list) or not boxes:
        problems.append("bounding_boxes must be a non-empty list")
    if not isinstance(classes, list) or not classes:
        problems.append("vehicle_classes must be a non-empty list")

    if isinstance(track_ids, list) and isinstance(boxes, list) and len(track_ids) != len(boxes):
        problems.append("track_ids and bounding_boxes length mismatch")
    if isinstance(track_ids, list) and isinstance(classes, list) and len(track_ids) != len(classes):
        problems.append("track_ids and vehicle_classes length mismatch")

    if isinstance(boxes, list):
        for index, box in enumerate(boxes):
            if not isinstance(box, list) or len(box) != 4:
                problems.append(f"bounding_boxes[{index}] must be [x1, y1, x2, y2]")

    confidence = payload.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        problems.append(f"confidence must be numeric, got {confidence!r}")
    elif not 0.0 <= float(confidence) <= 1.0:
        problems.append(f"confidence must be in [0, 1], got {confidence}")

    evidence = payload.get("evidence")
    if not isinstance(evidence, dict):
        problems.append("evidence must be an object")
    else:
        for key in ("snapshot_path", "clip_path"):
            if key not in evidence:
                problems.append(f"evidence missing key: {key}")

    if "zone_id" not in payload:
        problems.append("zone_id must be present (may be null)")
    if "rule_params" not in payload or not isinstance(payload.get("rule_params"), dict):
        problems.append("rule_params must be an object")

    return problems


def load_events(path: Path) -> List[Dict[str, Any]]:
    """Load events from a JSON document.

    Args:
        path: Path to ``events.json``.

    Returns:
        List of event dicts.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the document is malformed.
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"No events file at {target}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        raise ValueError(f"Malformed events document at {target}")
    return payload["events"]
