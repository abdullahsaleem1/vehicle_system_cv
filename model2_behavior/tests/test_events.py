"""Event contract: field presence, types, and JSON round-trip.

The downstream Node.js/Express backend consumes this contract, so these tests
are the guard against accidental breaking changes.

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from model2_behavior.src.events import (
    INCIDENT_TYPES,
    REQUIRED_EVENT_FIELDS,
    Evidence,
    EventWriter,
    IncidentEvent,
    load_events,
    utc_now_iso,
    validate_event_dict,
)


def sample_event(**overrides) -> IncidentEvent:
    """Build a valid event, with overrides applied.

    Args:
        **overrides: Fields to replace.

    Returns:
        A populated event.
    """
    fields = {
        "incident_type": "stopped_vehicle",
        "severity": "Medium",
        "camera_id": "cam_01",
        "timestamp": utc_now_iso(),
        "frame_index": 42,
        "track_ids": [7],
        "vehicle_classes": ["car"],
        "bounding_boxes": [[10.0, 20.0, 50.0, 80.0]],
        "confidence": 0.87,
        "rule_params": {"speed_thresh_bl_s": 0.05},
        "zone_id": "driving_lane",
    }
    fields.update(overrides)
    return IncidentEvent(**fields)


class TestIncidentEvent:
    """Event construction and validation."""

    def test_valid_event_constructs(self) -> None:
        event = sample_event()
        assert event.event_id
        uuid.UUID(event.event_id)

    def test_unknown_incident_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="incident_type"):
            sample_event(incident_type="jaywalking")

    def test_unknown_severity_rejected(self) -> None:
        with pytest.raises(ValueError, match="severity"):
            sample_event(severity="Critical")

    def test_mismatched_lengths_rejected(self) -> None:
        with pytest.raises(ValueError, match="bounding_boxes"):
            sample_event(track_ids=[1, 2], vehicle_classes=["car"], bounding_boxes=[[0, 0, 1, 1]])

    def test_out_of_range_confidence_rejected(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            sample_event(confidence=1.5)

    def test_every_incident_type_is_accepted(self) -> None:
        for name in INCIDENT_TYPES:
            assert sample_event(incident_type=name).incident_type == name


class TestEventDict:
    """Serialised shape consumed by the backend."""

    def test_contract_fields_all_present(self) -> None:
        payload = sample_event().to_dict()
        for field in REQUIRED_EVENT_FIELDS:
            assert field in payload, f"missing contract field: {field}"

    def test_no_extra_top_level_fields(self) -> None:
        payload = sample_event().to_dict()
        assert set(payload) == set(REQUIRED_EVENT_FIELDS)

    def test_evidence_shape(self) -> None:
        payload = sample_event().to_dict()
        assert set(payload["evidence"]) == {"snapshot_path", "clip_path"}

    def test_timestamp_is_iso_utc(self) -> None:
        payload = sample_event().to_dict()
        assert payload["timestamp"].endswith("Z")
        assert "T" in payload["timestamp"]

    def test_confidence_is_rounded(self) -> None:
        payload = sample_event(confidence=0.123456).to_dict()
        assert payload["confidence"] == pytest.approx(0.1235)

    def test_zone_id_may_be_null(self) -> None:
        payload = sample_event(zone_id=None).to_dict()
        assert payload["zone_id"] is None

    def test_payload_is_json_serialisable(self) -> None:
        payload = sample_event().to_dict()
        assert json.loads(json.dumps(payload)) == payload


class TestValidateEventDict:
    """The validator must accept valid payloads and flag specific problems."""

    def test_valid_payload_has_no_problems(self) -> None:
        assert validate_event_dict(sample_event().to_dict()) == []

    def test_missing_field_reported(self) -> None:
        payload = sample_event().to_dict()
        payload.pop("severity")
        problems = validate_event_dict(payload)
        assert any("severity" in problem for problem in problems)

    def test_bad_timestamp_reported(self) -> None:
        payload = sample_event().to_dict()
        payload["timestamp"] = "2026-01-01 00:00:00"
        assert any("timestamp" in problem for problem in validate_event_dict(payload))

    def test_bad_event_id_reported(self) -> None:
        payload = sample_event().to_dict()
        payload["event_id"] = "not-a-uuid"
        assert any("event_id" in problem for problem in validate_event_dict(payload))

    def test_empty_track_ids_reported(self) -> None:
        payload = sample_event().to_dict()
        payload["track_ids"] = []
        assert any("track_ids" in problem for problem in validate_event_dict(payload))

    def test_malformed_box_reported(self) -> None:
        payload = sample_event().to_dict()
        payload["bounding_boxes"] = [[1.0, 2.0, 3.0]]
        assert any("bounding_boxes" in problem for problem in validate_event_dict(payload))

    def test_missing_evidence_key_reported(self) -> None:
        payload = sample_event().to_dict()
        payload["evidence"] = {"snapshot_path": None}
        assert any("clip_path" in problem for problem in validate_event_dict(payload))

    def test_negative_frame_index_reported(self) -> None:
        payload = sample_event().to_dict()
        payload["frame_index"] = -1
        assert any("frame_index" in problem for problem in validate_event_dict(payload))


class TestEventWriter:
    """Writing events and snapshots to a run directory."""

    def test_writer_creates_directories(self, tmp_path: Path) -> None:
        writer = EventWriter(tmp_path / "run", camera_id="cam_01")
        assert writer.output_dir.is_dir()
        assert writer.snapshot_dir.is_dir()

    def test_add_flushes_immediately(self, tmp_path: Path) -> None:
        writer = EventWriter(tmp_path / "run", camera_id="cam_01")
        writer.add(sample_event())
        assert writer.events_path.is_file()
        events = load_events(writer.events_path)
        assert len(events) == 1

    def test_document_metadata(self, tmp_path: Path) -> None:
        writer = EventWriter(tmp_path / "run", camera_id="cam_01")
        writer.add(sample_event())
        document = json.loads(writer.events_path.read_text(encoding="utf-8"))
        assert document["camera_id"] == "cam_01"
        assert document["total_events"] == 1
        assert document["schema_version"] == 1

    def test_counts_by_type_includes_all_types(self, tmp_path: Path) -> None:
        writer = EventWriter(tmp_path / "run", camera_id="cam_01")
        writer.add(sample_event(incident_type="wrong_way", severity="High"))
        writer.add(sample_event(incident_type="wrong_way", severity="High"))
        writer.add(sample_event(incident_type="sudden_decel"))
        counts = writer.counts_by_type()
        assert counts["wrong_way"] == 2
        assert counts["sudden_decel"] == 1
        assert counts["collision_proxy"] == 0

    def test_snapshot_path_is_deterministic(self, tmp_path: Path) -> None:
        writer = EventWriter(tmp_path / "run", camera_id="cam_01")
        event = sample_event()
        first = writer.snapshot_path_for(event)
        second = writer.snapshot_path_for(event)
        assert first == second
        assert first.parent == writer.snapshot_dir

    def test_new_event_fills_camera_and_timestamp(self, tmp_path: Path) -> None:
        writer = EventWriter(tmp_path / "run", camera_id="cam_09")
        event = writer.new_event(
            incident_type="sudden_decel",
            severity="Medium",
            frame_index=1,
            track_ids=[1],
            vehicle_classes=["car"],
            bounding_boxes=[[0.0, 0.0, 1.0, 1.0]],
            confidence=0.5,
        )
        assert event.camera_id == "cam_09"
        assert event.timestamp.endswith("Z")

    def test_load_events_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_events(tmp_path / "absent.json")

    def test_evidence_defaults_are_null(self) -> None:
        assert Evidence().to_dict() == {"snapshot_path": None, "clip_path": None}
