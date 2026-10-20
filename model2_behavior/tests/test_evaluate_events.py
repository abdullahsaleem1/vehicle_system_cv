"""Unit tests for the Step 8 evaluation harness, on hand-made fixtures only.

The harness is written before any labelled footage exists, so its correctness
must be pinned now: a bug in the matcher would silently misreport precision and
recall the moment real data arrives. Every fixture here is a handful of literal
rows and events, so a failure points at the matcher rather than at the data.

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from model2_behavior.scripts.evaluate_events import (
    DEFAULT_MATCHES_PER_LABEL,
    DEFAULT_TOLERANCE_S,
    EvaluationError,
    Label,
    Prediction,
    build_clips,
    clip_name_matches,
    discover_clips,
    load_events_file,
    load_labels,
    match_predictions,
    parse_label_rows,
    run_evaluation,
    score_rules,
)


def write_labels(path: Path, rows: list) -> Path:
    """Write a labels.csv from header + row tuples.

    Args:
        path: Destination file.
        rows: Data rows, each ``(clip, camera_id, incident_type, start_s, end_s, notes)``.

    Returns:
        The written path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["clip", "camera_id", "incident_type", "start_s", "end_s", "notes"])
        writer.writerows(rows)
    return path


def write_events(path: Path, events: list) -> Path:
    """Write an events.json with the given event dicts.

    Args:
        path: Destination file.
        events: Event dictionaries.

    Returns:
        The written path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"events": events}, indent=2), encoding="utf-8")
    return path


def event(incident_type: str, timestamp_s: float, event_id: str = "e1") -> dict:
    """Build a minimal event dict.

    Args:
        incident_type: Rule name.
        timestamp_s: Stream timestamp in seconds.
        event_id: Event id.

    Returns:
        The event dictionary.
    """
    return {
        "event_id": event_id,
        "incident_type": incident_type,
        "timestamp_s": timestamp_s,
        "severity": "Medium",
        "confidence": 0.8,
    }


class TestParseLabelRows:
    """Labels-file parsing and its rejection of ambiguous rows."""

    def row(self, **overrides) -> dict:
        """Return a valid label row, optionally overridden.

        Args:
            **overrides: Column values to replace.

        Returns:
            A DictReader-style row.
        """
        base = {
            "clip": "clip_a.mp4",
            "camera_id": "cam_01",
            "incident_type": "wrong_way",
            "start_s": "10.0",
            "end_s": "12.0",
            "notes": "",
        }
        base.update(overrides)
        return base

    def test_parses_a_labelled_row(self) -> None:
        labels = parse_label_rows([self.row()], "labels.csv")
        assert labels[0].incident_type == "wrong_way"
        assert labels[0].start_s == 10.0
        assert labels[0].end_s == 12.0
        assert labels[0].is_negative is False

    def test_empty_incident_type_marks_a_negative_clip(self) -> None:
        labels = parse_label_rows(
            [self.row(incident_type="", start_s="", end_s="")], "labels.csv"
        )
        assert labels[0].is_negative is True

    def test_negative_row_with_timestamps_is_rejected(self) -> None:
        with pytest.raises(EvaluationError, match="must leave start_s and end_s empty"):
            parse_label_rows([self.row(incident_type="")], "labels.csv")

    def test_unknown_incident_type_is_rejected(self) -> None:
        with pytest.raises(EvaluationError, match="unknown incident_type"):
            parse_label_rows([self.row(incident_type="jaywalking")], "labels.csv")

    def test_missing_clip_is_rejected(self) -> None:
        with pytest.raises(EvaluationError, match="'clip' is empty"):
            parse_label_rows([self.row(clip="")], "labels.csv")

    def test_inverted_window_is_rejected(self) -> None:
        with pytest.raises(EvaluationError, match="precedes start_s"):
            parse_label_rows([self.row(start_s="20.0", end_s="10.0")], "labels.csv")

    def test_non_numeric_timestamp_is_rejected(self) -> None:
        with pytest.raises(EvaluationError, match="must be numbers"):
            parse_label_rows([self.row(start_s="ten")], "labels.csv")

    def test_instant_label_uses_start_when_end_absent(self) -> None:
        labels = parse_label_rows([self.row(end_s="")], "labels.csv")
        assert labels[0].start_s == labels[0].end_s == 10.0

    def test_missing_column_is_reported(self) -> None:
        incomplete = {key: value for key, value in self.row().items() if key != "end_s"}
        with pytest.raises(EvaluationError, match="missing column"):
            parse_label_rows([incomplete], "labels.csv")


class TestLoadLabels:
    """Reading labels from disk."""

    def test_loads_a_file(self, tmp_path: Path) -> None:
        path = write_labels(
            tmp_path / "labels.csv",
            [("clip_a.mp4", "cam_01", "wrong_way", "1.0", "2.0", "")],
        )
        assert len(load_labels(path)) == 1

    def test_missing_file_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(EvaluationError, match="not found"):
            load_labels(tmp_path / "absent.csv")

    def test_header_only_file_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.csv"
        path.write_text("clip,camera_id,incident_type,start_s,end_s,notes\n", encoding="utf-8")
        with pytest.raises(EvaluationError, match="no data rows"):
            load_labels(path)


class TestLoadEvents:
    """Reading predictions from disk."""

    def test_loads_an_events_object(self, tmp_path: Path) -> None:
        path = write_events(tmp_path / "events.json", [event("wrong_way", 4.0)])
        predictions = load_events_file(path, "clip_a")
        assert predictions[0].incident_type == "wrong_way"
        assert predictions[0].timestamp_s == 4.0

    def test_loads_a_bare_list(self, tmp_path: Path) -> None:
        path = tmp_path / "events.json"
        path.write_text(json.dumps([event("wrong_way", 4.0)]), encoding="utf-8")
        assert len(load_events_file(path, "clip_a")) == 1

    def test_loads_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        path.write_text(
            json.dumps(event("wrong_way", 4.0)) + "\n" + json.dumps(event("wrong_way", 9.0)),
            encoding="utf-8",
        )
        assert len(load_events_file(path, "clip_a")) == 2

    def test_empty_file_yields_no_predictions(self, tmp_path: Path) -> None:
        path = tmp_path / "events.json"
        path.write_text("", encoding="utf-8")
        assert load_events_file(path, "clip_a") == []

    def test_missing_incident_type_is_rejected(self, tmp_path: Path) -> None:
        path = write_events(tmp_path / "events.json", [{"timestamp_s": 1.0}])
        with pytest.raises(EvaluationError, match="incident_type"):
            load_events_file(path, "clip_a")

    def test_wall_clock_only_event_is_rejected(self, tmp_path: Path) -> None:
        path = write_events(
            tmp_path / "events.json",
            [{"incident_type": "wrong_way", "timestamp": "2026-10-20T00:00:00Z"}],
        )
        with pytest.raises(EvaluationError, match="needs stream seconds"):
            load_events_file(path, "clip_a")

    def test_invalid_json_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "events.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(EvaluationError, match="not valid JSON"):
            load_events_file(path, "clip_a")


class TestClipNameMatching:
    """Clip names are compared by basename without extension."""

    def test_extension_insensitive(self) -> None:
        assert clip_name_matches("clip_a.mp4", "clip_a") is True

    def test_path_insensitive(self) -> None:
        assert clip_name_matches("clip_a.mp4", "E:/clips/clip_a.mp4") is True

    def test_different_clips_do_not_match(self) -> None:
        assert clip_name_matches("clip_a.mp4", "clip_b.mp4") is False


class TestMatchPredictions:
    """The matching rule that decides TP/FP/FN."""

    def label(self, start_s: float, end_s: float, incident_type: str = "wrong_way") -> Label:
        """Build a label for a clip.

        Args:
            start_s: Window start.
            end_s: Window end.
            incident_type: Rule name.

        Returns:
            The label.
        """
        return Label("clip_a", "cam_01", incident_type, start_s, end_s, "")

    def prediction(self, timestamp_s: float, incident_type: str = "wrong_way") -> Prediction:
        """Build a prediction.

        Args:
            timestamp_s: Stream timestamp.
            incident_type: Rule name.

        Returns:
            The prediction.
        """
        return Prediction("clip_a", incident_type, timestamp_s)

    def clip(self, labels, predictions):
        """Build a ClipData for matching.

        Args:
            labels: Label list.
            predictions: Prediction list.

        Returns:
            The clip data.
        """
        from model2_behavior.scripts.evaluate_events import ClipData

        return ClipData(clip="clip_a", labels=list(labels), predictions=list(predictions))

    def test_exact_hit_is_a_true_positive(self) -> None:
        clip = self.clip([self.label(10.0, 12.0)], [self.prediction(11.0)])
        matches, false_positives, false_negatives = match_predictions(
            clip, "wrong_way", DEFAULT_TOLERANCE_S, DEFAULT_MATCHES_PER_LABEL
        )
        assert len(matches) == 1
        assert false_positives == []
        assert false_negatives == []

    def test_prediction_at_start_minus_tolerance_counts(self) -> None:
        clip = self.clip([self.label(10.0, 12.0)], [self.prediction(8.0)])
        matches, _, _ = match_predictions(
            clip, "wrong_way", DEFAULT_TOLERANCE_S, DEFAULT_MATCHES_PER_LABEL
        )
        assert len(matches) == 1

    def test_prediction_at_end_plus_tolerance_counts(self) -> None:
        clip = self.clip([self.label(10.0, 12.0)], [self.prediction(14.0)])
        matches, _, _ = match_predictions(
            clip, "wrong_way", DEFAULT_TOLERANCE_S, DEFAULT_MATCHES_PER_LABEL
        )
        assert len(matches) == 1

    def test_prediction_just_outside_tolerance_is_a_false_positive(self) -> None:
        clip = self.clip([self.label(10.0, 12.0)], [self.prediction(14.01)])
        matches, false_positives, false_negatives = match_predictions(
            clip, "wrong_way", DEFAULT_TOLERANCE_S, DEFAULT_MATCHES_PER_LABEL
        )
        assert matches == []
        assert len(false_positives) == 1
        assert len(false_negatives) == 1

    def test_wrong_type_never_matches(self) -> None:
        clip = self.clip([self.label(10.0, 12.0)], [self.prediction(11.0, "sudden_decel")])
        matches, false_positives, false_negatives = match_predictions(
            clip, "wrong_way", DEFAULT_TOLERANCE_S, DEFAULT_MATCHES_PER_LABEL
        )
        assert matches == []
        # The sudden_decel prediction is not a wrong_way prediction at all, so it
        # is neither a false positive nor a false negative for this rule.
        assert false_positives == []
        assert len(false_negatives) == 1

    def test_one_prediction_cannot_credit_two_labels(self) -> None:
        clip = self.clip(
            [self.label(10.0, 12.0), self.label(12.0, 14.0)], [self.prediction(13.0)]
        )
        matches, _, false_negatives = match_predictions(
            clip, "wrong_way", DEFAULT_TOLERANCE_S, DEFAULT_MATCHES_PER_LABEL
        )
        assert len(matches) == 1
        assert len(false_negatives) == 1

    def test_duplicate_alerts_do_not_inflate_recall(self) -> None:
        clip = self.clip(
            [self.label(10.0, 12.0)],
            [self.prediction(11.0), self.prediction(11.2), self.prediction(11.4)],
        )
        matches, false_positives, false_negatives = match_predictions(
            clip, "wrong_way", DEFAULT_TOLERANCE_S, DEFAULT_MATCHES_PER_LABEL
        )
        assert len(matches) == 1
        assert len(false_positives) == 2
        assert false_negatives == []

    def test_earliest_prediction_in_window_wins(self) -> None:
        clip = self.clip(
            [self.label(10.0, 12.0)], [self.prediction(11.5), self.prediction(10.1)]
        )
        matches, _, _ = match_predictions(
            clip, "wrong_way", DEFAULT_TOLERANCE_S, DEFAULT_MATCHES_PER_LABEL
        )
        assert matches[0][1].timestamp_s == 10.1

    def test_matching_is_deterministic_across_runs(self) -> None:
        clip = self.clip(
            [self.label(10.0, 12.0), self.label(10.5, 12.5)],
            [self.prediction(11.0), self.prediction(12.0)],
        )
        first = match_predictions(clip, "wrong_way", DEFAULT_TOLERANCE_S, 1)
        second = match_predictions(clip, "wrong_way", DEFAULT_TOLERANCE_S, 1)
        assert [pair[1].timestamp_s for pair in first[0]] == [
            pair[1].timestamp_s for pair in second[0]
        ]

    def test_predictions_with_no_labels_are_all_false_positives(self) -> None:
        clip = self.clip([], [self.prediction(11.0)])
        matches, false_positives, false_negatives = match_predictions(
            clip, "wrong_way", DEFAULT_TOLERANCE_S, DEFAULT_MATCHES_PER_LABEL
        )
        assert matches == []
        assert len(false_positives) == 1
        assert false_negatives == []


class TestScoreRules:
    """Per-rule aggregation, including false alarms per hour."""

    def build(self, negative_duration_s=0.0):
        """Build a two-clip scenario with one TP, one FP, one FN, one FA.

        Args:
            negative_duration_s: Duration to assign to the negative clip.

        Returns:
            ``(clips, scores)``.
        """
        from model2_behavior.scripts.evaluate_events import ClipData

        positive = ClipData(
            clip="clip_a",
            labels=[
                Label("clip_a", "cam_01", "wrong_way", 10.0, 12.0, "detected"),
                Label("clip_a", "cam_01", "wrong_way", 40.0, 42.0, "missed"),
            ],
            predictions=[
                Prediction("clip_a", "wrong_way", 11.0, "e1"),
                Prediction("clip_a", "wrong_way", 70.0, "e2"),
            ],
        )
        negative = ClipData(
            clip="clip_b",
            labels=[Label("clip_b", "cam_01", "", 0.0, 0.0, "negative")],
            predictions=[Prediction("clip_b", "wrong_way", 5.0, "e3")],
            duration_s=negative_duration_s,
        )
        clips = {"clip_a": positive, "clip_b": negative}
        return clips, score_rules(clips, ("wrong_way",), DEFAULT_TOLERANCE_S, 1)

    def test_counts(self) -> None:
        _, scores = self.build()
        score = scores["wrong_way"]
        assert score.true_positives == 1
        assert score.false_positives == 1
        assert score.false_negatives == 1
        assert score.false_alarms_on_negative_clips == 1

    def test_precision_recall_f1(self) -> None:
        _, scores = self.build()
        score = scores["wrong_way"]
        assert score.precision == pytest.approx(0.5)
        assert score.recall == pytest.approx(0.5)
        assert score.f1 == pytest.approx(0.5)

    def test_false_alarms_per_hour_uses_negative_duration(self) -> None:
        # One false alarm in 1800 s of negative footage is 2.0 alarms/hour.
        _, scores = self.build(negative_duration_s=1800.0)
        assert scores["wrong_way"].false_alarms_per_hour == pytest.approx(2.0)

    def test_false_alarms_per_hour_is_none_without_negative_duration(self) -> None:
        _, scores = self.build(negative_duration_s=0.0)
        assert scores["wrong_way"].false_alarms_per_hour is None

    def test_precision_is_none_with_no_predictions(self) -> None:
        from model2_behavior.scripts.evaluate_events import ClipData

        clips = {
            "clip_a": ClipData(
                clip="clip_a",
                labels=[Label("clip_a", "cam_01", "wrong_way", 10.0, 12.0, "")],
                predictions=[],
            )
        }
        scores = score_rules(clips, ("wrong_way",), DEFAULT_TOLERANCE_S, 1)
        assert scores["wrong_way"].precision is None
        assert scores["wrong_way"].recall == pytest.approx(0.0)

    def test_negative_clip_predictions_are_not_false_positives(self) -> None:
        # They are false ALARMS: counted separately so precision stays honest.
        _, scores = self.build()
        assert scores["wrong_way"].false_positives == 1

    def test_unmatched_predictions_are_listed(self) -> None:
        _, scores = self.build()
        ids = {prediction.event_id for prediction in scores["wrong_way"].unmatched_predictions}
        assert ids == {"e2", "e3"}

    def test_unseen_labels_are_listed(self) -> None:
        _, scores = self.build()
        assert len(scores["wrong_way"].unseen_labels) == 1
        assert scores["wrong_way"].unseen_labels[0].start_s == 40.0


class TestDiscoverClips:
    """Locating per-clip and per-run event files."""

    def test_per_clip_layout(self, tmp_path: Path) -> None:
        write_events(tmp_path / "clip_a.json", [event("wrong_way", 1.0)])
        found = discover_clips(tmp_path)
        assert "clip_a" in found
        assert len(found["clip_a"]) == 1

    def test_run_folder_layout(self, tmp_path: Path) -> None:
        run = tmp_path / "run_20260101T000000Z"
        write_events(run / "events.json", [event("wrong_way", 2.0)])
        (run / "run_summary.json").write_text(
            json.dumps({"source": "E:/clips/clip_c.mp4"}), encoding="utf-8"
        )
        found = discover_clips(tmp_path)
        assert "clip_c.mp4" in found

    def test_run_folder_without_summary_uses_folder_name(self, tmp_path: Path) -> None:
        run = tmp_path / "run_xyz"
        write_events(run / "events.json", [])
        found = discover_clips(tmp_path)
        assert "run_xyz" in found

    def test_run_summary_is_not_mistaken_for_events(self, tmp_path: Path) -> None:
        run = tmp_path / "run_xyz"
        write_events(run / "events.json", [event("wrong_way", 1.0)])
        (run / "run_summary.json").write_text(json.dumps({"source": ""}), encoding="utf-8")
        (run / "effective_config.json").write_text(json.dumps({}), encoding="utf-8")
        found = discover_clips(tmp_path)
        assert set(found) == {"run_xyz"}

    def test_two_files_for_one_clip_are_rejected(self, tmp_path: Path) -> None:
        write_events(tmp_path / "clip_a.json", [])
        write_events(tmp_path / "nested" / "clip_a.json", [])
        with pytest.raises(EvaluationError, match="same clip"):
            discover_clips(tmp_path)

    def test_events_jsonl_run_file_is_read(self, tmp_path: Path) -> None:
        run = tmp_path / "run_abc"
        run.mkdir(parents=True)
        (run / "events.jsonl").write_text(
            json.dumps(event("wrong_way", 3.0)) + "\n", encoding="utf-8"
        )
        found = discover_clips(tmp_path)
        assert "run_abc" in found

    def test_empty_directory_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(EvaluationError, match="no events files found"):
            discover_clips(tmp_path)

    def test_missing_directory_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(EvaluationError, match="not found"):
            discover_clips(tmp_path / "absent")


class TestBuildClips:
    """Joining labels to discovered predictions."""

    def test_joins_by_stem(self) -> None:
        labels = [Label("clip_a.mp4", "cam_01", "wrong_way", 1.0, 2.0, "")]
        events = {"clip_a.mp4": [Prediction("clip_a.mp4", "wrong_way", 1.5)]}
        clips = build_clips(labels, events)
        assert clips["clip_a.mp4"].predictions

    def test_predictions_are_rekeyed_to_the_label_clip_name(self) -> None:
        # Reports must name the clip the way labels.csv does, even when the
        # events file was keyed differently (e.g. clip_a.json vs clip_a.mp4).
        labels = [Label("clip_a.mp4", "cam_01", "wrong_way", 1.0, 2.0, "")]
        events = {"clip_a": [Prediction("clip_a", "wrong_way", 1.5)]}
        clips = build_clips(labels, events)
        assert clips["clip_a.mp4"].predictions[0].clip == "clip_a.mp4"

    def test_no_match_is_rejected(self) -> None:
        labels = [Label("clip_a.mp4", "cam_01", "wrong_way", 1.0, 2.0, "")]
        events = {"clip_z.mp4": []}
        with pytest.raises(EvaluationError, match="no labelled clip matched"):
            build_clips(labels, events)


class TestRunEvaluation:
    """End-to-end evaluation over a fixture directory tree."""

    def build_tree(self, tmp_path: Path) -> Path:
        """Write a small labels file plus two per-clip events files.

        Args:
            tmp_path: pytest temporary directory.

        Returns:
            The labels path.
        """
        labels = write_labels(
            tmp_path / "labels.csv",
            [
                ("clip_a.mp4", "cam_01", "wrong_way", "10.0", "12.0", "true incident"),
                ("clip_a.mp4", "cam_01", "sudden_decel", "30.0", "31.0", "true incident"),
                ("clip_b.mp4", "cam_01", "", "", "", "negative"),
            ],
        )
        events_dir = tmp_path / "events"
        write_events(
            events_dir / "clip_a.json",
            [
                event("wrong_way", 11.0, "a1"),  # TP
                event("wrong_way", 55.0, "a2"),  # FP
                event("sudden_decel", 30.4, "a3"),  # TP
                event("stopped_vehicle", 60.0, "a4"),  # FP: no such label
            ],
        )
        write_events(events_dir / "clip_b.json", [event("wrong_way", 4.0, "b1")])  # FA
        return labels

    def test_end_to_end_counts(self, tmp_path: Path) -> None:
        labels = self.build_tree(tmp_path)
        clips, scores = run_evaluation(labels, tmp_path / "events")

        assert scores["wrong_way"].true_positives == 1
        assert scores["wrong_way"].false_positives == 1
        assert scores["wrong_way"].false_negatives == 0
        assert scores["wrong_way"].false_alarms_on_negative_clips == 1

        assert scores["sudden_decel"].true_positives == 1
        assert scores["sudden_decel"].precision == pytest.approx(1.0)

        assert scores["stopped_vehicle"].false_positives == 1
        assert scores["stopped_vehicle"].recall is None

        assert clips["clip_b.mp4"].negative is True

    def test_tolerance_changes_the_result(self, tmp_path: Path) -> None:
        labels = self.build_tree(tmp_path)
        events_dir = tmp_path / "events"
        write_events(events_dir / "clip_a.json", [event("wrong_way", 15.0, "a1")])

        strict_clips, strict_scores = run_evaluation(labels, events_dir, tolerance_s=1.0)
        lenient_clips, lenient_scores = run_evaluation(labels, events_dir, tolerance_s=5.0)

        assert strict_scores["wrong_way"].true_positives == 0
        assert lenient_scores["wrong_way"].true_positives == 1

    def test_clip_filter_restricts_evaluation(self, tmp_path: Path) -> None:
        labels = self.build_tree(tmp_path)
        clips, _ = run_evaluation(labels, tmp_path / "events", clips_filter=["clip_a.mp4"])
        assert set(clips) == {"clip_a.mp4"}

    def test_clip_filter_matching_nothing_is_rejected(self, tmp_path: Path) -> None:
        labels = self.build_tree(tmp_path)
        with pytest.raises(EvaluationError, match="no label rows remain"):
            run_evaluation(labels, tmp_path / "events", clips_filter=["clip_zzz.mp4"])

    def test_negative_tolerance_is_rejected(self, tmp_path: Path) -> None:
        labels = self.build_tree(tmp_path)
        with pytest.raises(EvaluationError, match="tolerance_s must be"):
            run_evaluation(labels, tmp_path / "events", tolerance_s=-1.0)

    def test_zero_max_matches_is_rejected(self, tmp_path: Path) -> None:
        labels = self.build_tree(tmp_path)
        with pytest.raises(EvaluationError, match="max_matches_per_label must be"):
            run_evaluation(labels, tmp_path / "events", max_matches_per_label=0)

    def test_report_is_json_serialisable(self, tmp_path: Path) -> None:
        labels = self.build_tree(tmp_path)
        _, scores = run_evaluation(labels, tmp_path / "events")
        json.dumps({name: score.to_dict() for name, score in scores.items()})
