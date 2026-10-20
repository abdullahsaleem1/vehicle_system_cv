"""Evaluate emitted incident events against hand-labelled clips (no real data needed yet).

This is the Step 8 evaluation harness, wired up ahead of time so that the moment
labelled footage exists the numbers can be produced with one command. It needs no
GPU, no weights, and no network: it consumes only two things.

Inputs
------
``labels.csv``
    Ground truth, one row per labelled incident (or per negative clip)::

        clip,camera_id,incident_type,start_s,end_s,notes
        clip_a.mp4,cam_01,wrong_way,12.4,16.0,white corolla
        clip_a.mp4,cam_01,stopped_vehicle,30.0,30.0,
        clip_b.mp4,cam_01,,,,"no incidents - negative clip"

    * ``incident_type`` **empty** marks a NEGATIVE clip: a clip known to contain
      no incident of any kind. Negatives are what make a false-alarm rate
      meaningful, so they are required, not optional.
    * ``start_s`` / ``end_s`` bound the incident in stream seconds. A single
      instant may use the same value for both.
    * ``clip`` must match how the events file is keyed (see below).
    * A clip may appear on many rows (many incidents) or on one negative row.

``events``
    The ``events.json`` (or an ``events.jsonl``) written by
    :mod:`model2_behavior.scripts.run_pipeline`, either:

    * one events file per clip, named after the clip (``events/<clip>.json``), or
    * a directory of run output folders, each containing ``events.json`` plus a
      ``run_summary.json`` whose ``source``/name matches a clip, or
    * a single events file, when ``--clip`` is given to state which label row set
      it belongs to.

    Use ``--events-dir`` with a per-clip layout, or ``--events`` for a single file.
    Run metadata is loaded from the sibling ``run_summary.json`` when present so
    the config snapshot can be reported and the clip duration (for the
    false-alarms-per-hour denominator) can be recovered.

Matching rule (fixed, so results are reproducible)
--------------------------------------------------
A prediction is a **true positive** when ALL of the following hold:

    1. its ``incident_type`` equals the label's ``incident_type``, AND
    2. its ``timestamp`` falls inside ``[start_s - tolerance, end_s + tolerance]``,
       where the tolerance defaults to 2.0 s (``--tolerance-s``).

Each label may be matched by at most ``--max-matches-per-label`` predictions
(default 1), and each prediction may satisfy at most one label, so a rule that
fires 40 times during one stopped vehicle cannot inflate recall. Matched
predictions are TPs; unmatched predictions on a clip that has labels of that
type are false positives (including a same-type prediction whose timestamp falls
outside the label window); unmatched predictions on a NEGATIVE clip are counted
separately as false alarms, because those are the ones that become user-facing
noise.

Reported per rule
-----------------
    * precision, recall, F1
    * TP / FP / FN counts and the unmatched predictions themselves (with their
      timestamps), so a human can see *which* events were wrong
    * false alarms per hour on negative clips, using the TOTAL duration of
      negative clips as the denominator
    * the config snapshot used for each clip (thresholds + effective tracker
      parameters) AND a ``config_hash`` of the behaviour-affecting settings, so a
      metric is never quoted without the settings that made it

Tuning vs reporting (``--split``)
---------------------------------
``labels.csv`` may carry an optional ``split`` column with values ``tune`` or
``test``. ``--split tune`` and ``--split test`` then select disjoint row sets, so
thresholds can be chosen on one set and reported on held-out clips without
maintaining two label files. ``--split all`` (the default) uses every row. When
no row carries a split, ``--split tune/test`` is a no-op with a warning rather
than an error, so a labels file written before the column existed still works.

Usage
-----
    # Directory of per-clip run outputs plus run_summary.json siblings:
    python -m model2_behavior.scripts.evaluate_events \
        --labels outputs/eval/labels.csv --events-dir outputs/eval/runs

    # One events file, whose labels are all rows for one clip:
    python -m model2_behavior.scripts.evaluate_events \
        --labels labels.csv --events path/to/events.json --clip clip_a.mp4

Exit codes
----------
    0  evaluation completed and wrote its report
    1  the inputs were malformed or nothing could be matched up

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model2_behavior.src.logging_setup import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Every prediction is attributed to one of these, so a typo in a label file is
#: caught here rather than silently producing a rule nobody checks.
KNOWN_INCIDENT_TYPES = ("stopped_vehicle", "wrong_way", "sudden_decel", "collision_proxy")

#: Default match tolerance in seconds, applied to BOTH ends of a label window.
#: This is deliberately generous: detector-to-incident latency and the
#: debounce/min-consecutive delays mean the emitted timestamp lags the physical
#: event by a second or two. It is a fixed constant in the report so nobody has
#: to guess which value produced a number.
DEFAULT_TOLERANCE_S = 2.0

#: One prediction may satisfy at most this many labels. More than one only
#: happens when two label windows overlap, which is unusual; the cap keeps a
#: stray event from crediting two incidents.
DEFAULT_MATCHES_PER_LABEL = 1

LABEL_COLUMNS = ("clip", "camera_id", "incident_type", "start_s", "end_s", "notes")

#: Valid values for ``--split``. ``all`` evaluates every label row; ``tune`` and
#: ``test`` select their own rows so a threshold can be tuned on one set and
#: reported on another. An optional ``split`` column in ``labels.csv`` provides
#: the row-level assignment; when the column is absent, ``tune``/``test`` are
#: no-ops and every row is used.
SPLIT_CHOICES = ("all", "tune", "test")


class EvaluationError(RuntimeError):
    """Raised when the inputs cannot be interpreted."""


@dataclass(frozen=True)
class Label:
    """One ground-truth incident, or one negative clip.

    Attributes:
        clip: Clip identifier; must match how events are keyed.
        camera_id: Camera the clip came from.
        incident_type: Incident type, or ``""`` for a negative clip.
        start_s: Incident start in stream seconds (0.0 for a negative clip).
        end_s: Incident end in stream seconds (0.0 for a negative clip).
        notes: Free-text note, carried into the report.
        split: Optional ``tune``/``test`` assignment from an extra ``split``
            column, or ``""`` when the column is absent. Lets thresholds be
            tuned on one half of the data and reported on the other without
            maintaining two label files.
    """

    clip: str
    camera_id: str
    incident_type: str
    start_s: float
    end_s: float
    notes: str = ""
    split: str = ""

    @property
    def is_negative(self) -> bool:
        """Return True when this row asserts the clip has no incidents."""
        return self.incident_type == ""


@dataclass
class Prediction:
    """One emitted event, reduced to what evaluation needs.

    Attributes:
        clip: Clip the event belongs to.
        incident_type: Rule that fired.
        timestamp_s: Event timestamp in seconds of stream time.
        event_id: Original event id, kept for traceability.
        severity: Event severity, carried into the report.
        confidence: DETECTION confidence (not incident probability).
    """

    clip: str
    incident_type: str
    timestamp_s: float
    event_id: str = ""
    severity: str = ""
    confidence: float = 0.0


@dataclass
class ClipData:
    """Everything known about one clip.

    Attributes:
        clip: Clip identifier.
        labels: Ground-truth labels for this clip.
        predictions: Predictions for this clip.
        duration_s: Clip duration in seconds, when known.
        config_snapshot: Thresholds and tracker parameters used, when available.
        run_summary: Raw run summary, when available.
    """

    clip: str
    labels: List[Label] = field(default_factory=list)
    predictions: List[Prediction] = field(default_factory=list)
    duration_s: Optional[float] = None
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    run_summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def negative(self) -> bool:
        """Return True when this clip has at least one negative label row."""
        return any(label.is_negative for label in self.labels)


def parse_label_rows(rows: Iterable[Dict[str, str]], source: str) -> List[Label]:
    """Convert raw CSV rows into :class:`Label` objects.

    Args:
        rows: :mod:`csv` ``DictReader`` rows.
        source: Path of the labels file, for error messages.

    Returns:
        Parsed labels.

    Raises:
        EvaluationError: If a required column is missing, a timestamp is
            unparseable, a window is inverted, an incident type is unknown, or a
            negative row carries timestamps.
    """
    labels: List[Label] = []
    for line_number, row in enumerate(rows, start=2):
        missing = [column for column in LABEL_COLUMNS if column not in row]
        if missing:
            raise EvaluationError(
                f"{source}:{line_number}: missing column(s) {missing}; "
                f"expected header {LABEL_COLUMNS}"
            )

        clip = (row.get("clip") or "").strip()
        if not clip:
            raise EvaluationError(f"{source}:{line_number}: 'clip' is empty")

        split = (row.get("split") or "").strip().lower()
        if split and split not in ("tune", "test"):
            raise EvaluationError(
                f"{source}:{line_number}: split must be 'tune' or 'test' (or empty), got {split!r}"
            )

        incident_type = (row.get("incident_type") or "").strip()
        if incident_type and incident_type not in KNOWN_INCIDENT_TYPES:
            raise EvaluationError(
                f"{source}:{line_number}: unknown incident_type {incident_type!r}; "
                f"expected one of {KNOWN_INCIDENT_TYPES} or empty for a negative clip"
            )

        start_raw = (row.get("start_s") or "").strip()
        end_raw = (row.get("end_s") or "").strip()

        if not incident_type:
            if start_raw or end_raw:
                raise EvaluationError(
                    f"{source}:{line_number}: negative row (empty incident_type) must leave "
                    "start_s and end_s empty"
                )
            labels.append(
                Label(
                    clip=clip,
                    camera_id=(row.get("camera_id") or "").strip(),
                    incident_type="",
                    start_s=0.0,
                    end_s=0.0,
                    notes=(row.get("notes") or "").strip(),
                    split=split,
                )
            )
            continue

        try:
            start_s = float(start_raw)
            end_s = float(end_raw) if end_raw else start_s
        except ValueError as exc:
            raise EvaluationError(
                f"{source}:{line_number}: start_s/end_s must be numbers, got "
                f"{start_raw!r}/{end_raw!r}"
            ) from exc

        if end_s < start_s:
            raise EvaluationError(
                f"{source}:{line_number}: end_s ({end_s}) precedes start_s ({start_s})"
            )

        labels.append(
            Label(
                clip=clip,
                camera_id=(row.get("camera_id") or "").strip(),
                incident_type=incident_type,
                start_s=start_s,
                end_s=end_s,
                notes=(row.get("notes") or "").strip(),
                split=split,
            )
        )
    return labels


def load_labels(path: Path) -> List[Label]:
    """Read and validate a ``labels.csv`` file.

    Args:
        path: Path to the CSV.

    Returns:
        Parsed labels, in file order.

    Raises:
        EvaluationError: If the file is missing, empty, or malformed.
    """
    if not path.is_file():
        raise EvaluationError(f"labels file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise EvaluationError(f"labels file has no data rows: {path}")
    return parse_label_rows(rows, str(path))


def parse_event_dict(raw: Dict[str, Any], clip: str, source: str) -> Prediction:
    """Convert one raw event dict into a :class:`Prediction`.

    ``timestamp`` is read from the event's ISO timestamp by subtracting the run
    start, when the events file records a start; otherwise ``frame_index`` and the
    run FPS are used. Both are properties of the file, so the evaluation never
    needs the original video.

    Args:
        raw: One event object from ``events.json``.
        clip: Clip the events file belongs to.
        source: Path of the events file, for error messages.

    Returns:
        The prediction.

    Raises:
        EvaluationError: If the incident type or timestamp cannot be determined.
    """
    incident_type = str(raw.get("incident_type", "")).strip()
    if not incident_type:
        raise EvaluationError(f"{source}: event without 'incident_type': {raw!r}")

    timestamp = raw.get("timestamp_s")
    if timestamp is None:
        timestamp = raw.get("stream_time_s")
    if timestamp is None:
        # Fall back to the ISO timestamp when the run recorded a start time.
        raise EvaluationError(
            f"{source}: event without a numeric 'timestamp_s'; the evaluator needs stream "
            "seconds, not wall-clock. Re-run the pipeline to regenerate events.json."
        )

    return Prediction(
        clip=clip,
        incident_type=incident_type,
        timestamp_s=float(timestamp),
        event_id=str(raw.get("event_id", "")),
        severity=str(raw.get("severity", "")),
        confidence=float(raw.get("confidence", 0.0) or 0.0),
    )


def load_events_file(path: Path, clip: str) -> List[Prediction]:
    """Read predictions from an ``events.json`` or ``events.jsonl`` file.

    Args:
        path: Events file.
        clip: Clip the file belongs to.

    Returns:
        Predictions in file order.

    Raises:
        EvaluationError: If the file is missing or not valid JSON/JSONL.
    """
    if not path.is_file():
        raise EvaluationError(f"events file not found: {path}")

    # utf-8-sig strips a Windows BOM when present and is a no-op otherwise, so a
    # file written by PowerShell or Notepad is read rather than rejected.
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []

    try:
        if path.suffix.lower() == ".jsonl":
            payloads = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            loaded = json.loads(text)
            if isinstance(loaded, dict):
                payloads = loaded.get("events", [])
            else:
                payloads = loaded
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"{path}: not valid JSON: {exc}") from exc

    if not isinstance(payloads, list):
        raise EvaluationError(f"{path}: expected a list of events or an object with 'events'")

    return [parse_event_dict(raw, clip, str(path)) for raw in payloads]


def load_run_summary(path: Path) -> Dict[str, Any]:
    """Read a sibling ``run_summary.json`` if it exists.

    Args:
        path: Path to the summary file.

    Returns:
        The parsed summary, or an empty dict when absent or unreadable.
    """
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read run summary %s: %s", path, exc)
        return {}


def load_config_snapshot(path: Path) -> Dict[str, Any]:
    """Read a sibling ``effective_config.json`` if it exists.

    Args:
        path: Path to the config snapshot.

    Returns:
        The parsed snapshot, or an empty dict when absent or unreadable.
    """
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read config snapshot %s: %s", path, exc)
        return {}


def discover_clips(events_dir: Path) -> Dict[str, List[Prediction]]:
    """Find per-clip event files under a directory.

    Two layouts are supported:

        ``<events_dir>/<clip-stem>.json``          one file per clip
        ``<events_dir>/<run-folder>/events.json``  one run folder per clip, where
                                                   the clip name comes from the
                                                   sibling ``run_summary.json``'s
                                                   ``source`` when present,
                                                   otherwise the folder name.

    A file is treated as a run output when it is named ``events.json`` /
    ``events.jsonl`` (the pipeline's own name); anything else is treated as a
    per-clip file whose stem is the clip name. ``run_summary.json`` and
    ``effective_config.json`` are never events files, so they are skipped.

    Args:
        events_dir: Directory to scan.

    Returns:
        Mapping of clip identifier to predictions.

    Raises:
        EvaluationError: If the directory yields no events files at all.
    """
    if not events_dir.is_dir():
        raise EvaluationError(f"events directory not found: {events_dir}")

    reserved = {"run_summary", "effective_config"}
    discovered: Dict[str, List[Prediction]] = {}

    candidates = sorted(
        set(events_dir.rglob("*.json")) | set(events_dir.rglob("*.jsonl"))
    )
    for path in candidates:
        if path.stem in reserved:
            continue
        # An `events.jsonl` beside an `events.json` in the same folder would be
        # two views of one run; only the first is read, deterministically.
        is_run_file = path.name in ("events.json", "events.jsonl")
        if is_run_file:
            run_summary = load_run_summary(path.parent / "run_summary.json")
            clip = Path(str(run_summary.get("source", ""))).name or path.parent.name
            if clip in ("", "."):
                clip = path.parent.name
        else:
            clip = path.stem

        predictions = load_events_file(path, clip)
        if clip in discovered:
            # Duplicate clip names are a real ambiguity: report rather than merge
            # silently, because merging would double-count a rule's alerts.
            if is_run_file and path.with_suffix(".jsonl").exists():
                continue
            raise EvaluationError(
                f"two events files resolve to the same clip {clip!r}: "
                f"{path} collides with an already-discovered file. "
                "Rename one, or pass --only-clips."
            )
        discovered[clip] = predictions

    if not discovered:
        raise EvaluationError(
            f"no events files found under {events_dir}; "
            "point --events-dir at a directory of per-clip events files or run outputs"
        )
    return discovered


def clip_name_matches(label_clip: str, event_clip: str) -> bool:
    """Return True when a labels-file clip name refers to an events-file clip.

    Comparison is by basename with the extension dropped, so ``clip_a.mp4`` in
    ``labels.csv`` matches a run whose source was ``E:/clips/clip_a.mp4`` or an
    events file named ``clip_a.json``.

    Args:
        label_clip: Clip value from ``labels.csv``.
        event_clip: Clip key derived from the events layout.

    Returns:
        True when the two refer to the same clip.
    """
    return Path(label_clip).stem == Path(event_clip).stem


def build_clips(
    labels: Sequence[Label], events_by_clip: Dict[str, List[Prediction]]
) -> Dict[str, ClipData]:
    """Assemble clips from labels and discovered predictions.

    Args:
        labels: All ground-truth labels.
        events_by_clip: Predictions keyed by clip.

    Returns:
        Clip data keyed by the labels-file ``clip`` value.

    Raises:
        EvaluationError: If no label clip has any matching events file.
    """
    clips: Dict[str, ClipData] = {}
    for label in labels:
        clips.setdefault(label.clip, ClipData(clip=label.clip)).labels.append(label)

    matched_any = False
    for clip_name, clip in clips.items():
        for event_clip, predictions in events_by_clip.items():
            if clip_name_matches(clip_name, event_clip):
                # Re-key each prediction to the LABELS-side clip name so reports
                # name the clip the way the label file does, not the events file.
                clip.predictions.extend(
                    Prediction(
                        clip=clip_name,
                        incident_type=prediction.incident_type,
                        timestamp_s=prediction.timestamp_s,
                        event_id=prediction.event_id,
                        severity=prediction.severity,
                        confidence=prediction.confidence,
                    )
                    for prediction in predictions
                )
                matched_any = True

    if not matched_any:
        known = sorted(events_by_clip)
        raise EvaluationError(
            "no labelled clip matched any events file. "
            f"labelled clips: {sorted(clips)}; events clips: {known}. "
            "Names are compared by basename without extension."
        )
    return clips


def attach_run_metadata(clip: ClipData, events_dir: Path) -> None:
    """Load the run summary and config snapshot for a clip, when available.

    Args:
        clip: Clip to enrich (mutated in place).
        events_dir: Directory that holds the run outputs.
    """
    for summary_path in events_dir.rglob("run_summary.json"):
        summary = load_run_summary(summary_path)
        source = Path(str(summary.get("source", ""))).name
        folder_clip = summary_path.parent.name
        if (
            (source and clip_name_matches(clip.clip, source))
            or clip_name_matches(clip.clip, folder_clip)
            or summary_path.parent.name == clip.clip
        ):
            clip.run_summary = summary
            clip.duration_s = summary.get("duration_s") or clip.duration_s
            snapshot_path = summary_path.parent / "effective_config.json"
            clip.config_snapshot = load_config_snapshot(snapshot_path)
            return


def rule_config_for(snapshot: Dict[str, Any], rule_name: str) -> Dict[str, Any]:
    """Extract the thresholds a rule ran with from a config snapshot.

    Args:
        snapshot: Parsed ``effective_config.json``.
        rule_name: Rule to look up.

    Returns:
        The rule's threshold mapping, or an empty dict when unavailable.
    """
    rules = snapshot.get("rules") or {}
    entry = rules.get(rule_name) or {}
    return dict(entry.get("params") or {})


def config_hash(snapshot: Dict[str, Any]) -> Optional[str]:
    """Hash the parts of a config snapshot that affect rule decisions.

    A metric is only meaningful alongside the thresholds that produced it, and a
    hash is the cheapest way to prove two numbers came from the same settings. The
    hash deliberately covers only what changes behaviour — rule parameters, the
    rule-layer filter, detector settings, and effective tracker parameters — so a
    cosmetic change (a path, a timestamp) does not invalidate a comparison.

    Args:
        snapshot: Parsed ``effective_config.json``.

    Returns:
        A 16-hex-character digest, or None when there is no snapshot.
    """
    if not snapshot:
        return None
    payload = {
        "rules": snapshot.get("rules"),
        "rule_layer": snapshot.get("rule_layer"),
        "detector": snapshot.get("detector"),
        "tracker_params": snapshot.get("tracker_params"),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def match_predictions(
    clip: ClipData,
    incident_type: str,
    tolerance_s: float,
    max_matches_per_label: int,
) -> Tuple[List[Tuple[Label, Prediction]], List[Prediction], List[Label]]:
    """Match predictions of one rule to labels of the same rule for a clip.

    Matching is greedy on label order, which is deterministic: labels are
    processed in file order, and for each label the earliest still-unmatched
    prediction inside the tolerance window is taken. A prediction is used at most
    once, so duplicate alerts cannot inflate recall.

    Args:
        clip: Clip under evaluation.
        incident_type: Rule being scored.
        tolerance_s: Seconds added to BOTH ends of the label window.
        max_matches_per_label: Cap on predictions credited to one label.

    Returns:
        ``(matches, false_positives, false_negatives)`` where ``matches`` is a
        list of ``(label, prediction)`` pairs.
    """
    labels = [label for label in clip.labels if label.incident_type == incident_type]
    predictions = sorted(
        (item for item in clip.predictions if item.incident_type == incident_type),
        key=lambda item: item.timestamp_s,
    )

    used: set[int] = set()
    matches: List[Tuple[Label, Prediction]] = []
    false_negatives: List[Label] = []

    for label in labels:
        window_start = label.start_s - tolerance_s
        window_end = label.end_s + tolerance_s
        credited = 0
        for index, prediction in enumerate(predictions):
            if index in used:
                continue
            if window_start <= prediction.timestamp_s <= window_end:
                used.add(index)
                matches.append((label, prediction))
                credited += 1
                if credited >= max_matches_per_label:
                    break
        if credited == 0:
            false_negatives.append(label)

    false_positives = [item for index, item in enumerate(predictions) if index not in used]
    return matches, false_positives, false_negatives


@dataclass
class RuleScore:
    """Per-rule counts and derived metrics.

    Attributes:
        incident_type: Rule name.
        true_positives: Matched predictions.
        false_positives: Unmatched predictions on clips that DO have labels for
            this rule (a missed label is a recall problem, not a precision one,
            so those clips are scored normally).
        false_negatives: Labels with no matching prediction.
        false_alarms_on_negative_clips: Unmatched predictions on clips labelled
            as containing no incidents.
        negative_clip_seconds: Total labelled-negative duration seen.
        matches: The matched pairs, for reporting.
        unmatched_predictions: The un-matched predictions, for reporting.
        unseen_labels: The un-matched labels, for reporting.
    """

    incident_type: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    false_alarms_on_negative_clips: int = 0
    negative_clip_seconds: float = 0.0
    matches: List[Tuple[Label, Prediction]] = field(default_factory=list)
    unmatched_predictions: List[Prediction] = field(default_factory=list)
    unseen_labels: List[Label] = field(default_factory=list)

    @property
    def precision(self) -> Optional[float]:
        """Precision, or None when the rule made no positive predictions."""
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else None

    @property
    def recall(self) -> Optional[float]:
        """Recall, or None when there was nothing to recall."""
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else None

    @property
    def f1(self) -> Optional[float]:
        """Harmonic mean of precision and recall, or None when undefined."""
        precision, recall = self.precision, self.recall
        if precision is None or recall is None:
            return None
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @property
    def false_alarms_per_hour(self) -> Optional[float]:
        """False alarms per hour of negative footage, or None when no negatives."""
        if self.negative_clip_seconds <= 0:
            return None
        return self.false_alarms_on_negative_clips / (self.negative_clip_seconds / 3600.0)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable form."""
        return {
            "incident_type": self.incident_type,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "false_alarms_on_negative_clips": self.false_alarms_on_negative_clips,
            "negative_clip_seconds": round(self.negative_clip_seconds, 3),
            "false_alarms_per_hour": self.false_alarms_per_hour,
            "matches": [
                {
                    "label": {
                        "clip": label.clip,
                        "start_s": label.start_s,
                        "end_s": label.end_s,
                        "notes": label.notes,
                    },
                    "prediction": {
                        "event_id": prediction.event_id,
                        "timestamp_s": prediction.timestamp_s,
                        "severity": prediction.severity,
                    },
                }
                for label, prediction in self.matches
            ],
            "unmatched_predictions": [
                {
                    "event_id": prediction.event_id,
                    "clip": prediction.clip,
                    "timestamp_s": prediction.timestamp_s,
                    "severity": prediction.severity,
                }
                for prediction in self.unmatched_predictions
            ],
            "unseen_labels": [
                {
                    "clip": label.clip,
                    "start_s": label.start_s,
                    "end_s": label.end_s,
                    "notes": label.notes,
                }
                for label in self.unseen_labels
            ],
        }


def score_rules(
    clips: Dict[str, ClipData],
    incident_types: Sequence[str],
    tolerance_s: float,
    max_matches_per_label: int,
) -> Dict[str, RuleScore]:
    """Score every rule over every clip.

    Args:
        clips: Clip data keyed by clip name.
        incident_types: Rules to score.
        tolerance_s: Seconds added to BOTH ends of a label window.
        max_matches_per_label: Cap on predictions credited to one label.

    Returns:
        Mapping of rule name to its :class:`RuleScore`.
    """
    scores = {name: RuleScore(incident_type=name) for name in incident_types}

    for clip in clips.values():
        negative = clip.negative
        for name, score in scores.items():
            matches, false_positives, false_negatives = match_predictions(
                clip, name, tolerance_s, max_matches_per_label
            )
            score.matches.extend(matches)
            score.true_positives += len(matches)
            score.false_negatives += len(false_negatives)
            score.unseen_labels.extend(false_negatives)

            if negative:
                # A negative clip asserts no incidents of ANY type, so every
                # prediction on it is a false alarm for its own rule.
                score.false_alarms_on_negative_clips += len(false_positives)
                score.unmatched_predictions.extend(false_positives)
                if clip.duration_s:
                    score.negative_clip_seconds += clip.duration_s
            else:
                score.false_positives += len(false_positives)
                score.unmatched_predictions.extend(false_positives)
    return scores


def format_optional(value: Optional[float], digits: int = 3) -> str:
    """Render a metric that may be undefined as a string.

    Args:
        value: Metric value, or None when it is not defined.
        digits: Decimal places to use.

    Returns:
        The formatted number, or ``"n/a"``.
    """
    return "n/a" if value is None else f"{value:.{digits}f}"


def print_report(
    clips: Dict[str, ClipData],
    scores: Dict[str, RuleScore],
    tolerance_s: float,
    max_matches_per_label: int,
    labels_path: Path,
    split: str = "all",
) -> None:
    """Print the evaluation report.

    Args:
        clips: Clip data keyed by clip name.
        scores: Per-rule scores.
        tolerance_s: Tolerance used, printed so the numbers are never ambiguous.
        max_matches_per_label: Match cap used.
        labels_path: Labels file, for traceability.
        split: Split filter used, printed so a tune/test number is never
            mistaken for an all-data number.
    """
    negatives = [clip for clip in clips.values() if clip.negative]
    total_negative_seconds = sum(clip.duration_s or 0.0 for clip in negatives)

    print("=" * 78)
    print("MODEL 2 EVENT EVALUATION - FR-20 (pending supervisor sign-off)")
    print("=" * 78)
    print(f"Labels file            : {labels_path}")
    print(f"Split                  : {split}")
    print(f"Clips labelled         : {len(clips)}")
    print(f"Negative clips         : {len(negatives)} ({total_negative_seconds / 60.0:.2f} min)")
    print(f"Match tolerance        : +/- {tolerance_s:.2f} s")
    print(f"Max matches per label  : {max_matches_per_label}")
    print()
    print("Clips:")
    for clip in clips.values():
        state = "NEGATIVE" if clip.negative else "labelled"
        duration = f"{clip.duration_s:.2f}s" if clip.duration_s else "duration unknown"
        print(
            f"  {clip.clip:24s} {state:9s} labels={len(clip.labels):2d} "
            f"predictions={len(clip.predictions):2d} {duration}"
        )
    print()
    print("-" * 78)
    print(f"{'rule':18s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'prec':>7s} {'rec':>7s} {'f1':>7s} {'FA/h':>8s}")
    print("-" * 78)
    for name, score in scores.items():
        print(
            f"{name:18s} {score.true_positives:4d} {score.false_positives:4d} "
            f"{score.false_negatives:4d} {format_optional(score.precision):>7s} "
            f"{format_optional(score.recall):>7s} {format_optional(score.f1):>7s} "
            f"{format_optional(score.false_alarms_per_hour, 2):>8s}"
        )
    print("-" * 78)
    print("'FA/h' is false alarms per hour on NEGATIVE clips (needs clip durations).")

    for name, score in scores.items():
        if not (score.unmatched_predictions or score.unseen_labels):
            continue
        print()
        print(f"{name}:")
        negative_clips = {
            clip.clip for clip in clips.values() if clip.negative
        }
        for prediction in score.unmatched_predictions:
            # "false alarm" is reserved for negative clips; an unmatched event on
            # a clip that DOES contain this incident type is a plain false positive.
            kind = "false alarm" if prediction.clip in negative_clips else "false positive"
            print(
                f"  {kind:14s} clip={prediction.clip} t={prediction.timestamp_s:.2f}s "
                f"event_id={prediction.event_id}"
            )
        for label in score.unseen_labels:
            print(
                f"  missed         clip={label.clip} "
                f"[{label.start_s:.2f}, {label.end_s:.2f}] notes={label.notes!r}"
            )


def build_config_section(clips: Dict[str, ClipData], incident_types: Sequence[str]) -> Dict[str, Any]:
    """Collect the config snapshot actually used, per clip.

    A metric quoted without the thresholds that produced it invites the wrong
    conclusion, so the report always carries this section.

    Args:
        clips: Clip data keyed by clip name.
        incident_types: Rules to include.

    Returns:
        Mapping of clip name to its thresholds and effective tracker parameters.
    """
    section: Dict[str, Any] = {}
    for clip in clips.values():
        snapshot = clip.config_snapshot
        if not snapshot:
            section[clip.clip] = {"config_snapshot": None}
            continue
        section[clip.clip] = {
            "config_snapshot": {
                "path": None,
                "config_hash": config_hash(snapshot),
                "rule_params": {
                    name: rule_config_for(snapshot, name) for name in incident_types
                },
                "tracker_params": snapshot.get("tracker_params"),
                "detector": snapshot.get("detector"),
                "fps": snapshot.get("fps"),
                "source_fps": clip.run_summary.get("source_fps"),
                "used_fps_source": clip.run_summary.get("used_fps_source"),
                "clock_source": (clip.run_summary.get("clock") or {}).get("clock_source"),
            }
        }
    return section


def config_hash_used(clips: Dict[str, ClipData]) -> Optional[str]:
    """Return a single config hash for the whole evaluation, when unambiguous.

    Every clip should have run under the same thresholds, so one hash is the
    useful summary. If clips were produced by *different* configs the hash would
    be misleading, so the ambiguity is surfaced rather than hidden: the function
    returns ``None`` and the caller can see the per-clip hashes separately.

    Args:
        clips: Clip data keyed by clip name.

    Returns:
        The shared config hash, or ``None`` when there is none or it differs
        across clips.
    """
    hashes = {
        config_hash(clip.config_snapshot)
        for clip in clips.values()
        if clip.config_snapshot
    }
    if len(hashes) == 1:
        return next(iter(hashes))
    return None


def run_evaluation(
    labels_path: Path,
    events_dir: Path,
    clips_filter: Optional[Sequence[str]] = None,
    tolerance_s: float = DEFAULT_TOLERANCE_S,
    max_matches_per_label: int = DEFAULT_MATCHES_PER_LABEL,
    incident_types: Sequence[str] = KNOWN_INCIDENT_TYPES,
    split: str = "all",
) -> Tuple[Dict[str, ClipData], Dict[str, RuleScore]]:
    """Run the full evaluation in memory.

    Split out from :func:`main` so tests can call it directly on fixtures without
    shelling out.

    Args:
        labels_path: Path to ``labels.csv``.
        events_dir: Directory holding per-clip event files.
        clips_filter: Restrict evaluation to these clip names.
        tolerance_s: Seconds added to BOTH ends of a label window.
        max_matches_per_label: Cap on predictions credited to one label.
        incident_types: Rules to score.
        split: ``all`` to use every row, or ``tune``/``test`` to keep only rows
            whose ``split`` column matches. When no row carries a split, the
            filter is a no-op rather than an error, because a labels file that
            predates the column must still evaluate.

    Returns:
        ``(clips, scores)``.

    Raises:
        EvaluationError: If the inputs cannot be interpreted.
    """
    if tolerance_s < 0:
        raise EvaluationError(f"tolerance_s must be >= 0, got {tolerance_s}")
    if max_matches_per_label < 1:
        raise EvaluationError(f"max_matches_per_label must be >= 1, got {max_matches_per_label}")
    if split not in SPLIT_CHOICES:
        raise EvaluationError(f"split must be one of {SPLIT_CHOICES}, got {split!r}")

    labels = load_labels(labels_path)
    if clips_filter:
        wanted = {Path(name).stem for name in clips_filter}
        labels = [label for label in labels if Path(label.clip).stem in wanted]
        if not labels:
            raise EvaluationError(
                f"no label rows remain after filtering to {list(clips_filter)}"
            )

    if split != "all":
        labelled_splits = {label.split for label in labels}
        if labelled_splits - {""}:
            # The column exists somewhere, so honour it strictly: a row with no
            # split is not silently included in a split it was never assigned to.
            labels = [label for label in labels if label.split == split]
            if not labels:
                raise EvaluationError(
                    f"no label rows are assigned to split {split!r}; "
                    f"found splits {sorted(labelled_splits - {''})}"
                )
        else:
            logger.warning(
                "split=%r requested but no row carries a 'split' column; "
                "evaluating every row. Add a 'split' column to separate tune/test.",
                split,
            )

    events_by_clip = discover_clips(events_dir)
    clips = build_clips(labels, events_by_clip)
    for clip in clips.values():
        attach_run_metadata(clip, events_dir)

    scores = score_rules(clips, incident_types, tolerance_s, max_matches_per_label)
    return clips, scores


def main(argv: Optional[List[str]] = None) -> int:
    """Run the evaluation harness.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Score Model 2 incident events against labelled clips."
    )
    parser.add_argument("--labels", type=Path, required=True, help="Path to labels.csv.")
    parser.add_argument(
        "--events-dir",
        type=Path,
        default=None,
        help="Directory of per-clip events files or run output folders.",
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=None,
        help="A single events.json/jsonl file, whose clip is given by --clip.",
    )
    parser.add_argument("--clip", default=None, help="Clip name for --events.")
    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=DEFAULT_TOLERANCE_S,
        help=f"Seconds added to BOTH ends of a label window (default: {DEFAULT_TOLERANCE_S}).",
    )
    parser.add_argument(
        "--max-matches-per-label",
        type=int,
        default=DEFAULT_MATCHES_PER_LABEL,
        help=f"Max predictions credited to one label (default: {DEFAULT_MATCHES_PER_LABEL}).",
    )
    parser.add_argument(
        "--only-clips",
        default=None,
        help="Comma-separated clip names to restrict the evaluation to.",
    )
    parser.add_argument(
        "--split",
        choices=SPLIT_CHOICES,
        default="all",
        help=(
            "Evaluate only rows whose optional 'split' column matches. Use 'tune' to "
            "choose thresholds and 'test' to report them on held-out clips. Default: all."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the JSON report (default: alongside the labels file).",
    )
    args = parser.parse_args(argv)

    setup_logging(name="evaluate_events")
    logger.info("Model 2 event evaluation starting | FR-20 (pending supervisor sign-off)")

    if args.events is None and args.events_dir is None:
        print("ERROR: provide --events-dir or --events", file=sys.stderr)
        return 1
    if args.events is not None and not args.clip:
        print("ERROR: --events requires --clip to identify the clip.", file=sys.stderr)
        return 1

    # A single events file is staged into a temporary one-clip directory layout so
    # that both entry points share exactly the same code path.
    staged_dir: Optional[Path] = None
    events_dir = args.events_dir
    try:
        if args.events is not None:
            staged_dir = Path(tempfile.gettempdir()) / f"eval_{abs(hash(str(args.events))) % 10**8}"
            staged_dir.mkdir(parents=True, exist_ok=True)
            target = staged_dir / Path(args.clip).with_suffix(".json")
            target.write_text(args.events.read_text(encoding="utf-8"), encoding="utf-8")
            events_dir = staged_dir

        clips_filter = (
            [name.strip() for name in args.only_clips.split(",") if name.strip()]
            if args.only_clips
            else None
        )

        clips, scores = run_evaluation(
            labels_path=args.labels,
            events_dir=events_dir,
            clips_filter=clips_filter,
            tolerance_s=args.tolerance_s,
            max_matches_per_label=args.max_matches_per_label,
            split=args.split,
        )
    except EvaluationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print_report(clips, scores, args.tolerance_s, args.max_matches_per_label, args.labels, args.split)

    report = {
        "labels_file": str(args.labels),
        "events_dir": str(events_dir),
        "split": args.split,
        "tolerance_s": args.tolerance_s,
        "max_matches_per_label": args.max_matches_per_label,
        "clip_count": len(clips),
        "negative_clip_count": sum(1 for clip in clips.values() if clip.negative),
        "config_hash": config_hash_used(clips),
        "clips": {
            clip.clip: {
                "negative": clip.negative,
                "label_count": len(clip.labels),
                "prediction_count": len(clip.predictions),
                "duration_s": clip.duration_s,
            }
            for clip in clips.values()
        },
        "per_rule": {name: score.to_dict() for name, score in scores.items()},
        "config_snapshot_used": build_config_section(clips, list(scores)),
        "metric_notes": {
            "precision": "TP / (TP + FP); FP excludes false alarms on negative clips, which are reported separately.",
            "recall": "TP / (TP + FN); each label can be credited at most "
            f"{args.max_matches_per_label} prediction(s).",
            "matching": "One-to-one: each label is credited at most "
            f"{args.max_matches_per_label} prediction(s) and each prediction satisfies at "
            "most one label, so duplicate alerts on one incident are counted as false "
            "positives, not extra TPs.",
            "out_of_window": "A prediction whose type matches a clip label but whose "
            "timestamp falls outside [start_s - tol, end_s + tol] is a false positive.",
            "false_alarms_per_hour": "Unmatched predictions on negative clips, divided by "
            "the TOTAL duration of negative clips (not by the number of clips).",
            "split": "Rows are selected by the optional 'split' column; --split all uses "
            "every row.",
            "confidence_semantics": "event.confidence is DETECTION confidence, not incident probability.",
        },
    }

    output_path = args.output or (args.labels.parent / "evaluation_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print()
    print(f"Report written: {output_path}")

    if staged_dir is not None:
        shutil.rmtree(staged_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
