"""Local-only model weight resolution, verification, and loading.

Policy (hard requirement for Model 2, Tier A):
    * Weights are verified on local disk under ``model2_behavior/models/``.
    * Every load path is an absolute local filesystem path. A bare model name
      such as ``"yolo11m.pt"`` is never passed to :class:`ultralytics.YOLO`,
      because Ultralytics would then attempt a network download.
    * Runtime auto-download is disabled by forcing offline mode before the
      ``ultralytics`` package is imported.

The optional downloader lives in :mod:`model2_behavior.scripts.download_weights`
and is retained only as a fallback; it is not used by the pipeline.

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

# --------------------------------------------------------------------------
# Offline enforcement
# --------------------------------------------------------------------------
# This MUST run before `ultralytics` is imported anywhere in the process.
# `ultralytics.utils.is_online()` returns False as soon as YOLO_OFFLINE is set,
# which suppresses weight auto-downloads and update/telemetry checks.
os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("YOLO_VERBOSE", "False")


def project_root() -> Path:
    """Return the ``model2_behavior`` root directory."""
    return Path(__file__).resolve().parent.parent


def default_models_dir() -> Path:
    """Return the default local weights directory."""
    return project_root() / "models"


def enforce_offline() -> None:
    """Force offline mode and disable Ultralytics sync/telemetry.

    This mutates only the ``sync`` setting (per project decision) and leaves
    every other global Ultralytics setting untouched.
    """
    os.environ["YOLO_OFFLINE"] = "1"
    try:
        from ultralytics.utils import SETTINGS  # imported lazily by design

        if SETTINGS.get("sync") is not False:
            SETTINGS["sync"] = False
            try:
                SETTINGS.save()
            except Exception:  # pragma: no cover - settings persistence is best-effort
                pass
    except Exception:
        # ultralytics present but settings unavailable; offline env var still holds.
        pass


@dataclass
class WeightRecord:
    """Manifest entry describing one verified weight file."""

    filename: str
    path: str
    sha256: str
    size_bytes: int
    source_url: str
    verified_utc: str
    expected_size_bytes: Optional[int] = None
    expected_sha256: Optional[str] = None
    notes: str = ""
    class_map: Dict[str, str] = field(default_factory=dict)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute the SHA256 hex digest of a file using streamed reads.

    Args:
        path: File to hash.
        chunk_size: Read size in bytes.

    Returns:
        Lowercase hexadecimal SHA256 digest.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Cannot hash missing file: {target}")
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_zip_integrity(path: Path) -> bool:
    """Return True if ``path`` is a readable ZIP archive with valid CRCs.

    Ultralytics ``.pt`` files are ZIP containers (torch archive format). This
    is a strong structural integrity check that requires no network access.

    Args:
        path: Candidate weight file.

    Returns:
        True if the archive opens and every member passes its CRC check.
    """
    import zipfile

    target = Path(path)
    try:
        if not zipfile.is_zipfile(target):
            return False
        with zipfile.ZipFile(target) as archive:
            return archive.testzip() is None
    except Exception:
        return False


def resolve_weights_path(
    weights_path: str | Path | None = None,
    models_dir: Optional[Path] = None,
    filename: str = "yolo11m.pt",
) -> Path:
    """Resolve a weight file to an absolute local path and assert it exists.

    Resolution order:
        1. Explicit ``weights_path`` (absolute, or relative to project root).
        2. ``models_dir / filename``.
        3. ``default_models_dir() / filename``.

    A bare filename with no matching local file raises immediately, so the
    caller can never accidentally trigger an Ultralytics auto-download.

    Args:
        weights_path: Explicit path, if any.
        models_dir: Directory to search, if any.
        filename: Fallback weight filename.

    Returns:
        Absolute path to an existing weight file.

    Raises:
        FileNotFoundError: If no candidate exists on local disk.
    """
    candidates: list[Path] = []

    if weights_path is not None:
        candidate = Path(weights_path)
        if not candidate.is_absolute():
            candidate = project_root() / candidate
        candidates.append(candidate)

    if models_dir is not None:
        candidates.append(Path(models_dir) / filename)

    candidates.append(default_models_dir() / filename)

    checked: list[str] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        checked.append(str(resolved))
        if resolved.is_file():
            return resolved

    raise FileNotFoundError(
        "Model weights not found on local disk. Refusing to download at runtime. "
        f"Checked: {checked}. Place the file in {default_models_dir()} or set "
        "weights_path in configs/model2.yaml."
    )


def write_manifest(
    records: Iterable[WeightRecord],
    manifest_path: Optional[Path] = None,
) -> Path:
    """Write (or update) ``MANIFEST.json`` describing locally held weights.

    Existing entries for other files are preserved.

    Args:
        records: Records to insert or replace, keyed by filename.
        manifest_path: Destination file. Defaults to ``models/MANIFEST.json``.

    Returns:
        Path to the written manifest.

    Raises:
        OSError: If the manifest cannot be written.
    """
    target = Path(manifest_path) if manifest_path else default_models_dir() / "MANIFEST.json"
    target.parent.mkdir(parents=True, exist_ok=True)

    existing: Dict[str, Any] = {}
    if target.is_file():
        try:
            loaded = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("weights"), dict):
                existing = loaded["weights"]
        except (json.JSONDecodeError, OSError):
            existing = {}

    for record in records:
        existing[record.filename] = asdict(record)

    payload = {
        "schema_version": 1,
        "component": "model2_behavior",
        "requirement": "FR-20 (pending supervisor sign-off)",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "policy": {
            "source": "official Ultralytics release assets only",
            "location": "models/ (local disk)",
            "runtime_download": "disabled (YOLO_OFFLINE=1)",
            "loading": "absolute local path only; bare model names forbidden",
        },
        "weights": existing,
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    return target


TARGET_CLASS_IDS: Dict[int, str] = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def load_model(
    weights_path: str | Path | None = None,
    models_dir: Optional[Path] = None,
    filename: str = "yolo11m.pt",
    expected_class_ids: Optional[Dict[int, str]] = None,
):
    """Load a YOLO model strictly from a verified local path.

    Args:
        weights_path: Explicit weight path override.
        models_dir: Directory to resolve ``filename`` in.
        filename: Weight filename when ``weights_path`` is not given.
        expected_class_ids: COCO ids that must map to the expected names.
            Defaults to :data:`TARGET_CLASS_IDS`.

    Returns:
        A loaded :class:`ultralytics.YOLO` instance.

    Raises:
        FileNotFoundError: If weights are absent locally.
        RuntimeError: If the model's class mapping does not match expectations.
    """
    enforce_offline()

    resolved = resolve_weights_path(weights_path, models_dir, filename)

    from ultralytics import YOLO  # imported after offline enforcement

    model = YOLO(str(resolved))  # absolute local path, never a bare name

    expected = expected_class_ids if expected_class_ids is not None else TARGET_CLASS_IDS
    mismatches = {
        cid: (name, model.names.get(cid)) for cid, name in expected.items() if model.names.get(cid) != name
    }
    if mismatches:
        raise RuntimeError(f"Weight class map does not match expected COCO ids: {mismatches}")

    return model


def build_record(
    path: Path,
    source_url: str,
    expected_size_bytes: Optional[int] = None,
    expected_sha256: Optional[str] = None,
    notes: str = "",
    class_map: Optional[Dict[str, str]] = None,
) -> WeightRecord:
    """Build a :class:`WeightRecord` for a local weight file.

    Args:
        path: Local weight file.
        source_url: Official origin URL of the weight.
        expected_size_bytes: Size published by the source, if any.
        expected_sha256: SHA256 published by the source, if any.
        notes: Free-form notes, e.g. verification outcome.
        class_map: Mapping of relevant class ids to names.

    Returns:
        A populated :class:`WeightRecord`.
    """
    target = Path(path)
    return WeightRecord(
        filename=target.name,
        path=str(target.resolve()),
        sha256=sha256_file(target),
        size_bytes=target.stat().st_size,
        source_url=source_url,
        verified_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        expected_size_bytes=expected_size_bytes,
        expected_sha256=expected_sha256,
        notes=notes,
        class_map=class_map if class_map is not None else {str(k): v for k, v in TARGET_CLASS_IDS.items()},
    )
