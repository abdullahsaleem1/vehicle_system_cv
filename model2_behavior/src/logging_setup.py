"""Logging configuration for the Model 2 (Tier A) behavior/incident detector.

Provides a single :func:`setup_logging` entry point that installs a console
handler and a timestamped per-run log file. Import this module and call
:func:`setup_logging` once, as early as possible in any entry-point script.

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Module-level guard so repeated calls do not stack duplicate handlers.
_CONFIGURED = False


def _project_root() -> Path:
    """Return the ``model2_behavior`` package root directory."""
    return Path(__file__).resolve().parent.parent


def setup_logging(
    name: str = "model2",
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    run_tag: Optional[str] = None,
) -> Path:
    """Configure root logging with console and timestamped file handlers.

    Args:
        name: Base name used for the log file, e.g. ``"pipeline"``.
        log_dir: Directory for log files. Defaults to ``model2_behavior/logs``.
        level: Logging level for the root logger.
        run_tag: Optional extra tag appended to the log filename.

    Returns:
        Path to the log file that was created.

    Raises:
        OSError: If the log directory cannot be created.
    """
    global _CONFIGURED

    root = logging.getLogger()
    if _CONFIGURED:
        # Already configured by an earlier call; return the existing file handler path.
        for handler in root.handlers:
            if isinstance(handler, logging.FileHandler):
                return Path(handler.baseFilename)
        return _project_root() / "logs" / "model2.log"

    resolved_dir = Path(log_dir) if log_dir is not None else _project_root() / "logs"
    resolved_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"_{run_tag}" if run_tag else ""
    log_path = resolved_dir / f"{name}_{stamp}{suffix}.log"

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file_handler)

    # Third-party libraries are noisy at INFO; keep them at WARNING.
    for noisy in ("ultralytics", "matplotlib", "PIL", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    logging.getLogger(name).info("Logging initialised -> %s", log_path)
    return log_path


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger.

    Args:
        name: Logger name, conventionally ``__name__``.

    Returns:
        A :class:`logging.Logger` instance.
    """
    return logging.getLogger(name)
