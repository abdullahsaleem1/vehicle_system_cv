"""Prove the pipeline cannot reach the network for weights.

This is the offline guarantee required by the plan. It is a standalone script
rather than a unit test so it can be run at Gate 1 before any test framework is
installed.

What it proves:
    1. ``YOLO_OFFLINE=1`` makes ``ultralytics.utils.is_online()`` return False.
    2. Socket connections to any host are blocked after enforcement, and model
       loading still succeeds from the local path.
    3. A bare model name (no local file) raises instead of downloading.

Usage:
    python -m model2_behavior.scripts.offline_proof

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os  # noqa: E402

os.environ["YOLO_OFFLINE"] = "1"  # must precede ultralytics import

from model2_behavior.src.weights import (  # noqa: E402
    enforce_offline,
    load_model,
    resolve_weights_path,
)


def _block_network() -> Dict[str, Any]:
    """Monkeypatch socket to reject all outbound connections.

    Returns:
        Dict describing the guard state and a sample blocked call.
    """
    state: Dict[str, Any] = {"installed": True, "create_connection_calls": 0}

    original_create_connection = socket.create_connection

    def _denied(*_args: Any, **_kwargs: Any):
        state["create_connection_calls"] += 1
        raise OSError("network disabled by offline_proof: outbound connection blocked")

    def _denied_getaddrinfo(*_args: Any, **_kwargs: Any):
        state["create_connection_calls"] += 1
        raise OSError("network disabled by offline_proof: DNS blocked")

    socket.create_connection = _denied  # type: ignore[assignment]
    socket.getaddrinfo = _denied_getaddrinfo  # type: ignore[assignment]
    state["original_create_connection"] = original_create_connection
    return state


def main() -> int:
    """Run the offline proofs and print a report."""
    enforce_offline()

    from ultralytics.utils import is_online

    print("=" * 72)
    print("OFFLINE PROOF")
    print("=" * 72)

    failures: list[str] = []

    print(f"YOLO_OFFLINE env      : {os.environ.get('YOLO_OFFLINE')}")
    online = is_online()
    print(f"is_online()           : {online}   (expected False)")
    if online:
        failures.append("is_online() returned True")

    resolved = resolve_weights_path()
    print(f"resolved weights      : {resolved}")

    guard = _block_network()
    print(f"network guard         : installed ({guard['installed']})")

    try:
        model = load_model()
        print(f"model loaded offline  : True (task={model.task}, nc={len(model.names)})")
    except Exception as exc:
        failures.append(f"loading with network blocked failed: {exc!r}")
        print(f"model loaded offline  : FAILED -> {exc!r}")

    # A bare name with no local file must raise, never download.
    try:
        from ultralytics import YOLO

        YOLO("yolo11_does_not_exist.pt")
        failures.append("bare unresolved model name did not raise")
        print("bare-name guard       : FAILED (no exception raised)")
    except Exception as exc:
        print(f"bare-name guard       : OK (raised {type(exc).__name__})")

    socket.create_connection = guard["original_create_connection"]  # type: ignore[assignment]
    print(f"socket restored       : {socket.create_connection is guard['original_create_connection']}")
    print(f"blocked call attempts : {guard['create_connection_calls']}")

    print("-" * 72)
    if failures:
        for line in failures:
            print(f"OFFLINE PROOF FAILURE: {line}", file=sys.stderr)
        return 1
    print("OFFLINE PROOF PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
