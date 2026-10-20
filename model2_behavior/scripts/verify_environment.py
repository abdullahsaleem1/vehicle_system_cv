"""Environment verification for Model 2 (Tier A).

Reports interpreter version, CUDA availability, GPU details, and installed
versions of the libraries the pipeline depends on. Also exercises OpenCV video
read/write, because OpenCV 5.0.0 changed several video-capture behaviours.

Run:
    python -m model2_behavior.scripts.verify_environment

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

import json
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure the project root is importable when run as a bare script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REQUIRED_PACKAGES = ("torch", "torchvision", "ultralytics", "cv2", "numpy", "yaml", "requests")

OPTIONAL_PACKAGES = ("pytest", "psutil")


def _package_version(module_name: str) -> Optional[str]:
    """Return the ``__version__`` of an importable module, or None.

    Args:
        module_name: Import name, e.g. ``"cv2"``.

    Returns:
        Version string, or None if the module or its version is unavailable.
    """
    try:
        module = __import__(module_name)
    except Exception:
        return None
    version = getattr(module, "__version__", None)
    if version is None:
        try:
            from importlib.metadata import version as dist_version

            version = dist_version("opencv-python" if module_name == "cv2" else module_name)
        except Exception:
            version = "unknown"
    return str(version)


def check_opencv_video_roundtrip(width: int = 320, height: int = 240, frames: int = 5) -> Dict[str, Any]:
    """Smoke-test OpenCV video writing and reading in a temporary directory.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.
        frames: Number of frames to write.

    Returns:
        Dict with ``ok``, ``writer_backend``, ``frames_written``, ``frames_read``,
        ``details`` and ``error`` keys.
    """
    result: Dict[str, Any] = {
        "ok": False,
        "writer_backend": None,
        "frames_written": 0,
        "frames_read": 0,
        "details": "",
        "error": None,
    }
    try:
        import cv2
        import numpy as np
    except Exception as exc:  # pragma: no cover - import guard
        result["error"] = f"import failed: {exc}"
        return result

    tmp_dir = Path(tempfile.mkdtemp(prefix="model2_cv2_"))
    video_path = tmp_dir / "smoke.mp4"
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 20.0, (width, height))
        result["writer_backend"] = "mp4v"
        if not writer.isOpened():
            result["error"] = "VideoWriter failed to open with mp4v"
            return result

        for index in range(frames):
            frame = np.full((height, width, 3), index * 40 % 255, dtype=np.uint8)
            writer.write(frame)
            result["frames_written"] += 1
        writer.release()

        if not video_path.is_file() or video_path.stat().st_size == 0:
            result["error"] = "VideoWriter produced no output"
            return result

        capture = cv2.VideoCapture(str(video_path))
        count = 0
        while True:
            grabbed, _ = capture.read()
            if not grabbed:
                break
            count += 1
        capture.release()
        result["frames_read"] = count
        result["details"] = f"file size {video_path.stat().st_size} bytes"
        result["ok"] = count >= frames
        if not result["ok"]:
            result["error"] = f"wrote {result['frames_written']} but read back {count}"
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return result


def collect_environment() -> Dict[str, Any]:
    """Collect interpreter, GPU, and package information.

    Returns:
        Nested dict describing the environment.
    """
    info: Dict[str, Any] = {
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "platform": platform.platform(),
            "is_64bit": sys.maxsize > 2**32,
        },
        "packages": {},
        "cuda": {},
        "opencv_video": {},
    }

    for module_name in REQUIRED_PACKAGES:
        info["packages"][module_name] = _package_version(module_name)
    info["packages"]["optional"] = {name: _package_version(name) for name in OPTIONAL_PACKAGES}

    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        info["cuda"] = {
            "available": cuda_available,
            "torch_cuda_version": getattr(torch.version, "cuda", None),
            "device_count": torch.cuda.device_count() if cuda_available else 0,
            "devices": [],
        }
        if cuda_available:
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                info["cuda"]["devices"].append(
                    {
                        "index": index,
                        "name": props.name,
                        "total_vram_gb": round(props.total_memory / (1024**3), 2),
                        "compute_capability": f"{props.major}.{props.minor}",
                    }
                )
    except Exception as exc:  # pragma: no cover - import guard
        info["cuda"] = {"available": False, "error": repr(exc)}

    info["opencv_video"] = check_opencv_video_roundtrip()
    return info


def missing_packages(env: Optional[Dict[str, Any]] = None) -> List[str]:
    """Return required packages that are not importable.

    Args:
        env: Pre-collected environment dict; recomputed if omitted.

    Returns:
        Sorted list of missing module names.
    """
    data = env if env is not None else collect_environment()
    return sorted(name for name in REQUIRED_PACKAGES if not data["packages"].get(name))


def main() -> int:
    """Print the environment report and return an exit code."""
    info = collect_environment()
    print(json.dumps(info, indent=2))

    missing = missing_packages(info)
    if missing:
        print(f"\nMISSING REQUIRED PACKAGES: {', '.join(missing)}", file=sys.stderr)
        return 1

    if not info["opencv_video"].get("ok"):
        print(f"\nOPENCV VIDEO SMOKE TEST FAILED: {info['opencv_video'].get('error')}", file=sys.stderr)
        return 2

    print("\nEnvironment OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
