"""YOLO detection + tracking wrapper (local weights only).

Wraps ``model.track()`` so that:
    * weights are always loaded from a verified absolute local path,
    * the effective tracker YAML from the run output folder is the config used,
    * tracker output is converted into :class:`~model2_behavior.src.tracks.Observation`
      objects carrying the tracker-assigned id.

No network access is possible: offline mode is enforced before loading.

FR-20 (pending supervisor sign-off).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from model2_behavior.src.logging_setup import get_logger
from model2_behavior.src.tracks import Observation
from model2_behavior.src.weights import enforce_offline, resolve_weights_path

logger = get_logger(__name__)


@dataclass
class DetectorSettings:
    """Detector and tracker settings resolved from config.

    Attributes:
        imgsz: Inference size.
        conf: Detection confidence threshold (kept low so ByteTrack's second
            association stage can recover weak detections).
        iou: NMS IoU threshold.
        classes: Class ids to retain.
        max_det: Maximum detections per frame.
        device: Resolved torch device.
        vid_stride: Frame stride for video sources.
    """

    imgsz: int = 640
    conf: float = 0.10
    iou: float = 0.45
    classes: Sequence[int] = (2, 3, 5, 7)
    max_det: int = 100
    device: Any = 0
    vid_stride: int = 1


class DetectorTracker:
    """Owns the YOLO model and exposes per-frame tracking."""

    def __init__(
        self,
        weights_path: str | Path | None = None,
        tracker_yaml: Path | None = None,
        settings: Optional[DetectorSettings] = None,
        models_dir: Optional[Path] = None,
    ) -> None:
        """Load the model from local weights and prepare tracking.

        Args:
            weights_path: Explicit local weight path.
            tracker_yaml: Effective tracker YAML to use for tracking. Must exist.
            settings: Detector settings; defaults are used when omitted.
            models_dir: Directory to resolve default weights in.

        Raises:
            FileNotFoundError: If weights or the tracker YAML are missing locally.
        """
        enforce_offline()

        self.weights_path = resolve_weights_path(weights_path, models_dir)
        self.settings = settings if settings is not None else DetectorSettings()

        if tracker_yaml is None:
            raise ValueError("tracker_yaml is required; pass the effective tracker config path")
        self.tracker_yaml = Path(tracker_yaml).resolve()
        if not self.tracker_yaml.is_file():
            raise FileNotFoundError(f"Tracker YAML not found: {self.tracker_yaml}")

        from ultralytics import YOLO

        self.model = YOLO(str(self.weights_path))
        self._class_names: Dict[int, str] = dict(self.model.names)

        logger.info("Detector ready | weights=%s", self.weights_path)
        logger.info("Tracker YAML in use | %s", self.tracker_yaml)
        logger.info(
            "Detector settings | imgsz=%s conf=%s iou=%s classes=%s max_det=%s device=%s vid_stride=%s",
            self.settings.imgsz,
            self.settings.conf,
            self.settings.iou,
            list(self.settings.classes),
            self.settings.max_det,
            self.settings.device,
            self.settings.vid_stride,
        )

    @property
    def class_names(self) -> Dict[int, str]:
        """Class id to name mapping from the loaded model."""
        return dict(self._class_names)

    def track_frame(self, frame: Any, persist: bool = True) -> List[Observation]:
        """Run detection and tracking on a single frame.

        Args:
            frame: BGR image array.
            persist: Whether the tracker keeps state across calls. Must be True
                for continuous video so ids stay stable.

        Returns:
            Observations for this frame, including the tracker-assigned id.
            Detections without an id (not yet confirmed) are skipped.
        """
        results = self.model.track(
            source=frame,
            persist=persist,
            tracker=str(self.tracker_yaml),
            imgsz=self.settings.imgsz,
            conf=self.settings.conf,
            iou=self.settings.iou,
            classes=list(self.settings.classes),
            max_det=self.settings.max_det,
            device=self.settings.device,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        ids = getattr(boxes, "id", None)
        if ids is None:
            logger.warning("Tracker returned no ids; skipping frame (tracking not active)")
            return []

        coordinates = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy()
        track_ids = ids.cpu().numpy().astype(int).reshape(-1)

        observations: List[Observation] = []
        for index in range(len(track_ids)):
            class_id = int(class_ids[index])
            observations.append(
                Observation(
                    track_id=int(track_ids[index]),
                    frame_index=0,  # filled by the pipeline, which owns frame/time state
                    time_s=0.0,
                    box=(
                        float(coordinates[index][0]),
                        float(coordinates[index][1]),
                        float(coordinates[index][2]),
                        float(coordinates[index][3]),
                    ),
                    confidence=float(confidences[index]),
                    class_id=class_id,
                    class_name=self._class_names.get(class_id, str(class_id)),
                )
            )
        return observations

    def reset_tracker(self) -> None:
        """Reset tracker state so a new video starts with fresh ids."""
        predictor = getattr(self.model, "predictor", None)
        if predictor is not None and hasattr(predictor, "trackers"):
            try:
                predictor.trackers = []
            except Exception:  # pragma: no cover - defensive
                logger.debug("Could not clear predictor.trackers", exc_info=True)
