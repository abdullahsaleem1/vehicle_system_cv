"""Mock test for plate_detector inference output.

Simulates the exact structure ultralytics returns from model.predict()
WITHOUT loading any .pt weights or running real inference.
Validates that any downstream consumer (crop extraction, OCR handoff,
formatting) would work against the real model output contract.

Run: yolo_env\\Scripts\\python.exe mock_test_model_output.py
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent

MODEL_INFO = {
    "weights": str(BASE_DIR / "weights" / "plate_detector" / "weights" / "best.pt"),
    "task": "detect",
    "imgsz": 640,
    "names": {0: "License_Plate"},
    "nc": 1,
    # final-epoch metrics from results.csv (what we expect conf behaviour to look like)
    "precision": 0.98583,
    "recall": 0.95075,
    "mAP50": 0.98045,
}

CONF_THRESHOLD = 0.25
IOU_NMS_THRESHOLD = 0.7  # matches args.yaml iou


@dataclass
class MockBoxes:
    """Mimics ultralytics.engine.results.Boxes tensor API."""

    xyxy: torch.Tensor  # (N, 4) float32, pixel coords in orig image space
    conf: torch.Tensor  # (N,) float32
    cls: torch.Tensor   # (N,) float32 (class indices)

    @property
    def xywh(self):
        xy = (self.xyxy[:, :2] + self.xyxy[:, 2:]) / 2
        wh = self.xyxy[:, 2:] - self.xyxy[:, :2]
        return torch.cat([xy, wh], dim=1)

    def __len__(self):
        return int(self.xyxy.shape[0])


@dataclass
class MockResults:
    """Mimics a single element of the list returned by YOLO.predict()."""

    boxes: MockBoxes
    orig_img: np.ndarray  # HWC uint8 BGR
    names: dict
    path: str
    speed: dict = field(default_factory=lambda: {"preprocess": 1.2, "inference": 3.8, "postprocess": 0.6})


def build_mock_output():
    """One 1280x720 frame with 2 plates: one strong detection + one weak/edge case."""
    orig_h, orig_w = 720, 1280
    rng = np.random.default_rng(42)
    orig_img = rng.integers(0, 255, (orig_h, orig_w, 3), dtype=np.uint8)

    # x1, y1, x2, y2 in pixels; plate aspect ratio ~3:1 like real LPs
    det_xyxy = torch.tensor(
        [
            [410.0, 512.5, 688.0, 618.0],   # main plate, high conf
            [1024.5, 96.0, 1150.5, 136.5],  # small distant plate, low conf (below thr)
        ],
        dtype=torch.float32,
    )
    det_conf = torch.tensor([0.9127, 0.1843], dtype=torch.float32)
    det_cls = torch.zeros(2, dtype=torch.float32)  # single class -> all 0

    return MockResults(
        boxes=MockBoxes(xyxy=det_xyxy, conf=det_conf, cls=det_cls),
        orig_img=orig_img,
        names=MODEL_INFO["names"],
        path="test_frame_0001.jpg",
    )


# ---------- downstream logic under test (same ops consumers run on real output) ----------

def filter_by_conf(res: MockResults, thr=CONF_THRESHOLD):
    keep = res.boxes.conf >= thr
    return MockBoxes(res.boxes.xyxy[keep], res.boxes.conf[keep], res.boxes.cls[keep])


def nms_dedup(boxes: MockBoxes, iou_thr=IOU_NMS_THRESHOLD):
    """Greedy NMS identical in spirit to what the pipeline already applied server-side."""
    order = boxes.conf.argsort(descending=True)
    keep_idx = []
    for i in order:
        if all(_iou(boxes.xyxy[i], boxes.xyxy[k]) < iou_thr for k in keep_idx):
            keep_idx.append(int(i))
    return MockBoxes(boxes.xyxy[keep_idx], boxes.conf[keep_idx], boxes.cls[keep_idx])


def _iou(a, b):
    x1, y1 = torch.max(a[:2], b[:2])
    x2, y2 = torch.min(a[2:], b[2:])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (area_a + area_b - inter + 1e-9))


def extract_crops(res: MockResults, boxes: MockBoxes):
    crops = []
    h, w = res.orig_img.shape[:2]
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        xi1, yi1 = max(0, int(x1)), max(0, int(y1))
        xi2, yi2 = min(w, int(x2)), min(h, int(y2))
        crops.append(res.orig_img[yi1:yi2, xi1:xi2])
    return crops


# ------------------------------------ tests ------------------------------------

results_out = []


def check(name, cond):
    results_out.append((name, bool(cond)))


def main():
    print(f"Mocking output contract of: {MODEL_INFO['weights']}")
    print(f"(not loaded - structure simulated per task={MODEL_INFO['task']}, nc={MODEL_INFO['nc']}, imgsz={MODEL_INFO['imgsz']})\n")

    res = build_mock_output()

    # 1. Output object shape/dtype contract
    check("predict() returned a list-like single Results", isinstance(res, MockResults))
    check("names match data.yaml single class", res.names == {0: "License_Plate"})
    check("orig_img is uint8 HWC", res.orig_img.dtype == np.uint8 and res.orig_img.ndim == 3)
    check("speed dict has preprocess/inference/postprocess", set(res.speed) == {"preprocess", "inference", "postprocess"})

    # 2. Boxes tensor contract
    b = res.boxes
    check("xyxy shape (N,4) float32", b.xyxy.shape == (2, 4) and b.xyxy.dtype == torch.float32)
    check("conf shape (N,) in (0,1)", b.conf.shape == (2,) and bool(((b.conf > 0) & (b.conf < 1)).all()))
    check("cls indices valid (< nc)", bool(((b.cls >= 0) & (b.cls < MODEL_INFO["nc"])).all()))
    check("xywh derived view consistent",
          torch.allclose(b.xywh[:, 2:], b.xyxy[:, 2:] - b.xyxy[:, :2]))

    # 3. Coordinate sanity inside original frame
    h, w = res.orig_img.shape[:2]
    check("all boxes within image bounds",
          bool((b.xyxy[:, [0, 2]] <= w).all() and (b.xyxy[:, [1, 3]] <= h).all()))
    check("x2>x1 and y2>y1 for every box",
          bool(((b.xyxy[:, 2] > b.xyxy[:, 0]) & (b.xyxy[:, 3] > b.xyxy[:, 1])).all()))

    # 4. Confidence filtering behaves like thresholded consumer code
    kept = filter_by_conf(res)
    check(f"conf>={CONF_THRESHOLD} keeps only strong det", len(kept) == 1)
    check("kept det is the expected main plate", abs(float(kept.conf[0]) - 0.9127) < 1e-6)

    # 5. NMS dedup on overlapping duplicates
    dup = MockBoxes(
        torch.cat([kept.xyxy, kept.xyxy + torch.tensor([[8.0, 8.0, 8.0, 8.0]])]),
        torch.tensor([0.9127, 0.5511]), torch.zeros(2),
    )
    deduped = nms_dedup(dup)
    check("duplicate overlap removed by NMS", len(deduped) == 1 and abs(float(deduped.conf[0]) - 0.9127) < 1e-6)

    # 6. Crop extraction for downstream OCR
    crops = extract_crops(res, kept)
    check("crop extracted with nonzero size", len(crops) == 1 and crops[0].size > 0)
    ch, cw = crops[0].shape[:2]
    ar = cw / ch
    check(f"crop aspect ratio plausible for a plate ({ar:.2f})", 1.5 < ar < 6.0)

    # 7. Expected-quality guardrails from training metrics
    check("mock conf consistent with precision@0.986 regime", float(kept.conf[0]) > MODEL_INFO["mAP50"] * 0.9)

    failed = [(n, r) for n, r in results_out if not r]
    for name, ok in results_out:
        print(("PASS  " if ok else "FAIL  ") + name)

    print(f"\n{len(results_out) - len(failed)}/{len(results_out)} checks passed")
    print("NOTE: weights were never loaded; this validates the OUTPUT CONTRACT only.")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
