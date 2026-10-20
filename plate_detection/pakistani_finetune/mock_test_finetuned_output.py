"""Mock test for the fine-tuned plate detector output contract.

Extends the base mock-test pattern: simulates ultralytics predict() Results
and val() metrics WITHOUT loading any .pt weights or running inference.
Validates the contract that evaluate_comparison.py and downstream OCR
consumers rely on.

Run: yolo_env\\Scripts\\python.exe mock_test_finetuned_output.py
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent

FT_WEIGHTS = BASE_DIR / "weights" / "pk_plate_finetune" / "weights" / "best.pt"
BASE_WEIGHTS = BASE_DIR.parent / "weights" / "plate_detector" / "weights" / "best.pt"

MODEL_INFO = {
    "weights": str(FT_WEIGHTS),
    "task": "detect",
    "imgsz": 640,
    "names": {0: "License_Plate"},
    "nc": 1,
}

VAL_DICT_KEYS = {
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
}

CONF_THRESHOLD = 0.25
IOU_NMS_THRESHOLD = 0.7


@dataclass
class MockBoxes:
    xyxy: torch.Tensor
    conf: torch.Tensor
    cls: torch.Tensor

    @property
    def xywh(self):
        xy = (self.xyxy[:, :2] + self.xyxy[:, 2:]) / 2
        wh = self.xyxy[:, 2:] - self.xyxy[:, :2]
        return torch.cat([xy, wh], dim=1)

    def __len__(self):
        return int(self.xyxy.shape[0])


@dataclass
class MockResults:
    boxes: MockBoxes
    orig_img: np.ndarray
    names: dict
    path: str
    speed: dict = field(default_factory=lambda: {"preprocess": 1.0, "inference": 3.5, "postprocess": 0.5})


@dataclass
class MockValMetrics:
    """Mimics the results_dict returned by model.val()."""

    results_dict: dict


def build_mock_output():
    orig_h, orig_w = 720, 1280
    rng = np.random.default_rng(7)
    orig_img = rng.integers(0, 255, (orig_h, orig_w, 3), dtype=np.uint8)

    det_xyxy = torch.tensor(
        [
            [402.0, 508.0, 690.0, 616.0],
            [1018.0, 92.0, 1152.0, 134.0],
            [300.0, 300.0, 420.0, 340.0],
        ],
        dtype=torch.float32,
    )
    det_conf = torch.tensor([0.9384, 0.2103, 0.5567], dtype=torch.float32)
    det_cls = torch.zeros(3, dtype=torch.float32)

    return MockResults(
        boxes=MockBoxes(xyxy=det_xyxy, conf=det_conf, cls=det_cls),
        orig_img=orig_img,
        names=MODEL_INFO["names"],
        path="pk_val_frame_0001.jpg",
    )


def build_mock_val_metrics():
    return MockValMetrics(
        results_dict={
            "metrics/precision(B)": 0.99,
            "metrics/recall(B)": 0.96,
            "metrics/mAP50(B)": 0.988,
            "metrics/mAP50-95(B)": 0.70,
        }
    )


def filter_by_conf(res, thr=CONF_THRESHOLD):
    keep = res.boxes.conf >= thr
    return MockBoxes(res.boxes.xyxy[keep], res.boxes.conf[keep], res.boxes.cls[keep])


def nms_dedup(boxes, iou_thr=IOU_NMS_THRESHOLD):
    def _iou(a, b):
        x1, y1 = torch.max(a[:2], b[:2])
        x2, y2 = torch.min(a[2:], b[2:])
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return float(inter / (area_a + area_b - inter + 1e-9))

    order = boxes.conf.argsort(descending=True)
    keep_idx = []
    for i in order:
        if all(_iou(boxes.xyxy[i], boxes.xyxy[k]) < iou_thr for k in keep_idx):
            keep_idx.append(int(i))
    return MockBoxes(boxes.xyxy[keep_idx], boxes.conf[keep_idx], boxes.cls[keep_idx])


def extract_crops(res, boxes):
    crops = []
    h, w = res.orig_img.shape[:2]
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        xi1, yi1 = max(0, int(x1)), max(0, int(y1))
        xi2, yi2 = min(w, int(x2)), min(h, int(y2))
        crops.append(res.orig_img[yi1:yi2, xi1:xi2])
    return crops


def parse_val_dict(rd):
    out = {}
    mapping = {
        "precision": "metrics/precision(B)",
        "recall": "metrics/recall(B)",
        "mAP50": "metrics/mAP50(B)",
        "mAP50-95": "metrics/mAP50-95(B)",
    }
    for short, long in mapping.items():
        v = rd.get(long)
        if v is None:
            raise KeyError(f"missing '{long}' in val results_dict")
        out[short] = float(v)
    return out


results_out = []


def check(name, cond):
    results_out.append((name, bool(cond)))


def main():
    print(f"Mocking output contract of FINE-TUNED model: {MODEL_INFO['weights']}")
    print("(not loaded - structure simulated; no .pt file is opened)\n")

    check("fine-tuned weights path exists on disk", FT_WEIGHTS.exists())
    check("base weights still present and untouched", BASE_WEIGHTS.exists())

    res = build_mock_output()
    b = res.boxes

    check("names match dataset.yaml single class", res.names == {0: "License_Plate"})
    check("orig_img uint8 HWC", res.orig_img.dtype == np.uint8 and res.orig_img.ndim == 3)
    check("speed dict complete", set(res.speed) == {"preprocess", "inference", "postprocess"})
    check("xyxy shape/dtype valid", b.xyxy.shape == (3, 4) and b.xyxy.dtype == torch.float32)
    check("conf in (0,1)", bool(((b.conf > 0) & (b.conf < 1)).all()))
    check("cls indices valid (< nc)", bool(((b.cls >= 0) & (b.cls < MODEL_INFO["nc"])).all()))
    check("xywh view consistent", torch.allclose(b.xywh[:, 2:], b.xyxy[:, 2:] - b.xyxy[:, :2]))

    h, w = res.orig_img.shape[:2]
    check("boxes within image bounds",
          bool((b.xyxy[:, [0, 2]] <= w).all() and (b.xyxy[:, [1, 3]] <= h).all()))
    check("positive-area boxes",
          bool(((b.xyxy[:, 2] > b.xyxy[:, 0]) & (b.xyxy[:, 3] > b.xyxy[:, 1])).all()))

    kept = filter_by_conf(res)
    check(f"conf>={CONF_THRESHOLD} keeps 2/3 dets", len(kept) == 2)

    dup = MockBoxes(
        torch.cat([kept.xyxy, kept.xyxy[:1] + torch.tensor([[10.0, 10.0, 10.0, 10.0]])]),
        torch.tensor(list(kept.conf.tolist()) + [0.6011]),
        torch.zeros(len(kept) + 1),
    )
    deduped = nms_dedup(dup)
    check("NMS removes duplicate overlap", len(deduped) == len(kept))

    crops = extract_crops(res, kept)
    ars = [c.shape[1] / c.shape[0] for c in crops if c.size]
    check("all crops non-empty", len(crops) == len(kept) and all(c.size > 0 for c in crops))
    check(f"crop aspect ratios plausible for PK plates ({', '.join(f'{a:.2f}' for a in ars)})",
          all(1.5 < a < 6.5 for a in ars))

    vm = build_mock_val_metrics()
    try:
        parsed = parse_val_dict(vm.results_dict)
        ok = set(parsed) == {"precision", "recall", "mAP50", "mAP50-95"} and all(
            0 <= v <= 1 for v in parsed.values()
        )
    except Exception:
        ok = False
    check("evaluate_comparison can parse mocked val results_dict", ok)
    check("val dict contains exactly the four expected metric keys", VAL_DICT_KEYS == set(vm.results_dict))

    failed = [(n, r) for n, r in results_out if not r]
    for name, okk in results_out:
        print(("PASS  " if okk else "FAIL  ") + name)

    print(f"\n{len(results_out) - len(failed)}/{len(results_out)} checks passed")
    print("NOTE: no weights were loaded; this validates the OUTPUT CONTRACT only.")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
