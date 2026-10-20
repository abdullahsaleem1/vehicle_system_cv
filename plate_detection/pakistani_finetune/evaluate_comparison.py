"""Compare base vs fine-tuned plate detector on the same Pakistani val split.

Runs YOLO.val() for both checkpoints on pakistani_finetune/data (val split)
and prints a side-by-side metrics table.

Run after fine-tuning: yolo_env\\Scripts\\python.exe evaluate_comparison.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "dataset.yaml"
MODELS = [
    ("base", BASE_DIR.parent / "weights" / "plate_detector" / "weights" / "best.pt"),
    ("finetune_v1", BASE_DIR / "weights" / "pk_plate_finetune" / "weights" / "best.pt"),
    ("finetune_v2", BASE_DIR / "weights" / "pk_plate_finetune_v2" / "weights" / "best.pt"),
]

METRIC_KEYS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
}


def pick_device():
    import torch

    return 0 if torch.cuda.is_available() else "cpu"


def evaluate(weights: Path, device):
    model = YOLO(str(weights))
    m = model.val(data=str(DATA_YAML), split="val", imgsz=640, device=device, verbose=False)
    rd = m.results_dict
    out = {k: float(rd.get(vk, float("nan"))) for k, vk in METRIC_KEYS.items()}
    out["images"] = getattr(getattr(m, "box", None), "n", "") or ""
    return out


def main():
    available = [(tag, p) for tag, p in MODELS if p.exists()]
    missing = [str(p) for _, p in MODELS if not p.exists()]
    if missing:
        print("WARNING: weights not found (skipping):")
        for p in missing:
            print(f"  {p}")
    if len(available) < 2:
        print("FATAL: need at least two models to compare.")
        sys.exit(1)

    device = pick_device()
    print("=" * 72)
    print("MODEL COMPARISON - Pakistani validation split")
    print(f"data : {DATA_YAML}")
    print(f"date : {datetime.now():%Y-%m-%d %H:%M:%S} | device: {device}")
    print("=" * 72)

    results = {}
    for tag, weights in available:
        print(f"\nEvaluating {tag}:", weights)
        results[tag] = evaluate(weights, device)

    tags = [t for t, _ in available]
    col_w = 12
    hdr = f"{'metric':<12}" + "".join(f"{t:>{col_w}}" for t in tags)
    if "finetune_v1" in tags and "finetune_v2" in tags:
        hdr += f"{'v2-vs-v1':>{col_w}}"
    print("\n" + hdr)
    print("-" * len(hdr))
    rows = {}
    base_ref = results[tags[0]]
    for k in METRIC_KEYS:
        line = f"{k:<12}" + "".join(f"{results[t][k]:>{col_w}.4f}" for t in tags)
        row = {t: results[t][k] for t in tags}
        if "finetune_v1" in tags and "finetune_v2" in tags:
            d = results["finetune_v2"][k] - results["finetune_v1"][k]
            sign = "+" if d >= 0 else ""
            line += f"{sign + f'{d:.4f}':>{col_w}}"
        rows[k] = row
        print(line)

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data": str(DATA_YAML),
        "models": {t: str(p) for t, p in available},
        "metrics": rows,
    }
    out_file = BASE_DIR / "logs" / "comparison_results.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
