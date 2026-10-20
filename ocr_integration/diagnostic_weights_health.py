"""Comprehensive Weight Health and Prediction Diagnostic Script.
Analyzes:
1. Model checkpoint architecture, training args, and epoch metadata
2. Numerical health: NaN/Inf checks, weight distribution (mean, std, min, max, sparsity)
3. Layer-by-layer health breakdown (backbone vs neck vs head)
4. OCR model status (EasyOCR CRAFT + Recognizer weights)
5. Live inference prediction output on validation samples
"""

import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ultralytics import YOLO
CKPT_PATH = ROOT_DIR / "plate_detection" / "pakistani_finetune" / "weights" / "pk_plate_finetune_v2" / "weights" / "best.pt"
VAL_IMG_DIR = ROOT_DIR / "plate_detection" / "pakistani_finetune" / "data" / "images" / "val"
OUTPUT_DIR = ROOT_DIR / "ocr_integration" / "outputs"

def inspect_weights_health(ckpt_path: Path):
    print("=" * 80)
    print(f"1. CHECKPOINT NUMERICAL HEALTH AUDIT: {ckpt_path.name}")
    print(f"   Path: {ckpt_path}")
    print("=" * 80)

    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint does not exist at {ckpt_path}")
        return None

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    print(f"Checkpoint keys: {list(ckpt.keys())}")
    epoch = ckpt.get("epoch", -1)
    best_fitness = ckpt.get("best_fitness", None)
    train_args = ckpt.get("train_args", {})
    
    print(f"Saved at Epoch: {epoch} (1-indexed: {epoch + 1 if epoch >= 0 else 'N/A'})")
    print(f"Best Fitness: {best_fitness}")
    print(f"Model task: {train_args.get('task', 'detect')}")
    print(f"Optimizer used: {train_args.get('optimizer', 'AdamW')}")
    print(f"Initial LR (lr0): {train_args.get('lr0', 'N/A')}")
    print(f"Backbone freeze: {train_args.get('freeze', 0)} layers")

    # Extract state dict
    model_obj = ckpt.get("ema") or ckpt.get("model")
    if hasattr(model_obj, "state_dict"):
        state_dict = model_obj.state_dict()
    elif isinstance(model_obj, dict):
        state_dict = model_obj
    else:
        state_dict = model_obj.float().state_dict() if hasattr(model_obj, "float") else {}

    total_params = 0
    total_elements = 0
    nan_count = 0
    inf_count = 0
    zero_count = 0
    tensor_stats = []

    backbone_stats = []
    head_stats = []

    for name, param in state_dict.items():
        if not isinstance(param, torch.Tensor):
            continue
        t = param.detach().cpu().float()
        n_elem = t.numel()
        total_elements += n_elem
        if param.requires_grad or "weight" in name or "bias" in name:
            total_params += n_elem

        n_nan = torch.isnan(t).sum().item()
        n_inf = torch.isinf(t).sum().item()
        n_zero = (t == 0).sum().item()

        nan_count += n_nan
        inf_count += n_inf
        zero_count += n_zero

        mean_val = t.mean().item() if n_elem > 0 else 0.0
        std_val = t.std().item() if n_elem > 1 else 0.0
        min_val = t.min().item() if n_elem > 0 else 0.0
        max_val = t.max().item() if n_elem > 0 else 0.0
        norm_val = torch.norm(t).item()

        stat = {
            "name": name,
            "shape": list(t.shape),
            "numel": n_elem,
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "norm": norm_val,
            "zeros": n_zero,
            "zero_pct": (n_zero / n_elem * 100) if n_elem > 0 else 0,
            "n_nan": n_nan,
            "n_inf": n_inf
        }
        tensor_stats.append(stat)

        # Categorize
        if "model.0." <= name <= "model.9.":
            backbone_stats.append(stat)
        elif "model.22." in name or "model.23." in name or "head" in name:
            head_stats.append(stat)

    print("\n--- Numerical Sanity Check ---")
    print(f"Total Tensors Scanned: {len(tensor_stats)}")
    print(f"Total Parameters: {total_params:,}")
    print(f"NaN Values: {nan_count} {'[CLEAN PASS]' if nan_count == 0 else '[CRITICAL FAIL]'}")
    print(f"Inf Values: {inf_count} {'[CLEAN PASS]' if inf_count == 0 else '[CRITICAL FAIL]'}")
    sparsity = (zero_count / total_elements * 100) if total_elements > 0 else 0
    print(f"Total Exact Zeros: {zero_count:,} ({sparsity:.2f}% overall sparsity)")

    # Aggregate stats
    all_means = [s["mean"] for s in tensor_stats]
    all_stds = [s["std"] for s in tensor_stats]
    all_mins = [s["min"] for s in tensor_stats]
    all_maxs = [s["max"] for s in tensor_stats]

    print("\n--- Weight Distribution Metrics ---")
    print(f"Global Parameter Mean Range: [{min(all_means):.5f}, {max(all_means):.5f}] (Avg layer mean: {np.mean(all_means):.5f})")
    print(f"Global Parameter Std Range:  [{min(all_stds):.5f}, {max(all_stds):.5f}] (Avg layer std:  {np.mean(all_stds):.5f})")
    print(f"Global Parameter Extrema:    Min = {min(all_mins):.4f}, Max = {max(all_maxs):.4f}")

    # Outlier / Vanishing / Exploding weight check
    exploding_layers = [s for s in tensor_stats if abs(s["max"]) > 30.0 or abs(s["min"]) > 30.0]
    vanishing_layers = [s for s in tensor_stats if "weight" in s["name"] and s["std"] < 1e-5 and s["numel"] > 16]

    print("\n--- Anomaly & Health Status ---")
    if not exploding_layers:
        print("[PASS] No exploding weights detected (all weights bounded within [-30, +30])")
    else:
        print(f"[WARN] Potential high magnitude weights in {len(exploding_layers)} tensors:")
        for s in exploding_layers[:5]:
            print(f"       - {s['name']}: min={s['min']:.2f}, max={s['max']:.2f}")

    if not vanishing_layers:
        print("[PASS] No vanishing/collapsed weight tensors detected (all active layers maintain healthy variance)")
    else:
        print(f"[WARN] Vanishing variance detected in {len(vanishing_layers)} tensors:")
        for s in vanishing_layers[:5]:
            print(f"       - {s['name']}: std={s['std']:.6e}")

    # Inspect class names and metadata in YOLO object
    yolo_model = YOLO(str(ckpt_path))
    arch_name = yolo_model.model.__class__.__name__ if hasattr(yolo_model, "model") else "YOLO"
    print(f"\nUltralytics YOLO Architecture: {arch_name} (Task: {yolo_model.task})")
    print(f"Model Names / Classes: {yolo_model.names}")
    print(f"Input image size: {yolo_model.overrides.get('imgsz', 640)}")

    return yolo_model


def inspect_easyocr_weights():
    print("\n" + "=" * 80)
    print("2. EASYOCR ENGINE & WEIGHTS AUDIT")
    print("=" * 80)
    import easyocr
    user_home = Path.home()
    easyocr_model_dir = user_home / ".EasyOCR" / "model"
    print(f"EasyOCR Weight Cache Directory: {easyocr_model_dir}")
    if easyocr_model_dir.exists():
        for p in easyocr_model_dir.glob("*"):
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  - {p.name}: {size_mb:.2f} MB (Modified: {time.ctime(p.stat().st_mtime)})")
    else:
        print("  EasyOCR directory not yet created or located elsewhere.")

    t0 = time.time()
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    load_time = time.time() - t0
    print(f"EasyOCR GPU Reader Initialized in {load_time:.2f}s [STATUS: HEALTHY]")
    return reader


def run_model_predictions(model, reader, max_images=6):
    print("\n" + "=" * 80)
    print("3. LIVE MODEL INFERENCE & PREDICTION AUDIT")
    print("=" * 80)

    import cv2
    from ocr_integration.run_anpr_pipeline import preprocess_crop, clean_plate_text

    val_images = sorted(list(VAL_IMG_DIR.glob("*.jpg")))[:max_images]
    print(f"Running end-to-end inference on {len(val_images)} validation images...\n")

    charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
    pad_fraction = 0.08

    print(f"{'Image':<16} | {'Det Box [x1,y1,x2,y2]':<26} | {'DetConf':<7} | {'Aspect':<6} | {'OCR Conf':<8} | {'Predicted Text':<16} | {'Inference':<10}")
    print("-" * 105)

    for img_path in val_images:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        h, w = img_bgr.shape[:2]
        t_det0 = time.time()
        preds = model.predict(source=img_bgr, conf=0.25, imgsz=640, device=0, verbose=False)[0]
        det_time = (time.time() - t_det0) * 1000

        boxes = preds.boxes
        if len(boxes) == 0:
            print(f"{img_path.name:<16} | {'NO DETECTION':<26} | {'-':<7} | {'-':<6} | {'-':<8} | {'-':<16} | {det_time:.1f}ms")
            continue

        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            det_conf = float(b.conf[0])
            bw, bh = x2 - x1, y2 - y1
            aspect = bw / max(bh, 1e-6)

            pw, ph = bw * pad_fraction, bh * pad_fraction
            xi1, yi1 = max(0, int(x1 - pw)), max(0, int(y1 - ph))
            xi2, yi2 = min(w, int(x2 + pw)), min(h, int(y2 + ph))
            crop = img_bgr[yi1:yi2, xi1:xi2]

            t_ocr0 = time.time()
            if crop.size > 0:
                proc = preprocess_crop(crop)
                results = reader.readtext(proc, allowlist=charset, detail=1)
                frags = []
                for bbox, text, conf in results:
                    x_mean = float(np.mean([p[0] for p in bbox]))
                    frags.append((x_mean, str(text).strip(), float(conf)))
                frags.sort(key=lambda f: f[0])
                parts = [t for _, t, c in frags if c >= 0.3 and t]
                plate_str = " ".join(parts)
                ocr_conf = float(np.mean([c for _, _, c in frags])) if frags else 0.0
            else:
                plate_str = ""
                ocr_conf = 0.0
            ocr_time = (time.time() - t_ocr0) * 1000

            box_str = f"[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]"
            aspect_flag = f"{aspect:.2f}" if aspect >= 1.2 else f"{aspect:.2f}*"
            total_time_str = f"{det_time + ocr_time:.0f}ms"
            
            img_label = img_path.name if i == 0 else ""
            print(f"{img_label:<16} | {box_str:<26} | {det_conf:<7.3f} | {aspect_flag:<6} | {ocr_conf:<8.3f} | {plate_str:<16} | {total_time_str:<10}")

    print("\n* Note: Aspect < 1.2 marked with * indicates suspect non-plate / near-square crop (detector FP / extreme angle).")


def main():
    yolo_model = inspect_weights_health(CKPT_PATH)
    if yolo_model is None:
        return 1
    reader = inspect_easyocr_weights()
    run_model_predictions(yolo_model, reader, max_images=6)
    return 0

if __name__ == "__main__":
    sys.exit(main())
