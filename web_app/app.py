"""FastAPI Web Application for Interactive ANPR Plate Detection & OCR.

Provides:
- Real-time interactive UI to upload or pick sample Pakistani vehicle images
- Adjustable detector confidence and aspect-ratio filter toggles
- High-resolution annotated image visualization
- Individual plate crop inspection cards with confidence metrics
- System health and GPU diagnostics
"""

import base64
import io
import time
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO

# Project Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DETECTOR_PATH = ROOT_DIR / "plate_detection" / "pakistani_finetune" / "weights" / "pk_plate_finetune_v2" / "weights" / "best.pt"
VAL_DIR = ROOT_DIR / "plate_detection" / "pakistani_finetune" / "data" / "images" / "val"
CARS_DIR = ROOT_DIR / "Pakistani License Number Plates Data" / "Cars"

PLATE_CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
PAD_FRACTION = 0.08
TARGET_OCR_HEIGHT = 96

# Initialize FastAPI
app = FastAPI(title="Pakistani ANPR Live Prediction & Health Suite", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models
_detector = None
_reader = None

def get_detector():
    global _detector
    if _detector is None:
        if not DETECTOR_PATH.exists():
            raise RuntimeError(f"Detector weights not found at {DETECTOR_PATH}")
        _detector = YOLO(str(DETECTOR_PATH))
    return _detector

def get_reader():
    global _reader
    if _reader is None:
        import easyocr
        use_gpu = torch.cuda.is_available()
        _reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
    return _reader

def cluster_lines(frags):
    if not frags:
        return []
    # Sort by vertical centroid first
    frags_by_y = sorted(frags, key=lambda f: f["cy"])
    lines = []
    for f in frags_by_y:
        placed = False
        for line in lines:
            avg_cy = sum(item["cy"] for item in line) / len(line)
            avg_h = sum(item["h"] for item in line) / len(line)
            if abs(f["cy"] - avg_cy) < 0.50 * avg_h:
                line.append(f)
                placed = True
                break
        if not placed:
            lines.append([f])
    # Sort lines top-to-bottom
    lines.sort(key=lambda line: sum(item["cy"] for item in line) / len(line))
    # Sort items within each line left-to-right
    sorted_frags = []
    for line in lines:
        line.sort(key=lambda item: item["x1"])
        sorted_frags.extend(line)
    return sorted_frags

COMMON_CORRECTIONS = {
    "PUMJAB": "PUNJAB",
    "PUNJAB-": "PUNJAB",
}

def clean_plate_text(raw):
    cleaned = "".join(ch for ch in raw.upper() if ch in PLATE_CHARSET)
    return COMMON_CORRECTIONS.get(cleaned, cleaned)

def read_plate(crop_bgr):
    if crop_bgr.size == 0:
        return "", 0.0, []

    h, w = crop_bgr.shape[:2]
    reader = get_reader()

    # Pass A: High-fidelity Lanczos resize without harsh contrast artifacts (optimal for web/compressed JPEGs)
    scale = max(1.0, 110.0 / max(h, 1))
    proc_raw = cv2.resize(
        crop_bgr,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_LANCZOS4,
    )
    res_a = reader.readtext(proc_raw, allowlist=PLATE_CHARSET, detail=1)

    # Pass B: Mild CLAHE contrast normalization (optimal for low-light/shadowed camera footage)
    gray = cv2.cvtColor(proc_raw, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    proc_clahe = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
    res_b = reader.readtext(proc_clahe, allowlist=PLATE_CHARSET, detail=1)

    # Evaluate best candidate pass by composite confidence
    conf_a = sum(float(c) for _, _, c in res_a) if res_a else 0.0
    conf_b = sum(float(c) for _, _, c in res_b) if res_b else 0.0
    results = res_a if conf_a >= conf_b else res_b

    if not results:
        return "", 0.0, []

    frags = []
    for bbox, text, conf in results:
        t_clean = "".join(ch for ch in text.upper() if ch in PLATE_CHARSET).strip("- ")
        t_clean = COMMON_CORRECTIONS.get(t_clean, t_clean)
        if len(t_clean) == 0:
            continue
        xs = [float(p[0]) for p in bbox]
        ys = [float(p[1]) for p in bbox]
        frags.append({
            "text": t_clean,
            "conf": float(conf),
            "x1": min(xs),
            "y1": min(ys),
            "cx": float(np.mean(xs)),
            "cy": float(np.mean(ys)),
            "h": float(max(ys) - min(ys)),
        })

    if not frags:
        return "", 0.0, []

    # 2D line clustering: groups top-tier (district/year) and bottom-tier (number)
    ordered = cluster_lines(frags)
    valid_parts = [f["text"] for f in ordered if f["conf"] >= 0.20 and len(f["text"]) > 0]
    joined = " ".join(valid_parts)
    ocr_conf = float(np.mean([f["conf"] for f in ordered])) if ordered else 0.0
    return joined, ocr_conf, [f["text"] for f in ordered]

def b64_encode_img(img_bgr, quality=90):
    success, buffer = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

def annotate_image(img_bgr, detections):
    annotated = img_bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        is_suspect = d["is_suspect_aspect"]
        
        # Color: Bright Green for clean plates, Amber/Orange for suspect aspect
        color = (0, 165, 255) if is_suspect else (0, 230, 0)
        thickness = 3
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        
        # Label text
        plate_str = d["text"] or ("SUSPECT ASPECT" if is_suspect else "PLATE DETECTED")
        conf_str = f"{d['det_conf']:.2f}"
        if d["ocr_conf"] > 0:
            conf_str += f" | OCR {d['ocr_conf']:.2f}"
        label = f"{plate_str} [{conf_str}]"
        
        # Background box for label readability
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.65
        (lw, lh), baseline = cv2.getTextSize(label, font, font_scale, 1)
        ty = max(lh + 8, y1 - 10)
        
        cv2.rectangle(annotated, (x1, ty - lh - 6), (x1 + lw + 10, ty + baseline - 2), (20, 20, 20), -1)
        cv2.rectangle(annotated, (x1, ty - lh - 6), (x1 + lw + 10, ty + baseline - 2), color, 1)
        cv2.putText(annotated, label, (x1 + 5, ty - 2), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        
    return annotated

@app.on_event("startup")
def startup_event():
    # Warm up models on start
    print("Pre-warming YOLO detector and EasyOCR...")
    get_detector()
    get_reader()
    print("ANPR System ready for inference!")

@app.get("/api/health")
def get_health():
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    detector = get_detector()
    num_params = sum(p.numel() for p in detector.model.parameters())
    
    return {
        "status": "healthy",
        "weights_path": str(DETECTOR_PATH),
        "weights_size_mb": round(DETECTOR_PATH.stat().st_size / (1024 * 1024), 2),
        "parameters": num_params,
        "classes": detector.names,
        "cuda_available": cuda_avail,
        "device": device_name,
        "easyocr_status": "ready (en)",
    }

@app.get("/api/samples")
def list_samples():
    samples = []
    # Collect from val
    if VAL_DIR.exists():
        for p in sorted(VAL_DIR.glob("*.jpg"))[:8]:
            samples.append({"name": p.name, "category": "Validation Split", "source": "val"})
    # Collect from Cars dataset
    if CARS_DIR.exists():
        for p in sorted(CARS_DIR.glob("*.[jJ][pP][gG]"))[:8]:
            samples.append({"name": p.name, "category": "Unseen Real Vehicles", "source": "cars"})
    return {"samples": samples}

@app.get("/api/sample-image/{source}/{name}")
def get_sample_image(source: str, name: str):
    folder = VAL_DIR if source == "val" else CARS_DIR
    target = folder / name
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Sample image not found")
    return FileResponse(str(target))

@app.post("/api/predict")
async def predict_plate(
    file: Optional[UploadFile] = File(None),
    sample_source: Optional[str] = Form(None),
    sample_name: Optional[str] = Form(None),
    conf_threshold: float = Form(0.25),
    enable_aspect_filter: bool = Form(True),
    min_aspect: float = Form(1.20),
):
    t_start = time.time()
    
    # 1. Read input image
    if file and file.filename:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        filename = file.filename
    elif sample_name and sample_source:
        folder = VAL_DIR if sample_source == "val" else CARS_DIR
        target = folder / sample_name
        if not target.exists():
            raise HTTPException(status_code=404, detail="Sample not found")
        img_bgr = cv2.imread(str(target))
        filename = sample_name
    else:
        raise HTTPException(status_code=400, detail="No image file or sample provided")

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Failed to decode image")

    orig_h, orig_w = img_bgr.shape[:2]

    # 2. Run Plate Detection
    detector = get_detector()
    t_det0 = time.time()
    yolo_res = detector.predict(source=img_bgr, conf=conf_threshold, imgsz=640, device=0 if torch.cuda.is_available() else "cpu", verbose=False)[0]
    det_ms = (time.time() - t_det0) * 1000

    detections = []
    t_ocr_total = 0.0

    for b in yolo_res.boxes:
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
        det_conf = float(b.conf[0])
        bw, bh = x2 - x1, y2 - y1
        aspect = bw / max(bh, 1e-6)
        is_suspect = bool(aspect < min_aspect)

        # Pad crop
        pw, ph = bw * PAD_FRACTION, bh * PAD_FRACTION
        xi1, yi1 = max(0, int(x1 - pw)), max(0, int(y1 - ph))
        xi2, yi2 = min(orig_w, int(x2 + pw)), min(orig_h, int(y2 + ph))
        crop = img_bgr[yi1:yi2, xi1:xi2]

        crop_b64 = b64_encode_img(crop) if crop.size > 0 else ""

        # OCR reading: if aspect filter is enabled and box is suspect, skip OCR to avoid garbage
        plate_text = ""
        ocr_conf = 0.0
        fragments = []
        ocr_item_ms = 0.0

        if crop.size > 0:
            if enable_aspect_filter and is_suspect:
                # Flagged suspect, skip OCR
                pass
            else:
                t0_item = time.time()
                plate_text, ocr_conf, fragments = read_plate(crop)
                ocr_item_ms = (time.time() - t0_item) * 1000
                t_ocr_total += ocr_item_ms

        detections.append({
            "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "det_conf": round(det_conf, 3),
            "aspect": round(aspect, 2),
            "is_suspect_aspect": is_suspect,
            "text": plate_text,
            "clean_text": clean_plate_text(plate_text),
            "ocr_conf": round(ocr_conf, 3),
            "fragments": fragments,
            "crop_base64": crop_b64,
            "ocr_ms": round(ocr_item_ms, 1),
        })

    # Sort detections: valid high-conf plates first
    detections.sort(key=lambda d: (not d["is_suspect_aspect"], d["ocr_conf"], d["det_conf"]), reverse=True)

    # 3. Generate Visual Annotation
    annotated_bgr = annotate_image(img_bgr, detections)
    annotated_b64 = b64_encode_img(annotated_bgr)
    orig_b64 = b64_encode_img(img_bgr, quality=80)

    total_ms = (time.time() - t_start) * 1000

    return {
        "filename": filename,
        "image_size": {"width": orig_w, "height": orig_h},
        "timing": {
            "det_ms": round(det_ms, 1),
            "ocr_ms": round(t_ocr_total, 1),
            "total_ms": round(total_ms, 1),
        },
        "num_detections": len(detections),
        "num_plates_read": sum(1 for d in detections if d["text"]),
        "detections": detections,
        "annotated_image": annotated_b64,
        "original_image": orig_b64,
    }


# Embed Interactive UI
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Pakistani ANPR Live Prediction & Health Suite</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f17;
      --card-bg: #131a26;
      --card-border: #1f2a3c;
      --primary: #10b981;
      --primary-hover: #059669;
      --primary-glow: rgba(16, 185, 129, 0.25);
      --accent: #3b82f6;
      --warning: #f59e0b;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --font: 'Plus Jakarta Sans', sans-serif;
      --mono: 'JetBrains Mono', monospace;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      color: var(--text);
      font-family: var(--font);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    header {
      background: rgba(19, 26, 38, 0.8);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--card-border);
      padding: 1rem 2rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      position: sticky;
      top: 0;
      z-index: 50;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }

    .brand-icon {
      width: 40px;
      height: 40px;
      background: linear-gradient(135deg, #10b981, #047857);
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      font-size: 1.25rem;
      box-shadow: 0 0 20px var(--primary-glow);
    }

    .brand-title {
      font-weight: 800;
      font-size: 1.2rem;
      letter-spacing: -0.02em;
    }

    .brand-subtitle {
      font-size: 0.75rem;
      color: var(--text-muted);
    }

    .health-pill {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      background: #0f172a;
      border: 1px solid var(--card-border);
      padding: 0.4rem 0.9rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-family: var(--mono);
    }

    .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--primary);
      box-shadow: 0 0 10px var(--primary);
      animation: pulse 2s infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.5; transform: scale(0.85); }
    }

    main {
      flex: 1;
      padding: 2rem;
      max-width: 1600px;
      margin: 0 auto;
      width: 100%;
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 2rem;
    }

    @media (max-width: 1024px) {
      main { grid-template-columns: 1fr; }
    }

    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }

    .card-title {
      font-size: 0.95rem;
      font-weight: 700;
      letter-spacing: -0.01em;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      color: #e5e7eb;
    }

    .dropzone {
      border: 2px dashed #334155;
      border-radius: 12px;
      padding: 2rem 1.5rem;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s;
      background: rgba(15, 23, 42, 0.4);
    }

    .dropzone:hover, .dropzone.dragover {
      border-color: var(--primary);
      background: rgba(16, 185, 129, 0.05);
    }

    .drop-icon {
      font-size: 2rem;
      margin-bottom: 0.5rem;
    }

    .control-group {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }

    .control-label {
      font-size: 0.8rem;
      color: var(--text-muted);
      display: flex;
      justify-content: space-between;
    }

    input[type=range] {
      accent-color: var(--primary);
      cursor: pointer;
    }

    .switch-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 0.85rem;
    }

    .toggle {
      position: relative;
      display: inline-block;
      width: 44px;
      height: 24px;
    }

    .toggle input { opacity: 0; width: 0; height: 0; }

    .slider {
      position: absolute;
      cursor: pointer;
      top: 0; left: 0; right: 0; bottom: 0;
      background-color: #334155;
      transition: .3s;
      border-radius: 24px;
    }

    .slider:before {
      position: absolute;
      content: "";
      height: 18px;
      width: 18px;
      left: 3px;
      bottom: 3px;
      background-color: white;
      transition: .3s;
      border-radius: 50%;
    }

    input:checked + .slider { background-color: var(--primary); }
    input:checked + .slider:before { transform: translateX(20px); }

    .btn {
      background: var(--primary);
      color: #042f2e;
      border: none;
      padding: 0.85rem 1.5rem;
      border-radius: 10px;
      font-weight: 700;
      font-size: 0.95rem;
      cursor: pointer;
      transition: all 0.2s;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
    }

    .btn:hover {
      background: var(--primary-hover);
      box-shadow: 0 0 20px var(--primary-glow);
    }

    .samples-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(70px, 1fr));
      gap: 0.5rem;
      max-height: 220px;
      overflow-y: auto;
      padding-right: 4px;
    }

    .sample-btn {
      background: #0f172a;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 0.4rem;
      cursor: pointer;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 0.3rem;
      transition: all 0.2s;
    }

    .sample-btn:hover, .sample-btn.active {
      border-color: var(--primary);
      background: rgba(16, 185, 129, 0.1);
    }

    .sample-img {
      width: 100%;
      height: 48px;
      object-fit: cover;
      border-radius: 4px;
    }

    .sample-name {
      font-size: 0.65rem;
      color: var(--text-muted);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      max-width: 60px;
    }

    /* Right column */
    .view-container {
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
    }

    .stats-bar {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 1rem;
    }

    .stat-box {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
    }

    .stat-label {
      font-size: 0.75rem;
      color: var(--text-muted);
      text-transform: uppercase;
      font-weight: 600;
    }

    .stat-value {
      font-size: 1.4rem;
      font-weight: 800;
      font-family: var(--mono);
      color: #f9fafb;
    }

    .display-box {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }

    .display-header {
      padding: 1rem 1.5rem;
      border-bottom: 1px solid var(--card-border);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .tabs {
      display: flex;
      gap: 0.5rem;
      background: #0f172a;
      padding: 0.25rem;
      border-radius: 8px;
    }

    .tab-btn {
      background: none;
      border: none;
      color: var(--text-muted);
      padding: 0.35rem 0.8rem;
      font-size: 0.8rem;
      font-weight: 600;
      border-radius: 6px;
      cursor: pointer;
    }

    .tab-btn.active {
      background: var(--primary);
      color: #042f2e;
    }

    .image-stage {
      min-height: 440px;
      max-height: 600px;
      background: #000;
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
      overflow: hidden;
    }

    .stage-image {
      max-width: 100%;
      max-height: 600px;
      object-fit: contain;
    }

    .plate-cards {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 1rem;
    }

    .plate-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 0.85rem;
      transition: transform 0.2s;
    }

    .plate-card:hover {
      transform: translateY(-2px);
      border-color: #334155;
    }

    .pk-badge {
      background: linear-gradient(135deg, #064e3b, #047857);
      border: 2px solid #10b981;
      border-radius: 8px;
      padding: 0.75rem 1rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      color: #fff;
      font-family: var(--mono);
      box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }

    .pk-flag {
      font-size: 0.75rem;
      font-weight: 800;
      background: #022c22;
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      letter-spacing: 0.05em;
    }

    .pk-text {
      font-size: 1.35rem;
      font-weight: 800;
      letter-spacing: 0.1em;
    }

    .plate-meta {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.5rem;
      font-size: 0.75rem;
      font-family: var(--mono);
    }

    .meta-item {
      background: #0f172a;
      padding: 0.5rem;
      border-radius: 6px;
      border: 1px solid #1e293b;
    }

    .meta-title { color: var(--text-muted); margin-bottom: 2px; }
    .meta-val { font-weight: 700; color: #e2e8f0; }

    .crop-preview {
      width: 100%;
      height: 70px;
      object-fit: contain;
      background: #050811;
      border-radius: 6px;
      border: 1px solid #1e293b;
    }

    .loader {
      display: none;
      position: absolute;
      top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(11, 15, 23, 0.85);
      backdrop-filter: blur(4px);
      align-items: center;
      justify-content: center;
      flex-direction: column;
      gap: 1rem;
      z-index: 40;
    }

    .spinner {
      width: 48px;
      height: 48px;
      border: 4px solid #1e293b;
      border-top-color: var(--primary);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }

    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <div class="brand-icon">PK</div>
      <div>
        <div class="brand-title">Pakistani ANPR Live Studio</div>
        <div class="brand-subtitle">YOLOv8n Fine-Tuned v2 + EasyOCR 1.7.2 (CUDA Accelerated)</div>
      </div>
    </div>
    <div class="health-pill" id="health-badge">
      <div class="dot"></div>
      <span id="health-text">AUDITING WEIGHTS...</span>
    </div>
  </header>

  <main>
    <!-- Controls Sidebar -->
    <div style="display: flex; flex-direction: column; gap: 1.5rem;">
      <div class="card">
        <div class="card-title">Image Input</div>
        <div class="dropzone" id="dropzone">
          <div class="drop-icon">📷</div>
          <div style="font-weight: 600; font-size: 0.9rem;">Drop vehicle photo here</div>
          <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.25rem;">or click to browse local files</div>
          <input type="file" id="file-input" accept="image/*" style="display: none;">
        </div>

        <div class="control-group">
          <div class="control-label">
            <span>Detection Confidence</span>
            <span id="conf-val" style="font-family: var(--mono); color: var(--primary);">0.25</span>
          </div>
          <input type="range" id="conf-slider" min="0.10" max="0.90" step="0.05" value="0.25">
        </div>

        <div class="switch-row">
          <div>
            <div style="font-weight: 600;">Geometric Aspect Filter</div>
            <div style="font-size: 0.75rem; color: var(--text-muted);">Drop square false-positive crops</div>
          </div>
          <label class="toggle">
            <input type="checkbox" id="aspect-toggle" checked>
            <span class="slider"></span>
          </label>
        </div>

        <div class="control-group" id="aspect-slider-group">
          <div class="control-label">
            <span>Min Plate Aspect Ratio</span>
            <span id="aspect-val" style="font-family: var(--mono); color: var(--primary);">1.20</span>
          </div>
          <input type="range" id="aspect-slider" min="1.0" max="2.0" step="0.05" value="1.20">
        </div>

        <button class="btn" id="run-btn">
          <span>Run Live Inference</span>
        </button>
      </div>

      <!-- Sample Images Library -->
      <div class="card">
        <div class="card-title">Quick Test Gallery (Pakistani Vehicles)</div>
        <div style="font-size: 0.75rem; color: var(--text-muted);">Click any vehicle below to test instantly:</div>
        <div class="samples-grid" id="samples-container">
          <!-- Loaded via JS -->
        </div>
      </div>
    </div>

    <!-- Live Viewport -->
    <div class="view-container">
      <!-- Performance Stats -->
      <div class="stats-bar">
        <div class="stat-box">
          <div class="stat-label">Detector Latency</div>
          <div class="stat-value" id="stat-det">-- ms</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">EasyOCR Latency</div>
          <div class="stat-value" id="stat-ocr">-- ms</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">End-to-End Speed</div>
          <div class="stat-value" id="stat-total" style="color: var(--primary);">-- ms</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Plates Read</div>
          <div class="stat-value" id="stat-plates">0</div>
        </div>
      </div>

      <!-- Display Stage -->
      <div class="display-box">
        <div class="display-header">
          <div style="font-weight: 700; font-size: 0.9rem;" id="active-filename">No image selected</div>
          <div class="tabs">
            <button class="tab-btn active" id="tab-annotated">Annotated Overlays</button>
            <button class="tab-btn" id="tab-original">Original Input</button>
          </div>
        </div>

        <div class="image-stage">
          <div class="loader" id="loader">
            <div class="spinner"></div>
            <div style="font-weight: 600; font-size: 0.9rem;">Running YOLOv8n + EasyOCR...</div>
          </div>
          <img id="stage-img" class="stage-image" src="" alt="" style="display: none;">
          <div id="stage-placeholder" style="color: #475569; font-size: 0.9rem;">Upload an image or pick a sample to start</div>
        </div>
      </div>

      <!-- Plate Results Cards -->
      <div style="display: flex; flex-direction: column; gap: 0.75rem;">
        <div class="card-title">Extracted Plate Identifiers</div>
        <div class="plate-cards" id="plates-container">
          <!-- Populated via JS -->
        </div>
      </div>
    </div>
  </main>

  <script>
    let currentData = null;
    let selectedFile = null;
    let selectedSample = null;

    // Elements
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const confSlider = document.getElementById('conf-slider');
    const confVal = document.getElementById('conf-val');
    const aspectToggle = document.getElementById('aspect-toggle');
    const aspectSlider = document.getElementById('aspect-slider');
    const aspectVal = document.getElementById('aspect-val');
    const runBtn = document.getElementById('run-btn');
    const loader = document.getElementById('loader');
    const stageImg = document.getElementById('stage-img');
    const stagePlaceholder = document.getElementById('stage-placeholder');
    const tabAnnotated = document.getElementById('tab-annotated');
    const tabOriginal = document.getElementById('tab-original');
    const platesContainer = document.getElementById('plates-container');
    const samplesContainer = document.getElementById('samples-container');

    // Stats
    const statDet = document.getElementById('stat-det');
    const statOcr = document.getElementById('stat-ocr');
    const statTotal = document.getElementById('stat-total');
    const statPlates = document.getElementById('stat-plates');
    const activeFilename = document.getElementById('active-filename');

    // Health check
    async function checkHealth() {
      try {
        const res = await fetch('/api/health');
        const d = await res.json();
        document.getElementById('health-text').innerText = `${d.device} | ${d.weights_size_mb}MB (HEALTHY)`;
      } catch (e) {
        document.getElementById('health-text').innerText = 'OFFLINE';
      }
    }

    // Load samples
    async function loadSamples() {
      try {
        const res = await fetch('/api/samples');
        const d = await res.json();
        samplesContainer.innerHTML = '';
        d.samples.forEach(s => {
          const btn = document.createElement('div');
          btn.className = 'sample-btn';
          btn.innerHTML = `
            <img class="sample-img" src="/api/sample-image/${s.source}/${s.name}" alt="">
            <span class="sample-name" title="${s.name}">${s.name}</span>
          `;
          btn.onclick = () => {
            document.querySelectorAll('.sample-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            selectedFile = null;
            selectedSample = s;
            activeFilename.innerText = s.name;
            runPrediction();
          };
          samplesContainer.appendChild(btn);
        });

        // Trigger first sample automatically
        if (d.samples.length > 0) {
          samplesContainer.children[0].click();
        }
      } catch (e) {
        console.error(e);
      }
    }

    // Slider bindings
    confSlider.oninput = () => confVal.innerText = parseFloat(confSlider.value).toFixed(2);
    aspectSlider.oninput = () => aspectVal.innerText = parseFloat(aspectSlider.value).toFixed(2);

    // Dropzone
    dropzone.onclick = () => fileInput.click();
    fileInput.onchange = (e) => {
      if (e.target.files.length) {
        selectedFile = e.target.files[0];
        selectedSample = null;
        document.querySelectorAll('.sample-btn').forEach(b => b.classList.remove('active'));
        activeFilename.innerText = selectedFile.name;
        runPrediction();
      }
    };
    dropzone.ondragover = (e) => { e.preventDefault(); dropzone.classList.add('dragover'); };
    dropzone.ondragleave = () => dropzone.classList.remove('dragover');
    dropzone.ondrop = (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      if (e.dataTransfer.files.length) {
        selectedFile = e.dataTransfer.files[0];
        selectedSample = null;
        document.querySelectorAll('.sample-btn').forEach(b => b.classList.remove('active'));
        activeFilename.innerText = selectedFile.name;
        runPrediction();
      }
    };

    runBtn.onclick = () => runPrediction();

    async function runPrediction() {
      if (!selectedFile && !selectedSample) {
        alert('Please pick a sample image or upload a file first!');
        return;
      }

      loader.style.display = 'flex';
      const fd = new FormData();
      if (selectedFile) {
        fd.append('file', selectedFile);
      } else {
        fd.append('sample_source', selectedSample.source);
        fd.append('sample_name', selectedSample.name);
      }
      fd.append('conf_threshold', confSlider.value);
      fd.append('enable_aspect_filter', aspectToggle.checked);
      fd.append('min_aspect', aspectSlider.value);

      try {
        const res = await fetch('/api/predict', { method: 'POST', body: fd });
        currentData = await res.json();
        renderResults(currentData);
      } catch (e) {
        alert('Prediction failed: ' + e.message);
      } finally {
        loader.style.display = 'none';
      }
    }

    function renderResults(d) {
      statDet.innerText = d.timing.det_ms + ' ms';
      statOcr.innerText = d.timing.ocr_ms + ' ms';
      statTotal.innerText = d.timing.total_ms + ' ms';
      statPlates.innerText = d.num_plates_read;

      stagePlaceholder.style.display = 'none';
      stageImg.style.display = 'block';
      stageImg.src = d.annotated_image;
      tabAnnotated.classList.add('active');
      tabOriginal.classList.remove('active');

      // Render Plate Cards
      platesContainer.innerHTML = '';
      if (!d.detections.length) {
        platesContainer.innerHTML = '<div style="color: var(--text-muted); font-size: 0.9rem;">No vehicle plates detected at current confidence threshold.</div>';
        return;
      }

      d.detections.forEach((det, i) => {
        const card = document.createElement('div');
        card.className = 'plate-card';

        const isSuspect = det.is_suspect_aspect;
        const plateText = det.text || (isSuspect ? 'SUSPECT NON-PLATE' : 'READING FAILED');
        const badgeBg = isSuspect ? 'linear-gradient(135deg, #78350f, #b45309)' : 'linear-gradient(135deg, #064e3b, #047857)';
        const borderCol = isSuspect ? '#f59e0b' : '#10b981';

        card.innerHTML = `
          <div class="pk-badge" style="background: ${badgeBg}; border-color: ${borderCol};">
            <div class="pk-flag">${isSuspect ? 'REJECTED' : 'PAKISTAN'}</div>
            <div class="pk-text">${plateText}</div>
          </div>
          <img class="crop-preview" src="${det.crop_base64}" alt="Plate crop">
          <div class="plate-meta">
            <div class="meta-item">
              <div class="meta-title">Detector Conf</div>
              <div class="meta-val">${(det.det_conf * 100).toFixed(1)}%</div>
            </div>
            <div class="meta-item">
              <div class="meta-title">OCR Conf</div>
              <div class="meta-val">${(det.ocr_conf * 100).toFixed(1)}%</div>
            </div>
            <div class="meta-item">
              <div class="meta-title">Aspect Ratio</div>
              <div class="meta-val">${det.aspect.toFixed(2)} ${isSuspect ? '⚠️' : '✅'}</div>
            </div>
            <div class="meta-item">
              <div class="meta-title">OCR Latency</div>
              <div class="meta-val">${det.ocr_ms} ms</div>
            </div>
          </div>
          <div style="font-size: 0.7rem; color: var(--text-muted); font-family: var(--mono);">
            Bounding Box: [${det.box.join(', ')}]
          </div>
        `;
        platesContainer.appendChild(card);
      });
    }

    tabAnnotated.onclick = () => {
      if (currentData) {
        stageImg.src = currentData.annotated_image;
        tabAnnotated.classList.add('active');
        tabOriginal.classList.remove('active');
      }
    };

    tabOriginal.onclick = () => {
      if (currentData) {
        stageImg.src = currentData.original_image;
        tabOriginal.classList.add('active');
        tabAnnotated.classList.remove('active');
      }
    };

    // Init
    checkHealth();
    loadSamples();
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index_view():
    return HTMLResponse(content=INDEX_HTML)

def main():
    import uvicorn
    print(f"Starting Pakistani ANPR Live Studio server at http://127.0.0.1:8000 ...")
    uvicorn.run("web_app.app:app", host="127.0.0.1", port=8000, reload=False)

if __name__ == "__main__":
    main()
