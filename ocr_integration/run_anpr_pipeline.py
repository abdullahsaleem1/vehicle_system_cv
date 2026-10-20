"""Detect-then-read ANPR pipeline (FR-16).

Loads the fine-tuned v2 YOLO plate detector, crops detections with padding,
preprocesses crops for OCR, runs EasyOCR (English), and emits annotated images
plus structured JSON results.

Run: yolo_env\\Scripts\\python.exe run_anpr_pipeline.py --input <img|folder> [--out outputs] [--conf 0.25] [--limit N]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DETECTOR = BASE_DIR.parent / "plate_detection" / "pakistani_finetune" / "weights" / "pk_plate_finetune_v2" / "weights" / "best.pt"
DEFAULT_INPUT = BASE_DIR.parent / "plate_detection" / "pakistani_finetune" / "data" / "images" / "val"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PLATE_CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
PAD_FRACTION = 0.08
TARGET_OCR_HEIGHT = 96


_reader = None


def get_reader():
    global _reader
    if _reader is None:
        import easyocr

        _reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    return _reader


def cluster_lines(frags):
    if not frags:
        return []
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
    lines.sort(key=lambda line: sum(item["cy"] for item in line) / len(line))
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

    scale = max(1.0, 110.0 / max(h, 1))
    proc_raw = cv2.resize(
        crop_bgr,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_LANCZOS4,
    )
    res_a = reader.readtext(proc_raw, allowlist=PLATE_CHARSET, detail=1)

    gray = cv2.cvtColor(proc_raw, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    proc_clahe = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
    res_b = reader.readtext(proc_clahe, allowlist=PLATE_CHARSET, detail=1)

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

    ordered = cluster_lines(frags)
    valid_parts = [f["text"] for f in ordered if f["conf"] >= 0.20 and len(f["text"]) > 0]
    joined = " ".join(valid_parts)
    ocr_conf = float(np.mean([f["conf"] for f in ordered])) if ordered else 0.0
    return joined, ocr_conf, [f["text"] for f in ordered]


def annotate(img_bgr, det):
    x1, y1, x2, y2 = det["box"]
    cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
    label = f"{det['plate_text'] or '???'} [{det['ocr_conf']:.2f}]" if det["plate_text"] else f"det {det['det_conf']:.2f}"
    ty = int(y1) - 8 if int(y1) - 8 > 12 else int(y2) + 20
    cv2.putText(img_bgr, label, (int(x1), ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img_bgr, label, (int(x1), ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
    return img_bgr


def process_image(path: Path, model, out_dir: Path, det_conf_thr):
    img = cv2.imread(str(path))
    if img is None:
        return {"image": path.name, "error": "unreadable image"}

    dets = model.predict(source=img, conf=det_conf_thr, imgsz=640, device=0, verbose=False)[0]
    records = []
    for b in dets.boxes:
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
        bw, bh = x2 - x1, y2 - y1
        aspect = bw / max(bh, 1e-6)
        pw, ph = bw * PAD_FRACTION, bh * PAD_FRACTION
        H, W = img.shape[:2]
        xi1, yi1 = max(0, int(x1 - pw)), max(0, int(y1 - ph))
        xi2, yi2 = min(W, int(x2 + pw)), min(H, int(y2 + ph))
        crop = img[yi1:yi2, xi1:xi2]
        if crop.size == 0:
            continue
        t0 = time.time()
        text, ocr_conf, raw_frags = read_plate(crop)
        rec = {
            "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "det_conf": round(float(b.conf[0]), 4),
            "aspect": round(aspect, 2),
            "suspect_non_plate_aspect": bool(aspect < 1.2),
            "plate_text": text,
            "plate_text_clean": clean_plate_text(text),
            "ocr_conf": round(ocr_conf, 4),
            "raw_fragments": raw_frags,
            "ocr_ms": round((time.time() - t0) * 1000, 1),
        }
        records.append(rec)
        img = annotate(img, rec)

    out_path = out_dir / f"{path.stem}_annotated{path.suffix.lower()}"
    cv2.imwrite(str(out_path), img)
    return {"image": path.name, "annotated": str(out_path), "detections": records}


def main():
    ap = argparse.ArgumentParser(description="ANPR detect-then-read pipeline")
    ap.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="image file or folder")
    ap.add_argument("--out", type=str, default=str(BASE_DIR / "outputs"))
    ap.add_argument("--conf", type=float, default=0.25, help="detector confidence threshold")
    ap.add_argument("--limit", type=int, default=None, help="max images to process")
    args = ap.parse_args()

    if not DETECTOR.exists():
        print(f"FATAL: detector not found: {DETECTOR}")
        return 1

    inp = Path(args.input)
    if inp.is_file():
        imgs = [inp]
    elif inp.is_dir():
        imgs = sorted(p for p in inp.rglob("*") if p.suffix.lower() in IMG_EXTS)
    else:
        print(f"FATAL: input not found: {inp}")
        return 1
    if args.limit:
        imgs = imgs[: args.limit]
    if not imgs:
        print(f"FATAL: no images under {inp}")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    print(f"Detector : {DETECTOR}")
    print(f"Input    : {inp} ({len(imgs)} image(s))")
    print(f"Output   : {out_dir}")
    get_reader()
    print("EasyOCR ready (en, gpu)\n")

    model = YOLO(str(DETECTOR))

    results = []
    t_start = time.time()
    for i, p in enumerate(imgs, 1):
        r = process_image(p, model, out_dir, args.conf)
        results.append(r)
        dets = r.get("detections", [])
        summary = "; ".join(
            f"'{d['plate_text']}' (ocr {d['ocr_conf']:.2f}, det {d['det_conf']:.2f})" for d in dets
        ) or "no detection"
        print(f"[{i}/{len(imgs)}] {r['image']}: {summary}")

    json_path = out_dir / "anpr_results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    n_det = sum(len(r.get("detections", [])) for r in results)
    n_read = sum(1 for r in results for d in r.get("detections", []) if d["plate_text"])
    print(f"\nDone: {len(results)} images, {n_det} detections, {n_read} plates read "
          f"in {time.time() - t_start:.1f}s -> {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
