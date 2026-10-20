# Pakistani Vehicle ANPR System

A **detect-then-read** Automatic Number Plate Recognition (ANPR) pipeline specialized for
**Pakistani vehicle license plates**, paired with an interactive FastAPI web studio for
live inference, plate-crop inspection, and system diagnostics.

The system combines a fine-tuned YOLO plate detector with a multi-pass EasyOCR reader and a
geometric aspect-ratio gate that suppresses non-plate false positives before OCR runs.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Quick Start](#quick-start)
- [Web Studio API](#web-studio-api)
- [Offline Batch Inference](#offline-batch-inference)
- [Project Structure](#project-structure)
- [Model Weights](#model-weights)
- [Repository Layout & What Is Not Stored in Git](#repository-layout--what-is-not-stored-in-git)
- [Datasets](#datasets)
- [Training & Fine-Tuning](#training--fine-tuning)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [Known Limitations](#known-limitations)
- [Reports & Documentation](#reports--documentation)

---

## How It Works

The pipeline is a single forward flow with a cheap geometric guard in the middle:

```
            ┌─────────────────┐
 image ───▶ │  YOLO detector  │  fine-tuned PK plate detector, 1 class "License_Plate"
            └────────┬────────┘
                     │  boxes + detection confidence
                     ▼
            ┌─────────────────┐
            │ Aspect gate     │  aspect = width / height; boxes < min_aspect (1.20)
            │                 │  are flagged `is_suspect_aspect` and skipped by OCR
            └────────┬────────┘
                     │  padded plate crops (+8% padding)
                     ▼
            ┌─────────────────┐
            │  EasyOCR reader │  2 passes, charset A-Z 0-9 -, GPU when available
            └────────┬────────┘
                     │  fragments
                     ▼
            ┌─────────────────┐
            │ Line clustering │  group fragments by vertical centroid, order L→R
            └────────┬────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ Post-processing │  charset cleanup + common corrections (PUMJAB→PUNJAB)
            └────────┬────────┘
                     │
                     ▼
              plate text + confidence
```

### Stage 1 — Plate detection

A YOLO model fine-tuned on a single class, `License_Plate`, run at `imgsz=640` with a
configurable confidence threshold (`conf`, default `0.25`). Uses CUDA when available and
falls back to CPU automatically.

### Stage 2 — Geometric aspect gate

Pakistani plates are physically wide. The pipeline computes each box's aspect ratio
(`w / h`) and marks any box below `min_aspect` (default `1.20`) as
`is_suspect_aspect`. When `enable_aspect_filter` is on, suspect boxes are **not** sent to
OCR — this is what prevents square-ish detector false positives from producing garbage text.
Observed in practice: wide plate crops (≥ ~1.5:1) read reliably, square-ish crops are mostly
false positives or extreme viewing angles.

### Stage 3 — Multi-pass OCR

Each crop is upscaled to a target height with Lanczos and read twice by EasyOCR
(English, `allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"`):

| Pass | Pre-processing | Best for |
|---|---|---|
| A | Lanczos resize only (no harsh contrast) | clean / compressed images (web JPEGs) |
| B | Lanczos + mild CLAHE contrast normalization | low-light and shadowed footage |

The higher-confidence pass wins, its fragments are merged, then clustered into lines by
vertical centroid and ordered strictly left-to-right. This fixes reversed reads such as
`4787 LE` → `LE 4787`, which is essential for multi-line and multi-fragment plates.

### Stage 4 — Post-processing

Characters outside the allowlist are dropped and a small correction table is applied
(currently `PUMJAB` → `PUNJAB`), yielding a clean plate string for downstream use.

---

## Quick Start

### Requirements

- Python **3.10 – 3.12**
- ~4 GB disk for the environment (torch + easyocr + models), plus dataset space if training
- Optional but strongly recommended: an NVIDIA GPU with CUDA. The pipeline works on CPU,
  just slower. Verified here on an RTX 3060 (12 GB) at ~200 ms/image end-to-end.
- **Git LFS** — required to receive the model weights on clone

### 1. Clone with LFS

```bash
git lfs install          # once per machine
git clone https://github.com/abdullahsaleem1/vehicle_system_cv.git
cd vehicle_system_cv
```

> Without `git lfs install`, the `.pt` weight files download as small pointer text files
> and the app will fail to load the detector.

### 2. Create and activate a virtual environment

```bash
# Windows (PowerShell)
python -m venv yolo_env
.\yolo_env\Scripts\activate

# Linux / macOS
python3 -m venv yolo_env
source yolo_env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Install CUDA-enabled PyTorch if you have an NVIDIA GPU (the CPU wheel is the default on
some platforms and will silently run slowly):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 4. Run the Web Studio

```bash
# One-click on Windows
run_webapp.bat

# Or explicitly
python -m uvicorn web_app.app:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**.

On first request the app pre-warms the YOLO detector and the EasyOCR reader (a few seconds);
subsequent requests are fast. Check readiness with:

```bash
curl http://127.0.0.1:8000/api/health
```

---

## Web Studio API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Interactive HTML studio (upload, samples, sliders, crop cards) |
| `GET` | `/api/health` | Liveness + diagnostics: weights path/size, parameter count, classes, CUDA status, device name, EasyOCR status |
| `GET` | `/api/samples` | List built-in sample images (validation split + real vehicles) |
| `GET` | `/api/sample-image/{source}/{name}` | Fetch a sample image. `source` is `val` or `cars` |
| `POST` | `/api/predict` | Run the full detect → read pipeline on an upload or a sample |

### `POST /api/predict`

Accepts `multipart/form-data`. Provide **either** an uploaded `file` **or** a
`sample_source` + `sample_name` pair.

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | file | – | Uploaded image |
| `sample_source` | string | – | `val` or `cars` |
| `sample_name` | string | – | Filename from `/api/samples` |
| `conf_threshold` | float | `0.25` | Detector confidence threshold |
| `enable_aspect_filter` | bool | `true` | Skip OCR on suspect-aspect boxes |
| `min_aspect` | float | `1.20` | Suspect threshold for `w / h` |

Response (abridged):

```json
{
  "filename": "DSC_0968.JPG",
  "image_size": [h, w],
  "num_detections": 1,
  "num_plates_read": 1,
  "timing": { "det_ms": 39.7, "ocr_ms": 156.7, "total_ms": 206.0 },
  "detections": [
    {
      "box": [x1, y1, x2, y2],
      "det_conf": 0.827,
      "aspect": 1.37,
      "is_suspect_aspect": false,
      "text": "LEA 1856",
      "clean_text": "LEA1856",
      "ocr_conf": 0.725,
      "fragments": ["..."]
    }
  ],
  "original_image": "data:image/jpeg;base64,...",
  "annotated_image": "data:image/jpeg;base64,..."
}
```

Each detection also carries a `crop_base64` field for the individual plate crop, which is
what drives the crop inspection cards in the UI.

---

## Offline Batch Inference

Run the pipeline over an entire folder and get annotated images plus a structured JSON:

```bash
python ocr_integration/run_anpr_pipeline.py \
  --input "Pakistani License Number Plates Data/Cars" \
  --out "ocr_integration/outputs"
```

Outputs are written to the `--out` directory: `anpr_results.json` plus `*_annotated.jpg`
visualizations. (Batch outputs are gitignored — they are regenerated on each run.)

---

## Project Structure

```
vehicle_system_cv/
├── web_app/
│   └── app.py                     # FastAPI app: UI, health, samples, predict endpoints
├── ocr_integration/
│   ├── run_anpr_pipeline.py       # Offline batch detect-then-read pipeline
│   ├── evaluate_ocr.py            # OCR metrics (refuses to fabricate numbers without GT)
│   ├── diagnostic_weights_health.py
│   ├── logs_diag.py
│   └── ocr_notes.md               # Detailed OCR findings, failure cases, decisions
├── plate_detection/
│   ├── download_dataset.py        # Re-download the Roboflow base dataset
│   ├── train.py                   # Base detector training
│   ├── check_progress.py
│   ├── mock_test_model_output.py
│   ├── pakistani_finetune/
│   │   ├── prepare_pk_dataset.py  # Build the PK fine-tuning split
│   │   ├── finetune.py            # Fine-tune v1
│   │   ├── finetune_v2.py         # Fine-tune v2 (current production detector)
│   │   ├── evaluate_comparison.py # Compare base vs fine-tuned
│   │   ├── dataset.yaml
│   │   ├── finetune_summary.md    # v1/v2 results
│   │   └── weights/               # → weights live here (see Model Weights)
│   ├── weights/                   # Base detector checkpoints
│   └── datasets/                  # (NOT in git) Roboflow dataset
├── attribute_recognition/
│   ├── prepare_color_dataset.py   # Vehicle colour attribute dataset prep
│   └── prepare_type_dataset.py    # Vehicle type attribute dataset prep
├── pakistan_license_dataset_YOLO/ # PK plate dataset in YOLO format (tracked)
├── Pakistani License Number Plates Data/  # Real-world test vehicles & plates (tracked)
├── report_assets/                 # Figures and case-study images for the reports
├── generate_pdf_report.py
├── ANPR_SYSTEM_CLIENT_REPORT.md   # Technical report (MD)
├── ANPR_SYSTEM_CLIENT_REPORT.pdf  # Technical report (PDF)
├── ANPR_CLIENT_REPORT.html        # Client-facing report
├── ANPR_SUPERVISOR_REPORT.html    # Supervisor-facing report
├── FYP Phase_1.pdf                # Phase 1 project document
├── requirements.txt
├── run_webapp.bat                 # One-click Windows launcher
├── .gitattributes                 # Git LFS rules for *.pt / *.pth
└── .gitignore
```

---

## Model Weights

All tracked weights live under `plate_detection/` and are stored via **Git LFS**.

| Weight | Path | Role |
|---|---|---|
| **v2 detector (production)** | `plate_detection/pakistani_finetune/weights/pk_plate_finetune_v2/weights/best.pt` | **Loaded by `web_app/app.py`** |
| v1 detector | `plate_detection/pakistani_finetune/weights/pk_plate_finetune/weights/best.pt` | Earlier fine-tune, kept for comparison |
| Base detector | `plate_detection/weights/plate_detector/weights/best.pt` | Pre-fine-tuning baseline |
| Pretrained baselines | `yolov8n.pt`, `yolo26n.pt` | Stock YOLO weights used for training/eval |

> **To switch the production model**, change `DETECTOR_PATH` at the top of `web_app/app.py`.

### What is deliberately not stored

Raw `epoch*.pt` checkpoints (~16–23 MB each, ~1.1 GB total) and `last.pt` are gitignored.
Several exceed GitHub's 100 MB per-file limit and all are reproducible from training.

### Reported metrics (fine-tuned v2, as documented in the client report)

- Plate detection recall: **100.0%**
- Detection fitness (mAP@50): **0.9769**
- Box alignment (mAP@50-95): **0.6330**
- OCR confidence (example `LEF 3503`): **96.1%**
- End-to-end latency: **~183 ms** on RTX 3060

> These figures come from the project report and small evaluation sets. They have **not**
> been independently re-verified here, and the OCR evaluation lacks ground-truth labels —
> see [Known Limitations](#known-limitations). Treat them as indicative, not certified.

---

## Repository Layout & What Is Not Stored in Git

The working directory contains several large artifacts that are intentionally excluded so
the repository stays cloneable and pushable. The table below explains each.

| Path | Size | In git? | Why / how to restore |
|---|---|---|---|
| `plate_detection/datasets/license_plate_recognition_v4/` | ~946 MB (48k files) | ❌ | Roboflow base dataset. Restore with `download_dataset.py` |
| `plate_detection/**/weights/*/weights/epoch*.pt` | ~1.1 GB | ❌ | Per-epoch checkpoints; reproducible from training |
| `plate_detection/**/weights/*/weights/last.pt` | ~18 MB | ❌ | Recreated by Ultralytics during training |
| `yolo_env/` | multi-GB | ❌ | Virtual environment; recreate with `requirements.txt` |
| `**/__pycache__/` | small | ❌ | Python bytecode cache |
| `runs/` | varies | ❌ | Ultralytics run artifacts |
| `ocr_integration/outputs*/` | ~8 MB | ❌ | Batch inference output; regenerate by running the pipeline |
| `*.cache` | ~6 MB | ❌ | YOLO label caches; rebuilt on first training run |
| `*.log`, `*_pid.txt` | small | ❌ | Training logs and PIDs |
| `.freebuff/` | – | ❌ | Local tooling directory |

Everything else — source code, docs, reports, the real-world test images, the YOLO-format
PK dataset, and the production `best.pt` weights — **is** tracked.

### Why Git LFS

Weights are binary and change rarely. LFS stores them out-of-band and keeps a small pointer
in git history, so a future weight update doesn't permanently bloat the repository.

```bash
git lfs ls-files        # list LFS-tracked files
git lfs status          # check pending LFS objects
```

If a cloned `.pt` looks like a ~130-byte text file, LFS was not installed when you cloned.
Fix it with:

```bash
git lfs install
git lfs pull
```

---

## Datasets

### 1. Roboflow base dataset (not in git)

- **Project:** `roboflow-universe-projects/license-plate-recognition-rxg4e`, version 4
- **License:** CC BY 4.0
- **Format:** YOLOv8, 1 class (`License_Plate`)
- **Size:** ~48,481 files / ~946 MB (train 21,173 · valid 2,046 · test 1,019 images)
- **Splits:** `data.yaml` maps `train: ../train/images`, `val: ../valid/images`, `test: ../test/images`

Re-download:

```bash
# PowerShell
$env:ROBOFLOW_API_KEY = "your_api_key_here"
python plate_detection/download_dataset.py
```

Get a free key at https://app.roboflow.com/settings/api. The script skips the download if
the dataset is already present and cleans up incomplete downloads.

### 2. Pakistani license plate dataset (`pakistan_license_dataset_YOLO/`)

Tracked in the repo (~9.8 MB). YOLO-format images and labels for Pakistani plates,
plus the fine-tuning split under `plate_detection/pakistani_finetune/data/`
(76 train/val images, 78 label files).

### 3. Real-world test vehicles (`Pakistani License Number Plates Data/`)

Tracked (~48 MB, 108 files). Genuine road photos (`Cars/DSC_*.JPG`) used as the
"unseen real vehicles" samples in the web UI and for qualitative testing.

> **Ground-truth gap:** none of these datasets ship per-plate transcription strings.
> `ocr_integration/evaluate_ocr.py` therefore refuses to print accuracy metrics until
> `ground_truth.csv` is filled in by hand. See `ocr_integration/ocr_notes.md`.

---

## Training & Fine-Tuning

```bash
# 1. Base detector training
python plate_detection/train.py

# 2. Build the Pakistani fine-tuning split
python plate_detection/pakistani_finetune/prepare_pk_dataset.py

# 3. Fine-tune (v2 is the current production detector)
python plate_detection/pakistani_finetune/finetune.py
python plate_detection/pakistani_finetune/finetune_v2.py

# 4. Compare base vs fine-tuned
python plate_detection/pakistani_finetune/evaluate_comparison.py
```

Fine-tuning writes to `plate_detection/pakistani_finetune/weights/<run>/weights/`.
Copy the winning `best.pt` into a new run folder and point `DETECTOR_PATH` at it.
See `plate_detection/pakistani_finetune/finetune_summary.md` for the v1/v2 comparison.

### OCR evaluation

```bash
python ocr_integration/evaluate_ocr.py
```

This intentionally reports "NO GROUND TRUTH AVAILABLE — MANUAL VERIFICATION NEEDED" until
plate strings are transcribed. Fill `ground_truth.csv` (mark `verified=yes` only after
checking the annotated crops in `ocr_integration/outputs/`), then metrics compute
automatically. This is a deliberate anti-fabrication guard.

---

## Configuration Reference

| Setting | Location | Default | Effect |
|---|---|---|---|
| `DETECTOR_PATH` | `web_app/app.py` | v2 `best.pt` | Which detector weights to serve |
| `conf_threshold` | request / UI | `0.25` | Lower = more detections, more false positives |
| `enable_aspect_filter` | request / UI | `true` | Skip OCR on suspect-aspect boxes |
| `min_aspect` | request / UI | `1.20` | Suspect threshold for `w / h` |
| `PLATE_CHARSET` | `web_app/app.py` | `A-Z 0-9 -` | OCR allowlist and output filter |
| `PAD_FRACTION` | `web_app/app.py` | `0.08` | Crop padding around each detected box |
| `TARGET_OCR_HEIGHT` | `web_app/app.py` | `96` | Minimum upscaled height for OCR |
| `COMMON_CORRECTIONS` | `web_app/app.py` | `PUMJAB→PUNJAB` | Post-OCR text corrections |
| `imgsz` | `web_app/app.py` | `640` | Detector inference resolution |

Tuning intuition: if real plates are being missed, lower `conf_threshold`. If you see
garbage text, raise `min_aspect` (e.g. `1.35`). Turn the aspect filter off only for
diagnostics — it exists to suppress false positives.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Detector weights not found at ...` | LFS not pulled, or `DETECTOR_PATH` wrong | `git lfs install && git lfs pull`; verify the path exists |
| `.pt` file is ~130 bytes of text | Cloned without LFS | `git lfs install && git lfs pull` |
| `Address already in use` on port 8000 | Another process holds the port | `netstat -ano \| findstr :8000`, kill the PID, or pass `--port 8001` |
| First request is slow | YOLO + EasyOCR cold start | Expected once; models are cached after startup |
| Inference very slow, CPU listed in `/api/health` | CPU-only torch wheel installed | Reinstall CUDA torch (see Quick Start step 3) |
| Garbage plate text | Square-ish false-positive crop | Raise `min_aspect`, keep the aspect filter on |
| `ROBOFLOW_API_KEY` error | Env var not set | Set it before running `download_dataset.py` |
| EasyOCR tries to download models | First run only | Needs internet once; cached under `~/.EasyOCR` afterwards |

Quick health check:

```bash
curl -s http://127.0.0.1:8000/api/health
```

---

## Known Limitations

Documented honestly rather than papered over:

1. **No OCR ground truth.** Accuracy figures cannot be computed until plates are manually
   transcribed into `ground_truth.csv`. The reported 96.1% OCR confidence is a per-plate
   engine confidence on one example, not an accuracy metric.
2. **Small evaluation sample.** OCR behaviour was characterized on ~8 images / 12 detections —
   anecdotal rather than statistical.
3. **Failures track crop geometry, not the OCR engine.** Tiny crops (after upscaling) and
   near-square boxes dominate errors. Switching from EasyOCR to PaddleOCR would not fix
   detection-quality problems.
4. **No video / stream path.** The pipeline is single-image and folder-batch only; the
   `attribute_recognition/` scripts prepare colour and type datasets but those models are not
   part of the serving path yet.
5. **Adverse conditions untested.** Glare, damage, and motion blur were not in the sample;
   angle and distance already dominate errors and will worsen in moving footage.
6. **`report_assets/` figures** are report illustrations, not an automated evaluation suite.

---

## Reports & Documentation

| File | Contents |
|---|---|
| `ANPR_SYSTEM_CLIENT_REPORT.md` / `.pdf` | Detailed technical report |
| `ANPR_CLIENT_REPORT.html` | Client-facing summary |
| `ANPR_SUPERVISOR_REPORT.html` | Supervisor-facing summary |
| `FYP Phase_1.pdf` | Phase 1 project document |
| `ocr_integration/ocr_notes.md` | OCR integration findings, failure cases, engine decision |
| `plate_detection/pakistani_finetune/finetune_summary.md` | Fine-tuning v1 vs v2 results |

---

## License & Attribution

- Roboflow base dataset: **CC BY 4.0** (`license-plate-recognition-rxg4e` v4)
- Pretrained YOLO weights: subject to the Ultralytics license terms
- EasyOCR: Apache 2.0

Add a project license file if this code is to be redistributed.
