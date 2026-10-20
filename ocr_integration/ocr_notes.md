# OCR Integration Notes (FR-16)

Stage: detect-then-read ANPR. Detector: fine-tuned v2 (`pakistani_finetune/weights/
pk_plate_finetune_v2/weights/best.pt`). OCR: EasyOCR 1.7.2, English-only, GPU.
Code: `run_anpr_pipeline.py` (pipeline), `evaluate_ocr.py` (metrics), `ground_truth.csv` (manual GT).

## Ground truth status (important)

**No ground-truth plate strings exist in any project dataset** (checked filenames of all
three datasets — `vehicle_N.jpg`, hash names, `DSC_XXXX.JPG` — plus all label files and any
CSV/JSON metadata). Per spec this is flagged, not silently skipped:

- `evaluate_ocr.py` refuses to print metrics until GT exists ("NO GROUND TRUTH AVAILABLE -
  MANUAL VERIFICATION NEEDED") — no fabricated numbers.
- `ground_truth.csv` is scaffolded; filling it requires visually transcribing ~8–12 val
  images from `ocr_integration/outputs/*_annotated.jpg`. Mark rows `verified=yes` only
  after eyeballing; metrics then compute automatically.

## What worked

- EasyOCR installed cleanly alongside torch 2.6/CUDA 12.4 (no framework conflicts) and runs
  on GPU; ~0.3 s/image end-to-end for the whole pipeline at batch=1.
- Smoke test on 3 real PK plate close-ups read genuine Punjab district codes: `LEC 4800`,
  `LEA 16 4861 A`, `LEF .i2 1981` — engine clearly suitable for Latin-alphanumeric PK plates;
  English-only mode confirmed sufficient.
- Clean plate-aspect crops read very well: `LE 7592` @ conf 1.00, `RLD 2654` @ 0.89,
  `JGL 2195 Q4` @ 0.79.
- Preprocessing that measurably helped: small-crop upscaling to ≥96 px height + CLAHE
  contrast normalization + charset allowlist (`A-Z0-9-`). Left-to-right fragment ordering
  fixes reversed reads like `4787 LE` → `LE ... 4787`.

## What didn't / failure cases observed

From the 8-image sample run (12 detections, 6 readable):

| Image | Crop | Outcome | Diagnosis |
|---|---|---|---|
| vehicle_20 (2nd box) | 39x30 px | nothing read | crop too small even after upscale |
| vehicle_23 boxes 1&3, vehicle_35 | ~85-125 px, aspect ≈1:1 | garbage fragments (`RWO2D`, `KCT-SLALAE`) | near-square boxes = detector false positives or extreme-angle plates, not readable plates |
| vehicle_17 (2nd box) | 105x76 px | nothing usable | angle/distance |
| vehicle_4 | full-frame close-up | `PUMJAB LWC 4148` | state word misread (`PUNJAB`) but plate code correct |

Key insight: **failures correlate with crop geometry, not OCR-engine weakness** — wide
plate-aspect crops (≥~1.5:1) read reliably; square-ish crops are mostly detector FPs or
unusable angles. Pipeline now flags such boxes via `suspect_non_plate_aspect`
(aspect < 1.2) so downstream logic can skip them instead of trusting garbage text.
Adverse-condition tie-in (FR-16/NFR): glare/damage didn't appear in this tiny sample, but
angle and distance already dominate errors — expect those to worsen in motion footage.

## EasyOCR sufficient? PaddleOCR comparison?

EasyOCR is **sufficient for this stage**: real PK formats read correctly with high confidence,
installation was friction-free, and current failures trace to detection quality/crop geometry
that switching engines would not fix. A PaddleOCR head-to-head is only warranted if, *after*
(a) manual GT evaluation and (b) FP-crop filtering, char accuracy still lags the FYP target —
revisit then; not now.

## Known gaps

1. Manual verification set not yet transcribed (needs human eyes; template ready).
2. Sample size (8 images / 12 detections) is anecdotal; run the full 12-image val split +
   more once GT rows exist.
3. No video/stream path yet — current pipeline is single-image/folder based.
