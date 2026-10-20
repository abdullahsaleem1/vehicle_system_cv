# Pakistani Fine-Tune Summary

Auto-generated after completion of the fine-tuning stage.
Run folder: `plate_detection/pakistani_finetune/weights/pk_plate_finetune/`

## 1. Dataset

| Item | Value |
|---|---|
| Primary source | `pakistan_license_dataset_YOLO/` (76 img + 76 YOLO labels, all stems matched) |
| Conditional source | `Pakistani License Number Plates Data/` — **REJECTED** |
| Merge | Primary only → **64 train / 12 val** (85/15, seed=20260822) |
| Skipped/corrupt (primary) | 0 |
| Classes | 1 (`License_Plate`), `nc:1` |

### Conditional dataset rejection (quality gates)
108 JPGs in `Cars\`(104) + `Plates\`(4). Gate A passed on image integrity (108/108 readable),
but **Gate B FAILED: zero annotation files of any format exist in the tree (0/108 images labeled)**
→ rejected entirely per spec; no boxes were guessed; Gates C/D skipped as moot.
Internal near-duplicates also observed (`DSC_0971.JPG` / `DSC_0971x.jpg` pattern).

## 2. Training

| Item | Value |
|---|---|
| Start checkpoint | `plate_detection/weights/plate_detector/weights/best.pt` (base model untouched) |
| Config | epochs=30 · patience=10 · lr0=0.001 (requested) · imgsz=640 · batch=-1 · device=0 · cache=True |
| Overfit safeguards | `freeze=10` (backbone frozen), mosaic=1.0, mixup=0.15, degrees=10, translate=0.1, scale=0.3, shear=2.0, fliplr=0.5, flipud=0.0, hsv_h=0.015, hsv_s=0.5, hsv_v=0.3 |
| Duration | **18 epochs in ~56 s wall-clock** (early stop; ~1 s/epoch: 64 cached imgs + frozen backbone) |
| Early stopping | Triggered at epoch 18/30 — no fitness improvement in last 10 epochs; best epoch = 8 |
| Checkpoints | `last.pt`, `best.pt`, `epoch0.pt … epoch17.pt` all present and loadable |
| Logs | `finetune_log.txt` (timestamped per-epoch + checkpoint saves), `check_progress.py` operational |

## 3. Before / After (same held-out Pakistani val split: 12 images, 12 instances)

| Metric | Base `best.pt` | Fine-tuned `best.pt` | Δ |
|---|---|---|---|
| Precision | 0.9974 | 0.9165 | −0.0808 |
| Recall | 1.0000 | 0.9150 | −0.0850 |
| mAP50 | 0.9950 | 0.9486 | −0.0464 |
| **mAP50-95** | 0.4395 | **0.6531** | **+0.2136** |

Inference speed (val): base 7.3 ms/img vs fine-tuned 2.6 ms/img on RTX 3060.

### Interpretation
- The base model already localized PK plates almost perfectly at IoU=0.5 (mAP50 0.995).
- Fine-tuning delivered its intended effect on **box tightness**: mAP50-95 +21.4 points
  (0.44 → 0.65), i.e., tighter plate boxes → better crops for the OCR stage.
- P/R/mAP50 dipped slightly. With only 12 val instances each metric step is ±8%, so this is
  within small-sample noise; treat the comparison as directional, not definitive.

## 4. Flagged issues

1. **`optimizer=auto` overrode lr0 in BOTH stages.** Ultralytics ignored `lr0=0.001`
   (and the base stage's `lr0=0.01`) and selected AdamW(lr=0.002) for both runs. Effective
   fine-tune LR was therefore 2× the requested value — still stable, but if strict LR control
   is wanted re-run with explicit `optimizer='AdamW'`.
2. **Conditional PK dataset unusable as-is** (no annotations). If it becomes important,
   it needs manual labeling (~104 images) or replacement by DVLPD (see below).
3. **Tiny validation split (12 images)** makes metric deltas noisy; recommend expanding
   labeled PK data before drawing strong conclusions.
4. **DVLPD note (literature review):** highest-quality Pakistani dataset found (10k images,
   6 vehicle types, 98.5% mAP benchmark) is not publicly downloadable — YOLOv4 weights only
   on GitHub; requires contacting authors. Cited, not used.

## 5. Output contract

`mock_test_finetuned_output.py`: **17/17 PASS** (post-training; weights verified present,
base weights untouched). No live inference was needed for the contract checks.

---

# V2 RE-RUN (report-cited version)

## Why
v1's `optimizer=auto` silently overrode `lr0=0.001` → AdamW(lr=0.002), making the reported
hyperparameters inaccurate. v2 re-runs the identical experiment with `optimizer='AdamW'`
pinned explicitly so logged config == effective config. v1 artifacts were not modified.

## Confirmed hyperparameters (read back, not assumed)
- From Ultralytics runtime output: `optimizer: AdamW(lr=0.001, momentum=0.937)` — **no silent override; lr0=0.001 honored**
- From `pk_plate_finetune_v2/args.yaml`: `optimizer: AdamW`, `lr0: 0.001`, `lrf: 0.01`,
  `freeze: 10`, `mosaic: 1.0`, `mixup: 0.15`, `epochs: 30`, `patience: 10`
- Everything else identical to v1 (same augmentation set, imgsz=640, batch=-1, device=0,
  cache=True, same base checkpoint)
- Run: early stop at epoch 21/30, best epoch = 11, ~1.9 min wall-clock,
  weights in `weights/pk_plate_finetune_v2/weights/best.pt`

## Three-way comparison (same held-out val split: 12 images / 12 instances)

| Metric | Base | Fine-tune v1 (eff. LR 0.002) | Fine-tune v2 (LR 0.001) | v2 − v1 |
|---|---|---|---|---|
| Precision | 0.9974 | 0.9165 | 0.8431 | −0.0734 |
| Recall | 1.0000 | 0.9150 | 1.0000 | +0.0850 |
| mAP50 | 0.9950 | 0.9486 | 0.9769 | +0.0283 |
| mAP50-95 | 0.4395 | **0.6531** | 0.6330 | −0.0201 |

## Honest equivalence note
v2 and v1 are **essentially equivalent**. Every v2-vs-v1 delta (±0.02–0.09) is within the
noise floor of a 12-instance validation split, where one box shifting an IoU band moves
mAP50-95 by several points. The halved effective LR (0.002 → 0.001) did not produce a
meaningful performance difference at this dataset size — as expected. Both fine-tuned models
deliver the same substantive improvement over base: ~+20 pts mAP50-95 (tighter boxes for OCR).
v2 should be cited in the FYP report because its *stated* hyperparameters are verifiably exact;
if a single production checkpoint must be chosen, either is defensible (v1 has the marginally
best mAP50-95, v2 the best recall/mAP50 combination among fine-tunes).

## Cite-this-run block (FYP report)
> Fine-tuning: YOLO26n plate detector initialized from base `best.pt`;
> optimizer=AdamW, lr0=0.001, momentum=0.937, lrf=0.01; epochs=30 w/ early stopping
> (patience=10, stopped at 21); freeze=10; imgsz=640; batch=auto; augmentations:
> mosaic=1.0, mixup=0.15, degrees=10, translate=0.1, scale=0.3, shear=2.0,
> fliplr=0.5, hsv_h=0.015, hsv_s=0.5, hsv_v=0.3; dataset: 64 train / 12 val Pakistani plates.
