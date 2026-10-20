"""Fine-tune the trained base plate detector on Pakistani plates.

Loads ../weights/plate_detector/weights/best.pt as the starting checkpoint,
fine-tunes at lr0=0.001 for up to 30 epochs (patience=10), with the same
per-epoch checkpointing, timestamped logging, and auto-resume behavior as
the base training stage.

Run: yolo_env\\Scripts\\python.exe finetune.py
"""

import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import torch

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "dataset.yaml"
WEIGHTS_DIR = BASE_DIR / "weights"
RUN_NAME = "pk_plate_finetune"
RUN_DIR = WEIGHTS_DIR / RUN_NAME
RUN_WDIR = RUN_DIR / "weights"
LOG_FILE = BASE_DIR / "finetune_log.txt"

BASE_CKPT = BASE_DIR.parent / "weights" / "plate_detector" / "weights" / "best.pt"

EPOCHS = 30
PATIENCE = 10
LR0 = 0.001

RUN_START = time.time()
EPOCH_LINE_RE = re.compile(r"^epoch(\d+)$")


def now_stamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    line = f"[{now_stamp()}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def separator(kind):
    line = f"=== {kind} at {now_stamp()} ==="
    print("\n" + line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def fmt_elapsed(seconds):
    return str(timedelta(seconds=int(seconds)))


def fmt_val(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"


def read_ckpt_meta(path):
    """Return (last_done_epoch_zero_based, total_epochs, stored_data_path) or None if unreadable."""
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        epoch = int(ckpt.get("epoch", -1))
        args = ckpt.get("train_args") or {}
        total = int(args.get("epochs", EPOCHS))
        stored_data = args.get("data")
        return epoch, total, str(stored_data) if stored_data else None
    except Exception as e:
        log(f"WARNING: checkpoint {path} could not be loaded ({type(e).__name__}: {e})")
        return None


def epoch_ckpts_newest_first():
    def key(p):
        m = EPOCH_LINE_RE.match(p.stem.lower())
        return int(m.group(1)) if m else -1

    return sorted(RUN_WDIR.glob("epoch*.pt"), key=key, reverse=True)


def pick_resume_candidate():
    candidates = []
    if LAST_PT.exists():
        candidates.append(LAST_PT)
    candidates.extend(p for p in epoch_ckpts_newest_first() if p != LAST_PT)

    for cand in candidates:
        meta = read_ckpt_meta(cand)
        if meta is None:
            log(f"Skipping unreadable checkpoint {cand.name} (possible partial write during a power cut).")
            continue
        if cand != LAST_PT:
            log(f"FALLBACK: using periodic checkpoint {cand.name} because newer checkpoints are unreadable.")
        return cand, meta

    if LAST_PT.exists() or epoch_ckpts_newest_first():
        log(
            "FATAL: checkpoint files exist in "
            f"{RUN_WDIR} but none are loadable. To force a fresh start, rename or delete that folder, then re-run."
        )
        sys.exit(1)
    return None, None


def extract_losses(trainer):
    tloss = getattr(trainer, "tloss", None)
    if isinstance(tloss, dict):
        vals = [float(v) for v in tloss.values()]
    elif tloss is not None:
        try:
            vals = [float(v) for v in tloss]
        except TypeError:
            return None, None, None
    else:
        return None, None, None
    box = vals[0] if len(vals) > 0 else None
    cls = vals[1] if len(vals) > 1 else None
    dfl = vals[2] if len(vals) > 2 else None
    return box, cls, dfl


def on_fit_epoch_end(trainer):
    done = int(trainer.epoch) + 1
    total = int(trainer.epochs)
    if done > total:
        return
    box_loss, cls_loss, dfl_loss = extract_losses(trainer)
    map50 = (getattr(trainer, "metrics", None) or {}).get("metrics/mAP50(B)")
    pct = min(100, done * 100 // max(total, 1))
    elapsed = fmt_elapsed(time.time() - RUN_START)
    log(
        f"Epoch {done}/{total} complete ({pct}%) | "
        f"box_loss={fmt_val(box_loss)} cls_loss={fmt_val(cls_loss)} dfl_loss={fmt_val(dfl_loss)} | "
        f"mAP50={fmt_val(map50)} | elapsed this run {elapsed}"
    )
    saved = ["last.pt"]
    newest = epoch_ckpts_newest_first()
    if newest:
        saved.append(newest[0].name)
    log(f"Checkpoint saved: {' + '.join(saved)}")


LAST_PT = RUN_WDIR / "last.pt"
BEST_PT = RUN_WDIR / "best.pt"


def main():
    if not BASE_CKPT.exists():
        print(f"FATAL: base checkpoint not found: {BASE_CKPT}")
        print("       The base training stage must be completed first.")
        sys.exit(1)

    if not DATA_YAML.exists():
        print(f"FATAL: dataset config not found: {DATA_YAML}")
        print("       Run prepare_pk_dataset.py first.")
        sys.exit(1)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    resume_path, meta = pick_resume_candidate()

    if resume_path is not None:
        done_zero_based, ckpt_total, stored_data = meta
        done_human = done_zero_based + 1
        if done_human >= ckpt_total:
            separator(f"TRAINING ALREADY COMPLETE ({done_human}/{ckpt_total} epochs)")
            log(f"Nothing to resume. Final weights: {BEST_PT if BEST_PT.exists() else RUN_WDIR}")
            sys.exit(0)

        if stored_data and not Path(stored_data).exists():
            log(
                f"ERROR: checkpoint expects dataset at '{stored_data}' but it does not exist. "
                "Re-run prepare_pk_dataset.py before resuming."
            )
            sys.exit(1)

        separator(f"FINETUNE RESUMED (from {resume_path.name}, {done_human}/{ckpt_total} epochs already done)")
        log(f"Dataset config : {stored_data or DATA_YAML}")
        log(f"Run folder     : {RUN_DIR} (reused, no auto-increment)")
        model = YOLO(str(resume_path))
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        results = model.train(resume=str(resume_path))
    else:
        separator("FINETUNE STARTED (fresh)")
        log(f"Base checkpoint: {BASE_CKPT}")
        log(f"Dataset config : {DATA_YAML}")
        log(f"Run folder     : {RUN_DIR} (reused, no auto-increment)")
        log(
            f"Settings       : epochs={EPOCHS} patience={PATIENCE} lr0={LR0} imgsz=640 "
            "device=0 batch=-1(AutoBatch) cache=True save_period=1"
        )
        log("FINAL TRAINING CONFIG (all parameters):")
        log(f"  task/model    : detect, fine-tune from {BASE_CKPT}")
        log(f"  data          : {DATA_YAML}")
        log(f"  epochs        : {EPOCHS}")
        log(f"  patience      : {PATIENCE} (early stopping)")
        log(f"  lr0           : {LR0}")
        log(f"  imgsz         : 640")
        log(f"  batch         : -1 (AutoBatch)")
        log(f"  device        : 0")
        log(f"  cache         : True")
        log(f"  save_period   : 1 (per-epoch checkpoints)")
        log(f"  freeze        : 10 (backbone frozen - overfit safeguard for small dataset)")
        log(f"  mosaic        : 1.0")
        log(f"  mixup         : 0.15")
        log(f"  degrees       : 10")
        log(f"  translate     : 0.1")
        log(f"  scale         : 0.3")
        log(f"  shear         : 2.0")
        log(f"  fliplr        : 0.5 (plate localization only; mirror is geometrically valid)")
        log(f"  flipud        : 0.0 (off - plates are never upside-down in footage)")
        log(f"  hsv_h         : 0.015")
        log(f"  hsv_s         : 0.5")
        log(f"  hsv_v         : 0.3")
        model = YOLO(str(BASE_CKPT))
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        results = model.train(
            data=str(DATA_YAML),
            epochs=EPOCHS,
            patience=PATIENCE,
            lr0=LR0,
            imgsz=640,
            device=0,
            batch=-1,
            cache=True,
            save_period=1,
            project=str(WEIGHTS_DIR),
            name=RUN_NAME,
            exist_ok=True,
            freeze=10,
            mosaic=1.0,
            mixup=0.15,
            degrees=10,
            translate=0.1,
            scale=0.3,
            shear=2.0,
            fliplr=0.5,
            flipud=0.0,
            hsv_h=0.015,
            hsv_s=0.5,
            hsv_v=0.3,
        )

    separator("FINETUNE FINISHED")
    best = Path(getattr(results, "save_dir", RUN_DIR)) / "weights" / "best.pt"
    log(f"Best fine-tuned weights saved to: {best}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("INTERRUPTED by user (Ctrl+C). Progress through the last completed epoch is safe; re-run to resume.")
        separator("RUN INTERRUPTED")
        sys.exit(130)
    except Exception as e:
        import traceback

        log(f"CRASHED: {type(e).__name__}: {e}")
        for line in traceback.format_exc().strip().splitlines()[-6:]:
            log(f"  {line}")
        log("Progress through the last completed epoch is safe on disk; fix the issue and re-run to resume.")
        separator("RUN CRASHED")
        raise
