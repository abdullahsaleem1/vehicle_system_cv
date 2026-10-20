"""Prepare the vehicle COLOR classification dataset (VCoR).

Downloads landrykezebou/vcor-vehicle-color-recognition-dataset via the kaggle CLI
(or reuses an existing download), verifies class balance, and stages an
ImageFolder-style layout under attribute_recognition/data/color/.
VCoR ships with official train/val/test splits -> reused as-is (no re-split),
test kept aside for final evaluation.

Run: yolo_env\\Scripts\\python.exe prepare_color_dataset.py
"""

import json
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DOWNLOADS = BASE_DIR / "downloads"
DATA_DIR = BASE_DIR / "data" / "color"
KAGGLE_HANDLE = "landrykezebou/vcor-vehicle-color-recognition-dataset"

EXPECTED_CLASSES_15 = {
    "beige", "black", "blue", "brown", "gold", "green", "grey",
    "orange", "pink", "purple", "red", "silver", "tan", "white", "yellow",
}


def kaggle_available():
    try:
        import kaggle  # noqa
        return True
    except Exception:
        pass
    py = Path(sys.executable)
    cli = py.parent / "kaggle.exe"
    return cli.exists() or (py.parent / "kaggle").exists()


def download():
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    marker = DOWNLOADS / "vcor_downloaded.flag"
    if marker.exists() and any(DOWNLOADS.rglob("*.jpg")):
        print("VCoR download already present - skipping fetch.")
        return
    print(f"Downloading {KAGGLE_HANDLE} ...")
    cmd = [sys.executable, "-m", "kaggle", "datasets", "download", "-d", KAGGLE_HANDLE, "-p", str(DOWNLOADS), "--unzip"]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("FATAL: kaggle download failed (check ~/.kaggle/kaggle.json and internet).")
        sys.exit(1)
    marker.write_text("ok")


def locate_split_root():
    zips = list(DOWNLOADS.glob("*.zip"))
    if zips:
        import zipfile

        for z in zips:
            out = DOWNLOADS / z.stem
            if not out.exists():
                print(f"Extracting {z.name} ...")
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(out)
    candidates = []
    for p in DOWNLOADS.rglob("*"):
        if p.is_dir() and (p / "train").is_dir():
            candidates.append(p)
        if p.is_dir() and p.parent.name.lower() == "train":
            continue
    if any((DOWNLOADS / name).is_dir() for name in ("train", "val", "test")):
        return DOWNLOADS
    if candidates:
        return candidates[0]
    print(f"FATAL: could not locate VCoR train/val folders under {DOWNLOADS}")
    sys.exit(1)


def count_images(root: Path):
    counts = {}
    for cls_dir in sorted(root.iterdir()):
        if cls_dir.is_dir():
            n = sum(1 for f in cls_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
            if n:
                counts[cls_dir.name] = n
    return counts


def main():
    print("=" * 70)
    print("COLOR DATASET PREPARATION (VCoR)")
    print("=" * 70)
    if not kaggle_available():
        print("FATAL: kaggle CLI not installed in yolo_env.")
        sys.exit(1)

    download()
    root = locate_split_root()
    print(f"\nSplit root: {root}")

    report = {}
    ok = True
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        sdir = root / split
        if not sdir.is_dir():
            print(f"FATAL: expected split folder missing: {sdir}")
            sys.exit(1)
        counts = count_images(sdir)
        total = sum(counts.values())
        classes = set(counts)
        missing = EXPECTED_CLASSES_15 - classes
        extra = classes - EXPECTED_CLASSES_15
        min_c = min(counts.values()) if counts else 0
        max_c = max(counts.values()) if counts else 0
        imbalance = max_c / min_c if min_c else float("inf")
        print(f"\n[{split}] {total} images across {len(classes)} classes")
        for c in sorted(counts):
            share = counts[c] / total * 100 if total else 0
            flag = "  <-- LOW" if counts[c] < 200 else ""
            print(f"  {c:<10} {counts[c]:>6} ({share:4.1f}%){flag}")
        if missing:
            print(f"  WARNING: expected VCoR classes missing: {sorted(missing)}")
            ok = False
        if imbalance > 5:
            print(f"  WARNING: class imbalance ratio {imbalance:.1f}x (max/min)")
        link_dir = DATA_DIR / split
        if link_dir.exists() and any(link_dir.iterdir()):
            print(f"\ndata/color/{split} already staged - skipping copy.")
        else:
            import shutil

            shutil.copytree(sdir, link_dir, dirs_exist_ok=True)
            print(f"Staged -> data/color/{split}")
        report[split] = {"images": total, "classes": len(classes), "per_class": counts}

    test_dir = root / "test"
    if test_dir.is_dir():
        t_counts = count_images(test_dir)
        report["test_kept_aside"] = {"images": sum(t_counts.values()), "classes": len(t_counts)}
        print(f"\n[test] kept aside for final evaluation: {sum(t_counts.values())} images (not used for training/val)")

    (BASE_DIR / "logs" / "color_prep_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nCOLOR PREP " + ("COMPLETE" if ok else "COMPLETE WITH WARNINGS"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
