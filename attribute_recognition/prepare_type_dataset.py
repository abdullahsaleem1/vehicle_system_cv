"""Prepare the vehicle BODY-TYPE classification dataset.

Primary : MIO-TCD 50k subset (kaggle yash88600/miotcd-dataset-50000-imagesclassification)
Suppl.  : Kaggle ademboukhris/cars-body-type-cropped (adds Sedan/SUV distinction)

Applies the MIO-TCD -> FR-17 label mapping, merges both sources into the 6
FR-17 classes, prints + saves the mapping table and per-class counts,
and re-splits 85/15 train/val (stratified by class).

Run: yolo_env\\Scripts\\python.exe prepare_type_dataset.py
"""

import json
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DOWNLOADS = BASE_DIR / "downloads"
DATA_DIR = BASE_DIR / "data" / "type"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRAIN_RATIO = 0.85
SEED = 20260823

FR17_CLASSES = ["Sedan", "SUV", "Motorcycle", "Truck", "Van", "Bus"]

KAGGLE_SETS = {
    "miotcd": "yash88600/miotcd-dataset-50000-imagesclassification",
    "bodytype": "ademboukhris/cars-body-type-cropped",
}

MIO_TCD_TO_FR17 = {
    "car": "Sedan",
    "sedan": "Sedan",
    "pickup": "Truck",
    "pickup_truck": "Truck",
    "single_unit_truck": "Truck",
    "articulated_truck": "Truck",
    "truck": "Truck",
    "work_van": "Van",
    "van": "Van",
    "bus": "Bus",
    "motorcycle": "Motorcycle",
}

DROPPED_CLASSES = {"bicycle", "pedestrian", "non-motorized_vehicle", "non_motorized_vehicle",
                   "background", "motorized_vehicle", "hatchback", "coupe", "wagon",
                   "convertible", "minivan"}


def kaggle_available():
    try:
        import kaggle  # noqa
        return True
    except Exception:
        pass
    py = Path(sys.executable)
    return (py.parent / "kaggle.exe").exists() or (py.parent / "kaggle").exists()


def download_set(handle: str, dest_name: str):
    marker = DOWNLOADS / f"{dest_name}.flag"
    if marker.exists():
        print(f"[{dest_name}] already downloaded - skipping.")
        return DOWNLOADS / dest_name if (DOWNLOADS / dest_name).is_dir() else DOWNLOADS
    out_dir = DOWNLOADS / dest_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {handle} ...")
    r = subprocess.run([sys.executable, "-m", "kaggle", "datasets", "download", "-d", handle,
                        "-p", str(out_dir), "--unzip"])
    if r.returncode != 0:
        print(f"FATAL: kaggle download failed for {handle}")
        sys.exit(1)
    zips = list(out_dir.glob("*.zip"))
    if zips:
        import zipfile

        for z in zips:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(out_dir)
            z.unlink()
    marker.write_text("ok")
    return out_dir


def find_class_dirs(root: Path):
    """Return {class_folder_name: dir} for dirs that contain image files."""
    out = {}
    for d in sorted(root.rglob("*")):
        if d.is_dir() and not any(sub.is_dir() and not list(sub.iterdir()) == [] for sub in d.iterdir() if sub.is_dir()):
            n_imgs = sum(1 for f in d.iterdir() if f.suffix.lower() in IMG_EXTS) if any(d.iterdir()) else 0
            has_subdirs = any(s.is_dir() for s in d.iterdir())
            if n_imgs > 0 and not has_subdirs:
                rel = d.relative_to(root)
                out[str(rel).replace("\\", "/")] = d
    return out


def map_class(rel_key: str):
    name = rel_key.split("/")[-1].strip().lower().replace(" ", "_").replace("-", "_")
    if name in MIO_TCD_TO_FR17:
        return MIO_TCD_TO_FR17[name], False
    if name in DROPPED_CLASSES:
        return None, True
    return None, None


def main():
    print("=" * 70)
    print("BODY-TYPE DATASET PREPARATION (MIO-TCD 50k + cars-body-type-cropped)")
    print("=" * 70)
    if not kaggle_available():
        print("FATAL: kaggle CLI missing. pip install kaggle + place ~/.kaggle/kaggle.json")
        sys.exit(1)

    sources = {}
    for key, handle in KAGGLE_SETS.items():
        root = download_set(handle, key)
        class_dirs = find_class_dirs(root)
        sources[key] = class_dirs
        print(f"\n[{key}] found {len(class_dirs)} leaf class folders:")
        for rel in sorted(class_dirs):
            mapped, dropped = map_class(rel)
            tag = f"-> {mapped}" if mapped else ("(dropped)" if dropped else "(UNMAPPED - will abort)")
            print(f"    {rel:<40} {tag}")

        unmapped = [rel for rel in class_dirs if map_class(rel)[0] is None and map_class(rel)[1] is None]
        if unmapped:
            print(f"\nFATAL: unmapped class folders present: {unmapped}")
            print("       Update MIO_TCD_TO_FR17 / DROPPED_CLASSES after inspecting these names.")
            sys.exit(1)

    items = []
    per_class = Counter()
    per_source = {}
    for key in ("miotcd", "bodytype"):
        cnt = Counter()
        for rel, d in sources[key].items():
            mapped = map_class(rel)[0]
            if mapped is None:
                continue
            for f in d.iterdir():
                if f.suffix.lower() in IMG_EXTS:
                    items.append((f, mapped))
                    cnt[mapped] += 1
        per_source[key] = dict(cnt)

    print("\nMAPPING TABLE APPLIED:")
    print(f"{'FR-17 class':<12} | MIO-TCD source classes                          | Kaggle supp. classes")
    print("-" * 90)
    for fr in FR17_CLASSES:
        mio_parts = [k for k, v in MIO_TCD_TO_FR17.items() if v == fr]
        mio_n = per_source["miotcd"].get(fr, 0)
        kag_n = per_source["bodytype"].get(fr, 0)
        print(f"{fr:<12} | {', '.join(mio_parts):<47} | {'yes' if kag_n else '-'}")

    counts = Counter(m for _, m in items)
    total = len(items)
    print(f"\nFINAL PER-CLASS COUNTS (merged, before split): total={total}")
    min_c = min(counts.get(c, 0) for c in FR17_CLASSES)
    max_c = max(counts.get(c, 0) for c in FR17_CLASSES)
    ok = True
    for c in FR17_CLASSES:
        n = counts.get(c, 0)
        flag = ""
        if n < 500:
            flag = "  <-- LOW (consider more data)"
            ok = ok and n >= 200
        print(f"  {c:<12} {n:>7} ({n / total * 100:.1f}%){flag}")
    if max_c / max(min_c, 1) > 10:
        print("  WARNING: severe imbalance (>10x); training will use class-aware sampling caveat.")

    for split in ("train", "val"):
        (DATA_DIR / split).mkdir(parents=True, exist_ok=True)
        for c in FR17_CLASSES:
            (DATA_DIR / split / c).mkdir(parents=True, exist_ok=True)

    staged_marker = DATA_DIR / ".staged"
    if staged_marker.exists():
        print("\ndata/type already staged - skipping copy/split (idempotent rerun).")
    else:
        rng = random.Random(SEED)
        by_class = {}
        for p, c in items:
            by_class.setdefault(c, []).append(p)
        copied = {"train": 0, "val": 0}
        name_counters = Counter()
        for c, files in sorted(by_class.items()):
            files = sorted(files)
            rng.shuffle(files)
            n_train = int(len(files) * TRAIN_RATIO)
            for split, group in (("train", files[:n_train]), ("val", files[n_train:])):
                for src in group:
                    name_counters[(c, src.stem)] += 1
                    stem = src.stem
                    if name_counters[(c, src.stem)] > 1:
                        stem = f"{stem}_{name_counters[(c, src.stem)]}"
                    dst = DATA_DIR / split / c / f"{stem}{src.suffix.lower()}"
                    shutil.copy2(src, dst)
                    copied[split] += 1
        staged_marker.write_text(f"{copied['train']}/{copied['val']}")
        print(f"\nStaged+split: train={copied['train']} val={copied['val']} "
              f"(seed={SEED}, ratio={TRAIN_RATIO:.0%}, stratified per class)")

    report = {
        "mapping_mio_tcd": MIO_TCD_TO_FR17,
        "dropped": sorted(DROPPED_CLASSES),
        "per_class_final": dict(counts),
        "per_source": per_source,
        "total": total,
    }
    (BASE_DIR / "logs" / "type_prep_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nTYPE PREP " + ("COMPLETE" if ok else "COMPLETE WITH LOW-COUNT WARNINGS"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
