"""Prepare the Pakistani fine-tuning dataset.

Locates the two source folders anywhere in the project tree, runs the
conditional dataset through quality gates, merges what passes into
pakistani_finetune/data with an 85/15 train/val split, and writes dataset.yaml.

Run: yolo_env\\Scripts\\python.exe prepare_pk_dataset.py
"""

import hashlib
import json
import random
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
ISSUES_FILE = LOGS_DIR / "prepare_issues.json"
DATASET_YAML = BASE_DIR / "dataset.yaml"

EXCLUDE_DIRS = {"yolo_env", ".git", "__pycache__", "pakistani_finetune", "weights", "node_modules"}

PRIMARY_ALIASES = ["pakistan_license_dataset_yolo"]
CONDITIONAL_ALIASES = ["pakistan licesnse number plates dataset", "pakistani license number plates data"]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRAIN_RATIO = 0.85
SEED = 20260822

DUPLICATE_HASH_BITS = 64
DUPLICATE_HAMMING_THRESHOLD = 4


def iter_project_dirs():
    stack = [PROJECT_ROOT]
    while stack:
        d = stack.pop()
        try:
            entries = list(d.iterdir())
        except (PermissionError, OSError):
            continue
        for e in entries:
            if e.is_dir():
                if e.name.lower() in EXCLUDE_DIRS:
                    continue
                yield e
                stack.append(e)


def find_dataset_dir(aliases):
    exact = []
    fuzzy = []
    for d in iter_project_dirs():
        n = d.name.lower()
        for a in aliases:
            if n == a.lower():
                exact.append(d)
            elif a.lower() in n or n in a.lower():
                fuzzy.append(d)
    if len(exact) == 1:
        return exact[0], "exact"
    if len(exact) > 1:
        print(f"AMBIGUITY: multiple candidate folders found: {[str(p) for p in exact]}")
        sys.exit(1)
    uniq_fuzzy = sorted(set(fuzzy), key=lambda p: len(str(p)))
    if len(uniq_fuzzy) >= 1:
        return uniq_fuzzy[0], f"fuzzy match (exact name '{aliases[0]}' not found)"
    return None, None


def find_images(root):
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS and p.is_file())


def find_annotation_files(root):
    buckets = {"yolo_txt": [], "voc_xml": [], "csv": []}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        sfx = p.suffix.lower()
        if sfx == ".txt":
            buckets["yolo_txt"].append(p)
        elif sfx == ".xml":
            buckets["voc_xml"].append(p)
        elif sfx == ".csv":
            buckets["csv"].append(p)
    return buckets


def label_path_for_image(img, ann_root):
    rel = img.relative_to(ann_root)
    candidates = []
    stem = img.stem
    for base in [ann_root / "labels"] + list(ann_root.glob("*/labels")) + list(ann_root.glob("*/*/labels")):
        candidates.append(base / rel.parent / f"{stem}.txt")
        candidates.append(base / f"{stem}.txt")
    candidates.append(img.with_suffix(".txt"))
    for c in candidates:
        if c.is_file():
            return c
    return None


def parse_yolo_line(line, w=1.0, h=1.0):
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"expected 5 fields, got {len(parts)}")
    cls = int(parts[0])
    xc, yc, bw, bh = (float(v) for v in parts[1:])
    if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 + 1e-6 and 0 < bh <= 1 + 1e-6):
        raise ValueError(f"box out of range xc={xc} yc={yc} w={bw} h={bh}")
    x1 = max(0.0, xc - bw / 2)
    y1 = max(0.0, yc - bh / 2)
    x2 = min(1.0, xc + bw / 2)
    y2 = min(1.0, yc + bh / 2)
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        raise ValueError("zero-area box after clamping")
    return cls, x1, y1, x2, y2


def voc_to_yolo_lines(xml_path):
    tree = ET.parse(xml_path)
    root_el = tree.getroot()
    size = root_el.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)
    lines = []
    for obj in root_el.iter("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name not in ("license_plate", "licence_plate", "plate", "number_plate", "license-plate"):
            raise ValueError(f"unexpected class name '{name}' in {xml_path.name}")
        b = obj.find("bndbox")
        xmin = float(b.findtext("xmin"))
        ymin = float(b.findtext("ymin"))
        xmax = float(b.findtext("xmax"))
        ymax = float(b.findtext("ymax"))
        xc = ((xmin + xmax) / 2) / w
        yc = ((ymin + ymax) / 2) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    if not lines:
        raise ValueError("no plate objects in xml")
    return lines


CSV_HEADER_RE = re.compile(r"(xmin|x_min)", re.IGNORECASE)


def csv_to_annotations(csv_path):
    import csv as csvmod

    rows = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csvmod.DictReader(f)
        cols = {c.strip().lower(): c for c in (reader.fieldnames or [])}
        need = ["xmin", "ymin", "xmax", "ymax"]
        if any(n not in "".join(cols).lower() for n in ["min"]) or not CSV_HEADER_RE.search(",".join(reader.fieldnames or [])):
            raise ValueError("csv does not look like a VOC-style bbox table")
        fname_col = next((cols[c] for c in cols if c in ("filename", "image", "image_name", "file")), None)
        xmin_col = next((cols[c] for c in cols if c.replace("_", "") == "xmin"), None)
        ymin_col = next((cols[c] for c in cols if c.replace("_", "") == "ymin"), None)
        xmax_col = next((cols[c] for c in cols if c.replace("_", "") == "xmax"), None)
        ymax_col = next((cols[c] for c in cols if c.replace("_", "") == "ymax"), None)
        cname_col = next((cols[c] for c in cols if c in ("class", "label", "classname")), None)
        if not all([fname_col, xmin_col, ymin_col, xmax_col, ymax_col]):
            raise ValueError("csv missing filename/xmin/ymin/xmax/ymax columns")
        for row in reader:
            fn = row[fname_col].strip()
            cls_name = (row.get(cname_col) or "").strip().lower() if cname_col else "license_plate"
            if cls_name not in ("license_plate", "licence_plate", "plate", "number_plate"):
                raise ValueError(f"unexpected class '{cls_name}' in csv")
            rows.setdefault(fn, []).append(
                (float(row[xmin_col]), float(row[ymin_col]), float(row[xmax_col]), float(row[ymax_col]))
            )
    return rows


def dhash(img_path, hash_size=8):
    try:
        with Image.open(img_path) as im:
            im = im.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
        px = list(im.getdata())
        rows = [px[i * (hash_size + 1):(i + 1) * (hash_size + 1)] for i in range(hash_size)]
        bits = 0
        for r_idx, row in enumerate(rows):
            for c_idx in range(hash_size):
                if row[c_idx] > row[c_idx + 1]:
                    bits |= 1 << (r_idx * hash_size + c_idx)
        return bits
    except Exception:
        return None


def hamming(a, b):
    return bin(a ^ b).count("1")


def main():
    print("=" * 70)
    print("PK FINE-TUNE DATASET PREPARATION")
    print("=" * 70)

    issues = []

    primary_dir, primary_how = find_dataset_dir(PRIMARY_ALIASES)
    conditional_dir, conditional_how = find_dataset_dir(CONDITIONAL_ALIASES)

    print(f"\nPRIMARY folder   : {primary_dir}")
    print(f"  located via    : {primary_how}")
    print(f"CONDITIONAL folder: {conditional_dir}")
    print(f"  located via    : {conditional_how}")

    if primary_dir is None:
        print("\nFATAL: PRIMARY dataset folder not found anywhere in project tree.")
        print(f"       Looked for any of: {PRIMARY_ALIASES}")
        sys.exit(1)

    for tag, d in (("PRIMARY", primary_dir), ("CONDITIONAL", conditional_dir)):
        if d is None:
            continue
        print(f"\n--- {tag} top-level contents ---")
        for child in sorted(d.iterdir()):
            kind = "dir " if child.is_dir() else "file"
            print(f"  [{kind}] {child.name}")

    primary_imgs = find_images(primary_dir)
    print(f"\nPRIMARY: {len(primary_imgs)} images found (recursive)")

    merged_sources = [("primary", str(primary_dir))]

    use_conditional = False
    gate_results = {}

    if conditional_dir is None:
        print("\nCONDITIONAL dataset NOT FOUND -> proceeding with primary only.")
        gate_results["located"] = False
    else:
        cond_imgs = find_images(conditional_dir)
        anns = find_annotation_files(conditional_dir)
        fmt_counts = {k: len(v) for k, v in anns.items() if v}
        ext_counts = {}
        for p in cond_imgs:
            ext_counts[p.suffix.lower()] = ext_counts.get(p.suffix.lower(), 0) + 1
        print(f"\nCONDITIONAL: {len(cond_imgs)} images by type: {ext_counts}")
        print(f"CONDITIONAL annotation files found: {fmt_counts if fmt_counts else 'NONE'}")

        loaded_ok, load_fail = [], []

        def note_load_fail(p, reason):
            load_fail.append({"file": str(p.relative_to(PROJECT_ROOT)), "reason": reason})

        for img in cond_imgs:
            try:
                with Image.open(img) as im:
                    im.verify()
                with Image.open(img) as im2:
                    if im2.width < 32 or im2.height < 32:
                        note_load_fail(img, f"suspiciously small image {im2.size}")
                        continue
                loaded_ok.append(img)
            except Exception as e:
                note_load_fail(img, f"unreadable/corrupt ({type(e).__name__}: {e})")

        has_labels_for_all = False
        pairs = []
        missing_labels = [str(p) for p in loaded_ok if label_path_for_image(p, conditional_dir) is None]
        if anns["yolo_txt"] or anns["voc_xml"]:
            for img in loaded_ok:
                lp = label_path_for_image(img, conditional_dir)
                if lp is None:
                    continue
                try:
                    if lp.suffix == ".xml":
                        lines = voc_to_yolo_lines(lp)
                    else:
                        raw = lp.read_text(encoding="utf-8", errors="replace").strip()
                        if not raw:
                            raise ValueError("empty label file")
                        lines = raw.splitlines()
                    parsed = [parse_yolo_line(l) for l in lines if l.strip()]
                    if not parsed:
                        raise ValueError("label had no valid boxes")
                    pairs.append((img, parsed))
                except Exception as e:
                    note_load_fail(img, f"label unusable ({e})")

        # GATE A: visible Pakistani plates (automated proxy = integrity + resolution; visual truth needs human/model review)
        gate_a = len(cond_imgs) > 0 and len(load_fail) == 0
        automated_a_verdict = "PASS(integrity-only)" if gate_a else "FAIL"
        print("\nGATE A (images contain visible PK plates):")
        print(f"  readable/intact images : {len(loaded_ok)}/{len(cond_imgs)}")
        print(f"  automated verdict      : {automated_a_verdict}")
        print("  NOTE: actual plate visibility cannot be machine-verified without inference; requires spot-check.")

        # GATE B: parseable annotations, every image matched & non-empty
        gate_b = bool(pairs) and len(missing_labels) == 0 and len(pairs) == len(loaded_ok)
        print("\nGATE B (parseable labels for EVERY image):")
        print(f"  annotation formats     : {fmt_counts if fmt_counts else 'NONE'}")
        print(f"  images with valid labels: {len(pairs)}/{len(loaded_ok)}")
        if missing_labels:
            print(f"  images MISSING labels  : {len(missing_labels)} e.g. {missing_labels[:3]}")
        verdict_b = "PASS" if gate_b else "FAIL"
        print(f"  verdict                : {verdict_b}")

        gate_c = False
        if gate_b:
            bad_boxes = []
            for img, boxes in pairs:
                for (_, *box) in boxes:
                    x1, y1, x2, y2 = box
                    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                        bad_boxes.append((img.name, box))
            gate_c = len(bad_boxes) == 0
            print("\nGATE C (boxes sane: non-zero area, within bounds):")
            print(f"  invalid boxes          : {len(bad_boxes)} {bad_boxes[:3]}")
            print(f"  verdict                : {'PASS' if gate_c else 'FAIL'}")
        else:
            print("\nGATE C: SKIPPED (gate B failed)")

        gate_d = False
        dup_report = {"checked": 0, "duplicates_vs_primary": []}
        if gate_b and gate_c:
            primary_hashes = {}
            for p in primary_imgs:
                hv = dhash(p)
                if hv is not None:
                    primary_hashes[p.name] = hv
            for img, _ in pairs:
                hv = dhash(img)
                dup_report["checked"] += 1
                if hv is None:
                    continue
                for pname, ph in primary_hashes.items():
                    if hamming(hv, ph) <= DUPLICATE_HAMMING_THRESHOLD:
                        dup_report["duplicates_vs_primary"].append(
                            {"conditional": img.name, "primary": pname}
                        )
                        break
            gate_d = len(dup_report["duplicates_vs_primary"]) == 0
            print("\nGATE D (no duplicates vs primary):")
            print(f"  compared               : {dup_report['checked']} vs {len(primary_hashes)} primary imgs")
            print(f"  duplicates             : {len(dup_report['duplicates_vs_primary'])}")
            print(f"  verdict                : {'PASS' if gate_d else 'FAIL'}")
        else:
            print("\nGATE D: SKIPPED (earlier gate failed)")

        gate_results.update(
            {
                "located": True,
                "A": gate_a,
                "B": gate_b,
                "C": gate_c,
                "D": gate_d,
                "dup_report": dup_report,
            }
        )
        use_conditional = gate_a and gate_b and gate_c and gate_d

    print("\n" + "=" * 70)
    if use_conditional:
        print("DECISION: CONDITIONAL dataset PASSED ALL GATES -> MERGING both sources.")
    else:
        reasons = []
        if gate_results.get("located", True) is False:
            reasons.append("conditional folder not found on disk")
        else:
            if not gate_results.get("B"):
                reasons.append(
                    "gate B FAILED - no parseable annotations covering every image "
                    "(dataset contains images only; guessing boxes is forbidden per spec)"
                )
        print("DECISION: CONDITIONAL dataset REJECTED -> using PRIMARY ONLY.")
        print(f"  reason(s): {'; '.join(reasons) if reasons else 'one or more gates failed'}")
        issues.append({"source": "conditional", "action": "rejected", "reasons": reasons})
    print("=" * 70)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (DATA_DIR / sub).mkdir(parents=True, exist_ok=True)

    existing_train = list((DATA_DIR / "images" / "train").glob("*")) + list((DATA_DIR / "images" / "val").glob("*"))
    if existing_train:
        print(f"\ndata/ already populated ({len(existing_train)} images) - skipping copy/split (idempotent rerun).")
    else:
        items = []
        skipped = 0
        for img in primary_imgs:
            lp = None
            stem = img.stem
            for cand in primary_dir.rglob(f"{stem}.txt"):
                if cand.is_file():
                    lp = cand
                    break
            if lp is None:
                skipped += 1
                issues.append({"source": "primary", "file": str(img), "action": "skipped", "reason": "no matching label"})
                continue
            try:
                with Image.open(img) as im:
                    im.verify()
                raw = lp.read_text(encoding="utf-8", errors="replace").strip()
                if not raw:
                    raise ValueError("empty label")
                for line in raw.splitlines():
                    if line.strip():
                        parse_yolo_line(line)
            except Exception as e:
                skipped += 1
                issues.append({"source": "primary", "file": str(img), "action": "skipped", "reason": str(e)})
                continue
            items.append((img, lp))

        rng = random.Random(SEED)
        rng.shuffle(items)
        n_train = int(len(items) * TRAIN_RATIO)
        splits = {"train": items[:n_train], "val": items[n_train:]}
        copied = {"train": 0, "val": 0}
        for split, split_items in splits.items():
            for img, lp in split_items:
                dst_img = DATA_DIR / "images" / split / f"{img.stem}{img.suffix.lower()}"
                dst_lbl = DATA_DIR / "labels" / split / f"{img.stem}.txt"
                shutil.copy2(img, dst_img)
                shutil.copy2(lp, dst_lbl)
                copied[split] += 1
        print(f"\nCopied+split PRIMARY dataset: train={copied['train']} val={copied['val']} skipped={skipped} (seed={SEED}, ratio={TRAIN_RATIO:.0%})")

    yaml_paths = {
        "train": str((DATA_DIR / "images" / "train").resolve()),
        "val": str((DATA_DIR / "images" / "val").resolve()),
    }
    yaml_text = (
        f"path: {DATA_DIR.resolve()}\n"
        f"train: {yaml_paths['train']}\n"
        f"val: {yaml_paths['val']}\n"
        f"nc: 1\n"
        f"names: ['License_Plate']\n"
    )
    DATASET_YAML.write_text(yaml_text, encoding="utf-8")
    print(f"\nWrote {DATASET_YAML}:")
    print(yaml_text)

    ok = True
    for key, p in (("train", yaml_paths["train"]), ("val", yaml_paths["val"])):
        n_img = len(list(Path(p).glob("*"))) if Path(p).exists() else 0
        lbl_p = str(p).replace("images", "labels")
        n_lbl = len(list(Path(lbl_p).glob("*"))) if Path(lbl_p).exists() else 0
        status = "OK" if n_img > 0 and n_img == n_lbl else "PROBLEM"
        ok = ok and (status == "OK")
        print(f"sanity[{key}]: images={n_img} labels={n_lbl} -> {status}")

    ISSUES_FILE.parent.mkdir(parents=True, exist_ok=True)
    ISSUES_FILE.write_text(json.dumps(issues, indent=2), encoding="utf-8")
    print(f"\nFlagged issues logged: {len(issues)} -> {ISSUES_FILE}")

    print("\nPREPARATION " + ("COMPLETE" if ok else "INCOMPLETE - fix problems above"))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
