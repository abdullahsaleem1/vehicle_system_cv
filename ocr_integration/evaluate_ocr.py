"""Evaluate OCR plate-reading accuracy against ground truth.

Reads ground_truth.csv (image,plate_text,verified) and compares against the
pipeline's anpr_results.json. Metrics are computed ONLY over rows marked
verified=yes; anything else is reported as pending manual verification.

ground_truth.csv columns: image,plate_text,verified,notes
Run: yolo_env\\Scripts\\python.exe evaluate_ocr.py [--results outputs/anpr_results.json]
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = BASE_DIR / "outputs" / "anpr_results.json"
GT_FILE = BASE_DIR / "ground_truth.csv"
PLATE_CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"


def normalize(s):
    return "".join(ch for ch in s.upper() if ch in PLATE_CHARSET)


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str, default=str(DEFAULT_RESULTS))
    args = ap.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"FATAL: results file not found: {results_path}")
        print("       Run run_anpr_pipeline.py first.")
        return 1

    if not GT_FILE.exists():
        print("NO GROUND TRUTH AVAILABLE - MANUAL VERIFICATION NEEDED")
        print(f"       Expected ground-truth file: {GT_FILE}")
        print("       Format: image,plate_text,verified(yes/no),notes")
        return 2

    gt = {}
    pending = []
    with open(GT_FILE, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            img = (row.get("image") or "").strip()
            txt = normalize(row.get("plate_text") or "")
            verified = (row.get("verified") or "").strip().lower() in ("yes", "y", "true", "1")
            if not img:
                continue
            if not txt:
                pending.append((img, "empty plate_text"))
            elif not verified:
                pending.append((img, f"'{txt}' not yet visually verified"))
            else:
                gt[img] = txt

    preds = {}
    for r in json.loads(results_path.read_text(encoding="utf-8")):
        dets = r.get("detections", [])
        best = max(dets, key=lambda d: d["ocr_conf"], default=None)
        if best:
            preds[r["image"]] = normalize(best["plate_text_clean"] or best["plate_text"])

    evaluated_imgs = sorted(set(gt) & set(preds))
    missing_gt = sorted(set(preds) - set(gt))

    if not evaluated_imgs:
        print("NO GROUND TRUTH AVAILABLE - MANUAL VERIFICATION NEEDED")
        print(f"       {len(pending)} entries pending verification in {GT_FILE.name}:")
        for img, why in pending[:10]:
            print(f"         - {img}: {why}")
        if missing_gt:
            print(f"       {len(missing_gt)} pipeline predictions have no GT row yet "
                  f"(transcribe these): {missing_gt[:10]}")
        return 2

    exact = 0
    char_correct = 0
    char_total = 0
    print(f"{'image':<28}{'gt':>14}{'pred':>14}{'ok':>6}{'dist':>6}")
    print("-" * 68)
    for img in evaluated_imgs:
        g, p = gt[img], preds[img]
        ok = g == p
        exact += ok
        d = levenshtein(g, p)
        char_total += max(len(g), len(p))
        char_correct += max(len(g), len(p)) - d
        print(f"{img:<28}{g or '(none)':>14}{p or '(none)':>14}{('Y' if ok else 'N'):>6}{d:>6}")

    n = len(evaluated_imgs)
    print("-" * 68)
    print(f"images evaluated : {n}")
    print(f"exact-match acc  : {exact}/{n} = {exact / n:.2%}")
    print(f"char-level acc   : {char_correct}/{char_total} = "
          f"{(char_correct / char_total if char_total else 0):.2%}")

    if pending:
        print(f"\nNOTE: {len(pending)} GT entries still pending manual verification - "
              "metrics cover verified rows only.")
    if missing_gt:
        print(f"NOTE: {len(missing_gt)} predictions lack GT rows (not scored): {missing_gt[:10]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
