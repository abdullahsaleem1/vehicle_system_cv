import json

data = json.load(open("ocr_integration/outputs/anpr_results.json"))
for r in data:
    for d in r.get("detections", []):
        x1, y1, x2, y2 = d["box"]
        w, h = x2 - x1, y2 - y1
        status = "READ" if d["plate_text"] else "FAILED"
        print(
            f"{r['image']:<18} crop {w:.0f}x{h:.0f}px det={d['det_conf']:.2f} "
            f"ocr={d['ocr_conf']:.2f} {status:>6} '{d['plate_text']}' frags={d['raw_fragments']}"
        )
