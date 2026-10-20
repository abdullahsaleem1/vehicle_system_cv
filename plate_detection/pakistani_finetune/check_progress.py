"""Quick progress checker for the Pakistani fine-tuning run.

Run: yolo_env\\Scripts\\python.exe check_progress.py
"""

import re
from pathlib import Path

LOG_FILE = Path(__file__).resolve().parent / "finetune_log.txt"
EPOCH_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] Epoch (\d+)/(\d+) complete")
RUN_RE = re.compile(r"^=== (FINETUNE (?:STARTED|RESUMED|FINISHED)[^\n]*) at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ===$")


def main():
    if not LOG_FILE.exists():
        print("No finetune_log.txt found - fine-tuning has never been started.")
        return
    last_epoch = None
    last_run = None
    finished = False
    with open(LOG_FILE, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            m = EPOCH_RE.search(line)
            if m:
                last_epoch = m.groups()
            m = RUN_RE.match(line)
            if m and m.group(1).startswith("FINETUNE"):
                last_run = m.group(1)
            if "FINETUNE FINISHED" in line or "TRAINING ALREADY COMPLETE" in line:
                finished = True

    if last_epoch is None:
        ctx = f" | last event: {last_run}" if last_run else ""
        print(f"No epochs completed yet - fine-tuning in early startup/first epoch.{ctx}")
        return

    ts, done, total = last_epoch[0], int(last_epoch[1]), int(last_epoch[2])
    remaining = max(total - done, 0)
    suffix = " | ALL EPOCHS COMPLETE (early stop may have applied)" if finished or remaining == 0 else ""
    print(f"Last completed: epoch {done}/{total} at {ts} | remaining: {remaining}{suffix}")


if __name__ == "__main__":
    main()
