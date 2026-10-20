import os
import shutil
import sys
from pathlib import Path

from roboflow import Roboflow

WORKSPACE = "roboflow-universe-projects"
PROJECT = "license-plate-recognition-rxg4e"
VERSION = 4
FORMAT = "yolov8"

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets" / "license_plate_recognition_v4"


def main():
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("ERROR: ROBOFLOW_API_KEY environment variable is not set.")
        print()
        print("Get your free API key at: https://app.roboflow.com/settings/api")
        print("Then set it in PowerShell before running this script:")
        print('  $env:ROBOFLOW_API_KEY = "your_api_key_here"')
        sys.exit(1)

    if (DATASET_DIR / "data.yaml").exists():
        print(f"Dataset already present at {DATASET_DIR}, skipping download.")
        return

    if DATASET_DIR.exists():
        print(f"Removing incomplete download folder {DATASET_DIR} ...")
        shutil.rmtree(DATASET_DIR)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    version = project.version(VERSION)
    dataset = version.download(FORMAT, location=str(DATASET_DIR))

    if not (DATASET_DIR / "data.yaml").exists():
        print(f"ERROR: download reported success but {DATASET_DIR} is missing data.yaml.")
        sys.exit(1)

    print()
    print(f"Download complete: {dataset.location}")
    print(f"data.yaml is at: {Path(dataset.location) / 'data.yaml'}")


if __name__ == "__main__":
    main()
