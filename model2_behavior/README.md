# Model 2, Tier A — Behavior / Incident Detector

**FR-20 (pending supervisor sign-off)**

A no-training behavior/incident detector: a COCO-pretrained YOLO vehicle detector,
a multi-object tracker, and a configurable rule engine that emits structured JSON
incident events for a downstream Node.js/Express backend.

> **Status: Tier A, Step 4 complete. Not yet validated on real footage.**
> The pipeline runs end to end, all rules and the event contract are covered by
> tests, but **no threshold has been validated against real video**. Every
> motion/behavior threshold is a **PLACEHOLDER** until the Step 8 evaluation
> harness reports per-rule precision/recall on labelled clips.

---

## Read this first

**`event.confidence` is DETECTION confidence — NOT the probability that the
incident occurred.**

Behavior rules are kinematic heuristics, not calibrated classifiers. A
`confidence` of `0.9` means *"the detector was confident this was a vehicle"*, not
*"this incident is 90% certain to have happened"*. Do not present it to end users
as an incident probability. If a calibrated incident probability is ever needed,
that requires a trained model and labelled data — explicitly out of scope for
Tier A.

**Speeds are in body-lengths per second (`bl/s`), not km/h.**
Without homography / camera calibration there is no metric speed. `bl/s` is a
scale-invariant proxy: pixel speed divided by bounding-box height. Raw pixel
values appear in logs for diagnostics only. **Never** infer km/h from these
numbers.

**Evidence snapshots may contain license plates.**
Event snapshots are full-frame JPEG crops of the scene, so they can include
number plates of the vehicles involved. They are evidence artifacts subject to the
project's **90-day retention / purge policy (NFR-25)**; nothing in this module
retains them beyond that, and Model 1's plate-crop images are handled separately.

**Timestamps come from the frame clock, not `frame_index / fps`.**

The clock is chosen by source type, and a **file source never falls back to
wall-clock time**:

| Source | Clock | Fallback if PTS is unusable |
|---|---|---|
| File (`.mp4`, `.avi`, …) | container PTS (`CAP_PROP_POS_MSEC`) | container frame clock: `frame_index / container_fps` |
| Live (RTSP/RTMP/HTTP, webcam) | wall-clock | — |

Wall-clock time on a file measures how fast the CPU decodes, not what happened in
the video, so using it would make every time-based threshold meaningless. A file
therefore degrades to the **container** frame clock, logs a warning, and records
`clock_source: "frame_count"` plus `degraded_to_container_clock: true` in
`run_summary.json`. Wall-clock timing is used **only** for live sources. Dropped
frames and variable frame rates do not corrupt time-based thresholds. See
`src/frame_clock.py`.

**All file paths use `pathlib`.** `Path` objects are used throughout the module
(`config.py`, `pipeline.py`, `events.py`, `geometry.py`, `weights.py`, and every
script); `open()`/`read_text()`/`write_text()` are called on `Path` objects, and
`cv2` calls stringify explicitly with `str(path)`.

**The camera is assumed to be STATIC.**

Every rule here assumes a fixed camera: lane polygons are authored once and never
move, `wrong_way` compares a vehicle's displacement against a fixed lane
direction, and `stopped_vehicle` / `sudden_decel` measure motion *relative to the
image*, which is only motion relative to the ground while the camera is fixed. On
a moving camera (a vehicle-mounted or panning unit, or a camera knocked by wind)
those assumptions break silently.

The pipeline therefore runs a **static-camera check** (`src/motion_check.py`): it
estimates the scene's global shift between the current frame and the **first frame
of the run** using sparse optical flow, and takes the **median** displacement of
the tracked corners so that vehicles moving independently cannot masquerade as
camera motion. When that median reaches `static_camera_check.threshold` (default
`0.01`, i.e. 1% of the frame width) the run logs a warning and records
`camera_motion_suspected: true` (plus a `camera_motion` block detailing how the
decision was made) in `run_summary.json`.

The check **never** blocks a run or alters an event; it flags footage that must be
reviewed rather than trusted. Reference frames for lane authoring **must come from
the same fixed camera** the pipeline will run on — authoring lanes on any other
view (for example a different camera or a moving shot) produces geometry that
cannot be valid.

---

## What it does

```
video / RTSP
   -> YOLO (COCO, 4 vehicle classes)      detector conf = 0.10
   -> ByteTrack (effective YAML)          assigns stable track ids
   -> track store (fps-aware history)     speeds, displacement, IoU
   -> rule layer                          precision filter + 4 rules
   -> events.json + snapshot JPEGs        stable JSON contract
```

Target classes (COCO ids): `car=2`, `motorcycle=3`, `bus=5`, `truck=7`.

This pipeline handles **full-vehicle** boxes only. It is deliberately separate
from the Model 1 ANPR plate-crop pipeline and shares no weights or checkpoints.

### The four rules

| Rule | Fires when | Required geometry |
|---|---|---|
| `stopped_vehicle` | Speed stays below `speed_thresh_bl_s` for `min_consecutive_s`, inside **any** configured lane | **at least one lane polygon (mandatory)** |
| `wrong_way` | Movement opposes the direction of **the lane the vehicle is inside**, beyond `angle_tolerance_deg` | **at least one lane with a `direction` (mandatory)** |
| `sudden_decel` | Speed drops faster than `decel_thresh_bl_s2` within `window_s` | none |
| `collision_proxy` | Two boxes overlap ≥ `iou_thresh` for `min_overlap_s`, **both were moving**, then one decelerates | none |

`collision_proxy` is a **proxy**: it uses box overlap and kinematics, not contact
forces. It is not a physically validated collision detector.

`wrong_way` is evaluated **per lane**: a vehicle is judged only against the lane
its ground-contact point currently sits in, and the event's `zone_id` is that
lane's name. A vehicle outside every lane cannot be wrong-way.

A rule **refuses to construct** when its required geometry is missing. Passing
`--allow-missing-geometry` skips such rules with a loud warning instead.

### Geometry validation at load time

Lane geometry is validated the moment it is loaded, so a bad configuration fails
loudly and early instead of producing silent misses during a run. Invalid
geometry raises and the run does not start:

- every lane polygon has **≥ 3 points**;
- polygons are **non-degenerate** (non-zero area) and **non-self-intersecting**;
- every polygon lies **inside the reference frame** (`reference_resolution`);
- lane **direction vectors are non-zero**, and two lanes sharing an identical
  direction vector are flagged (usually a copied lane whose direction was never
  updated).

**Overlapping lanes** are detected with an exact intersection-area computation
(convex case, which is what lane quadrilaterals are). They are **reported**, not
rejected, because there is a documented deterministic resolution:

> When two lanes overlap, the **first lane in config order wins**. A vehicle in
> the shared region is attributed to that lane, always, independent of dict
> ordering. This is logged as a warning at load time with the intersection area.

Touching lanes that merely share a boundary line are **not** flagged — that is
the normal way to lay out adjacent lanes. The authoring tool
(`define_camera_geometry.py`) surfaces overlaps as errors and refuses to save them.

---

## Setup

Python 3.12, CUDA-capable GPU (developed on an RTX 3060 12 GB). All commands run
from the repository root.

```powershell
# Offline mode must be set before Python starts.
$env:YOLO_OFFLINE="1"

# Dev-only test dependency.
.\yolo_env\Scripts\python.exe -m pip install pytest

# ByteTrack requires the lap package for its assignment step.
# Pinned in requirements-model2.txt; install only after sign-off.
.\yolo_env\Scripts\python.exe -m pip install -r model2_behavior\requirements-model2.txt
```

**Dependency policy.** `model2_behavior/requirements-model2.txt` lists the extra
runtime dependencies (currently just `lap==0.5.13`) with the justification for
each. No script installs anything automatically, and no package is added to the
environment without listing it and getting sign-off first.

### About `lap` and the network

`lap` is a hard runtime requirement of ByteTrack, not an optional extra:
`ultralytics/trackers/byte_tracker.py` calls `linear_assignment(...)` with the
default `use_lap=True` at all three association stages, so the NumPy fallback is
unreachable through the normal code path.

Importing that module also calls `check_requirements("lap>=0.5.12")`. With
`YOLO_OFFLINE=1` set, Ultralytics prints *"AutoUpdate skipped (offline)"* and
fails cleanly — it does **not** reach the network. Installing `lap` once, ahead of
time, is therefore both required and safe.

Verified facts about the blocked network attempts: normal weight loading makes
**zero** network calls. The only calls ever observed came from a deliberate
bare-name guard test, which probes `https://api.github.com` via a hardcoded
`attempts = 2` retry loop in `ultralytics/utils/downloads.py:477`.

---

## Weights

Weights live at `model2_behavior/models/yolo11m.pt` and are loaded **only** from
that absolute local path. A bare name like `"yolo11m.pt"` is never passed to
`YOLO()`, because Ultralytics would then try to download it.

| Property | Value |
|---|---|
| File | `models/yolo11m.pt` |
| Size | `40684120` bytes (exact match to the official v8.3.0 asset) |
| SHA256 | `d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95` |
| Source | `https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt` |
| `expected_sha256` | `null` — **see below** |

**Why `expected_sha256` is null.** The GitHub release-assets API *does* expose a
`digest` field, and it is explicitly `null` for `yolo11m.pt`, while being
populated for `.ts` and `.onnx` assets. Ultralytics publishes SHA256 only for
`.ts`/`.onnx`, never for `.pt` weights. No comparison is possible, and none is
claimed. Verification instead relied on an exact byte-size match plus ZIP/CRC
integrity checking.

`scripts/download_weights.py` is retained as an **unused fallback** and is not
part of the pipeline. The pipeline never downloads anything at runtime.

---

## Camera geometry (required for two rules)

`stopped_vehicle` and `wrong_way` are inert until geometry is authored. Author it
by clicking on a reference frame:

```powershell
.\yolo_env\Scripts\python.exe -m model2_behavior.scripts.define_camera_geometry --source path\to\frame.jpg
.\yolo_env\Scripts\python.exe -m model2_behavior.scripts.define_camera_geometry --source path\to\clip.mp4 --config model2_behavior\configs\model2.yaml
```

> **The reference frame must come from the fixed camera the pipeline will run
> on.** Lanes are stored in that camera's pixel coordinates and never move, so a
> frame from a different view — or from a moving shot — yields geometry that
> cannot be valid. See the static-camera assumption above.

Controls: left click to add points; right click or Enter to finish the polygon;
then two clicks for the lane direction (tail → head). `u` undoes, `r` restarts,
`q` aborts without writing. After each lane you are offered the chance to author
**another lane**; previously captured lanes are drawn dimmed for context.

### Multiple lanes

A camera can have any number of lanes. Each lives in `rois.lanes` as:

```yaml
rois:
  lanes:
    - name: "northbound"
      polygon: [[120, 300], [520, 300], [560, 470], [80, 470]]
      direction: [[300, 460], [300, 320]]   # tail -> head
    - name: "southbound"
      polygon: [[560, 300], [950, 300], [980, 470], [600, 470]]
      direction: [[760, 320], [760, 460]]
  reference_resolution: [1280, 720]
```

**`rois.lanes[].direction` is the SINGLE SOURCE OF TRUTH for lane direction.**
There is deliberately no direction key anywhere under `rules`, so a direction can
never drift out of sync with the lane it belongs to. (`define_camera_geometry`
writes nothing under `rules`.)

Geometry is stored in **pixel coordinates of the reference frame resolution**,
with `reference_resolution` recorded alongside. At runtime, if the live stream
resolution differs, the geometry is **rescaled** automatically and a warning is
logged.

`full_frame` is rejected everywhere as a lane polygon: a vehicle stopped outside a
designated lane is not an incident. The legacy single-lane keys `driving_lane` and
`lane_direction` are **rejected outright** by the loader, so a stale config cannot
silently half-work.

---

## Running

```powershell
$env:YOLO_OFFLINE="1"

# Real run (needs lane geometry authored first).
.\yolo_env\Scripts\python.exe -m model2_behavior.scripts.run_pipeline `
    --source path\to\clip.mp4 --fps 25

# Run the rules that can run, when geometry is not yet authored.
.\yolo_env\Scripts\python.exe -m model2_behavior.scripts.run_pipeline `
    --source path\to\clip.mp4 --fps 25 --allow-missing-geometry
```

Pass `--fps` explicitly. If omitted, the container FPS is used, and a `25.0`
fallback is applied with a warning when the container reports nothing.

### Validating the tracker before authoring geometry (`--tracking-only`)

Detector + tracker + annotated output only. **No rules are evaluated and no
lane geometry is required**, so you can confirm track-id stability on raw footage
*before* configuring lanes:

```powershell
.\yolo_env\Scripts\python.exe -m model2_behavior.scripts.run_pipeline `
    --source path\to\clip.mp4 --tracking-only
```

The run reports:

- **measured throughput** both including and **excluding the first
  `--warmup-frames` frames** (default 10), because early frames include
  CUDA/tracker initialisation;
- **p95 frame time** (and p50/max) excluding warm-up, because a mean FPS figure
  hides stalls that a real-time deployment cares about;
- **per-track lifetimes** in `run_summary.json` (`tracks[]`, with
  `first_time_s`, `last_time_s`, `duration_s`, `observations`), plus
  `median_track_lifetime_s` and `tracks_shorter_than_1s`;
- **`suspected_id_switches (heuristic)`** — a *heuristic* that flags short-lived
  tracks ending long before the run finishes that share a class with a later
  track.

> **`suspected_id_switches (heuristic)` is NOT a tracking-accuracy metric.**
> It is a review hint, not a measurement of identity switches. A true id-switch
> count requires ground-truth identity labels, which this project does not have.
> Only the honestly-named key exists in `run_summary.json`; the earlier
> `id_switch_indicators` alias has been **removed** so no reader can mistake the
> heuristic for a tracking-accuracy metric.

### Reporting throughput honestly

A throughput number is only meaningful alongside the conditions it was measured
under, so every run's `run_summary.json` records:

| Field | Meaning |
|---|---|
| `avg_fps` | Mean throughput **including** warm-up (not a benchmark figure) |
| `avg_fps_excl_warmup` | Mean throughput excluding the first `warmup_frames` |
| `p95_frame_time_ms` | 95th-percentile per-frame time, warm-up excluded |
| `frame_time_percentiles_ms` | `p50` / `p95` / `max` frame time, warm-up excluded |
| `source_resolution` | The resolution actually processed |
| `clip_duration_s` | Stream seconds processed (from the frame clock) |
| `camera_motion_suspected` | Static-camera check verdict: `true` when the scene's global shift crossed the threshold |
| `camera_motion` | The check's detail: `checked`, `threshold_fraction_of_width`, `max_shift_fraction`, `median_shift_fraction`, `samples_taken`, `skipped_frames`, `first_suspected_frame`, and the config used |

**Warm-up is always excluded from the percentile and `avg_fps_excl_warmup`
figures.** A single "fps" number measured with initialisation included is not
reported as a benchmark anywhere in this module or its docs. Any performance
claim must state the clip, its resolution, the device, and that warm-up was
excluded.

### Verifying before you run

```powershell
# 1. Validate the output contract with synthetic data. No GPU, no footage.
.\yolo_env\Scripts\python.exe -m model2_behavior.scripts.mock_test_contract

# 2. Prove annotated output renders with the network blocked.
.\yolo_env\Scripts\python.exe -m model2_behavior.scripts.annotation_offline_proof

# 3. Full test suite.
$env:YOLO_OFFLINE="1"; .\yolo_env\Scripts\python.exe -m pytest

# 4. Gate 3 evidence: dependency pin, Model 1 still runs, frame clock,
#    track_buffer_s conversion, multi-lane geometry, tracking-only mode, contract.
.\yolo_env\Scripts\python.exe -m model2_behavior.scripts.gate3_report

# 5. Model 1 regression check: Model 1 plate detection must be byte-identical to
#    its recorded baseline (Model 2 shares Model 1's environment).
.\yolo_env\Scripts\python.exe -m model2_behavior.scripts.model1_regression_check
```

Run the mock contract test **before** any live-inference run. It is the
project-convention gate that catches contract breakage without needing footage.

---

## Output contract

Every run writes to `outputs/run_<UTC timestamp>/`:

```
events.json             the contract the backend consumes
effective_config.json   every effective value, including fps and frame conversions
effective_tracker.yaml  the exact tracker YAML passed to model.track()
run_summary.json        counts, timing, throughput
snapshots/              JPEG per event
annotated.mp4           annotated video (if enabled)
```

Each event object:

```json
{
  "event_id": "d6994a82-7219-4838-b844-6f15be6355a9",
  "incident_type": "stopped_vehicle",
  "severity": "Medium",
  "camera_id": "cam_01",
  "zone_id": "driving_lane",
  "timestamp": "2026-10-20T11:29:16.086Z",
  "frame_index": 100,
  "track_ids": [1],
  "vehicle_classes": ["car"],
  "bounding_boxes": [[280.0, 370.0, 340.0, 430.0]],
  "confidence": 0.85,
  "evidence": {
    "snapshot_path": "...\\snapshots\\00000100_stopped_vehicle_d6994a82.jpg",
    "clip_path": null
  },
  "rule_params": { "fps": 25.0, "min_consecutive_s": 3.0, "min_consecutive_frames": 75 }
}
```

- `incident_type` ∈ `stopped_vehicle | wrong_way | sudden_decel | collision_proxy`
- `severity` ∈ `High | Medium | Low`
- `confidence` — **detection** confidence, `[0, 1]`, not incident probability
- `zone_id` — the matched lane name, or `null`
- `timestamp` — taken from the **frame clock** (video PTS for files, wall-clock for
  live), never `frame_index / fps`, so it survives dropped frames
- `clip_path` — always `null` in Tier A; clip extraction is a Tier B item
- `rule_params` — the thresholds **actually used**, so a decision is auditable

The event document is flushed to disk after **every** event, so a crash mid-run
still leaves a valid JSON file containing everything up to that point.

---

## Configuration

Everything lives in `configs/model2.yaml`. There are **no silent defaults**: the
run writes `effective_config.json` recording every resolved value.

All durations are in **seconds** and are converted to frame counts at runtime
using the measured FPS. Frame counts must never be specified in the config;
`min_consecutive_frames` in a rule is rejected by the loader with an explicit
error, and `tracker_params.track_buffer` (a frame count) is likewise rejected in
favour of **`tracker_params.track_buffer_s`** (default `3.0` seconds). The
conversion `track_buffer_s × fps → track_buffer` is logged and recorded in
`effective_tracker.yaml`.

The effective tracker config is the merge of `tracker_config_path` with
`tracker_params`, written to the run directory and passed to `model.track()`.
Detector `conf` is deliberately low (`0.10`, ≥ ByteTrack's `track_low_thresh`) so
the tracker's second association stage can recover weak detections. Precision
filtering happens in the **rule layer** via `rule_layer.min_track_mean_conf`
(`0.35`), applied to each track's mean detection confidence.

The **`static_camera_check`** block configures the static-camera precondition
check (`enabled`, `work_width`, `max_corners`, `min_distance`, `threshold`,
`min_tracked_corners`, `check_every_n_frames`). An unknown key or an out-of-range
value is rejected at load time rather than silently defaulting. See the
static-camera assumption above.

### Threshold status

| Rule | Parameter | Status |
|---|---|---|
| `stopped_vehicle` | `speed_thresh_bl_s` | **PLACEHOLDER, unvalidated** |
| `wrong_way` | `angle_tolerance_deg`, `min_displacement_bl` | **PLACEHOLDER, unvalidated** |
| `sudden_decel` | `decel_thresh_bl_s2`, `window_s`, `min_speed_before_bl_s` | **PLACEHOLDER, unvalidated** |
| `collision_proxy` | `iou_thresh`, `min_overlap_s`, `min_pre_contact_speed_bl_s`, `post_contact_speed_drop_bl_s`, `window_s` | **PLACEHOLDER, unvalidated** |

Step 8 adds an evaluation harness that reports per-rule precision/recall on
labelled clips. Until that has run and the numbers have been reviewed, treat
every value above as a starting guess.

### Evaluating on labelled clips (`scripts/evaluate_events.py`)

The harness is written and fixture-tested now; it needs no GPU and no real data to
exercise. It reads:

- **`labels.csv`** — `clip,camera_id,incident_type,start_s,end_s,notes` plus an
  **optional `split` column**. An **empty `incident_type` marks a NEGATIVE clip**
  (no incidents of any type); negative clips are what make a false-alarm rate
  meaningful, so they are required.
- **events JSON** per clip — from `run_pipeline.py`'s `events.json` (or
  `events.jsonl`), either one file per clip (`<clip>.json`) or one run folder per
  clip (`<run>/events.json` + `run_summary.json`).

**Matching rule (fixed):** a prediction is a **TP** when its `incident_type`
matches the label's **and** its timestamp falls inside
`[start_s − 2 s, end_s + 2 s]` (tolerance configurable with `--tolerance-s`).
Matching is **one-to-one**: a label can be credited at most one prediction and a
prediction can satisfy at most one label, so a rule that fires repeatedly during
one incident produces **duplicate false positives, never extra TPs**. A same-type
event whose timestamp falls outside the window on a clip that *does* have that
label is a **false positive**. Unmatched events on a **negative** clip are counted
separately as **false alarms** so they cannot flatter precision.

It reports **per-rule precision, recall, F1**, TP/FP/FN counts, the unmatched
predictions and missed labels themselves, **false alarms per hour on negative
clips** (numerator: false alarms on negative clips; denominator: the **total
duration of negative clips**, recovered from each `run_summary.json`), and the
**config snapshot actually used** — thresholds and effective tracker parameters,
alongside a **`config_hash`** of the behaviour-affecting settings — so a metric is
never quoted without the settings that produced it. If clips were produced by
different configs the single `config_hash` is reported as `null` rather than
hiding the mismatch.

#### Tuning vs reporting: `--split`

`labels.csv` may carry a `split` column with values `tune` or `test`. `--split
tune` and `--split test` then select disjoint row sets, so thresholds can be
chosen on one set and **reported on held-out clips** without maintaining two label
files. `--split all` (the default) uses every row; if no row carries a `split`
column the filter is a no-op with a warning, so a labels file written before the
column existed still evaluates.

```powershell
# Directory of per-clip events files (or run output folders):
.\yolo_env\Scripts\python.exe -m model2_behavior.scripts.evaluate_events `
    --labels outputs\eval\labels.csv --events-dir outputs\eval\runs

# Choose thresholds on the tuning clips, then report on held-out test clips:
.\yolo_env\Scripts\python.exe -m model2_behavior.scripts.evaluate_events `
    --labels outputs\eval\labels.csv --events-dir outputs\eval\runs --split test

# A single events file for one clip:
.\yolo_env\Scripts\python.exe -m model2_behavior.scripts.evaluate_events `
    --labels labels.csv --events path\to\events.json --clip clip_a.mp4
```

---

## Tests

```powershell
$env:YOLO_OFFLINE="1"; .\yolo_env\Scripts\python.exe -m pytest
```

**346 tests**, no GPU, no footage, no network. They cover geometry parsing,
load-time geometry validation (point count, self-intersection, in-frame,
zero-length direction, overlapping lanes and their deterministic tie-break),
multi-lane containment and rescaling, speed normalization, all four rules on
synthetic trajectories (including per-lane `wrong_way` and a vehicle outside every
lane), rule refusal without required geometry, fps-aware threshold conversion,
`track_buffer_s` conversion, the frame clock (PTS validity, monotonicity,
dropped-frame detection, and that a file source degrades to the container clock
and never to wall-clock), tracking-only mode, the frame-time percentile and track
lifetime statistics, the event contract, the Model 1 regression comparison, and
the evaluation harness's matcher (including one-to-one duplicate handling,
out-of-window false positives, the negative-duration false-alarm denominator, and
`--split`), and the static-camera check (still scene quiet, synthetic panning
scene flagged, then the same flag observed end to end in `run_summary.json`).
Rules are tested with **synthetic geometry and lane directions supplied by the
tests** — nothing depends on real data being present.

The Model 1 regression baseline lives at
`tests/fixtures/model1_regression_baseline.json` so it is **tracked by git** and
reviewed alongside the code. It was recorded **after `lap==0.5.13` was installed**
into `yolo_env`, so it captures the post-dependency environment Model 1 actually
runs in from that point on; `scripts/model1_regression_check.py` compares against
it by default.

### Model 1 isolation

Model 2 adds one new directory (`model2_behavior/`) and touches **two files
outside it**, both at the repository root, neither of which is Model 1 code:

| Root file | Change | Why it is safe for Model 1 |
|---|---|---|
| `.gitignore` | **modified** (appended Model 2 ignore rules) | Additive ignore patterns only; no existing rule removed or changed. |
| `pytest.ini` | **added** | New file; configures `testpaths = model2_behavior/tests`. Model 1 ships no pytest suite (its checks are `plate_detection/mock_test_*.py` scripts, which do not match `python_files = test_*.py`), so this cannot alter Model 1's test workflow. It exists because pytest options must be read from a file at the `rootdir` — a config placed *inside* `model2_behavior/` is silently ignored when pytest is invoked from the root, which would leave `--strict-markers` and the `filterwarnings = error` rule inactive without any warning. |

No file under `plate_detection/`, no Model 1 weight, no Model 1 checkpoint, and no
Model 1 output is modified. `git diff main --stat -- . ":!model2_behavior"` shows
exactly those two files and nothing else.

---

## Project layout

```
model2_behavior/
  configs/        model2.yaml, bytetrack.yaml, botsort.yaml
  models/         yolo11m.pt (git-ignored), MANIFEST.json
  requirements-model2.txt   extra runtime deps (lap) with justification
  src/
    weights.py            offline enforcement, local-path resolution
    config.py             loading, validation, effective tracker YAML
    geometry.py           multi-lane polygons, directions, rescaling
    frame_clock.py        PTS / wall-clock timing, dropped-frame detection
    motion_check.py       static-camera precondition check (global shift)
    tracks.py             observations, track state, normalized speed, IoU
    detector_tracker.py   YOLO + ByteTrack wrapper
    events.py             incident schema and JSON writer
    pipeline.py           end-to-end run, tracking-only mode
    rules/                base + the four rules
  scripts/        verify_environment, verify_weights, offline_proof,
                  gate1_report, gate2_report, gate3_report,
                  define_camera_geometry, mock_test_contract,
                  annotation_offline_proof, run_pipeline,
                  evaluate_events, model1_regression_check
  tests/          pytest suite
  tests/fixtures/ tracked Model 1 regression baseline
```

---

## Known limitations

- **No homography.** Speeds are `bl/s` proxies, not metric. Thresholds do not
  transfer between camera angles without re-tuning.
- **Thresholds are unvalidated.** See the table above.
- **No clip evidence.** Tier A writes JPEG snapshots only; `clip_path` is `null`
  until Tier B.
- **Collision detection is a proxy**, based on overlap and kinematics.
- **Lane assignment uses the box's ground-contact point**, so `wrong_way` and
  `stopped_vehicle` judge a vehicle by the lane that point falls in. Overlapping
  lanes are resolved by the first match in config order.
- **ID-switch reporting is heuristic**, not ground truth. (`id_switch_indicators`
  was removed; only `suspected_id_switches` remains.)
- **A static camera is assumed.** A moving camera invalidates fixed lane geometry
  and every image-relative speed rule. The static-camera check flags suspected
  motion in `run_summary.json` (`camera_motion_suspected`), but it is a warning,
  not a correction — flagged footage must be reviewed, and the check is a
  heuristic on the median global shift, not a calibration.
- **Not yet run on real footage.** Step 7 is pending a sample clip or RTSP URL.
