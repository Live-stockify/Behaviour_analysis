================================================================================
  LIVESTOCKIFY — BIRD BEHAVIOUR DETECTION SYSTEM
  README & EXECUTION GUIDE
================================================================================

PROJECT OVERVIEW
----------------
This system detects and counts bird behaviours (Drinking, Eating, Sitting,
Standing) from farm video footage or live RTSP camera streams.

It uses a YOLOv8 model enhanced with Level 3 detection techniques that were
developed and tested on real farm footage to maximize bird detection recall.

Output: JSON Lines (.jsonl) files — one detection record per processed frame.
No video output. No image output. Clean and lightweight.

================================================================================
  NEW TECHNIQUES ADDED — LEVEL 3 DETECTION
================================================================================

The original baseline used a single YOLOv8 pass at conf=0.15.
Level 3 adds 4 techniques that together find 2–3x more birds per frame.

------------------------------------------------------------------------
TECHNIQUE 1 — CLAHE (Contrast Limited Adaptive Histogram Equalization)
------------------------------------------------------------------------
What it does:
  Enhances local contrast in the frame BEFORE feeding it to the model.
  Works only on the luminance channel (LAB color space) so colors stay natural.

Why it matters:
  Farm cameras often have harsh overhead lighting with bright spots and dark
  corners. Birds standing in shadows may have confidence scores of 0.04–0.07
  (invisible to the model at any threshold). After CLAHE, those same birds score
  0.10–0.15+ and become detectable.

Settings:
  clipLimit   = 2.0     (how aggressive the boost is)
  tileGridSize= (8, 8)  (local region for adaptive processing)

Where in code: detector.py → apply_clahe()
Config toggle: inference.yaml → model.enable_clahe: true

------------------------------------------------------------------------
TECHNIQUE 2 — 2×2 TILED INFERENCE (Sliced Detection)
------------------------------------------------------------------------
What it does:
  Splits the frame into 4 overlapping tiles (top-left, top-right,
  bottom-left, bottom-right) and runs the YOLO model on each tile separately.
  Each tile detection is translated back to full-frame coordinates.

  +----------+----------+
  |  Tile 00 |  Tile 01 |   Each tile = half width x half height
  |          |          |   with 25% overlap at borders
  +----------+----------+
  |  Tile 10 |  Tile 11 |
  |          |          |
  +----------+----------+

Why it matters:
  YOLO input is always resized to 640×640. A bird that is 40px wide in a
  1920×1080 frame gets squashed to ~13px after resize — below reliable detection
  range. Tiling gives each bird 4x more pixels in the model's view, making
  small/distant/background birds detectable.

  The 25% overlap ensures birds at tile borders appear fully in at least
  one tile and are not missed.

Settings:
  grid    = [2, 2]   (2 rows x 2 cols = 4 tiles)
  overlap = 0.25     (25% border overlap)

Where in code: detector.py → predict() — PASS 2 block
Config toggle: inference.yaml → tiling.enabled: true

------------------------------------------------------------------------
TECHNIQUE 3 — PER-CLASS CONFIDENCE THRESHOLDS
------------------------------------------------------------------------
What it does:
  Instead of one global confidence cutoff, each behaviour class gets its own
  minimum threshold tuned to how visually distinct that behaviour is.

  Drinking : 0.08   (subtle — beak down, partially occluded)
  Eating   : 0.08   (subtle — similar posture to Drinking)
  Sitting  : 0.12   (clearer — distinct body shape)
  Standing : 0.12   (clearest — most common posture)

Why it matters:
  The model is less confident on subtle postures like Drinking and Eating.
  A single global threshold of 0.12 would discard many real Drinking/Eating
  detections. Per-class thresholds recover these without opening false positives
  for the easier classes.

  The model call itself uses min(all thresholds) = 0.08 to get every candidate.
  Each candidate is then filtered against its own class threshold individually.

Where in code: detector.py → class_conf_thresholds dict + per-box filtering
Config: inference.yaml → model.class_thresholds

------------------------------------------------------------------------
TECHNIQUE 4 — BOX-UNION FUSION (Cross-Pass Deduplication)
------------------------------------------------------------------------
What it does:
  After the full-frame pass and the 4 tile passes all run, the same bird may
  appear in multiple results. Standard NMS would arbitrarily pick the smallest
  box (usually from a tile). Box-Union instead:

    1. Finds pairs of same-class boxes that overlap significantly
       (IoU > 0.45 OR one covers > 75% of the other)
    2. Merges them into their UNION — the outer bounding rectangle
    3. Keeps the HIGHEST confidence of the merged pair

  Example:
    Full-frame box : [100, 200, 300, 400]  (correct size, lower conf)
    Tile box       : [110, 210, 250, 380]  (slightly clipped, higher conf)
    Fused result   : [100, 200, 300, 400]  (largest extent + best conf)

Why it matters:
  Without fusion, tile-scale detections produce slightly smaller bounding boxes
  than the real bird size because tile boundaries clip the detection area.
  Union fusion guarantees the final box is always at least as large as the
  full-frame detection — giving accurate, correctly-sized bounding boxes.

Where in code: detector.py → predict() — FUSION block

------------------------------------------------------------------------
COMBINED EFFECT — LEVEL 1 vs LEVEL 3
------------------------------------------------------------------------
  Level 1 (Baseline YOLO, conf=0.15)          :  8–12 birds/frame
  Level 2 (CLAHE + Tiling, NMS dedup)         : 18–22 birds/frame
  Level 3 (CLAHE + Tiling + Box-Union Fusion) : 24–30 birds/frame

  Level 3 consistently finds 2–3x more birds than the original baseline
  and produces correctly-sized bounding boxes.

================================================================================
  PROJECT STRUCTURE
================================================================================

  Behaviour_analysis-main/
  │
  ├── configs/
  │   ├── inference.yaml     ← Main config (adjust source, thresholds, etc.)
  │   └── classes.yaml       ← Class definitions (Drinking, Eating, etc.)
  │
  ├── src/livestockify/
  │   ├── inference/
  │   │   ├── detector.py    ← Level 3 logic: CLAHE + Tiling + Box-Union
  │   │   ├── runner.py      ← Main loop: reads frames, runs detector, writes JSON
  │   │   ├── aggregator.py  ← Converts detections into JSON count records
  │   │   └── video_source.py← RTSP / file video source abstraction
  │   └── storage/
  │       └── json_writer.py ← Writes detection records to .jsonl files
  │
  ├── models/
  │   └── livestockify_v1.pt ← Trained YOLOv8 model (required)
  │
  ├── data/output/detections/← Output folder — JSONL files written here
  │
  ├── docs/
  │   ├── level3_techniques.md  ← Detailed technical documentation
  │   └── README.txt            ← This file
  │
  ├── scripts/
  │   ├── run_inference.sh   ← Shell helper for Linux/Pi
  │   └── test_with_video.sh ← Shell helper for video testing
  │
  └── test-1.mp4 … test-5.mp4   ← Test video files

================================================================================
  INSTALLATION
================================================================================

Step 1: Install dependencies
  pip install ultralytics loguru pyyaml psutil opencv-python

Step 2: Install the package in editable mode (required for imports to work)
  pip install -e .

  OR add src/ to PYTHONPATH manually:
  set PYTHONPATH=src     (Windows)
  export PYTHONPATH=src  (Linux/Mac)

Step 3: Confirm model file exists
  models/livestockify_v1.pt   ← must be present

================================================================================
  EXECUTION — HOW TO RUN
================================================================================

--- Option A: Test on a local video file ---

  python -m livestockify.inference.runner --source test-1.mp4
  python -m livestockify.inference.runner --source test-2.mp4
  python -m livestockify.inference.runner --source test-4_n.mp4

  The --source flag overrides whatever is set in inference.yaml.

--- Option B: Run a live RTSP stream ---

  python -m livestockify.inference.runner --source rtsp://user:pass@192.168.1.100:554/stream1

  The system auto-reconnects if the stream drops (up to 100 attempts).

--- Option C: Edit inference.yaml and run without --source ---

  Edit configs/inference.yaml:

    source:
      type: file              # change to "rtsp" for live stream
      file_path: test-1.mp4

    # OR for RTSP:
    source:
      type: rtsp
      rtsp_url: "rtsp://user:pass@192.168.1.100:554/stream1"

  Then run:
    python -m livestockify.inference.runner

--- Option D: Run from the scripts folder ---

  bash scripts/run_inference.sh       (Linux/Pi)
  bash scripts/test_with_video.sh     (Linux/Pi)

================================================================================
  OUTPUT FORMAT
================================================================================

Output file: data/output/detections/detections_YYYY-MM-DD.jsonl
Each line is one JSON record written once per processed frame.

Example record:
  {
    "timestamp":        "2026-04-20T09:30:00+00:00",
    "farm_id":          "saibabu",
    "farm_name":        "Saibabu Farm",
    "cam_id":           "cam1",
    "counts": {
      "Drinking":  3,
      "Eating":   12,
      "Sitting":   5,
      "Standing":  8
    },
    "total":            28,
    "percentages": {
      "Drinking": 10.71,
      "Eating":   42.86,
      "Sitting":  17.86,
      "Standing": 28.57
    },
    "avg_confidence":   0.1423,
    "inference_time_ms": 4231.5,
    "frame_index":       7,
    "cpu_usage":         82.3
  }

A new file is created each day (daily_rotation: true in config).
Records are appended — existing data is never overwritten.

================================================================================
  CONFIG REFERENCE — inference.yaml
================================================================================

  model:
    weights:          models/livestockify_v1.pt   # Path to model weights
    conf_threshold:   0.10    # Base confidence (fallback)
    iou_threshold:    0.45    # NMS IoU threshold
    imgsz:            640     # Model input size (do not change)
    device:           cpu     # cpu | cuda | mps
    enable_clahe:     true    # Level 3: lighting normalization
    class_thresholds:         # Level 3: per-class sensitivity
      Drinking: 0.08
      Eating:   0.08
      Sitting:  0.12
      Standing: 0.12

  tiling:
    enabled: true             # Level 3: sliced inference
    grid:    [2, 2]           # 2x2 tiles
    overlap: 0.25             # 25% tile border overlap

  source:
    type:      file           # "file" or "rtsp"
    file_path: test-1.mp4    # used when type=file
    rtsp_url:  "rtsp://..."  # used when type=rtsp

  sampling:
    interval_seconds: 5       # Process 1 frame every N seconds
    warmup_frames:    3       # Skip first N frames (camera stabilization)

  output:
    json_dir:        data/output/detections
    daily_rotation:  true

  farm:
    id:       saibabu
    name:     Saibabu Farm
    cam_id:   cam1
    location: Telangana, India

================================================================================
  CONSOLE OUTPUT (what you see while running)
================================================================================

  2026-04-20 09:30:00 | INFO     | ... — [1/4] Setting up video source...
  2026-04-20 09:30:01 | INFO     | ... — [2/4] Loading detector...
  2026-04-20 09:30:05 | INFO     | ... — [3/4] Creating aggregator...
  2026-04-20 09:30:05 | INFO     | ... — [4/4] Creating storage writer...
  2026-04-20 09:30:05 | INFO     | ... — Setup complete. Starting inference loop...
  2026-04-20 09:30:10 | INFO     | ... — Frame 1 | total=24 | Dr:3 | Ea:12 | Si:5 | St:4 | inference=4231ms | cpu=82%
  2026-04-20 09:30:15 | INFO     | ... — Frame 2 | total=27 | Dr:2 | Ea:14 | Si:6 | St:5 | inference=4180ms | cpu=79%
  ...

  Press Ctrl+C to stop at any time. Cleanup is automatic.

================================================================================
  STOPPING THE RUNNER
================================================================================

  - Video file:    Automatically stops when the file ends.
  - RTSP stream:   Press Ctrl+C — cleanup is handled gracefully.
  - Both modes:    Send SIGTERM to stop from another process (e.g. systemd).

================================================================================
