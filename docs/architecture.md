# LiveStockify Architecture

## High-Level Data Flow

```
┌─────────────────┐    RTSP    ┌──────────────────────┐    JSON    ┌──────────────┐
│   CCTV Camera   │ ─────────> │    Raspberry Pi 5    │ ────────> │  Local Disk  │
│  (Hikvision/    │            │                      │            │  (.jsonl)    │
│   Dahua/etc)    │            │  ┌────────────────┐ │            └──────┬───────┘
└─────────────────┘            │  │ video_source   │ │                   │
                               │  │     ↓          │ │                   │ (future)
                               │  │ detector       │ │                   ↓
                               │  │     ↓          │ │            ┌──────────────┐
                               │  │ aggregator     │ │            │  Cloud (S3 / │
                               │  │     ↓          │ │            │  REST API)   │
                               │  │ json_writer    │ │            └──────────────┘
                               │  └────────────────┘ │
                               └──────────────────────┘
```

## Component Responsibilities

### 1. `video_source.py` — Input Abstraction

**Job:** Hide the difference between RTSP streams and MP4 files behind a single interface.

**Why this matters:** The rest of the code shouldn't care whether frames come from a camera or a file. This lets us test on local videos and deploy on real cameras without changing any inference code.

**Key classes:**
- `VideoSource` (abstract base) — defines `read()` and `release()`
- `FileSource` — wraps `cv2.VideoCapture` for MP4 files
- `RTSPSource` — wraps `cv2.VideoCapture` for RTSP streams + auto-reconnect

### 2. `detector.py` — Model Wrapper

**Job:** Load YOLOv8 once at startup, run inference, return parsed `Detection` objects.

**Why this matters:** YOLOv8's raw output is a complex tensor structure. Parsing it inline everywhere would be ugly. The wrapper gives us clean Python dataclasses.

**Key classes:**
- `Detection` (dataclass) — one bounding box with class, confidence, coordinates
- `Detector` — loads model, exposes `predict(frame) -> List[Detection]`

### 3. `aggregator.py` — Detection Counter

**Job:** Convert raw detections into a structured count record with timestamp, percentages, and farm metadata.

**Why this matters:** This is the bridge between "raw model output" and "what the cloud actually wants." All the business logic about counting and structuring lives here.

**Key classes:**
- `CountRecord` (dataclass) — the final JSON shape (timestamp, counts, percentages, metadata)
- `Aggregator` — exposes `aggregate(detections, inference_time) -> CountRecord`

### 4. `runner.py` — Main Loop

**Job:** Orchestrate everything. Read frame → detect → aggregate → write → sleep → repeat.

**Why this matters:** This is the only file that knows about ALL the other components. Everything else is independent and unit-testable.

**Key classes:**
- `InferenceRunner` — owns the loop, handles signals (SIGINT/SIGTERM) for clean shutdown

### 5. `json_writer.py` — Storage

**Job:** Append count records to a daily JSONL file.

**Why JSONL?**
- Append-only (no parsing on write)
- Each line is a complete record (no incomplete writes if power dies)
- Easy to tail/stream
- Easy to upload to S3 as-is
- Easy to parse line-by-line later

### 6. `cloud_uploader.py` — Cloud Push (Stub)

**Job:** Eventually push records to S3 or a REST API.

**Why a stub?** We're keeping the architecture clean for the future. Right now we just write to disk. When the cloud endpoint is ready, we implement this class without touching anything else.

## Configuration Strategy

We use **YAML configs** in `configs/`, not Python files:

- `classes.yaml` — class definitions (rarely changes)
- `farms.yaml` — farm metadata (rarely changes)
- `inference.yaml` — runtime config (per-Pi, frequently changes)

**Why YAML?** Non-developers can edit it. You don't need to redeploy code to switch cameras or change confidence thresholds. SSH into the Pi, edit the YAML, restart the service.

## Data Flow Through One Frame

Let's trace a single frame from camera to disk:

```
1. RTSP camera produces frame
   ↓
2. RTSPSource.read() returns (True, np.ndarray)
   ↓
3. Detector.predict(frame) runs YOLOv8
   ↓ returns List[Detection]
4. Aggregator.aggregate(detections, inference_ms)
   ↓ counts per class, computes percentages
   ↓ returns CountRecord
5. JsonLinesWriter.write(record)
   ↓ appends one line to detections_2026-04-08.jsonl
6. Runner sleeps for 5 seconds
   ↓
7. Loop back to step 1
```

## What's Intentionally NOT Here

**Real-time video display.** The Pi has no monitor in production. Display is for debugging only.

**Web dashboard on the Pi.** We're keeping the Pi minimal. Visualization happens elsewhere (separate dashboard service that reads the JSONL files or queries the cloud).

**Multi-camera support on one Pi.** One Pi handles one camera. If you need 3 cameras, you need 3 Pis. This keeps the code simple and the Pi load predictable.

**Online learning / model updates from production.** The Pi is read-only ML. Training happens elsewhere on your Mac (or a GPU box). New models are deployed by `git pull` + `systemctl restart`.

## Future Extensions

When you're ready, the architecture supports adding:

1. **Cloud uploader** — Implement `cloud_uploader.py`. The runner already calls it after each write.
2. **Multiple cameras per Pi** — Refactor `runner.py` to spawn one thread per source.
3. **Tracking (not just detection)** — Add `tracker.py` between detector and aggregator.
4. **Alerts** — Add `alerter.py` that watches for anomalies (e.g., 0 birds eating for 30 minutes).
5. **Tiled inference** — Replace `detector.py` with a tiled version. The runner doesn't need to know.

The clean separation between components means each of these is a small, isolated change.
