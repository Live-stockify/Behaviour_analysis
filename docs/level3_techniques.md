# Level 3 Detection Techniques — Why They Matter

## Overview

The production pipeline (`detector.py`) uses **Level 3 Hybrid Detection**, a 4-stage
algorithm that dramatically increases the number of birds detected per frame compared
to a plain single-pass YOLO call.

Each technique targets a specific real-world problem seen in farm footage.

---

## Technique 1 — CLAHE (Contrast Limited Adaptive Histogram Equalization)

**Where in code:** `detector.py → apply_clahe()` | Config: `enable_clahe: true`

### What it does
Applies adaptive contrast enhancement **only on the luminance channel** (LAB color
space). It boosts local contrast in dark or unevenly-lit areas without washing out
bright regions.

### Why it matters for bird detection
Farm footage often has:
- Harsh overhead lighting creating hotspots and deep shadows
- Birds standing in shaded corners that appear nearly invisible to the model
- Low-contrast environments (dusty air, overcast sky)

CLAHE normalizes these conditions **before the model ever sees the frame**, making
every bird equally visible regardless of where in the shed it stands.

### Settings used
```yaml
clipLimit: 2.0        # How aggressive the contrast boost is (higher = stronger)
tileGridSize: (8, 8)  # Local region size — smaller = more adaptive
```

### Impact observed
Birds in dark corners that scored 0.04–0.07 confidence (below any threshold)
were pushed to 0.10–0.15+ after CLAHE, becoming detectable.

---

## Technique 2 — 2×2 Tiled Inference (Sliced Detection)

**Where in code:** `detector.py → predict()` — Pass 2 | Config: `tiling.enabled: true, grid: [2,2]`

### What it does
The full frame is split into **4 overlapping tiles** (top-left, top-right,
bottom-left, bottom-right), and the model runs separately on each tile.
Results are then translated back to full-frame coordinates.

```
+----------+----------+
|  Tile 00 |  Tile 01 |
|          |          |
+----------+----------+
|  Tile 10 |  Tile 11 |
|          |          |
+----------+----------+
         ↑ 25% overlap between adjacent tiles
```

### Why it matters for bird detection
YOLOv8's input is always resized to 640×640. When a 1920×1080 farm frame is
crammed into 640px, small birds (which may only be 20–40 pixels wide) get
**downscaled to 8–16 pixels** — below the model's reliable detection range.

Tiling gives each bird **4× more pixels** within the model's field of view,
making small/distant birds detectable.

### 25% overlap prevents missed detections at tile boundaries
A bird sitting on the border between two tiles appears fully in **both** tiles.
Without overlap, it could fall between the cracks entirely.

### Impact observed
+40–80% more birds detected per frame compared to full-frame only pass,
especially birds in the far corners or background of wide-angle shots.

---

## Technique 3 — Per-Class Confidence Thresholds

**Where in code:** `detector.py → class_conf_thresholds` | Config: `model.class_thresholds`

### What it does
Instead of one global confidence threshold, each behavior class gets its own
minimum confidence value.

```yaml
class_thresholds:
  Drinking: 0.08   # Lower — drinking birds are often partially occluded
  Eating:   0.08   # Lower — beak-down posture is subtle
  Sitting:  0.12   # Higher — sitting is visually clearer
  Standing: 0.12   # Higher — standing is the most common / easiest pose
```

### Why it matters
The model is not equally confident across all behaviors. Drinking and Eating
are **physically subtle** (small posture changes, beak direction) and the model
systematically underpredicts them with a single global threshold.

Lowering the threshold **only for subtle classes** recovers real detections
without opening the door to false positives for the already-easy classes.

### How the model call uses it
The model is called with `conf = min(all thresholds) = 0.08` to catch every
possible detection. Then each box is **individually filtered** against its
class-specific threshold — not discarded at model level.

---

## Technique 4 — Box-Union Fusion (Cross-Pass Deduplication)

**Where in code:** `detector.py → predict()` — FUSION block

### What it does
After the full-frame pass and the tiled pass both run, we have **duplicate
detections** for the same bird (one from each pass). Standard NMS would
arbitrarily keep the smaller tile-pass box, discarding the full-frame box.

Box-Union instead:
1. Groups boxes of the same class that overlap significantly (IoU > 0.45 OR
   one covers > 75% of the other)
2. **Merges them into their union** — the outer bounding rectangle of both boxes
3. Takes the **highest confidence** of the merged pair

```
Full-frame box:  [100, 200, 300, 400]   (correctly sized but lower conf)
Tile box:        [110, 210, 250, 380]   (slightly cropped, higher conf)
                          ↓ Union
Merged box:      [100, 200, 300, 400]   (largest extent, best conf)
```

### Why it matters — the "small box" problem
Standard NMS between tile and full detections of the same bird tends to pick the
**tile-scale box** which, after coordinate translation, is correctly placed but
**slightly smaller** than the true bird outline (because the tile boundary clipped it).

Union fusion **fixes the box scale** — the merged box is always at least as large
as the full-frame detection, giving accurate bounding boxes.

### Fusion thresholds
```python
iou_score   > 0.45   # Standard overlap check
overlap_ratio > 0.75  # One box almost entirely inside another
```

---

## Combined Effect — Level 1 → Level 3 Comparison

| Technique | Birds Found (example frame) | Notes |
|---|---|---|
| Level 1: Baseline YOLO (conf=0.15) | ~8–12 | Misses dark, small, subtle birds |
| Level 2: CLAHE + Tiling (NMS fusion) | ~18–22 | More birds but small boxes |
| **Level 3: CLAHE + Tiling + Union Fusion** | **~24–30** | Full coverage, correct box sizes |

> Numbers are approximate and depend on video/frame content.
> In dense enclosures, Level 3 consistently finds **2–3× more birds** than Level 1.

---

## How to Test (Video or Live Stream)

### Test on a local video file
```powershell
# From project root
python -m livestockify.inference.runner --source test-1.mp4

# Or specify any video
python -m livestockify.inference.runner --source test-4_n.mp4
```

### Test on a live RTSP stream
```powershell
python -m livestockify.inference.runner --source rtsp://username:password@192.168.1.100:554/stream1
```

### Or edit the config and run without --source flag
```yaml
# configs/inference.yaml
source:
  type: rtsp                          # ← change to rtsp for live stream
  rtsp_url: "rtsp://...your url..."
```
```powershell
python -m livestockify.inference.runner
```

### Output
Results are written to `data/output/detections/detections_YYYY-MM-DD.jsonl`.
Each line is one JSON record:
```json
{
  "timestamp": "2026-04-20T09:30:00+00:00",
  "farm_id": "saibabu",
  "cam_id": "cam1",
  "counts": {"Drinking": 3, "Eating": 12, "Sitting": 5, "Standing": 8},
  "total": 28,
  "percentages": {"Drinking": 10.71, "Eating": 42.86, "Sitting": 17.86, "Standing": 28.57},
  "avg_confidence": 0.1423,
  "inference_time_ms": 4231.5,
  "frame_index": 7,
  "cpu_usage": 82.3
}
```
