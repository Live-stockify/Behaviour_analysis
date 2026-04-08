# Training Guide

How to train a new LiveStockify model from scratch (or improve an existing one).

## Prerequisites

- macOS or Linux machine (NOT the Pi — training is too slow on Pi)
- Python 3.9+
- ~10 GB disk space for training data + intermediate files
- Patience (training takes 4–8 hours on M-series Mac CPU)

## Step 1: Install Training Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements/train.txt
pip install -e .
```

## The Full Training Pipeline

```
Raw CCTV clips
      ↓
[1] Frame extraction      → extracted_frames/
      ↓
[2] Strategic sampling    → sampled_frames/  (~1000 frames)
      ↓
[3] Annotation in Roboflow (manual or semi-automated)
      ↓
[4] Export from Roboflow  → roboflow_export/
      ↓
[5] Convert + train       → runs/detect/round_N/weights/best.pt
      ↓
[6] Promote to production → models/livestockify_vN.pt
```

## Step 2: Frame Extraction

```bash
python -m livestockify.data.extract_frames \
  --input-dir ./raw_clips \
  --output-dir ./extracted_frames \
  --log-dir ./logs
```

This extracts 1 frame per 10 seconds from each MP4. Roughly 60 frames per 10-min clip.

The script auto-skips files that fail validation (date mismatches, duplicates, mislabeled NIT clips). Check `logs/skipped_files.csv` to see what was skipped.

## Step 3: Strategic Sampling

```bash
python -m livestockify.data.sample_frames \
  --frames-dir ./extracted_frames \
  --extraction-log ./logs/extraction_log.csv \
  --output-dir ./sampled_frames \
  --log-dir ./logs \
  --target 1000
```

This picks ~1000 frames using stratified sampling: balanced across cameras, days, growth stages, and clip types. Morning clips (MOR1/MOR2) get higher weight because they capture peak eating behavior.

## Step 4: Annotation in Roboflow

1. Upload `sampled_frames/` to Roboflow (Object Detection project)
2. Annotate the 4 classes: Drinking, Eating, Sitting, Standing
3. Use the GREEN/YELLOW/RED zone rules from the labeling guidelines
4. Export as **YOLOv8** format

For faster labeling, use the bootstrap workflow:
1. Label 30–50 frames manually
2. Train a quick model (~10 min)
3. Use it to pre-annotate the rest
4. Correct the pre-annotations (3–5x faster than from scratch)

## Step 5: Convert and Train

After exporting from Roboflow:

```bash
# Unzip the export
mkdir -p round2_dataset
unzip ~/Downloads/roboflow-export.zip -d round2_dataset/

# Train
python -m livestockify.training.train \
  --dataset round2_dataset/data.yaml \
  --epochs 100 \
  --device cpu
```

Training output goes to `runs/detect/round_N/weights/best.pt`.

### Training Hyperparameters

These are baked into `train.py`:

| Parameter | Value | Why |
|-----------|-------|-----|
| Model | YOLOv8s | Best size/accuracy tradeoff |
| Image size | 640 | Standard YOLOv8 resolution |
| Batch size | 4–8 | Limited by CPU memory |
| Epochs | 100 | Early stopping at 20 epochs no improvement |
| `flipud` | 0.0 | Birds don't appear upside down |
| `degrees` | 0.0 | Camera is fixed, no rotation |
| `fliplr` | 0.5 | Horizontal flip is fine |
| `mosaic` | 1.0 | Helps with small datasets |
| `scale` | 0.4 | Moderate zoom variation |

## Step 6: Evaluate the Model

```bash
python -c "
from ultralytics import YOLO
model = YOLO('runs/detect/round_N/weights/best.pt')
results = model.val(data='round2_dataset/data.yaml')
print(results.box.maps)  # per-class mAP50-95
"
```

Compare against the previous model in `models/README.md`. Only promote if the new model is meaningfully better on:
- Overall mAP50
- Per-class recall (especially Drinking, which has been weakest)

## Step 7: Promote to Production

```bash
# Copy the new model to the models/ directory with a versioned name
cp runs/detect/round_N/weights/best.pt models/livestockify_v2.pt

# Update models/README.md with new stats

# Update inference config to point to it
sed -i '' 's|livestockify_v1.pt|livestockify_v2.pt|' configs/inference.yaml

# Commit
git add models/livestockify_v2.pt models/README.md configs/inference.yaml
git commit -m "Add v2 model: trained on 1500 frames, mAP50 0.55"
git push

# On the Pi
ssh pi@your-pi
cd livestockify
git pull
sudo systemctl restart livestockify
```

## Tips for Better Models

1. **More data > better hyperparameters.** If your model is at 0.45 mAP50, don't tune learning rates — go label more frames.

2. **Class balance matters.** If Drinking is 2% of your labels, the model will struggle with it. Either oversample drinking-rich frames or augment them.

3. **Bootstrap loop is your friend.** Don't try to label 1000 frames perfectly upfront. Label 50 → train → predict → correct → repeat. Each round is 3–5x faster than the last.

4. **Check failure modes visually.** After training, run the model on 10 random frames and look at the predictions. You'll learn more in 5 minutes than from any metric.

5. **Don't trust mAP alone.** A model with 0.6 mAP50 but 0.2 recall on Drinking is useless for a drinking-focused dashboard. Look at per-class performance.
