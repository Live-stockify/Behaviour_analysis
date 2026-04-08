# LiveStockify

AI-powered poultry behavioral monitoring system. Detects and classifies bird behaviors (eating, drinking, standing, sitting) from CCTV footage in real-time using YOLOv8.

## What This Does

LiveStockify watches a poultry farm via CCTV and tells you, at any moment, what percentage of birds are eating, drinking, standing, or sitting. This helps farmers detect health issues, feed shortages, and stress events early.

The system runs on a Raspberry Pi 5 inside the farm, processes the camera feed locally, and reports behavioral counts to the cloud.

## Architecture

```
[CCTV Camera] --RTSP--> [Pi 5 Inference] --JSON--> [Cloud Storage]
                              |
                       [YOLOv8 Model]
                              |
                      Count birds per behavior
                              |
            {timestamp, eating: 12, drinking: 3,
             standing: 28, sitting: 15, total: 58}
```

## Repository Structure

```
livestockify/
├── configs/                    # YAML config files (farms, classes, runtime)
├── models/                     # Trained model weights (committed)
├── src/livestockify/           # Main Python package
│   ├── data/                   # Data prep utilities (training-side)
│   ├── training/               # Model training scripts
│   ├── inference/              # Pi-side inference engine
│   └── storage/                # Output handlers (JSON, future cloud)
├── scripts/                    # CLI entry points
├── tests/                      # Unit tests
└── docs/                       # Documentation
```

## Quick Start — Inference on Pi 5

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/livestockify.git
cd livestockify

# 2. Install inference dependencies (lightweight, no training tools)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements/inference.txt
pip install -e .

# 3. Edit the inference config with your camera details
nano configs/inference.yaml

# 4. Run inference on a test video
python -m livestockify.inference.runner --config configs/inference.yaml --source path/to/test.mp4

# 5. Or run on the live RTSP stream
python -m livestockify.inference.runner --config configs/inference.yaml
```

## Quick Start — Training on Mac

```bash
# 1. Install training dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements/train.txt
pip install -e .

# 2. Prepare dataset (assumes you have raw_clips/ from CCTV downloads)
python -m livestockify.data.extract_frames \
    --input-dir ./raw_clips \
    --output-dir ./extracted_frames \
    --log-dir ./logs

python -m livestockify.data.sample_frames \
    --frames-dir ./extracted_frames \
    --extraction-log ./logs/extraction_log.csv \
    --output-dir ./sampled_frames \
    --log-dir ./logs \
    --target 1000

# 3. After labeling in Roboflow, train
python -m livestockify.training.train \
    --dataset ./round2_dataset/data.yaml \
    --epochs 100 \
    --device cpu
```

## Model Status

| Version | mAP50 | Notes |
|---------|-------|-------|
| v1 (current) | 0.454 | Trained on 730 Saibabu farm frames |

See `models/README.md` for more details on the production model.

## Documentation

- [Pi Setup Guide](docs/pi_setup.md) — How to set up the Raspberry Pi 5
- [Training Guide](docs/training.md) — How to train a new model
- [Architecture](docs/architecture.md) — System design and data flow

## License

Private — Kuppi Smart Solutions.
