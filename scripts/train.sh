#!/usr/bin/env bash
# Train a new YOLOv8 model
# Usage: ./scripts/train.sh path/to/data.yaml [epochs]

set -e

DATA_YAML="${1:-round2_dataset/data.yaml}"
EPOCHS="${2:-100}"

if [ ! -f "$DATA_YAML" ]; then
  echo "Error: dataset YAML not found: $DATA_YAML"
  echo "Usage: $0 path/to/data.yaml [epochs]"
  exit 1
fi

echo "Training with:"
echo "  Dataset: $DATA_YAML"
echo "  Epochs:  $EPOCHS"
echo ""

python -m livestockify.training.train \
  --dataset "$DATA_YAML" \
  --epochs "$EPOCHS" \
  --device cpu
