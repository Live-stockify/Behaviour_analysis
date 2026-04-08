#!/usr/bin/env bash
# Test inference on a local video file
# Usage: ./scripts/test_with_video.sh path/to/video.mp4

set -e

VIDEO_PATH="${1:-data/test_clips/saibabu_test.mp4}"

if [ ! -f "$VIDEO_PATH" ]; then
  echo "Error: video file not found: $VIDEO_PATH"
  echo "Usage: $0 path/to/video.mp4"
  exit 1
fi

echo "Running inference on: $VIDEO_PATH"
echo ""

python -m livestockify.inference.runner \
  --config configs/inference.yaml \
  --classes configs/classes.yaml \
  --source "$VIDEO_PATH"
