#!/usr/bin/env bash
# Run inference in production mode (uses RTSP from config)
# Designed to be called by systemd on the Pi 5

set -e

# Activate venv if present
if [ -d "venv" ]; then
  source venv/bin/activate
fi

# Make sure log dir exists
mkdir -p logs

# Run forever (systemd will restart on failure)
exec python -m livestockify.inference.runner \
  --config configs/inference.yaml \
  --classes configs/classes.yaml
