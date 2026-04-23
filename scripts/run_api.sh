#!/usr/bin/env bash
# Start the LiveStockify API server
# Usage: ./scripts/run_api.sh [--port 8000] [--api-key YOUR_KEY]

set -e

# Activate venv if present
if [ -d "venv" ]; then
  source venv/bin/activate
fi

# Make sure log dir exists
mkdir -p logs

# Default API key from env or fallback
API_KEY="${LIVESTOCKIFY_API_KEY:-change-me-in-production}"

exec python -m livestockify.api.server \
  --config configs/inference.yaml \
  --classes configs/classes.yaml \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key "$API_KEY" \
  "$@"
