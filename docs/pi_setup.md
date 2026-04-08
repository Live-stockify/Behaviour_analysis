# Raspberry Pi 5 Setup Guide

This guide walks you through setting up the LiveStockify inference service on a Raspberry Pi 5.

## Hardware Requirements

- **Raspberry Pi 5** (4GB or 8GB RAM — 8GB recommended)
- **MicroSD card** (32GB minimum, 64GB recommended, Class 10 or better)
- **Active cooling** (heatsink + fan) — inference will heat the CPU
- **Power supply** (official 27W USB-C PSU recommended)
- **Network connectivity** (Ethernet preferred over WiFi for RTSP stability)

## Performance Expectations

| Model | Input Size | Inference Time (Pi 5 CPU) | Frames/sec |
|-------|-----------|--------------------------|------------|
| YOLOv8n | 640×640 | ~400 ms | ~2.5 fps |
| YOLOv8s | 640×640 | ~900 ms | ~1.1 fps |

The Pi is **NOT real-time**. We process 1 frame every 5 seconds, which is plenty for behavioral monitoring.

## Step 1: Flash Raspberry Pi OS

1. Download [Raspberry Pi Imager](https://www.raspberrypi.com/software/)
2. Flash **Raspberry Pi OS (64-bit)** — Bookworm or newer
3. Pre-configure SSH, WiFi, hostname, and user via the imager's settings
4. Boot the Pi and SSH in

## Step 2: System Updates

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git ffmpeg libatlas-base-dev
```

## Step 3: Clone the Repository

```bash
cd ~
git clone https://github.com/YOUR_USERNAME/livestockify.git
cd livestockify
```

## Step 4: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

## Step 5: Install Inference Dependencies

```bash
# Install ARM-optimized PyTorch first (saves memory)
pip install --upgrade pip
pip install -r requirements/inference.txt
pip install -e .
```

This takes ~10–15 minutes on Pi 5. Be patient.

## Step 6: Verify the Model Loads

```bash
python -c "
from ultralytics import YOLO
model = YOLO('models/livestockify_v1.pt')
print('Model loaded successfully')
print(f'Classes: {model.names}')
"
```

You should see all 4 class names listed.

## Step 7: Configure Your Camera

Edit `configs/inference.yaml`:

```yaml
source:
  type: rtsp
  rtsp_url: rtsp://username:password@192.168.1.100:554/stream1
```

Find your camera's RTSP URL from its documentation. Common patterns:
- Hikvision: `rtsp://user:pass@IP:554/Streaming/Channels/101`
- Dahua: `rtsp://user:pass@IP:554/cam/realmonitor?channel=1&subtype=0`
- TP-Link: `rtsp://user:pass@IP:554/stream1`

Also update the farm metadata:

```yaml
farm:
  id: saibabu
  name: Saibabu Farm
  cam_id: cam1
  location: Telangana, India
```

## Step 8: Test with a Local Video First

Before connecting to the camera, test that everything works with a test MP4:

```bash
# Copy a test video to the Pi (from your Mac)
scp test.mp4 pi@your-pi-ip:~/livestockify/data/test.mp4

# Run inference on it
./scripts/test_with_video.sh data/test.mp4
```

You should see log lines like:
```
Frame 1 | total=18 | D:0 | E:6 | Si:5 | St:7 | inference=890ms
```

Check that `data/output/detections/detections_2026-04-08.jsonl` is being written.

## Step 9: Test with the RTSP Stream

Once the file test works:

```bash
# Make sure config has source.type: rtsp
./scripts/run_inference.sh
```

If it connects, you'll see:
```
RTSP stream connected: rtsp://***@192.168.1.100:554/stream1
```

## Step 10: Install as a System Service

So the inference runs automatically on boot and restarts on failure:

```bash
sudo cp scripts/livestockify.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable livestockify
sudo systemctl start livestockify
```

Check it's running:

```bash
sudo systemctl status livestockify
```

View live logs:

```bash
sudo journalctl -u livestockify -f
```

## Step 11: Verify Detection Records

```bash
# Tail the detection records
tail -f data/output/detections/detections_$(date +%Y-%m-%d).jsonl

# Count records from today
wc -l data/output/detections/detections_$(date +%Y-%m-%d).jsonl
```

## Troubleshooting

### "Cannot connect to RTSP stream"

- Verify the camera IP is reachable: `ping 192.168.1.100`
- Test the URL with VLC first
- Check username/password
- Some cameras need port forwarding configured
- Ethernet is more reliable than WiFi for RTSP

### High inference latency (>2 seconds)

- Make sure active cooling is installed
- Check CPU temperature: `vcgencmd measure_temp`
- If temp > 75°C, the Pi is throttling — add cooling
- Consider reducing `imgsz` from 640 to 480

### Out of memory

- Check usage: `free -h`
- Reduce `imgsz` in `configs/inference.yaml`
- Stop other services on the Pi
- Consider switching to YOLOv8n (smaller model)

### Service won't start

- Check logs: `sudo journalctl -u livestockify -n 50`
- Run manually first: `./scripts/run_inference.sh`
- Verify all paths in `livestockify.service` match your install location

## Updating

```bash
cd ~/livestockify
git pull
source venv/bin/activate
pip install -r requirements/inference.txt
sudo systemctl restart livestockify
```
