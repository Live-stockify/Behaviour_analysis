"""
LiveStockify Data Pusher
=========================
Runs detection on all cameras periodically and pushes results
to the backend API in the required payload format.

This runs as a background service on the Pi 5.

Usage:
    python -m livestockify.api.pusher --config configs/inference.yaml

    # Or with custom interval:
    python -m livestockify.api.pusher --config configs/inference.yaml --interval 300
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import requests
import yaml
from loguru import logger

from livestockify.inference.detector import Detector


# ============================================================
# CONFIG
# ============================================================

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_class_names(classes_path: str) -> List[str]:
    with open(classes_path, "r") as f:
        data = yaml.safe_load(f)
    classes = sorted(data["classes"], key=lambda c: c["id"])
    return [c["name"] for c in classes]


# ============================================================
# CAMERA FRAME CAPTURE + DETECTION
# ============================================================

def grab_frame(rtsp_url: str) -> Optional[np.ndarray]:
    """Grab a single frame from an RTSP URL."""
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        cap.release()
        return None
    for _ in range(3):
        cap.grab()
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def detect_camera(detector: Detector, cam_config: dict, class_names: List[str]) -> dict:
    """Run detection on a single camera and return result in backend format."""
    empty_counts = {name: 0 for name in class_names}
    empty_pcts = {name: 0.0 for name in class_names}

    rtsp_url = cam_config.get("rtsp_url", "")
    if not rtsp_url:
        return {
            "status": "failed",
            "error": "No RTSP URL configured",
            "inference_time_ms": 0,
            "total": 0,
            "counts": empty_counts,
            "percentages": empty_pcts,
        }

    try:
        frame = grab_frame(rtsp_url)
    except Exception as e:
        return {
            "status": "failed",
            "error": f"RTSP connection failed: {str(e)}",
            "inference_time_ms": 0,
            "total": 0,
            "counts": empty_counts,
            "percentages": empty_pcts,
        }

    if frame is None:
        return {
            "status": "failed",
            "error": "Failed to grab frame from camera",
            "inference_time_ms": 0,
            "total": 0,
            "counts": empty_counts,
            "percentages": empty_pcts,
        }

    inference_start = time.time()
    try:
        detections = detector.predict(frame)
    except Exception as e:
        return {
            "status": "failed",
            "error": f"Inference failed: {str(e)}",
            "inference_time_ms": 0,
            "total": 0,
            "counts": empty_counts,
            "percentages": empty_pcts,
        }
    inference_ms = (time.time() - inference_start) * 1000

    counts = {name: 0 for name in class_names}
    for det in detections:
        counts[det.class_name] = counts.get(det.class_name, 0) + 1

    total = len(detections)
    percentages = {
        name: round((cnt / total * 100), 1) if total > 0 else 0.0
        for name, cnt in counts.items()
    }

    return {
        "status": "ok",
        "inference_time_ms": round(inference_ms, 0),
        "total": total,
        "counts": counts,
        "percentages": percentages,
    }


# ============================================================
# BUILD PAYLOAD IN BACKEND FORMAT
# ============================================================

def build_payload(
    config: dict,
    detector: Detector,
    class_names: List[str],
) -> dict:
    """
    Run detection on all farms/cameras and build the payload
    in the exact format the backend API expects.
    """
    start_time = time.time()

    pi_id = config.get("pusher", {}).get("pi_id", "PI-001")
    api_key = config.get("pusher", {}).get("backend_api_key", "pidata2026")
    farms_config = config.get("farms", [])

    farms_payload = []

    for farm in farms_config:
        farm_name = farm.get("name_backend", farm.get("id", "unknown"))
        cameras = farm.get("cameras", [])

        cameras_payload = {}
        cameras_ok = 0
        cameras_failed = 0
        farm_total = 0
        farm_counts = {name: 0 for name in class_names}

        for cam in cameras:
            if not cam.get("enabled", True):
                continue

            cam_id = cam["id"]
            logger.info(f"  [{farm_name}] Processing {cam_id}...")

            result = detect_camera(detector, cam, class_names)
            cameras_payload[cam_id] = result

            if result["status"] == "ok":
                cameras_ok += 1
                farm_total += result["total"]
                for name, cnt in result["counts"].items():
                    farm_counts[name] = farm_counts.get(name, 0) + cnt
                logger.info(
                    f"  [{farm_name}] {cam_id}: {result['total']} birds | "
                    f"{result['inference_time_ms']:.0f}ms"
                )
            else:
                cameras_failed += 1
                logger.warning(f"  [{farm_name}] {cam_id}: {result.get('error', 'unknown error')}")

        # Farm-level percentages
        farm_percentages = {
            name: round((cnt / farm_total * 100), 1) if farm_total > 0 else 0.0
            for name, cnt in farm_counts.items()
        }

        farm_entry = {
            "farm_name": farm_name,
            "cameras": cameras_payload,
            "summary": {
                "total_birds": farm_total,
                "counts": farm_counts,
                "percentages": farm_percentages,
                "cameras_ok": cameras_ok,
                "cameras_failed": cameras_failed,
            },
        }

        farms_payload.append(farm_entry)

    processing_time = (time.time() - start_time) * 1000

    payload = {
        "api_key": api_key,
        "pi_id": pi_id,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "processing_time_ms": round(processing_time, 0),
        "farms": farms_payload,
    }

    return payload


# ============================================================
# PUSH TO BACKEND
# ============================================================

def push_to_backend(payload: dict, backend_url: str, timeout: int = 30) -> bool:
    """
    POST the payload to the backend API.
    Returns True on success, False on failure.
    """
    try:
        response = requests.post(
            backend_url,
            json=payload,
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 200:
            logger.info(f"Push successful: HTTP {response.status_code}")
            return True
        else:
            logger.warning(
                f"Push failed: HTTP {response.status_code} — {response.text[:200]}"
            )
            return False

    except requests.exceptions.Timeout:
        logger.error(f"Push timeout after {timeout}s to {backend_url}")
        return False
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Push connection error: {e}")
        return False
    except Exception as e:
        logger.error(f"Push unexpected error: {e}")
        return False


# ============================================================
# MAIN LOOP
# ============================================================

class DataPusher:
    """
    Main loop: detect → build payload → push → sleep → repeat.
    """

    def __init__(self, config: dict, class_names: List[str]):
        self.config = config
        self.class_names = class_names
        self.should_stop = False
        self.detector: Optional[Detector] = None

    def setup(self) -> None:
        logger.info("=" * 60)
        logger.info("LiveStockify Data Pusher")
        logger.info("=" * 60)

        pusher_cfg = self.config.get("pusher", {})
        logger.info(f"  Pi ID:        {pusher_cfg.get('pi_id', 'PI-001')}")
        logger.info(f"  Backend URL:  {pusher_cfg.get('backend_url', 'NOT SET')}")
        logger.info(f"  Interval:     {pusher_cfg.get('interval_seconds', 300)}s")

        # Count farms and cameras
        farms = self.config.get("farms", [])
        total_cams = sum(
            len([c for c in f.get("cameras", []) if c.get("enabled", True) and c.get("rtsp_url", "")])
            for f in farms
        )
        logger.info(f"  Farms:        {len(farms)}")
        logger.info(f"  Cameras:      {total_cams}")

        # Load model
        model_cfg = self.config["model"]
        logger.info(f"  Model:        {model_cfg['weights']}")

        self.detector = Detector(
            weights_path=model_cfg["weights"],
            class_names=self.class_names,
            conf_threshold=model_cfg.get("conf_threshold", 0.15),
            iou_threshold=model_cfg.get("iou_threshold", 0.45),
            imgsz=model_cfg.get("imgsz", 640),
            device=model_cfg.get("device", "cpu"),
        )

        logger.info("Setup complete. Starting push loop...")

    def run(self) -> None:
        pusher_cfg = self.config.get("pusher", {})
        interval = pusher_cfg.get("interval_seconds", 300)
        backend_url = pusher_cfg.get("backend_url", "")

        if not backend_url:
            logger.error("No backend_url configured in pusher section. Exiting.")
            return

        cycle_count = 0
        push_success = 0
        push_fail = 0

        while not self.should_stop:
            cycle_count += 1
            logger.info(f"\n--- Cycle {cycle_count} ---")

            # Build payload (runs detection on all cameras)
            try:
                payload = build_payload(self.config, self.detector, self.class_names)
            except Exception as e:
                logger.error(f"Failed to build payload: {e}")
                time.sleep(interval)
                continue

            # Log summary
            total_birds = sum(
                f["summary"]["total_birds"] for f in payload["farms"]
            )
            total_ok = sum(
                f["summary"]["cameras_ok"] for f in payload["farms"]
            )
            total_failed = sum(
                f["summary"]["cameras_failed"] for f in payload["farms"]
            )
            logger.info(
                f"Detection done: {total_birds} birds, "
                f"{total_ok} cameras OK, {total_failed} failed, "
                f"{payload['processing_time_ms']:.0f}ms"
            )

            # Push to backend
            success = push_to_backend(payload, backend_url)
            if success:
                push_success += 1
            else:
                push_fail += 1

            logger.info(
                f"Push stats: {push_success} success, {push_fail} failed "
                f"(total cycles: {cycle_count})"
            )

            # Sleep until next cycle
            logger.info(f"Next push in {interval}s...")
            sleep_start = time.time()
            while not self.should_stop and (time.time() - sleep_start) < interval:
                time.sleep(1)

    def stop(self) -> None:
        logger.info("Stop signal received")
        self.should_stop = True


# ============================================================
# LOGGING
# ============================================================

def setup_logging(log_config: dict) -> None:
    logger.remove()

    logger.add(
        sys.stderr,
        level=log_config.get("level", "INFO"),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan> - <level>{message}</level>",
    )

    log_file = log_config.get("log_file")
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level=log_config.get("level", "INFO"),
            rotation=log_config.get("rotation", "10 MB"),
            retention=log_config.get("retention", "7 days"),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} - {message}",
        )


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="LiveStockify Data Pusher")
    parser.add_argument("--config", "-c", default="configs/inference.yaml")
    parser.add_argument("--classes", default="configs/classes.yaml")
    parser.add_argument("--interval", type=int, default=None,
                        help="Override push interval in seconds")

    args = parser.parse_args()

    config = load_config(args.config)
    class_names = load_class_names(args.classes)

    # CLI override for interval
    if args.interval:
        if "pusher" not in config:
            config["pusher"] = {}
        config["pusher"]["interval_seconds"] = args.interval

    setup_logging(config.get("logging", {}))

    pusher = DataPusher(config, class_names)

    def handle_signal(sig, frame):
        pusher.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        pusher.setup()
        pusher.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        logger.info("Data pusher stopped. Goodbye.")


if __name__ == "__main__":
    main()
