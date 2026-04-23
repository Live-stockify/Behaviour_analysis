"""
Main inference runner.

Supports two modes:
1. SINGLE: One video source (file or RTSP) — for testing or single-camera deployment
2. MULTI: Cycle through multiple RTSP cameras — for production multi-camera farms

The runner loops continuously:
  read frame -> detect -> aggregate -> write -> next camera -> sleep
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from loguru import logger

from livestockify.inference.aggregator import Aggregator, CountRecord
from livestockify.inference.detector import Detector
from livestockify.inference.video_source import (
    VideoSource,
    FileSource,
    RTSPSource,
    create_source,
)
from livestockify.storage.json_writer import JsonLinesWriter


# ============================================================
# CONFIG LOADING
# ============================================================

def load_config(config_path: str | Path) -> dict:
    """Load and validate the inference YAML config."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    required_sections = ["model", "source", "sampling", "output", "farm"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Config missing required section: {section}")

    return config


def load_class_names(classes_yaml: str | Path) -> list[str]:
    """Load ordered class names from classes.yaml."""
    with open(classes_yaml, "r") as f:
        data = yaml.safe_load(f)

    classes = sorted(data["classes"], key=lambda c: c["id"])
    return [c["name"] for c in classes]


# ============================================================
# LOGGING SETUP
# ============================================================

def setup_logging(log_config: dict) -> None:
    """Configure loguru based on config."""
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
# MULTI-CAMERA SOURCE MANAGER
# ============================================================

class MultiCameraManager:
    """
    Manages multiple RTSP camera connections.

    Cycles through cameras in order: cam1 -> cam2 -> ... -> cam8 -> cam1 -> ...
    Handles individual camera failures without stopping the whole system.
    """

    def __init__(self, cameras_config: list[dict], reconnect_config: dict):
        self.cameras_config = [c for c in cameras_config if c.get("enabled", True)]
        self.reconnect_config = reconnect_config
        self.sources: Dict[str, Optional[RTSPSource]] = {}
        self.current_index = 0

        if not self.cameras_config:
            raise ValueError("No enabled cameras found in config")

        logger.info(f"MultiCameraManager: {len(self.cameras_config)} cameras configured")
        for cam in self.cameras_config:
            cam_id = cam["id"]
            url = cam.get("rtsp_url", "")
            if url:
                logger.info(f"  {cam_id}: {RTSPSource._mask_url(url)}")
            else:
                logger.warning(f"  {cam_id}: NO URL configured — will be skipped")

    def connect_all(self) -> int:
        """
        Attempt to connect to all cameras.
        Returns the number of successfully connected cameras.
        """
        connected = 0
        for cam in self.cameras_config:
            cam_id = cam["id"]
            url = cam.get("rtsp_url", "")

            if not url:
                self.sources[cam_id] = None
                continue

            try:
                source = RTSPSource(
                    rtsp_url=url,
                    reconnect_delay=self.reconnect_config.get("reconnect_delay_seconds", 5),
                    max_reconnects=self.reconnect_config.get("max_reconnect_attempts", 100),
                )
                self.sources[cam_id] = source
                connected += 1
            except Exception as e:
                logger.error(f"Failed to connect {cam_id}: {e}")
                self.sources[cam_id] = None

        logger.info(f"Connected to {connected}/{len(self.cameras_config)} cameras")
        return connected

    def get_next_frame(self) -> tuple[Optional[str], bool, Optional["np.ndarray"]]:
        """
        Read one frame from the next camera in the cycle.

        Returns:
            (cam_id, success, frame) — cam_id is None if no cameras are available
        """
        import numpy as np

        attempts = 0
        total_cameras = len(self.cameras_config)

        while attempts < total_cameras:
            cam_config = self.cameras_config[self.current_index]
            cam_id = cam_config["id"]

            # Advance index for next call (round-robin)
            self.current_index = (self.current_index + 1) % total_cameras

            source = self.sources.get(cam_id)
            if source is None:
                attempts += 1
                continue

            try:
                ret, frame = source.read()
                if ret and frame is not None:
                    return cam_id, True, frame
                else:
                    logger.warning(f"{cam_id}: failed to read frame")
                    attempts += 1
                    continue
            except Exception as e:
                logger.error(f"{cam_id}: error reading frame: {e}")
                attempts += 1
                continue

        # All cameras failed
        return None, False, None

    def release_all(self) -> None:
        """Disconnect from all cameras."""
        for cam_id, source in self.sources.items():
            if source is not None:
                try:
                    source.release()
                except Exception:
                    pass
        logger.info("All cameras disconnected")


# ============================================================
# MAIN LOOP
# ============================================================

class InferenceRunner:
    """
    The main inference loop.

    Supports single-camera and multi-camera modes.
    """

    def __init__(self, config: dict, class_names: list[str]):
        self.config = config
        self.class_names = class_names
        self.should_stop = False

        self.source: Optional[VideoSource] = None
        self.multi_cam: Optional[MultiCameraManager] = None
        self.detector: Optional[Detector] = None
        self.aggregators: Dict[str, Aggregator] = {}
        self.writer: Optional[JsonLinesWriter] = None

        self.mode = config["source"].get("mode", "single")

    def setup(self) -> None:
        """Initialize all components."""
        logger.info("=" * 60)
        logger.info("LiveStockify Inference Runner")
        logger.info(f"Mode: {self.mode.upper()}")
        logger.info("=" * 60)

        # Video source(s)
        if self.mode == "multi":
            logger.info("[1/4] Setting up multi-camera sources...")
            self.multi_cam = MultiCameraManager(
                cameras_config=self.config["source"].get("cameras", []),
                reconnect_config=self.config["source"],
            )
            connected = self.multi_cam.connect_all()
            if connected == 0:
                raise RuntimeError("No cameras could be connected. Check RTSP URLs.")

            # Create one aggregator per camera
            farm_cfg = self.config["farm"]
            for cam in self.config["source"]["cameras"]:
                if cam.get("enabled", True):
                    self.aggregators[cam["id"]] = Aggregator(
                        farm_id=farm_cfg["id"],
                        farm_name=farm_cfg["name"],
                        cam_id=cam["id"],
                        class_names=self.class_names,
                    )
        else:
            logger.info("[1/4] Setting up single video source...")
            self.source = create_source(self.config["source"])

            farm_cfg = self.config["farm"]
            self.aggregators["default"] = Aggregator(
                farm_id=farm_cfg["id"],
                farm_name=farm_cfg["name"],
                cam_id=farm_cfg.get("cam_id", "cam1"),
                class_names=self.class_names,
            )

        # Detector
        logger.info("[2/4] Loading detector...")
        model_cfg = self.config["model"]
        self.detector = Detector(
            weights_path=model_cfg["weights"],
            class_names=self.class_names,
            conf_threshold=model_cfg.get("conf_threshold", 0.15),
            iou_threshold=model_cfg.get("iou_threshold", 0.45),
            imgsz=model_cfg.get("imgsz", 640),
            device=model_cfg.get("device", "cpu"),
        )

        # Aggregator (already created above)
        logger.info("[3/4] Aggregators created")

        # Storage
        logger.info("[4/4] Creating storage writer...")
        out_cfg = self.config["output"]
        self.writer = JsonLinesWriter(
            output_dir=out_cfg["json_dir"],
            daily_rotation=out_cfg.get("daily_rotation", True),
        )

        logger.info("Setup complete. Starting inference loop...")

    def run(self) -> None:
        """Main inference loop — dispatches to single or multi mode."""
        if self.mode == "multi":
            self._run_multi()
        else:
            self._run_single()

    def _run_single(self) -> None:
        """Single camera inference loop."""
        sampling_cfg = self.config["sampling"]
        interval_seconds = sampling_cfg.get("interval_seconds", 5)
        warmup_frames = sampling_cfg.get("warmup_frames", 3)

        # Warmup
        for i in range(warmup_frames):
            ret, _ = self.source.read()
            if ret:
                logger.debug(f"Warmup frame {i+1}/{warmup_frames}")

        processed_count = 0
        aggregator = self.aggregators["default"]

        while not self.should_stop and self.source.is_opened:
            loop_start = time.time()

            ret, frame = self.source.read()
            if not ret or frame is None:
                if self.source.__class__.__name__ == "FileSource":
                    logger.info("End of file reached. Stopping.")
                    break
                time.sleep(1)
                continue

            # Inference
            inference_start = time.time()
            try:
                detections = self.detector.predict(frame)
            except Exception as e:
                logger.error(f"Inference failed: {e}")
                continue
            inference_ms = (time.time() - inference_start) * 1000

            # Aggregate + write
            record = aggregator.aggregate(detections, inference_ms)
            self._write_and_log(record)

            processed_count += 1

            # Sleep
            elapsed = time.time() - loop_start
            sleep_time = max(0, interval_seconds - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info(f"Loop ended. Processed {processed_count} frames total.")

    def _run_multi(self) -> None:
        """Multi-camera cycling inference loop."""
        sampling_cfg = self.config["sampling"]
        interval_seconds = sampling_cfg.get("interval_seconds", 1)

        processed_count = 0
        cycle_count = 0
        num_cameras = len([c for c in self.config["source"]["cameras"]
                          if c.get("enabled", True)])

        logger.info(
            f"Multi-camera mode: {num_cameras} cameras, "
            f"{interval_seconds}s between each, "
            f"~{num_cameras * (interval_seconds + 1)}s per full cycle"
        )

        while not self.should_stop:
            cam_id, success, frame = self.multi_cam.get_next_frame()

            if not success or frame is None:
                if cam_id is None:
                    logger.error("All cameras failed. Waiting 10s before retry...")
                    time.sleep(10)
                continue

            # Inference
            inference_start = time.time()
            try:
                detections = self.detector.predict(frame)
            except Exception as e:
                logger.error(f"[{cam_id}] Inference failed: {e}")
                continue
            inference_ms = (time.time() - inference_start) * 1000

            # Aggregate with the camera-specific aggregator
            aggregator = self.aggregators.get(cam_id)
            if aggregator is None:
                continue

            record = aggregator.aggregate(detections, inference_ms)
            self._write_and_log(record, cam_id)

            processed_count += 1

            # Track full cycles
            if processed_count % num_cameras == 0:
                cycle_count += 1
                if cycle_count % 10 == 0:  # Log every 10 cycles
                    logger.info(f"Completed {cycle_count} full camera cycles")

            # Sleep between cameras
            time.sleep(interval_seconds)

        logger.info(
            f"Loop ended. Processed {processed_count} frames "
            f"({cycle_count} full cycles)."
        )

    def _write_and_log(self, record: CountRecord, cam_prefix: str = "") -> None:
        """Write record to storage and log a summary."""
        try:
            self.writer.write(record)
        except Exception as e:
            logger.error(f"Failed to write record: {e}")

        counts_str = " | ".join(
            f"{k[:2]}:{v}" for k, v in record.counts.items()
        )
        cam_label = f"[{cam_prefix}] " if cam_prefix else ""
        logger.info(
            f"{cam_label}Frame {record.frame_index} | "
            f"total={record.total} | {counts_str} | "
            f"inference={record.inference_time_ms:.0f}ms"
        )

    def stop(self) -> None:
        """Signal the loop to stop."""
        logger.info("Stop signal received")
        self.should_stop = True

    def cleanup(self) -> None:
        """Release all resources."""
        logger.info("Cleaning up...")
        if self.source:
            self.source.release()
        if self.multi_cam:
            self.multi_cam.release_all()
        if self.writer:
            self.writer.close()
        logger.info("Cleanup complete. Goodbye.")


# ============================================================
# CLI ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="LiveStockify - Pi 5 inference runner"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/inference.yaml",
        help="Path to inference config YAML",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="configs/classes.yaml",
        help="Path to classes config YAML",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Override the video source (e.g., test.mp4 or rtsp://...)",
    )
    args = parser.parse_args()

    # Load configs
    config = load_config(args.config)
    class_names = load_class_names(args.classes)

    # CLI override for source (forces single mode)
    if args.source:
        config["source"]["mode"] = "single"
        if args.source.startswith("rtsp://"):
            config["source"]["type"] = "rtsp"
            config["source"]["rtsp_url"] = args.source
        else:
            config["source"]["type"] = "file"
            config["source"]["file_path"] = args.source

    # Setup logging
    setup_logging(config.get("logging", {}))

    # Create and run
    runner = InferenceRunner(config, class_names)

    # Handle Ctrl+C gracefully
    def handle_signal(sig, frame):
        runner.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        runner.setup()
        runner.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        runner.cleanup()


if __name__ == "__main__":
    main()