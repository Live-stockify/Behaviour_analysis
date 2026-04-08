"""
Main inference runner.

This is the entry point that ties everything together:
1. Load config
2. Create video source (RTSP or file)
3. Create detector (YOLOv8)
4. Create aggregator
5. Create storage writer
6. Loop: read frame -> detect -> aggregate -> write -> sleep
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger

from livestockify.inference.aggregator import Aggregator
from livestockify.inference.detector import Detector
from livestockify.inference.video_source import VideoSource, create_source
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
    
    # Basic validation
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
    logger.remove()  # Remove default handler
    
    # Console output
    logger.add(
        sys.stderr,
        level=log_config.get("level", "INFO"),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan> - <level>{message}</level>",
    )
    
    # File output
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
# MAIN LOOP
# ============================================================

class InferenceRunner:
    """
    The main inference loop.
    
    Wraps everything in a class so we can handle clean shutdown via signals.
    """

    def __init__(self, config: dict, class_names: list[str]):
        self.config = config
        self.class_names = class_names
        self.should_stop = False
        
        # Components (initialized in start())
        self.source: Optional[VideoSource] = None
        self.detector: Optional[Detector] = None
        self.aggregator: Optional[Aggregator] = None
        self.writer: Optional[JsonLinesWriter] = None

    def setup(self) -> None:
        """Initialize all components."""
        logger.info("=" * 60)
        logger.info("LiveStockify Inference Runner")
        logger.info("=" * 60)
        
        # Video source
        logger.info("[1/4] Setting up video source...")
        self.source = create_source(self.config["source"])
        
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
        
        # Aggregator
        logger.info("[3/4] Creating aggregator...")
        farm_cfg = self.config["farm"]
        self.aggregator = Aggregator(
            farm_id=farm_cfg["id"],
            farm_name=farm_cfg["name"],
            cam_id=farm_cfg["cam_id"],
            class_names=self.class_names,
        )
        
        # Storage
        logger.info("[4/4] Creating storage writer...")
        out_cfg = self.config["output"]
        self.writer = JsonLinesWriter(
            output_dir=out_cfg["json_dir"],
            daily_rotation=out_cfg.get("daily_rotation", True),
        )
        
        logger.info("Setup complete. Starting inference loop...")

    def run(self) -> None:
        """Main inference loop."""
        sampling_cfg = self.config["sampling"]
        interval_seconds = sampling_cfg.get("interval_seconds", 5)
        warmup_frames = sampling_cfg.get("warmup_frames", 3)
        
        # Warmup: discard first few frames
        for i in range(warmup_frames):
            ret, _ = self.source.read()
            if ret:
                logger.debug(f"Warmup frame {i+1}/{warmup_frames}")
        
        last_inference_time = 0.0
        processed_count = 0
        
        while not self.should_stop and self.source.is_opened:
            loop_start = time.time()
            
            # Read frame
            ret, frame = self.source.read()
            if not ret or frame is None:
                logger.warning("No frame received, source may be exhausted")
                if isinstance(self.source.__class__.__name__, str) and \
                   self.source.__class__.__name__ == "FileSource":
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
            last_inference_time = inference_ms
            
            # Aggregate
            record = self.aggregator.aggregate(detections, inference_ms)
            
            # Write
            try:
                self.writer.write(record)
            except Exception as e:
                logger.error(f"Failed to write record: {e}")
            
            # Log a quick summary
            counts_str = " | ".join(
                f"{k[:2]}:{v}" for k, v in record.counts.items()
            )
            logger.info(
                f"Frame {record.frame_index} | total={record.total} | "
                f"{counts_str} | inference={inference_ms:.0f}ms"
            )
            
            processed_count += 1
            
            # Sleep until next interval
            elapsed = time.time() - loop_start
            sleep_time = max(0, interval_seconds - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        logger.info(f"Loop ended. Processed {processed_count} frames total.")

    def stop(self) -> None:
        """Signal the loop to stop."""
        logger.info("Stop signal received")
        self.should_stop = True

    def cleanup(self) -> None:
        """Release all resources."""
        logger.info("Cleaning up...")
        if self.source:
            self.source.release()
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
        help="Override the video source path (e.g., test.mp4 or rtsp://...)",
    )
    args = parser.parse_args()
    
    # Load configs
    config = load_config(args.config)
    class_names = load_class_names(args.classes)
    
    # CLI override for source (handy for testing)
    if args.source:
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
