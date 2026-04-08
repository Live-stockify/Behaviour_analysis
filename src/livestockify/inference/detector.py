"""
Detector — wraps the YOLOv8 model for inference.

Loads the model once at startup and provides a clean `predict()` method
that returns parsed detections (not raw tensors).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from loguru import logger
from ultralytics import YOLO


@dataclass
class Detection:
    """A single detection from the model."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class Detector:
    """
    YOLOv8 detector wrapper.
    
    Loads the model once and provides a simple prediction interface.
    All inference parameters (conf, iou, imgsz) are baked in at init.
    """

    def __init__(
        self,
        weights_path: str | Path,
        class_names: List[str],
        conf_threshold: float = 0.15,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: str = "cpu",
    ):
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
        
        self.weights_path = weights_path
        self.class_names = class_names
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.device = device
        
        logger.info(f"Loading YOLOv8 model from {weights_path}")
        logger.info(
            f"Settings: conf={conf_threshold}, iou={iou_threshold}, "
            f"imgsz={imgsz}, device={device}"
        )
        
        self.model = YOLO(str(weights_path))
        
        # Warm up the model with a dummy frame to avoid first-call latency
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        self.model.predict(
            source=dummy, conf=conf_threshold, imgsz=imgsz,
            device=device, verbose=False
        )
        logger.info("Model loaded and warmed up")

    def predict(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a single frame.
        
        Args:
            frame: BGR image as numpy array (H, W, 3)
        
        Returns:
            List of Detection objects
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        
        detections: List[Detection] = []
        
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = (
                    self.class_names[cls_id]
                    if cls_id < len(self.class_names)
                    else f"class_{cls_id}"
                )
                
                detections.append(Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                ))
        
        return detections
