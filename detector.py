"""
Detector — wraps the YOLOv8 model for inference.

Loads the model once at startup and provides a clean `predict()` method
that returns parsed detections (not raw tensors).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
import cv2
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
        conf_threshold: float = 0.10,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: str = "cpu",
        enable_clahe: bool = True,
        tiling_enabled: bool = True,
        tiling_grid: tuple[int, int] = (2, 2),
        tiling_overlap: float = 0.25,
        class_conf_thresholds: Optional[Dict[str, float]] = None,
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
        self.enable_clahe = enable_clahe
        self.tiling_enabled = tiling_enabled
        self.tiling_grid = tiling_grid
        self.tiling_overlap = tiling_overlap
        
        # Build per-class threshold map (Level 3 optimization)
        # Default fallback values for behavior model
        self.class_conf_thresholds = {
            "Drinking": 0.08,
            "Eating": 0.08,
            "Sitting": 0.12,
            "Standing": 0.12
        }
        if class_conf_thresholds:
            self.class_conf_thresholds.update(class_conf_thresholds)
        
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

    def apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE preprocessing to normalize lighting and enhance contrast."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        enhanced = cv2.merge((l_enhanced, a, b))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def predict(self, frame: np.ndarray) -> List[Detection]:
        """
        Run Level 3 Hybrid Detection on a single frame.
        
        Algorithm:
          1. Apply CLAHE (Level 3 contrast boost)
          2. Run Full-Frame Pass (imgsz=640)
          3. Run 2x2 Tiled Pass (imgsz=640)
          4. Fusion: Weighted Box Union to merge partials and fix box scale.
        """
        original_frame = frame.copy()
        if self.enable_clahe:
            frame = self.apply_clahe(frame)

        h, w = frame.shape[:2]
        all_raw_boxes = [] # List of [x1, y1, x2, y2, conf, cls_id]

        # --- PASS 1: Full Image ---
        results_full = self.model.predict(
            source=frame,
            conf=min(self.class_conf_thresholds.values()),
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        for r in results_full:
            if r.boxes:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else "unknown"
                    
                    if conf >= self.class_conf_thresholds.get(cls_name, self.conf_threshold):
                        all_raw_boxes.append([x1, y1, x2, y2, conf, cls_id])

        # --- PASS 2: Tiled (if enabled) ---
        if self.tiling_enabled:
            grid_y, grid_x = self.tiling_grid
            sy, sx = h // grid_y, w // grid_x
            oy, ox = int(sy * self.tiling_overlap), int(sx * self.tiling_overlap)
            
            for i in range(grid_y):
                for j in range(grid_x):
                    tx1, ty1 = max(0, j * sx - ox), max(0, i * sy - oy)
                    tx2, ty2 = min(w, (j + 1) * sx + ox), min(h, (i + 1) * sy + oy)
                    tile = frame[ty1:ty2, tx1:tx2]
                    
                    res_tile = self.model.predict(
                        source=tile,
                        conf=min(self.class_conf_thresholds.values()),
                        imgsz=self.imgsz,
                        device=self.device,
                        verbose=False
                    )
                    for r in res_tile:
                        if r.boxes:
                            for box in r.boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                                conf = float(box.conf[0].cpu().numpy())
                                cls_id = int(box.cls[0].cpu().numpy())
                                cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else "unknown"
                                
                                if conf >= self.class_conf_thresholds.get(cls_name, self.conf_threshold):
                                    all_raw_boxes.append([x1 + tx1, y1 + ty1, x2 + tx1, y2 + ty1, conf, cls_id])

        # --- FUSION: Box Union ---
        detections: List[Detection] = []
        if all_raw_boxes:
            used = [False] * len(all_raw_boxes)
            for i in range(len(all_raw_boxes)):
                if used[i]: continue
                
                curr = all_raw_boxes[i]
                curr_box = curr[:4]
                curr_conf = curr[4]
                curr_cls = int(curr[5])
                
                # Look for merges
                for j in range(i + 1, len(all_raw_boxes)):
                    if used[j] or int(all_raw_boxes[j][5]) != curr_cls: continue
                    
                    boxB = all_raw_boxes[j][:4]
                    # IOU & Overlap check for fusion
                    xA, yA = max(curr_box[0], boxB[0]), max(curr_box[1], boxB[1])
                    xB, yB = min(curr_box[2], boxB[2]), min(curr_box[3], boxB[3])
                    inter = max(0, xB - xA) * max(0, yB - yA)
                    areaA = (curr_box[2]-curr_box[0]) * (curr_box[3]-curr_box[1])
                    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
                    
                    iou_score = inter / float(areaA + areaB - inter + 1e-6)
                    overlap_ratio = inter / float(min(areaA, areaB) + 1e-6)
                    
                    if iou_score > 0.45 or overlap_ratio > 0.75:
                        # UNION MERGE: Take the outer bounds
                        curr_box = [
                            min(curr_box[0], boxB[0]),
                            min(curr_box[1], boxB[1]),
                            max(curr_box[2], boxB[2]),
                            max(curr_box[3], boxB[3])
                        ]
                        curr_conf = max(curr_conf, all_raw_boxes[j][4])
                        used[j] = True
                
                detections.append(Detection(
                    x1=curr_box[0], y1=curr_box[1], x2=curr_box[2], y2=curr_box[3],
                    confidence=curr_conf,
                    class_id=curr_cls,
                    class_name=self.class_names[curr_cls] if curr_cls < len(self.class_names) else f"class_{curr_cls}"
                ))
                used[i] = True
        
        return detections


