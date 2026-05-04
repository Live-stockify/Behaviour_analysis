"""
ONNX Detector — Pure onnxruntime inference without PyTorch/ultralytics.

Loads a YOLOv8 ONNX model and runs inference using only onnxruntime,
numpy, and OpenCV. No torch dependency.

This solves the ARM64 PyTorch segfault issue on Raspberry Pi 5.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
import onnxruntime as ort
from loguru import logger


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
    YOLOv8 detector using pure onnxruntime.

    No PyTorch, no ultralytics — just ONNX + numpy + OpenCV.
    """

    def __init__(
        self,
        weights_path: str | Path,
        class_names: List[str],
        conf_threshold: float = 0.15,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: str = "cpu",  # kept for config compatibility, always CPU on Pi
    ):
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")

        if not str(weights_path).endswith(".onnx"):
            raise ValueError(
                f"This detector requires an .onnx model file, got: {weights_path}. "
                f"Export your .pt model to ONNX on your Mac using: "
                f"model.export(format='onnx', imgsz=640, simplify=True)"
            )

        self.class_names = class_names
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz

        logger.info(f"Loading ONNX model from {weights_path}")
        logger.info(
            f"Settings: conf={conf_threshold}, iou={iou_threshold}, "
            f"imgsz={imgsz}, runtime=onnxruntime"
        )

        # Create ONNX session
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4  # Use 4 CPU threads

        self.session = ort.InferenceSession(
            str(weights_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Warmup with a dummy frame
        dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
        self.session.run([self.output_name], {self.input_name: dummy})

        logger.info("ONNX model loaded and warmed up")

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, float, int, int]:
        """
        Preprocess a BGR frame for YOLOv8 ONNX input.

        1. Letterbox resize to imgsz×imgsz (preserve aspect ratio)
        2. BGR → RGB
        3. HWC → CHW
        4. Normalize to 0-1
        5. Add batch dimension

        Returns:
            (input_tensor, ratio, (pad_w, pad_h), orig_w, orig_h)
        """
        orig_h, orig_w = frame.shape[:2]

        # Compute scale to fit in imgsz×imgsz
        ratio = min(self.imgsz / orig_w, self.imgsz / orig_h)
        new_w = int(orig_w * ratio)
        new_h = int(orig_h * ratio)

        # Resize
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Letterbox pad (center the image)
        pad_w = (self.imgsz - new_w) // 2
        pad_h = (self.imgsz - new_h) // 2

        padded = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # BGR → RGB, HWC → CHW, normalize, add batch
        blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)

        return blob, ratio, pad_w, pad_h, orig_w, orig_h

    def _postprocess(
        self,
        output: np.ndarray,
        ratio: float,
        pad_w: int,
        pad_h: int,
        orig_w: int,
        orig_h: int,
    ) -> List[Detection]:
        """
        Parse YOLOv8 ONNX output into Detection objects.

        YOLOv8 output shape: [1, 4+num_classes, 8400]
        - First 4 values: cx, cy, w, h (in input image coords)
        - Remaining values: class scores

        Steps:
        1. Transpose to [8400, 4+num_classes]
        2. Extract boxes and class scores
        3. Filter by confidence
        4. Apply NMS
        5. Scale boxes back to original image coordinates
        """
        # output shape: [1, 8, 8400] → [8400, 8]
        predictions = output[0].T  # [8400, 4+num_classes]

        num_classes = len(self.class_names)

        # Split into boxes and scores
        boxes_xywh = predictions[:, :4]           # [8400, 4] — cx, cy, w, h
        class_scores = predictions[:, 4:4 + num_classes]  # [8400, num_classes]

        # Get best class and confidence for each box
        max_scores = np.max(class_scores, axis=1)     # [8400]
        class_ids = np.argmax(class_scores, axis=1)    # [8400]

        # Filter by confidence
        mask = max_scores >= self.conf_threshold
        if not np.any(mask):
            return []

        boxes_xywh = boxes_xywh[mask]
        scores = max_scores[mask]
        class_ids = class_ids[mask]

        # Convert cx,cy,w,h → x1,y1,x2,y2
        boxes_xyxy = np.zeros_like(boxes_xywh)
        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2

        # Remove letterbox padding and scale to original image
        boxes_xyxy[:, 0] = (boxes_xyxy[:, 0] - pad_w) / ratio
        boxes_xyxy[:, 1] = (boxes_xyxy[:, 1] - pad_h) / ratio
        boxes_xyxy[:, 2] = (boxes_xyxy[:, 2] - pad_w) / ratio
        boxes_xyxy[:, 3] = (boxes_xyxy[:, 3] - pad_h) / ratio

        # Clamp to image bounds
        boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_w)
        boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_h)
        boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_w)
        boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_h)

        # Apply NMS (class-aware)
        detections = self._nms(boxes_xyxy, scores, class_ids)

        return detections

    def _nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
    ) -> List[Detection]:
        """Apply class-aware Non-Maximum Suppression using OpenCV."""
        if len(boxes) == 0:
            return []

        # Convert to format OpenCV expects: [x, y, w, h]
        boxes_xywh = np.zeros_like(boxes)
        boxes_xywh[:, 0] = boxes[:, 0]
        boxes_xywh[:, 1] = boxes[:, 1]
        boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
        boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # height

        # OpenCV NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(),
            scores.tolist(),
            score_threshold=self.conf_threshold,
            nms_threshold=self.iou_threshold,
        )

        if len(indices) == 0:
            return []

        indices = np.array(indices).flatten()

        detections = []
        for idx in indices:
            cls_id = int(class_ids[idx])
            cls_name = (
                self.class_names[cls_id]
                if cls_id < len(self.class_names)
                else f"class_{cls_id}"
            )
            detections.append(Detection(
                x1=float(boxes[idx, 0]),
                y1=float(boxes[idx, 1]),
                x2=float(boxes[idx, 2]),
                y2=float(boxes[idx, 3]),
                confidence=float(scores[idx]),
                class_id=cls_id,
                class_name=cls_name,
            ))

        return detections

    def predict(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a single frame.

        Args:
            frame: BGR image as numpy array (H, W, 3)

        Returns:
            List of Detection objects
        """
        # Preprocess
        blob, ratio, pad_w, pad_h, orig_w, orig_h = self._preprocess(frame)

        # Run inference
        output = self.session.run([self.output_name], {self.input_name: blob})[0]

        # Postprocess
        detections = self._postprocess(output, ratio, pad_w, pad_h, orig_w, orig_h)

        return detections
