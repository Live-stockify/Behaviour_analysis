"""
Aggregator — converts a list of detections into a JSON-friendly count record.

This is the bridge between raw detections and the structured output
that gets stored or pushed to the cloud.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List

from .detector import Detection


@dataclass
class CountRecord:
    """
    A single behavioral count record at a point in time.
    
    This is the structure that gets serialized to JSON and eventually
    pushed to the cloud.
    """
    timestamp: str               # ISO 8601 UTC
    farm_id: str
    farm_name: str
    cam_id: str
    
    # Counts per class
    counts: Dict[str, int]       # {"Eating": 12, "Drinking": 3, ...}
    total: int                   # Total birds detected
    
    # Percentages (computed)
    percentages: Dict[str, float]
    
    # Metadata
    avg_confidence: float
    inference_time_ms: float
    frame_index: int             # Which frame this came from
    cpu_usage: float = 0.0       # CPU percent used for monitoring
    
    def to_dict(self) -> dict:
        return asdict(self)


class Aggregator:
    """
    Aggregates raw detections into count records.
    
    Tracks frame index across calls so each record has a sequence number.
    """

    def __init__(
        self,
        farm_id: str,
        farm_name: str,
        cam_id: str,
        class_names: List[str],
    ):
        self.farm_id = farm_id
        self.farm_name = farm_name
        self.cam_id = cam_id
        self.class_names = class_names
        self.frame_index = 0

    def aggregate(
        self,
        detections: List[Detection],
        inference_time_ms: float = 0.0,
        cpu_usage: float = 0.0,
    ) -> CountRecord:
        """
        Convert detections to a count record.
        
        Args:
            detections: list of Detection objects from the detector
            inference_time_ms: how long detection took (for monitoring)
        
        Returns:
            CountRecord ready to serialize
        """
        self.frame_index += 1
        
        # Initialize all classes to 0 so the JSON shape is always consistent
        counts: Dict[str, int] = {name: 0 for name in self.class_names}
        
        confidence_sum = 0.0
        for det in detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1
            confidence_sum += det.confidence
        
        total = len(detections)
        avg_conf = confidence_sum / total if total > 0 else 0.0
        
        # Percentages
        percentages: Dict[str, float] = {}
        for name, cnt in counts.items():
            percentages[name] = round((cnt / total * 100), 2) if total > 0 else 0.0
        
        return CountRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            farm_id=self.farm_id,
            farm_name=self.farm_name,
            cam_id=self.cam_id,
            counts=counts,
            total=total,
            percentages=percentages,
            avg_confidence=round(avg_conf, 4),
            inference_time_ms=round(inference_time_ms, 2),
            frame_index=self.frame_index,
            cpu_usage=round(cpu_usage, 2),
        )
