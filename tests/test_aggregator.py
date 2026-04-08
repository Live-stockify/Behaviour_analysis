"""
Tests for the Aggregator class.

Run with: pytest tests/test_aggregator.py -v
"""

import pytest

from livestockify.inference.aggregator import Aggregator, CountRecord
from livestockify.inference.detector import Detection


CLASS_NAMES = ["Drinking", "Eating", "Sitting", "Standing"]


@pytest.fixture
def aggregator():
    return Aggregator(
        farm_id="test_farm",
        farm_name="Test Farm",
        cam_id="cam1",
        class_names=CLASS_NAMES,
    )


def make_detection(class_id: int, confidence: float = 0.5) -> Detection:
    """Helper to create a detection."""
    return Detection(
        x1=0, y1=0, x2=100, y2=100,
        confidence=confidence,
        class_id=class_id,
        class_name=CLASS_NAMES[class_id],
    )


def test_empty_detections(aggregator):
    """Aggregator with zero detections should still produce a record with 0 counts."""
    record = aggregator.aggregate([])
    
    assert record.total == 0
    assert record.counts == {"Drinking": 0, "Eating": 0, "Sitting": 0, "Standing": 0}
    assert record.percentages == {"Drinking": 0.0, "Eating": 0.0, "Sitting": 0.0, "Standing": 0.0}
    assert record.avg_confidence == 0.0


def test_single_detection(aggregator):
    """One Eating detection should give 100% Eating."""
    record = aggregator.aggregate([make_detection(class_id=1, confidence=0.8)])
    
    assert record.total == 1
    assert record.counts["Eating"] == 1
    assert record.counts["Drinking"] == 0
    assert record.percentages["Eating"] == 100.0
    assert record.avg_confidence == 0.8


def test_mixed_detections(aggregator):
    """Mix of classes should produce correct counts and percentages."""
    detections = [
        make_detection(1, 0.9),  # Eating
        make_detection(1, 0.8),  # Eating
        make_detection(2, 0.7),  # Sitting
        make_detection(3, 0.6),  # Standing
    ]
    record = aggregator.aggregate(detections)
    
    assert record.total == 4
    assert record.counts["Eating"] == 2
    assert record.counts["Sitting"] == 1
    assert record.counts["Standing"] == 1
    assert record.counts["Drinking"] == 0
    
    assert record.percentages["Eating"] == 50.0
    assert record.percentages["Sitting"] == 25.0
    assert record.percentages["Standing"] == 25.0
    
    # Average confidence: (0.9 + 0.8 + 0.7 + 0.6) / 4 = 0.75
    assert record.avg_confidence == 0.75


def test_frame_index_increments(aggregator):
    """Frame index should increment with each call."""
    r1 = aggregator.aggregate([])
    r2 = aggregator.aggregate([])
    r3 = aggregator.aggregate([])
    
    assert r1.frame_index == 1
    assert r2.frame_index == 2
    assert r3.frame_index == 3


def test_metadata_preserved(aggregator):
    """Farm metadata should appear in every record."""
    record = aggregator.aggregate([])
    
    assert record.farm_id == "test_farm"
    assert record.farm_name == "Test Farm"
    assert record.cam_id == "cam1"


def test_to_dict_serializable(aggregator):
    """to_dict() output should be JSON-serializable."""
    import json
    
    record = aggregator.aggregate([make_detection(1, 0.5)])
    d = record.to_dict()
    
    # Should not raise
    json_str = json.dumps(d)
    assert "Eating" in json_str
    assert "test_farm" in json_str
