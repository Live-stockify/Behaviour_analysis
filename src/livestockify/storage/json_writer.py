"""
JSON Lines writer for local detection storage.

Writes one JSON record per line (jsonl format), which is:
- Easy to append (no parsing needed)
- Easy to stream/tail
- Easy to parse line-by-line later
- Easy to upload to S3/cloud as-is

Files are rotated daily by default.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from livestockify.inference.aggregator import CountRecord


class JsonLinesWriter:
    """
    Append-only JSONL writer with daily rotation.
    
    Each day gets its own file: detections_2026-04-08.jsonl
    """

    def __init__(self, output_dir: str | Path, daily_rotation: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.daily_rotation = daily_rotation
        
        self._current_date: Optional[str] = None
        self._current_file: Optional[Path] = None
        
        logger.info(f"JsonLinesWriter initialized at {self.output_dir}")

    def _get_filepath(self) -> Path:
        """Get the file path for today's records."""
        if not self.daily_rotation:
            return self.output_dir / "detections.jsonl"
        
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._current_date:
            self._current_date = today
            self._current_file = self.output_dir / f"detections_{today}.jsonl"
            logger.info(f"Rotated to new file: {self._current_file}")
        
        return self._current_file

    def write(self, record: CountRecord) -> None:
        """Append a single record to the current file."""
        filepath = self._get_filepath()
        
        try:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to write record to {filepath}: {e}")
            raise

    def close(self) -> None:
        """No-op (we open/close per write to be safe)."""
        pass
