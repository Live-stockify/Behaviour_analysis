"""
Video source abstraction.

Provides a single interface for reading frames from either:
1. An MP4 file (testing)
2. An RTSP stream (production)

Both sources expose the same `read()` method, so the runner doesn't
need to know which one it's using.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger


class VideoSource(ABC):
    """Abstract base class for any video input."""

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame.
        
        Returns:
            (success, frame) tuple. If success is False, frame is None.
        """
        ...

    @abstractmethod
    def release(self) -> None:
        """Clean up resources."""
        ...

    @property
    @abstractmethod
    def is_opened(self) -> bool:
        """Whether the source is currently active."""
        ...


class FileSource(VideoSource):
    """
    Reads frames from a video file.
    
    Used for testing and offline processing. When the file ends,
    `read()` returns (False, None).
    """

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.file_path}")
        
        logger.info(f"Opening video file: {self.file_path}")
        self.cap = cv2.VideoCapture(str(self.file_path))
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video file: {self.file_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(
            f"Video opened: {self.width}x{self.height} @ {self.fps:.1f}fps, "
            f"{self.total_frames} frames"
        )
        self._frame_count = 0

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ret, frame = self.cap.read()
        if ret:
            self._frame_count += 1
        return ret, frame if ret else None

    def release(self) -> None:
        if self.cap.isOpened():
            self.cap.release()
            logger.info(f"Closed video file. Read {self._frame_count} frames.")

    @property
    def is_opened(self) -> bool:
        return self.cap.isOpened()

    def seek_to_time(self, seconds: float) -> bool:
        """Jump to a specific timestamp in the video."""
        target_frame = int(seconds * self.fps)
        return self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)


class RTSPSource(VideoSource):
    """
    Reads frames from an RTSP stream.
    
    Production source. Auto-reconnects on failure.
    Note: RTSP streams are real-time, so we always grab the latest frame
    (we don't process every frame).
    """

    def __init__(
        self,
        rtsp_url: str,
        reconnect_delay: float = 5.0,
        max_reconnects: int = 100,
    ):
        self.rtsp_url = rtsp_url
        self.reconnect_delay = reconnect_delay
        self.max_reconnects = max_reconnects
        self.reconnect_count = 0
        self.cap: Optional[cv2.VideoCapture] = None
        
        self._connect()

    def _connect(self) -> None:
        """Open the RTSP stream."""
        # Mask credentials in logs
        safe_url = self._mask_url(self.rtsp_url)
        logger.info(f"Connecting to RTSP stream: {safe_url}")
        
        # Use FFMPEG backend for better RTSP handling
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        
        # Reduce buffering — we want the latest frame, not stale ones
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not connect to RTSP stream: {safe_url}")
        
        logger.info(f"RTSP stream connected: {safe_url}")

    @staticmethod
    def _mask_url(url: str) -> str:
        """Hide credentials in URL for logging."""
        if "@" in url:
            scheme_part, rest = url.split("://", 1)
            if "@" in rest:
                _, host_part = rest.split("@", 1)
                return f"{scheme_part}://***@{host_part}"
        return url

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.cap is None or not self.cap.isOpened():
            if not self._reconnect():
                return False, None
        
        ret, frame = self.cap.read()
        
        if not ret:
            logger.warning("Failed to read frame from RTSP, attempting reconnect")
            if self._reconnect():
                ret, frame = self.cap.read()
        
        return ret, frame if ret else None

    def _reconnect(self) -> bool:
        """Attempt to reconnect to the stream."""
        if self.reconnect_count >= self.max_reconnects:
            logger.error(
                f"Max reconnect attempts ({self.max_reconnects}) reached. Giving up."
            )
            return False
        
        self.reconnect_count += 1
        logger.warning(
            f"Reconnect attempt {self.reconnect_count}/{self.max_reconnects} "
            f"in {self.reconnect_delay}s..."
        )
        
        if self.cap is not None:
            self.cap.release()
        
        time.sleep(self.reconnect_delay)
        
        try:
            self._connect()
            self.reconnect_count = 0  # Reset on successful reconnect
            return True
        except Exception as e:
            logger.error(f"Reconnect failed: {e}")
            return False

    def release(self) -> None:
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            logger.info("Closed RTSP stream")

    @property
    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()


def create_source(config: dict) -> VideoSource:
    """
    Factory function: create the right source from config.
    
    Args:
        config: source section of the inference config dict
    
    Returns:
        VideoSource instance (either FileSource or RTSPSource)
    """
    source_type = config.get("type", "file").lower()
    
    if source_type == "file":
        file_path = config.get("file_path")
        if not file_path:
            raise ValueError("source.file_path required when source.type is 'file'")
        return FileSource(file_path)
    
    elif source_type == "rtsp":
        rtsp_url = config.get("rtsp_url")
        if not rtsp_url:
            raise ValueError("source.rtsp_url required when source.type is 'rtsp'")
        return RTSPSource(
            rtsp_url=rtsp_url,
            reconnect_delay=config.get("reconnect_delay_seconds", 5.0),
            max_reconnects=config.get("max_reconnect_attempts", 100),
        )
    
    else:
        raise ValueError(
            f"Unknown source type: {source_type}. Must be 'file' or 'rtsp'."
        )
