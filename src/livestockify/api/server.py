"""
LiveStockify API Server
========================
FastAPI server that runs on the Pi 5.
Exposes on-demand inference endpoints for the UI team.

Usage:
    # Start the server
    python -m livestockify.api.server --config configs/inference.yaml

    # Or via uvicorn directly
    uvicorn livestockify.api.server:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health              → Health check
    GET  /api/cameras         → List configured cameras
    POST /api/detect          → Run detection on all cameras
    POST /api/detect/{cam_id} → Run detection on one camera
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

from livestockify.inference.detector import Detector


# ============================================================
# PYDANTIC MODELS (API Response Shapes)
# ============================================================

class CameraResult(BaseModel):
    """Detection result for a single camera."""
    status: str                          # "ok" or "error"
    error: Optional[str] = None
    inference_time_ms: float = 0
    total: int = 0
    counts: Dict[str, int]
    percentages: Dict[str, float]


class FarmSummary(BaseModel):
    """Aggregated result across all cameras."""
    total_birds: int
    counts: Dict[str, int]
    percentages: Dict[str, float]
    cameras_ok: int
    cameras_failed: int


class DetectionResponse(BaseModel):
    """Full response for /api/detect."""
    farm_id: str
    farm_name: str
    timestamp: str
    processing_time_ms: float
    cameras: Dict[str, CameraResult]
    summary: FarmSummary


class CameraInfo(BaseModel):
    """Camera info for /api/cameras."""
    id: str
    enabled: bool
    has_url: bool


class CamerasResponse(BaseModel):
    """Response for /api/cameras."""
    farm_id: str
    farm_name: str
    cameras: List[CameraInfo]
    total: int
    enabled: int


class HealthResponse(BaseModel):
    """Response for /health."""
    status: str
    farm_id: str
    model_loaded: bool
    uptime_seconds: float


# ============================================================
# GLOBAL STATE
# ============================================================

class AppState:
    """Holds the shared state across requests."""
    config: dict = {}
    class_names: List[str] = []
    detector: Optional[Detector] = None
    start_time: float = 0
    api_key: str = ""


state = AppState()


# ============================================================
# CONFIG & STARTUP
# ============================================================

def load_config(config_path: str = "configs/inference.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_class_names(classes_path: str = "configs/classes.yaml") -> List[str]:
    with open(classes_path, "r") as f:
        data = yaml.safe_load(f)
    classes = sorted(data["classes"], key=lambda c: c["id"])
    return [c["name"] for c in classes]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    config_path = os.environ.get("LIVESTOCKIFY_CONFIG", "configs/inference.yaml")
    classes_path = os.environ.get("LIVESTOCKIFY_CLASSES", "configs/classes.yaml")

    logger.info("=" * 60)
    logger.info("LiveStockify API Server starting...")
    logger.info("=" * 60)

    # Load config
    state.config = load_config(config_path)
    state.class_names = load_class_names(classes_path)

    # Load API key from config or environment
    state.api_key = os.environ.get(
        "LIVESTOCKIFY_API_KEY",
        state.config.get("api", {}).get("api_key", "change-me-in-production")
    )

    if state.api_key == "change-me-in-production":
        logger.warning(
            "Using default API key! Set LIVESTOCKIFY_API_KEY env var "
            "or update configs/inference.yaml"
        )

    # Load model
    model_cfg = state.config["model"]
    logger.info(f"Loading model: {model_cfg['weights']}")

    state.detector = Detector(
        weights_path=model_cfg["weights"],
        class_names=state.class_names,
        conf_threshold=model_cfg.get("conf_threshold", 0.15),
        iou_threshold=model_cfg.get("iou_threshold", 0.45),
        imgsz=model_cfg.get("imgsz", 640),
        device=model_cfg.get("device", "cpu"),
    )

    state.start_time = time.time()
    logger.info("API server ready!")

    yield

    # Cleanup
    logger.info("API server shutting down...")


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="LiveStockify API",
    description="On-demand poultry behavioral detection from CCTV cameras",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow UI team's frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """Validate the API key from the request header."""
    if not api_key or api_key != state.api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Pass it as X-API-Key header.",
        )
    return api_key


# ============================================================
# CAMERA FRAME CAPTURE
# ============================================================

def grab_frame_from_rtsp(rtsp_url: str, timeout_seconds: float = 10.0) -> Optional[np.ndarray]:
    """
    Grab a single frame from an RTSP URL.

    Opens the stream, grabs one frame, closes immediately.
    This is the on-demand approach — we don't keep streams open.
    """
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        cap.release()
        return None

    # Grab a few frames to get past any stale buffer
    for _ in range(3):
        cap.grab()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return None

    return frame


def grab_frame_from_file(file_path: str) -> Optional[np.ndarray]:
    """Grab one frame from a video file (for testing)."""
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return None

    ret, frame = cap.read()
    cap.release()

    return frame if ret else None


def detect_on_camera(cam_id: str, cam_config: dict) -> CameraResult:
    """
    Run detection on a single camera.
    Opens RTSP → grabs frame → runs model → returns result.
    """
    empty_counts = {name: 0 for name in state.class_names}
    empty_pcts = {name: 0.0 for name in state.class_names}

    rtsp_url = cam_config.get("rtsp_url", "")
    if not rtsp_url:
        return CameraResult(
            status="error",
            error="No RTSP URL configured",
            counts=empty_counts,
            percentages=empty_pcts,
        )

    # Grab frame
    try:
        frame = grab_frame_from_rtsp(rtsp_url)
    except Exception as e:
        return CameraResult(
            status="error",
            error=f"RTSP connection failed: {str(e)}",
            counts=empty_counts,
            percentages=empty_pcts,
        )

    if frame is None:
        return CameraResult(
            status="error",
            error="Failed to grab frame from RTSP stream",
            counts=empty_counts,
            percentages=empty_pcts,
        )

    # Run inference
    inference_start = time.time()
    try:
        detections = state.detector.predict(frame)
    except Exception as e:
        return CameraResult(
            status="error",
            error=f"Inference failed: {str(e)}",
            counts=empty_counts,
            percentages=empty_pcts,
        )
    inference_ms = (time.time() - inference_start) * 1000

    # Count per class
    counts = {name: 0 for name in state.class_names}
    for det in detections:
        counts[det.class_name] = counts.get(det.class_name, 0) + 1

    total = len(detections)
    percentages = {
        name: round((cnt / total * 100), 2) if total > 0 else 0.0
        for name, cnt in counts.items()
    }

    return CameraResult(
        status="ok",
        inference_time_ms=round(inference_ms, 2),
        total=total,
        counts=counts,
        percentages=percentages,
    )


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check — no auth required."""
    return HealthResponse(
        status="ok",
        farm_id=state.config.get("farm", {}).get("id", "unknown"),
        model_loaded=state.detector is not None,
        uptime_seconds=round(time.time() - state.start_time, 1),
    )


@app.get("/api/cameras", response_model=CamerasResponse, dependencies=[Depends(verify_api_key)])
async def list_cameras():
    """List all configured cameras and their status."""
    farm_cfg = state.config.get("farm", {})
    cameras_cfg = state.config.get("source", {}).get("cameras", [])

    cameras = []
    enabled_count = 0
    for cam in cameras_cfg:
        is_enabled = cam.get("enabled", True)
        if is_enabled:
            enabled_count += 1
        cameras.append(CameraInfo(
            id=cam["id"],
            enabled=is_enabled,
            has_url=bool(cam.get("rtsp_url", "")),
        ))

    return CamerasResponse(
        farm_id=farm_cfg.get("id", "unknown"),
        farm_name=farm_cfg.get("name", "Unknown Farm"),
        cameras=cameras,
        total=len(cameras),
        enabled=enabled_count,
    )


@app.post("/api/detect", response_model=DetectionResponse, dependencies=[Depends(verify_api_key)])
async def detect_all_cameras():
    """
    Run detection on ALL enabled cameras.

    This processes cameras sequentially (not parallel) to avoid
    overloading the Pi's CPU. Takes ~1-2 seconds per camera.
    """
    start_time = time.time()
    farm_cfg = state.config.get("farm", {})
    cameras_cfg = state.config.get("source", {}).get("cameras", [])

    # Process each enabled camera
    camera_results: Dict[str, CameraResult] = {}
    for cam in cameras_cfg:
        if not cam.get("enabled", True):
            continue

        cam_id = cam["id"]
        logger.info(f"Processing {cam_id}...")

        result = detect_on_camera(cam_id, cam)
        camera_results[cam_id] = result

        if result.status == "ok":
            logger.info(
                f"  {cam_id}: {result.total} birds | "
                f"inference={result.inference_time_ms:.0f}ms"
            )
        else:
            logger.warning(f"  {cam_id}: {result.error}")

    # Build summary across all cameras
    total_birds = 0
    summary_counts = {name: 0 for name in state.class_names}
    cameras_ok = 0
    cameras_failed = 0

    for cam_id, result in camera_results.items():
        if result.status == "ok":
            cameras_ok += 1
            total_birds += result.total
            for name, cnt in result.counts.items():
                summary_counts[name] = summary_counts.get(name, 0) + cnt
        else:
            cameras_failed += 1

    summary_percentages = {
        name: round((cnt / total_birds * 100), 2) if total_birds > 0 else 0.0
        for name, cnt in summary_counts.items()
    }

    processing_time = (time.time() - start_time) * 1000

    logger.info(
        f"Detection complete: {cameras_ok} cameras OK, "
        f"{cameras_failed} failed, {total_birds} total birds, "
        f"{processing_time:.0f}ms total"
    )

    return DetectionResponse(
        farm_id=farm_cfg.get("id", "unknown"),
        farm_name=farm_cfg.get("name", "Unknown Farm"),
        timestamp=datetime.now(timezone.utc).isoformat(),
        processing_time_ms=round(processing_time, 2),
        cameras=camera_results,
        summary=FarmSummary(
            total_birds=total_birds,
            counts=summary_counts,
            percentages=summary_percentages,
            cameras_ok=cameras_ok,
            cameras_failed=cameras_failed,
        ),
    )


@app.post(
    "/api/detect/{cam_id}",
    response_model=DetectionResponse,
    dependencies=[Depends(verify_api_key)],
)
async def detect_single_camera(cam_id: str):
    """Run detection on a single specific camera."""
    start_time = time.time()
    farm_cfg = state.config.get("farm", {})
    cameras_cfg = state.config.get("source", {}).get("cameras", [])

    # Find the camera
    cam_config = None
    for cam in cameras_cfg:
        if cam["id"] == cam_id:
            cam_config = cam
            break

    if cam_config is None:
        raise HTTPException(
            status_code=404,
            detail=f"Camera '{cam_id}' not found. Use GET /api/cameras to list available cameras.",
        )

    if not cam_config.get("enabled", True):
        raise HTTPException(
            status_code=400,
            detail=f"Camera '{cam_id}' is disabled in config.",
        )

    # Run detection
    result = detect_on_camera(cam_id, cam_config)

    processing_time = (time.time() - start_time) * 1000
    cameras_ok = 1 if result.status == "ok" else 0
    cameras_failed = 1 - cameras_ok

    return DetectionResponse(
        farm_id=farm_cfg.get("id", "unknown"),
        farm_name=farm_cfg.get("name", "Unknown Farm"),
        timestamp=datetime.now(timezone.utc).isoformat(),
        processing_time_ms=round(processing_time, 2),
        cameras={cam_id: result},
        summary=FarmSummary(
            total_birds=result.total,
            counts=result.counts,
            percentages=result.percentages,
            cameras_ok=cameras_ok,
            cameras_failed=cameras_failed,
        ),
    )


# ============================================================
# TEST ENDPOINT (uses file instead of RTSP)
# ============================================================

@app.post("/api/test", response_model=DetectionResponse, dependencies=[Depends(verify_api_key)])
async def detect_test_file():
    """
    Run detection on the test video file configured in inference.yaml.
    Used for testing when no cameras are connected.
    """
    start_time = time.time()
    farm_cfg = state.config.get("farm", {})
    source_cfg = state.config.get("source", {})
    file_path = source_cfg.get("file_path", "")

    if not file_path or not Path(file_path).exists():
        raise HTTPException(
            status_code=400,
            detail=f"Test file not found: {file_path}. Update source.file_path in config.",
        )

    # Grab one frame from the file
    frame = grab_frame_from_file(file_path)
    if frame is None:
        raise HTTPException(status_code=500, detail="Could not read frame from test file")

    # Run inference
    inference_start = time.time()
    detections = state.detector.predict(frame)
    inference_ms = (time.time() - inference_start) * 1000

    # Build result
    counts = {name: 0 for name in state.class_names}
    for det in detections:
        counts[det.class_name] = counts.get(det.class_name, 0) + 1

    total = len(detections)
    percentages = {
        name: round((cnt / total * 100), 2) if total > 0 else 0.0
        for name, cnt in counts.items()
    }

    processing_time = (time.time() - start_time) * 1000

    result = CameraResult(
        status="ok",
        inference_time_ms=round(inference_ms, 2),
        total=total,
        counts=counts,
        percentages=percentages,
    )

    return DetectionResponse(
        farm_id=farm_cfg.get("id", "unknown"),
        farm_name=farm_cfg.get("name", "Unknown Farm"),
        timestamp=datetime.now(timezone.utc).isoformat(),
        processing_time_ms=round(processing_time, 2),
        cameras={"test_file": result},
        summary=FarmSummary(
            total_birds=total,
            counts=counts,
            percentages=percentages,
            cameras_ok=1,
            cameras_failed=0,
        ),
    )


# ============================================================
# CLI ENTRY POINT
# ============================================================

def main():
    """Start the API server from command line."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="LiveStockify API Server")
    parser.add_argument("--config", "-c", default="configs/inference.yaml", help="Config file path")
    parser.add_argument("--classes", default="configs/classes.yaml", help="Classes config path")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--api-key", default=None, help="API key (or set LIVESTOCKIFY_API_KEY env var)")

    args = parser.parse_args()

    # Set env vars for the lifespan handler to pick up
    os.environ["LIVESTOCKIFY_CONFIG"] = args.config
    os.environ["LIVESTOCKIFY_CLASSES"] = args.classes

    if args.api_key:
        os.environ["LIVESTOCKIFY_API_KEY"] = args.api_key

    uvicorn.run(
        "livestockify.api.server:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
