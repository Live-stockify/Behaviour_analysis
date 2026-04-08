"""
Tiled Inference for Dense Object Detection
============================================
Splits images into overlapping tiles, runs YOLO on each tile,
and stitches predictions back with NMS deduplication.

Compares 3 strategies: full-image, 2x2 tiles, 3x3 tiles.

Usage:
    # Compare all 3 strategies on a few test frames:
    python tiled_inference.py --compare \
        --model "runs/detect/runs/detect/round2_v3/weights/best.pt" \
        --input "sampled_frames" \
        --num-test 5

    # Run 2x2 tiled inference on all frames:
    python tiled_inference.py --tile 2 \
        --model "runs/detect/runs/detect/round2_v3/weights/best.pt" \
        --input "sampled_frames" \
        --output "tiled_predictions_2x2"

    # Run 3x3 tiled inference:
    python tiled_inference.py --tile 3 \
        --model "runs/detect/runs/detect/round2_v3/weights/best.pt" \
        --input "sampled_frames" \
        --output "tiled_predictions_3x3"
"""

import os
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

CLASS_NAMES = ["Drinking", "Eating", "Sitting", "Standing"]
CLASS_COLORS = {
    0: (255, 100, 100),    # Drinking - light blue (BGR)
    1: (100, 255, 100),    # Eating - green
    2: (100, 100, 255),    # Sitting - red
    3: (255, 255, 100),    # Standing - cyan
}

# Tile overlap fraction (15% overlap between adjacent tiles)
OVERLAP_FRACTION = 0.15

# Confidence and NMS thresholds
CONF_THRESHOLD = 0.20
NMS_IOU = 0.45  # IoU threshold for global NMS after stitching


# ============================================================
# TILING FUNCTIONS
# ============================================================

def generate_tiles(img_width, img_height, n_tiles, overlap=OVERLAP_FRACTION):
    """
    Generate tile coordinates with overlap.
    
    Args:
        img_width, img_height: original image dimensions
        n_tiles: 2 (for 2x2) or 3 (for 3x3)
        overlap: fraction of tile size to overlap (0.15 = 15%)
    
    Returns:
        List of (x1, y1, x2, y2) tuples
    """
    base_w = img_width // n_tiles
    base_h = img_height // n_tiles
    
    overlap_w = int(base_w * overlap)
    overlap_h = int(base_h * overlap)
    
    tiles = []
    for row in range(n_tiles):
        for col in range(n_tiles):
            # Base position
            x1 = col * base_w
            y1 = row * base_h
            x2 = x1 + base_w
            y2 = y1 + base_h
            
            # Add overlap (extend boundaries inward except at image edges)
            if col > 0:
                x1 -= overlap_w
            if row > 0:
                y1 -= overlap_h
            if col < n_tiles - 1:
                x2 += overlap_w
            if row < n_tiles - 1:
                y2 += overlap_h
            
            # Clamp to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width, x2)
            y2 = min(img_height, y2)
            
            tiles.append((x1, y1, x2, y2))
    
    return tiles


def predict_on_tile(model, tile_img, tile_coords, conf=CONF_THRESHOLD):
    """
    Run model on a single tile and return detections in ORIGINAL image coordinates.
    
    Args:
        model: YOLO model
        tile_img: cropped tile (numpy array)
        tile_coords: (x1, y1, x2, y2) of tile in original image
        conf: confidence threshold
    
    Returns:
        List of [x1, y1, x2, y2, conf, class_id] in original image coordinates
    """
    tile_x1, tile_y1, _, _ = tile_coords
    
    results = model.predict(
        source=tile_img,
        conf=conf,
        imgsz=640,
        device='cpu',
        verbose=False,
    )
    
    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            # Get box in tile coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Translate to original image coordinates
            x1 += tile_x1
            y1 += tile_y1
            x2 += tile_x1
            y2 += tile_y1
            
            conf_val = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            
            detections.append([x1, y1, x2, y2, conf_val, cls_id])
    
    return detections


def global_nms(detections, iou_threshold=NMS_IOU):
    """
    Apply NMS across all detections to remove duplicates from tile overlaps.
    Uses class-aware NMS (only suppresses boxes of the same class).
    """
    if not detections:
        return []
    
    detections = np.array(detections)
    
    # Separate by class for class-aware NMS
    final = []
    for cls_id in range(len(CLASS_NAMES)):
        cls_dets = detections[detections[:, 5] == cls_id]
        if len(cls_dets) == 0:
            continue
        
        boxes = cls_dets[:, :4].astype(np.float32)
        scores = cls_dets[:, 4].astype(np.float32)
        
        # Use OpenCV NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            score_threshold=CONF_THRESHOLD,
            nms_threshold=iou_threshold,
        )
        
        if len(indices) > 0:
            indices = np.array(indices).flatten()
            for idx in indices:
                final.append(cls_dets[idx].tolist())
    
    return final


# ============================================================
# INFERENCE STRATEGIES
# ============================================================

def predict_full_image(model, img, conf=CONF_THRESHOLD):
    """Standard single-pass inference on full image."""
    results = model.predict(
        source=img,
        conf=conf,
        imgsz=640,
        device='cpu',
        verbose=False,
    )
    
    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf_val = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            detections.append([x1, y1, x2, y2, conf_val, cls_id])
    
    return detections


def predict_tiled(model, img, n_tiles, conf=CONF_THRESHOLD):
    """Run tiled inference and stitch results."""
    h, w = img.shape[:2]
    tiles = generate_tiles(w, h, n_tiles)
    
    all_detections = []
    for tile_coords in tiles:
        x1, y1, x2, y2 = tile_coords
        tile_img = img[y1:y2, x1:x2]
        if tile_img.size == 0:
            continue
        
        tile_dets = predict_on_tile(model, tile_img, tile_coords, conf)
        all_detections.extend(tile_dets)
    
    # Apply global NMS to remove duplicates from overlapping regions
    final = global_nms(all_detections)
    
    return final


# ============================================================
# VISUALIZATION
# ============================================================

def draw_detections(img, detections, label=""):
    """Draw bounding boxes and labels on image."""
    img_copy = img.copy()
    
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls_id = int(cls_id)
        
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        
        # Label
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "?"
        text = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_copy, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img_copy, text, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Add summary banner at top
    banner_h = 60
    banner = np.zeros((banner_h, img_copy.shape[1], 3), dtype=np.uint8)
    
    # Count by class
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for det in detections:
        counts[int(det[5])] += 1
    
    summary = f"{label} | Total: {len(detections)} | "
    summary += f"D:{counts[0]} E:{counts[1]} Si:{counts[2]} St:{counts[3]}"
    
    cv2.putText(banner, summary, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return np.vstack([banner, img_copy])


# ============================================================
# COMPARISON MODE
# ============================================================

def compare_strategies(model_path, input_dir, num_test=5):
    """
    Run all 3 strategies (full, 2x2, 3x3) on test frames and save side-by-side comparison.
    """
    print(f"\n{'='*65}")
    print(f"  TILED INFERENCE COMPARISON")
    print(f"{'='*65}")
    print(f"  Model:  {model_path}")
    print(f"  Input:  {input_dir}")
    print(f"  Tests:  {num_test} frames")
    print(f"{'='*65}\n")
    
    # Load model
    print(f"  Loading model...")
    model = YOLO(model_path)
    
    # Pick diverse test frames (different cameras, days, clip types)
    input_path = Path(input_dir)
    all_frames = sorted(input_path.glob("*.jpg"))
    
    if len(all_frames) == 0:
        print(f"  ERROR: No .jpg files found in {input_dir}")
        return
    
    # Pick frames spread across the dataset
    step = max(1, len(all_frames) // num_test)
    test_frames = all_frames[::step][:num_test]
    
    print(f"  Selected {len(test_frames)} test frames\n")
    
    # Output directory
    output_dir = Path("comparison_results")
    output_dir.mkdir(exist_ok=True)
    
    stats = {
        "full":   {"total": 0, "time": 0},
        "tile2":  {"total": 0, "time": 0},
        "tile3":  {"total": 0, "time": 0},
    }
    
    for i, frame_path in enumerate(test_frames):
        print(f"  [{i+1}/{len(test_frames)}] {frame_path.name}")
        
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        
        # Strategy 1: Full image
        t0 = time.time()
        dets_full = predict_full_image(model, img)
        t_full = time.time() - t0
        
        # Strategy 2: 2x2 tiles
        t0 = time.time()
        dets_2 = predict_tiled(model, img, n_tiles=2)
        t_2 = time.time() - t0
        
        # Strategy 3: 3x3 tiles
        t0 = time.time()
        dets_3 = predict_tiled(model, img, n_tiles=3)
        t_3 = time.time() - t0
        
        print(f"    Full:  {len(dets_full):>4} detections  ({t_full:.2f}s)")
        print(f"    2x2:   {len(dets_2):>4} detections  ({t_2:.2f}s)  +{len(dets_2) - len(dets_full):+d}")
        print(f"    3x3:   {len(dets_3):>4} detections  ({t_3:.2f}s)  +{len(dets_3) - len(dets_full):+d}")
        
        stats["full"]["total"] += len(dets_full)
        stats["full"]["time"] += t_full
        stats["tile2"]["total"] += len(dets_2)
        stats["tile2"]["time"] += t_2
        stats["tile3"]["total"] += len(dets_3)
        stats["tile3"]["time"] += t_3
        
        # Draw each strategy
        img_full = draw_detections(img, dets_full, "FULL IMAGE")
        img_2 = draw_detections(img, dets_2, "2x2 TILES")
        img_3 = draw_detections(img, dets_3, "3x3 TILES")
        
        # Resize for side-by-side
        target_h = 600
        def resize(im):
            ratio = target_h / im.shape[0]
            new_w = int(im.shape[1] * ratio)
            return cv2.resize(im, (new_w, target_h))
        
        img_full = resize(img_full)
        img_2 = resize(img_2)
        img_3 = resize(img_3)
        
        # Stack horizontally
        max_w = max(img_full.shape[1], img_2.shape[1], img_3.shape[1])
        def pad(im):
            if im.shape[1] < max_w:
                pad_w = max_w - im.shape[1]
                return cv2.copyMakeBorder(im, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            return im
        
        comparison = np.hstack([pad(img_full), pad(img_2), pad(img_3)])
        
        out_path = output_dir / f"compare_{i+1:02d}_{frame_path.stem}.jpg"
        cv2.imwrite(str(out_path), comparison)
        print(f"    Saved: {out_path}\n")
    
    # Final summary
    print(f"\n{'='*65}")
    print(f"  COMPARISON SUMMARY ({len(test_frames)} frames)")
    print(f"{'='*65}")
    print(f"  Strategy    | Total Detections | Avg/frame | Total Time | Time/frame")
    print(f"  " + "-"*63)
    
    for strategy, label in [("full", "Full image"), ("tile2", "2x2 tiles "), ("tile3", "3x3 tiles ")]:
        s = stats[strategy]
        avg = s["total"] / len(test_frames)
        time_per = s["time"] / len(test_frames)
        print(f"  {label} | {s['total']:>16} | {avg:>9.1f} | {s['time']:>10.2f}s | {time_per:>10.2f}s")
    
    # Improvement
    full_avg = stats["full"]["total"] / len(test_frames)
    t2_avg = stats["tile2"]["total"] / len(test_frames)
    t3_avg = stats["tile3"]["total"] / len(test_frames)
    
    print(f"\n  Detection improvement vs full image:")
    print(f"    2x2: +{(t2_avg/max(full_avg,1) - 1)*100:.1f}% more detections per frame")
    print(f"    3x3: +{(t3_avg/max(full_avg,1) - 1)*100:.1f}% more detections per frame")
    
    print(f"\n  Speed cost vs full image:")
    print(f"    2x2: {stats['tile2']['time']/max(stats['full']['time'],0.01):.1f}x slower")
    print(f"    3x3: {stats['tile3']['time']/max(stats['full']['time'],0.01):.1f}x slower")
    
    print(f"\n  Comparison images saved to: {output_dir.absolute()}")
    print(f"{'='*65}\n")


# ============================================================
# BATCH MODE
# ============================================================

def run_batch(model_path, input_dir, output_dir, n_tiles):
    """Run tiled inference on all images in a directory."""
    print(f"\n{'='*65}")
    print(f"  TILED INFERENCE - {n_tiles}x{n_tiles} mode")
    print(f"{'='*65}")
    
    model = YOLO(model_path)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    images = sorted(input_path.glob("*.jpg"))
    print(f"  Processing {len(images)} images...\n")
    
    total_dets = 0
    start = time.time()
    
    for i, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        dets = predict_tiled(model, img, n_tiles=n_tiles)
        total_dets += len(dets)
        
        # Save visualization
        img_drawn = draw_detections(img, dets, f"{n_tiles}x{n_tiles}")
        out_path = output_path / img_path.name
        cv2.imwrite(str(out_path), img_drawn)
        
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            print(f"    {i+1}/{len(images)} processed ({elapsed:.0f}s elapsed)")
    
    elapsed = time.time() - start
    print(f"\n  Done!")
    print(f"  Total detections: {total_dets}")
    print(f"  Avg per frame: {total_dets/max(len(images),1):.1f}")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/max(len(images),1):.2f}s per frame)")
    print(f"  Output: {output_path}")
    print(f"{'='*65}\n")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Tiled Inference for Dense Detection")
    parser.add_argument("--model", required=True, help="Path to trained model weights")
    parser.add_argument("--input", required=True, help="Input image directory")
    parser.add_argument("--output", default="tiled_predictions", help="Output directory")
    parser.add_argument("--compare", action="store_true", help="Run comparison mode")
    parser.add_argument("--num-test", type=int, default=5, help="Number of frames for comparison mode")
    parser.add_argument("--tile", type=int, default=2, choices=[2, 3], help="Tile grid size (2 or 3)")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_strategies(args.model, args.input, args.num_test)
    else:
        run_batch(args.model, args.input, args.output, args.tile)


if __name__ == "__main__":
    main()
