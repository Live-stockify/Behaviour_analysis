"""
Polygon-to-BBox Converter + YOLOv8 Training Prep
==================================================
Converts Roboflow SAM polygon annotations to bounding box format,
sets up train/val split, and runs YOLOv8 training.

Usage:
    # Step 1: Convert polygons to bboxes
    python prepare_and_train.py --convert \
        --roboflow-dir "./roboflow_anuj" \
        --output-dir "./yolo_dataset"

    # Step 2: Train YOLOv8
    python prepare_and_train.py --train \
        --dataset-dir "./yolo_dataset"

    # Step 3: Pre-annotate remaining frames
    python prepare_and_train.py --predict \
        --model-path "./runs/detect/train/weights/best.pt" \
        --frames-dir "./sampled_frames" \
        --output-dir "./pre_annotations"

    # Or do all steps at once:
    python prepare_and_train.py --all \
        --roboflow-dir "./roboflow_anuj" \
        --output-dir "./yolo_dataset" \
        --frames-dir "./sampled_frames"
"""

import os
import re
import csv
import shutil
import argparse
import random
from pathlib import Path

# ============================================================
# CLASS MAPPING (from Roboflow export)
# ============================================================
# Roboflow order: 0=Drinking, 1=Eating, 2=Sitting, 3=Standing
# We keep this order throughout
CLASS_NAMES = ["Drinking", "Eating", "Sitting", "Standing"]


# ============================================================
# POLYGON TO BBOX CONVERSION
# ============================================================

def polygon_to_bbox(coords):
    """
    Convert polygon coordinates to bounding box.
    
    Input: list of floats [x1, y1, x2, y2, x3, y3, ...]
    Output: (x_center, y_center, width, height) normalized 0-1
    """
    xs = coords[0::2]  # every other starting from 0
    ys = coords[1::2]  # every other starting from 1
    
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    return x_center, y_center, width, height


def is_bbox_format(parts):
    """Check if a label line is already in bbox format (5 values: class x y w h)."""
    return len(parts) == 5


def is_polygon_format(parts):
    """Check if a label line is in polygon format (class + many xy pairs)."""
    return len(parts) > 5 and len(parts) % 2 == 1  # odd count = class + even pairs


def convert_label_file(input_path, output_path):
    """
    Convert a single label file from polygon/mixed format to pure bbox format.
    Returns: (total_annotations, converted_polygons, kept_bboxes, skipped)
    """
    total = 0
    converted = 0
    kept = 0
    skipped = 0
    
    output_lines = []
    
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            total += 1
            
            try:
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
            except (ValueError, IndexError):
                skipped += 1
                continue
            
            if len(coords) == 4:
                # Already bbox format: class x_center y_center width height
                x_c, y_c, w, h = coords
                kept += 1
            elif len(coords) >= 6 and len(coords) % 2 == 0:
                # Polygon format: convert to bbox
                x_c, y_c, w, h = polygon_to_bbox(coords)
                converted += 1
            else:
                skipped += 1
                continue
            
            # Validate bounds
            if w <= 0 or h <= 0 or w > 1 or h > 1:
                skipped += 1
                continue
            
            # Clamp to 0-1
            x_c = max(0, min(1, x_c))
            y_c = max(0, min(1, y_c))
            w = max(0.001, min(1, w))
            h = max(0.001, min(1, h))
            
            output_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(output_lines) + '\n' if output_lines else '')
    
    return total, converted, kept, skipped


# ============================================================
# DATASET PREPARATION
# ============================================================

def convert_dataset(roboflow_dir, output_dir):
    """
    Convert Roboflow export to clean YOLOv8 bbox dataset with train/val split.
    """
    roboflow_path = Path(roboflow_dir)
    output_path = Path(output_dir)
    
    print(f"\n{'='*60}")
    print(f"  POLYGON → BBOX CONVERTER")
    print(f"{'='*60}")
    
    # Find images and labels
    img_dir = roboflow_path / "train" / "images"
    lbl_dir = roboflow_path / "train" / "labels"
    
    if not img_dir.exists():
        print(f"ERROR: {img_dir} not found")
        return
    
    images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    print(f"  Found {len(images)} images in {img_dir}")
    
    # Create output structure
    for split in ["train", "val"]:
        (output_path / split / "images").mkdir(parents=True, exist_ok=True)
        (output_path / split / "labels").mkdir(parents=True, exist_ok=True)
    
    # Split: 85% train, 15% val
    random.seed(42)
    indices = list(range(len(images)))
    random.shuffle(indices)
    val_count = max(3, int(len(images) * 0.15))  # at least 3 for validation
    val_indices = set(indices[:val_count])
    
    total_annotations = 0
    total_converted = 0
    total_kept = 0
    total_skipped = 0
    class_counts = {i: 0 for i in range(len(CLASS_NAMES))}
    
    for i, img_path in enumerate(images):
        split = "val" if i in val_indices else "train"
        
        # Find matching label file
        stem = img_path.stem
        label_path = lbl_dir / f"{stem}.txt"
        
        if not label_path.exists():
            print(f"  WARNING: No label for {img_path.name}, skipping")
            continue
        
        # Convert label
        out_label = output_path / split / "labels" / f"{stem}.txt"
        ann, conv, kept, skip = convert_label_file(label_path, out_label)
        
        total_annotations += ann
        total_converted += conv
        total_kept += kept
        total_skipped += skip
        
        # Copy image
        out_image = output_path / split / "images" / img_path.name
        shutil.copy2(str(img_path), str(out_image))
        
        # Count classes
        with open(out_label, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls = int(parts[0])
                    if cls in class_counts:
                        class_counts[cls] += 1
    
    # Create data.yaml
    yaml_content = f"""train: {output_path.absolute()}/train/images
val: {output_path.absolute()}/val/images

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    
    yaml_path = output_path / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    # Summary
    train_count = len(images) - val_count
    
    print(f"\n  Conversion Results:")
    print(f"    Total annotations:  {total_annotations}")
    print(f"    Polygons → BBox:    {total_converted}")
    print(f"    Already BBox:       {total_kept}")
    print(f"    Skipped (invalid):  {total_skipped}")
    
    print(f"\n  Class Distribution:")
    for cls_id, name in enumerate(CLASS_NAMES):
        cnt = class_counts.get(cls_id, 0)
        pct = cnt / max(total_annotations - total_skipped, 1) * 100
        print(f"    {cls_id}: {name:<12} {cnt:>5} ({pct:.1f}%)")
    
    print(f"\n  Dataset Split:")
    print(f"    Train: {train_count} images")
    print(f"    Val:   {val_count} images")
    
    print(f"\n  Output: {output_path}")
    print(f"  data.yaml: {yaml_path}")
    print(f"{'='*60}\n")
    
    return str(yaml_path)


# ============================================================
# TRAINING
# ============================================================

def train_model(dataset_dir, epochs=80):
    """Train YOLOv8n on the prepared dataset."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        return None
    
    yaml_path = Path(dataset_dir) / "data.yaml"
    if not yaml_path.exists():
        print(f"ERROR: {yaml_path} not found. Run --convert first.")
        return None
    
    print(f"\n{'='*60}")
    print(f"  TRAINING YOLOv8n (Bootstrap Model)")
    print(f"{'='*60}")
    print(f"  Dataset:  {yaml_path}")
    print(f"  Epochs:   {epochs}")
    print(f"  Device:   MPS (Apple Silicon)")
    print(f"{'='*60}\n")
    
    model = YOLO("yolov8n.pt")
    
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=640,
        batch=8,            # Small batch for limited data
        device="mps",       # Apple Silicon GPU
        patience=20,        # Early stopping
        
        # Augmentation tuned for CCTV
        flipud=0.0,         # No vertical flip (birds don't appear upside down)
        fliplr=0.5,         # Horizontal flip OK
        degrees=0.0,        # No rotation (camera is fixed)
        scale=0.3,          # Moderate scale variation
        mosaic=1.0,         # Mosaic helps with small dataset
        mixup=0.1,
        
        # Save settings
        project="runs/detect",
        name="bootstrap_v1",
        exist_ok=True,
    )
    
    best_model = Path("runs/detect/bootstrap_v1/weights/best.pt")
    print(f"\n  Best model: {best_model}")
    return str(best_model)


# ============================================================
# PRE-ANNOTATION (PREDICTION)
# ============================================================

def predict_and_export(model_path, frames_dir, output_dir, labeled_dir=None):
    """
    Run the trained model on unlabeled frames and export predictions
    in YOLO label format for import into Roboflow.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed.")
        return
    
    frames_path = Path(frames_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create output structure for Roboflow import
    img_out = output_path / "images"
    lbl_out = output_path / "labels"
    img_out.mkdir(exist_ok=True)
    lbl_out.mkdir(exist_ok=True)
    
    # Get list of already-labeled images (to skip them)
    already_labeled = set()
    if labeled_dir:
        labeled_path = Path(labeled_dir)
        for f in labeled_path.rglob("*.jpg"):
            # Extract the original frame name from Roboflow's renamed file
            # Roboflow adds .rf.XXXXX before the extension
            original = f.stem.split(".rf.")[0] if ".rf." in f.stem else f.stem
            already_labeled.add(original)
        for f in labeled_path.rglob("*.png"):
            original = f.stem.split(".rf.")[0] if ".rf." in f.stem else f.stem
            already_labeled.add(original)
    
    # Find all frames to predict on
    all_frames = sorted(list(frames_path.glob("*.jpg")) + list(frames_path.glob("*.png")))
    
    # Filter out already-labeled
    frames_to_predict = []
    for f in all_frames:
        stem = f.stem.split(".rf.")[0] if ".rf." in f.stem else f.stem
        if stem not in already_labeled:
            frames_to_predict.append(f)
    
    print(f"\n{'='*60}")
    print(f"  PRE-ANNOTATION GENERATOR")
    print(f"{'='*60}")
    print(f"  Model:           {model_path}")
    print(f"  Frames dir:      {frames_path}")
    print(f"  Total frames:    {len(all_frames)}")
    print(f"  Already labeled: {len(already_labeled)}")
    print(f"  To predict:      {len(frames_to_predict)}")
    print(f"  Output:          {output_path}")
    print(f"{'='*60}\n")
    
    if not frames_to_predict:
        print("  No frames to predict on. All frames are already labeled.")
        return
    
    # Load model
    model = YOLO(model_path)
    
    # Run prediction
    print(f"  Running predictions...")
    results = model.predict(
        source=[str(f) for f in frames_to_predict],
        conf=0.15,          # Low confidence threshold to catch more birds
        iou=0.4,            # NMS IoU threshold
        imgsz=640,
        device="mps",
        verbose=False,
    )
    
    # Export predictions as YOLO label files
    total_detections = 0
    class_counts = {i: 0 for i in range(len(CLASS_NAMES))}
    frames_with_detections = 0
    
    for i, (frame_path, result) in enumerate(zip(frames_to_predict, results)):
        # Copy image
        shutil.copy2(str(frame_path), str(img_out / frame_path.name))
        
        # Write label file
        label_file = lbl_out / f"{frame_path.stem}.txt"
        boxes = result.boxes
        
        if boxes is not None and len(boxes) > 0:
            frames_with_detections += 1
            lines = []
            for box in boxes:
                cls = int(box.cls.item())
                x_c, y_c, w, h = box.xywhn[0].tolist()
                conf = box.conf.item()
                lines.append(f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
                total_detections += 1
                if cls in class_counts:
                    class_counts[cls] += 1
            
            with open(label_file, 'w') as f:
                f.write('\n'.join(lines) + '\n')
        else:
            # Write empty label file
            with open(label_file, 'w') as f:
                f.write('')
        
        if (i + 1) % 100 == 0:
            print(f"    Processed {i+1}/{len(frames_to_predict)}...")
    
    # Summary
    print(f"\n  {'='*55}")
    print(f"  PRE-ANNOTATION COMPLETE")
    print(f"  {'='*55}")
    print(f"  Frames processed:       {len(frames_to_predict)}")
    print(f"  Frames with detections: {frames_with_detections} ({frames_with_detections/len(frames_to_predict)*100:.1f}%)")
    print(f"  Total detections:       {total_detections}")
    print(f"  Avg detections/frame:   {total_detections/max(len(frames_to_predict),1):.1f}")
    
    print(f"\n  Class Distribution (predictions):")
    for cls_id, name in enumerate(CLASS_NAMES):
        cnt = class_counts.get(cls_id, 0)
        pct = cnt / max(total_detections, 1) * 100
        print(f"    {cls_id}: {name:<12} {cnt:>6} ({pct:.1f}%)")
    
    print(f"\n  Output:")
    print(f"    Images: {img_out} ({len(frames_to_predict)} files)")
    print(f"    Labels: {lbl_out} ({len(frames_to_predict)} files)")
    print(f"\n  NEXT STEP: Upload {output_path} to Roboflow for correction")
    print(f"  {'='*55}\n")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Polygon→BBox Converter + YOLOv8 Bootstrap Training + Pre-annotation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  1. python prepare_and_train.py --convert --roboflow-dir ./roboflow_anuj --output-dir ./yolo_dataset
  2. python prepare_and_train.py --train --dataset-dir ./yolo_dataset
  3. python prepare_and_train.py --predict --model-path ./runs/detect/bootstrap_v1/weights/best.pt \\
       --frames-dir ./sampled_frames --output-dir ./pre_annotations --labeled-dir ./roboflow_anuj/train/images
        """
    )
    
    # Mode flags
    parser.add_argument("--convert", action="store_true", help="Convert Roboflow polygons to bbox")
    parser.add_argument("--train", action="store_true", help="Train YOLOv8n bootstrap model")
    parser.add_argument("--predict", action="store_true", help="Pre-annotate unlabeled frames")
    parser.add_argument("--all", action="store_true", help="Run all steps sequentially")
    
    # Paths
    parser.add_argument("--roboflow-dir", type=str, help="Roboflow export directory")
    parser.add_argument("--output-dir", type=str, default="./yolo_dataset", help="Output dataset directory")
    parser.add_argument("--dataset-dir", type=str, default="./yolo_dataset", help="Dataset directory for training")
    parser.add_argument("--model-path", type=str, help="Path to trained model weights")
    parser.add_argument("--frames-dir", type=str, help="Directory with frames to pre-annotate")
    parser.add_argument("--labeled-dir", type=str, help="Directory with already-labeled images (to skip)")
    
    # Training params
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs (default: 80)")
    
    args = parser.parse_args()
    
    if args.all:
        if not args.roboflow_dir or not args.frames_dir:
            print("ERROR: --all requires --roboflow-dir and --frames-dir")
            return
        
        # Step 1: Convert
        yaml_path = convert_dataset(args.roboflow_dir, args.output_dir)
        
        # Step 2: Train
        model_path = train_model(args.output_dir, args.epochs)
        if not model_path:
            return
        
        # Step 3: Predict
        predict_and_export(
            model_path, args.frames_dir,
            "./pre_annotations",
            labeled_dir=Path(args.roboflow_dir) / "train" / "images"
        )
        return
    
    if args.convert:
        if not args.roboflow_dir:
            print("ERROR: --convert requires --roboflow-dir")
            return
        convert_dataset(args.roboflow_dir, args.output_dir)
    
    if args.train:
        train_model(args.dataset_dir, args.epochs)
    
    if args.predict:
        if not args.model_path or not args.frames_dir:
            print("ERROR: --predict requires --model-path and --frames-dir")
            return
        predict_and_export(
            args.model_path, args.frames_dir,
            args.output_dir or "./pre_annotations",
            labeled_dir=args.labeled_dir
        )
    
    if not any([args.convert, args.train, args.predict, args.all]):
        parser.print_help()


if __name__ == "__main__":
    main()
