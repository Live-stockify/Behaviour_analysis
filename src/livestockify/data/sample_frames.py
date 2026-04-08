"""
Poultry Frame Sampler
======================
Strategically samples frames from extracted images for annotation.
Ensures balanced coverage across cameras, days, clip types, and quality grades.

Usage:
    python sample_frames.py \
        --frames-dir "./extracted_frames" \
        --extraction-log "./logs/extraction_log.csv" \
        --output-dir "./sampled_frames" \
        --log-dir "./logs" \
        --target 1000
"""

import os
import re
import csv
import random
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

FRAME_PATTERN = re.compile(
    r'^(saibabu|naveen|vamsi)_(cam[1-3])_d(\d{2})_(MOR1|MOR2|AFT|EVE|NIT)_(\d{8})_(\d{4})_f(\d{4})\.jpg$'
)

# Cam3 early days to skip (storage area, no birds)
CAM3_SKIP_DAYS = {6, 7, 8, 9}

# Sampling weights by clip type (higher = sample more from this clip)
# MOR1 and MOR2 get highest weight because they have the most eating/drinking
CLIP_WEIGHTS = {
    "MOR1": 3.0,   # Peak eating — highest priority
    "MOR2": 2.5,   # Active eating continues
    "EVE":  2.0,   # Pre-dark eating peak
    "AFT":  1.5,   # Mid-day variety
    "NIT":  1.0,   # Night — needed but lower priority
}

# Growth stage definitions for Saibabu (starts d08 effectively)
def get_growth_stage(day_num):
    if day_num <= 10:
        return "early"      # d06-d10: early chick
    elif day_num <= 17:
        return "growing"    # d11-d17: growing
    else:
        return "juvenile"   # d18+: juvenile


# ============================================================
# QUALITY GRADING (inline, no OpenCV dependency for sampling)
# ============================================================

def load_quality_grades(extraction_log):
    """
    Load quality grades from extraction_log.csv.
    Returns dict: clip_prefix -> {grade_A, grade_B, grade_C, avg_sharpness}
    """
    grades = {}
    try:
        with open(extraction_log, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                prefix = Path(row.get("filename", "")).stem
                grades[prefix] = {
                    "grade_A": int(row.get("grade_A", 0)),
                    "grade_B": int(row.get("grade_B", 0)),
                    "grade_C": int(row.get("grade_C", 0)),
                    "avg_sharpness": float(row.get("avg_sharpness", 0)),
                    "avg_brightness": float(row.get("avg_brightness", 0)),
                }
    except FileNotFoundError:
        print(f"  WARNING: Extraction log not found at {extraction_log}")
        print(f"  Sampling without quality grades.")
    return grades


# ============================================================
# FRAME INVENTORY
# ============================================================

def scan_frames(frames_dir):
    """
    Scan all extracted frame JPGs and parse metadata.
    Returns list of dicts with frame info.
    """
    frames_path = Path(frames_dir)
    all_frames = []

    for jpg in frames_path.rglob("*.jpg"):
        match = FRAME_PATTERN.match(jpg.name)
        if not match:
            continue

        farm, cam, day_str, clip, date, time, fnum = match.groups()
        day_num = int(day_str)

        # Skip Cam3 early days (storage area)
        if cam == "cam3" and day_num in CAM3_SKIP_DAYS:
            continue

        all_frames.append({
            "path": str(jpg),
            "filename": jpg.name,
            "farm": farm,
            "cam": cam,
            "day_num": day_num,
            "day": f"d{day_str}",
            "clip": clip,
            "date": date,
            "time": time,
            "frame_num": int(fnum),
            "stage": get_growth_stage(day_num),
            "clip_prefix": f"{farm}_{cam}_d{day_str}_{clip}_{date}_{time}",
        })

    return all_frames


# ============================================================
# SAMPLING STRATEGY
# ============================================================

def stratified_sample(all_frames, quality_grades, target_count):
    """
    Strategically sample frames with balanced coverage:
    1. Weight by clip type (MOR1/MOR2 > EVE > AFT > NIT)
    2. Balance across cameras
    3. Balance across growth stages
    4. Prefer Grade A > Grade B > Grade C frames
    5. Spread across days (don't cluster on one day)
    """

    # Group frames by (cam, day, clip)
    groups = defaultdict(list)
    for f in all_frames:
        key = (f["cam"], f["day_num"], f["clip"])
        groups[key].append(f)

    # Score each group
    scored_groups = []
    for key, frames in groups.items():
        cam, day_num, clip = key
        clip_prefix = frames[0]["clip_prefix"]

        # Base weight from clip type
        weight = CLIP_WEIGHTS.get(clip, 1.0)

        # Quality bonus: prefer clips with more Grade A frames
        if clip_prefix in quality_grades:
            qg = quality_grades[clip_prefix]
            total = qg["grade_A"] + qg["grade_B"] + qg["grade_C"]
            if total > 0:
                a_ratio = qg["grade_A"] / total
                weight *= (0.5 + a_ratio)  # 0.5 to 1.5x multiplier

        scored_groups.append({
            "key": key,
            "cam": cam,
            "day_num": day_num,
            "clip": clip,
            "stage": frames[0]["stage"],
            "frames": frames,
            "weight": weight,
            "num_frames": len(frames),
        })

    # Calculate how many frames to sample per group
    total_weight = sum(g["weight"] for g in scored_groups)
    for g in scored_groups:
        # Proportional allocation based on weight
        g["allocation"] = max(1, round(target_count * g["weight"] / total_weight))

    # Balance across cameras (ensure no camera dominates)
    cam_counts = defaultdict(int)
    for g in scored_groups:
        cam_counts[g["cam"]] += g["allocation"]

    max_per_cam = target_count * 0.45  # No camera gets more than 45%
    for cam, count in cam_counts.items():
        if count > max_per_cam:
            scale = max_per_cam / count
            for g in scored_groups:
                if g["cam"] == cam:
                    g["allocation"] = max(1, round(g["allocation"] * scale))

    # Balance across growth stages
    stage_counts = defaultdict(int)
    for g in scored_groups:
        stage_counts[g["stage"]] += g["allocation"]

    # Ensure each stage gets at least 15% of target
    min_per_stage = target_count * 0.15
    for stage, count in stage_counts.items():
        if count < min_per_stage and count > 0:
            boost = min_per_stage / count
            for g in scored_groups:
                if g["stage"] == stage:
                    g["allocation"] = max(1, round(g["allocation"] * boost))

    # Now sample frames from each group
    sampled = []
    for g in scored_groups:
        n = min(g["allocation"], len(g["frames"]))
        if n <= 0:
            continue

        frames = g["frames"]

        # Within each group, prefer frames that are evenly spaced
        # (don't pick consecutive frames which look nearly identical)
        if n >= len(frames):
            chosen = frames
        else:
            # Evenly space the selection across frame numbers
            step = len(frames) / n
            indices = [int(i * step) for i in range(n)]
            chosen = [frames[idx] for idx in indices]

        sampled.extend(chosen)

    # If we have too many, trim randomly (preserving balance)
    if len(sampled) > target_count * 1.1:
        random.shuffle(sampled)
        sampled = sampled[:target_count]

    # If we have too few, add more from underrepresented groups
    if len(sampled) < target_count * 0.9:
        already_selected = {f["filename"] for f in sampled}
        remaining = [f for f in all_frames if f["filename"] not in already_selected]
        random.shuffle(remaining)
        needed = target_count - len(sampled)
        sampled.extend(remaining[:needed])

    return sampled


# ============================================================
# MAIN
# ============================================================

def run_sampling(frames_dir, extraction_log, output_dir, log_dir, target):
    frames_path = Path(frames_dir)
    output_path = Path(output_dir)
    log_path = Path(log_dir)

    output_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  POULTRY FRAME SAMPLER")
    print(f"{'='*65}")
    print(f"  Frames dir:  {frames_path}")
    print(f"  Output dir:  {output_path}")
    print(f"  Target:      {target} frames")
    print(f"{'='*65}")

    # Step 1: Scan frames
    print(f"\n[STEP 1] Scanning frames...")
    all_frames = scan_frames(frames_dir)
    print(f"  Total usable frames: {len(all_frames)}")

    # Breakdown
    cam_counts = defaultdict(int)
    clip_counts = defaultdict(int)
    stage_counts = defaultdict(int)
    day_counts = defaultdict(int)

    for f in all_frames:
        cam_counts[f["cam"]] += 1
        clip_counts[f["clip"]] += 1
        stage_counts[f["stage"]] += 1
        day_counts[f["day_num"]] += 1

    print(f"\n  By camera:")
    for cam in sorted(cam_counts):
        print(f"    {cam}: {cam_counts[cam]} frames")

    print(f"\n  By clip type:")
    for clip in ["MOR1", "MOR2", "AFT", "EVE", "NIT"]:
        print(f"    {clip}: {clip_counts.get(clip, 0)} frames (weight: {CLIP_WEIGHTS.get(clip, 0)}x)")

    print(f"\n  By growth stage:")
    for stage in ["early", "growing", "juvenile"]:
        print(f"    {stage}: {stage_counts.get(stage, 0)} frames")

    # Step 2: Load quality grades
    print(f"\n[STEP 2] Loading quality grades...")
    quality_grades = load_quality_grades(extraction_log)
    print(f"  Loaded grades for {len(quality_grades)} clips")

    # Step 3: Sample
    print(f"\n[STEP 3] Sampling {target} frames...")
    random.seed(42)  # Reproducible
    sampled = stratified_sample(all_frames, quality_grades, target)
    print(f"  Selected: {len(sampled)} frames")

    # Step 4: Copy sampled frames to output
    print(f"\n[STEP 4] Copying to {output_path}...")
    for f in sampled:
        src = Path(f["path"])
        dst = output_path / f["filename"]
        if not dst.exists():
            shutil.copy2(str(src), str(dst))

    # Step 5: Analyze and log
    s_cam = defaultdict(int)
    s_clip = defaultdict(int)
    s_stage = defaultdict(int)
    s_day = defaultdict(int)

    for f in sampled:
        s_cam[f["cam"]] += 1
        s_clip[f["clip"]] += 1
        s_stage[f["stage"]] += 1
        s_day[f["day_num"]] += 1

    print(f"\n  Sampled distribution:")
    print(f"\n  By camera:")
    for cam in sorted(s_cam):
        pct = s_cam[cam] / len(sampled) * 100
        print(f"    {cam}: {s_cam[cam]} ({pct:.1f}%)")

    print(f"\n  By clip type:")
    for clip in ["MOR1", "MOR2", "AFT", "EVE", "NIT"]:
        cnt = s_clip.get(clip, 0)
        pct = cnt / len(sampled) * 100
        print(f"    {clip}: {cnt} ({pct:.1f}%)")

    print(f"\n  By growth stage:")
    for stage in ["early", "growing", "juvenile"]:
        cnt = s_stage.get(stage, 0)
        pct = cnt / len(sampled) * 100
        print(f"    {stage}: {cnt} ({pct:.1f}%)")

    print(f"\n  By day:")
    for d in sorted(s_day):
        print(f"    d{d:02d}: {s_day[d]} frames", end="")
        if d % 5 == 0:
            print()
        else:
            print("  |  ", end="")
    print()

    # Step 6: Write sampling log
    sampling_log = log_path / "sampling_log.csv"
    with open(sampling_log, 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["filename", "farm", "cam", "day", "clip", "date", "time",
                         "frame_num", "stage", "clip_prefix"])
        for f in sorted(sampled, key=lambda x: x["filename"]):
            writer.writerow([
                f["filename"], f["farm"], f["cam"], f["day"], f["clip"],
                f["date"], f["time"], f["frame_num"], f["stage"], f["clip_prefix"]
            ])

    # Summary
    print(f"\n{'='*65}")
    print(f"  SAMPLING COMPLETE")
    print(f"{'='*65}")
    print(f"  Sampled frames:  {len(sampled)}")
    print(f"  Output folder:   {output_path}")
    print(f"  Sampling log:    {sampling_log}")
    print(f"\n  Morning clips (MOR1+MOR2): {s_clip.get('MOR1',0) + s_clip.get('MOR2',0)} ({(s_clip.get('MOR1',0) + s_clip.get('MOR2',0))/len(sampled)*100:.1f}%) — eating/drinking source")
    print(f"  Evening clips (EVE):        {s_clip.get('EVE',0)} ({s_clip.get('EVE',0)/len(sampled)*100:.1f}%) — pre-dark eating + sitting")
    print(f"  Afternoon clips (AFT):      {s_clip.get('AFT',0)} ({s_clip.get('AFT',0)/len(sampled)*100:.1f}%) — standing/sitting variety")
    print(f"  Night clips (NIT):          {s_clip.get('NIT',0)} ({s_clip.get('NIT',0)/len(sampled)*100:.1f}%) — IR robustness")
    print(f"\n  Next step: Upload sampled_frames/ to CVAT for annotation")
    print(f"{'='*65}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Poultry Frame Sampler — strategic frame selection for annotation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sample_frames.py \\
    --frames-dir "./extracted_frames" \\
    --extraction-log "./logs/extraction_log.csv" \\
    --output-dir "./sampled_frames" \\
    --log-dir "./logs" \\
    --target 1000
        """
    )

    parser.add_argument("--frames-dir", required=True, help="Directory with extracted frame JPGs")
    parser.add_argument("--extraction-log", required=True, help="Path to extraction_log.csv")
    parser.add_argument("--output-dir", required=True, help="Output directory for sampled frames")
    parser.add_argument("--log-dir", required=True, help="Directory for sampling_log.csv")
    parser.add_argument("--target", type=int, default=1000, help="Target number of frames to sample (default: 1000)")

    args = parser.parse_args()

    run_sampling(
        frames_dir=args.frames_dir,
        extraction_log=args.extraction_log,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        target=args.target,
    )


if __name__ == "__main__":
    main()
