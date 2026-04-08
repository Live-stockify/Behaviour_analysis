"""
Poultry CCTV Frame Extractor
==============================
Extracts frames from 10-minute video clips, validates filenames,
skips known problematic files, grades quality, and logs everything.

Usage:
    python extract_frames.py --input-dir "/path/to/raw_clips/saibabu_farm" \
                             --output-dir "/path/to/extracted_frames" \
                             --log-dir "/path/to/logs"

    # Skip quality grading for faster extraction:
    python extract_frames.py --input-dir "..." --output-dir "..." --log-dir "..." --no-grade

    # Custom extraction rate (default: 1 frame per 10 seconds):
    python extract_frames.py --input-dir "..." --output-dir "..." --log-dir "..." --interval 5
"""

import os
import re
import csv
import subprocess
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_INTERVAL = 10  # 1 frame every 10 seconds

CLIP_PATTERN = re.compile(
    r'^(saibabu|naveen|vamsi)_(cam[1-3])_d(\d{2})_(MOR1|MOR2|AFT|EVE|NIT)_(\d{8})_(\d{4})\.mp4$'
)

FARM_CONFIG = {
    "saibabu": {"start": "2026-02-09", "days": 26},
    "naveen":  {"start": "2026-02-10", "days": 24},
    "vamsi":   {"start": "2026-02-04", "days": 28},
}

STANDARD_NIT_DAYS = {1, 4, 7, 10, 13, 16, 19, 22, 25, 28}


# ============================================================
# VALIDATION
# ============================================================

def validate_file(filename):
    """
    Validate a filename. Returns (is_valid, skip_reason_or_parsed_info).
    """
    match = CLIP_PATTERN.match(filename)
    if not match:
        return False, "Filename does not match naming pattern"

    farm, cam, day_str, clip, date_str, time_str = match.groups()
    day_num = int(day_str)

    if farm not in FARM_CONFIG:
        return False, f"Unknown farm: {farm}"

    # Validate date matches day number
    start_date = datetime.strptime(FARM_CONFIG[farm]["start"], "%Y-%m-%d")
    expected_date = start_date + timedelta(days=day_num - 1)
    expected_date_str = expected_date.strftime("%Y%m%d")

    if date_str != expected_date_str:
        return False, f"Date mismatch: d{day_str} should be {expected_date_str}, got {date_str}"

    # Skip NIT clips on non-standard days (confirmed daytime content)
    if clip == "NIT" and day_num not in STANDARD_NIT_DAYS:
        return False, f"NIT on non-standard day d{day_str} — confirmed daytime content, skipping"

    # Skip NIT clips with daytime timestamps (6 AM to 6 PM)
    if clip == "NIT":
        hour = int(time_str[:2])
        if 6 <= hour <= 18:
            return False, f"NIT clip at {time_str} is daytime — mislabeled"

    return True, {
        "farm": farm, "cam": cam, "day": f"d{day_str}",
        "day_num": day_num, "clip": clip, "date": date_str,
        "time": time_str, "filename": filename,
    }


def check_duplicates(file_list):
    """Find duplicate clip keys (same farm_cam_day_clip)."""
    seen = defaultdict(list)
    for f in file_list:
        match = CLIP_PATTERN.match(f)
        if match:
            farm, cam, day, clip, date, time = match.groups()
            key = f"{farm}_{cam}_d{day}_{clip}"
            seen[key].append(f)
    return {k: v for k, v in seen.items() if len(v) > 1}


def resolve_duplicate(files):
    """Pick the most likely correct file from duplicates."""
    preferred_times = {
        "MOR1": "0600", "MOR2": "0700",
        "AFT": "1200", "EVE": "1700", "NIT": "2130"
    }

    match = CLIP_PATTERN.match(files[0])
    if not match:
        return files[0], files[1:]

    clip = match.groups()[3]
    target_time = preferred_times.get(clip, "0000")

    best = None
    rejects = []
    for f in files:
        m = CLIP_PATTERN.match(f)
        if m and m.groups()[5] == target_time:
            best = f
        else:
            rejects.append(f)

    if best is None:
        best = files[0]
        rejects = files[1:]

    return best, rejects


# ============================================================
# FRAME EXTRACTION
# ============================================================

def check_ffmpeg():
    """Verify FFmpeg is installed."""
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def extract_frames(video_path, output_dir, interval=10):
    """
    Extract frames from a video using FFmpeg.
    Returns: (success, num_frames, error_message)
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = video_path.stem
    output_pattern = str(output_dir / f"{prefix}_f%04d.jpg")
    fps_value = 1.0 / interval

    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={fps_value}",
        "-q:v", "2", "-y",
        output_pattern
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            err = result.stderr[-300:] if result.stderr else "Unknown error"
            return False, 0, f"FFmpeg error: {err}"
        frames = list(output_dir.glob(f"{prefix}_f*.jpg"))
        return True, len(frames), ""
    except subprocess.TimeoutExpired:
        return False, 0, "FFmpeg timeout (>5 min)"
    except Exception as e:
        return False, 0, str(e)


# ============================================================
# QUALITY GRADING
# ============================================================

def grade_frame(img_path):
    """Grade frame quality: A (clear), B (usable), C (unusable)."""
    try:
        import cv2
        import numpy as np
    except ImportError as e:
        print("OpenCV and NumPy are required for grading. Install with: pip install opencv-python numpy")
        return '?', 0, 0, 0

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 'C', 0, 0, 0

    sharpness = cv2.Laplacian(img, cv2.CV_64F).var()
    brightness = np.mean(img)
    contrast = np.std(img)

    if sharpness < 75 or brightness < 30 or brightness > 240 or contrast < 15:
        return 'C', round(sharpness, 1), round(brightness, 1), round(contrast, 1)
    if sharpness >= 150 and 40 <= brightness <= 220 and contrast >= 25:
        return 'A', round(sharpness, 1), round(brightness, 1), round(contrast, 1)
    return 'B', round(sharpness, 1), round(brightness, 1), round(contrast, 1)


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_extraction(input_dir, output_dir, log_dir, interval=10, do_grade=True):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Find all .mp4 files
    mp4_paths = {f.name: f for f in input_path.rglob("*.mp4")}
    mp4_files = sorted(mp4_paths.keys())

    print(f"\n{'='*65}")
    print(f"  POULTRY FRAME EXTRACTOR")
    print(f"{'='*65}")
    print(f"  Input:    {input_path}")
    print(f"  Output:   {output_path}")
    print(f"  Logs:     {log_path}")
    print(f"  Interval: 1 frame every {interval} seconds")
    print(f"  Grading:  {'YES' if do_grade else 'NO'}")
    print(f"  Files found: {len(mp4_files)}")
    print(f"{'='*65}")

    # Step 2: Validate
    print(f"\n[PHASE 1] Validating {len(mp4_files)} files...")
    valid_files = {}
    invalid_files = []

    for fname in mp4_files:
        is_valid, result = validate_file(fname)
        if is_valid:
            valid_files[fname] = result
        else:
            invalid_files.append((fname, result))

    print(f"  Valid: {len(valid_files)} | Invalid: {len(invalid_files)}")

    # Step 3: Resolve duplicates
    duplicates = check_duplicates(list(valid_files.keys()))
    duplicate_rejects = []

    for key, dup_files in duplicates.items():
        keep, rejects = resolve_duplicate(dup_files)
        for r in rejects:
            if r in valid_files:
                del valid_files[r]
                duplicate_rejects.append((r, f"Duplicate of {keep} — keeping the one with expected time"))

    if duplicate_rejects:
        print(f"  Duplicates resolved: {len(duplicate_rejects)} rejected")

    # Step 4: Write skipped_files.csv
    all_skipped = invalid_files + duplicate_rejects
    skipped_csv = log_path / "skipped_files.csv"

    with open(skipped_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "reason"])
        for fname, reason in sorted(all_skipped):
            writer.writerow([fname, reason])

    print(f"\n  {'='*55}")
    print(f"  VALIDATION RESULT")
    print(f"  {'='*55}")
    print(f"  To extract:  {len(valid_files)} clips")
    print(f"  Skipped:     {len(all_skipped)} clips")
    print(f"  {'='*55}")

    if all_skipped:
        print(f"\n  SKIPPED FILES:")
        for fname, reason in sorted(all_skipped):
            print(f"    \u2717 {fname}")
            print(f"      {reason}")

    print(f"\n  EXTRACTING FILES:")
    for fname in sorted(valid_files.keys()):
        print(f"    \u2713 {fname}")

    # Step 5: Extract frames
    print(f"\n[PHASE 2] Extracting frames from {len(valid_files)} clips...")
    expected_frames = len(valid_files) * (600 // interval)
    print(f"  Expected: ~{expected_frames} frames total\n")

    extraction_log = log_path / "extraction_log.csv"
    ext_headers = [
        "timestamp", "filename", "farm", "cam", "day", "clip", "date", "time",
        "frames_extracted", "grade_A", "grade_B", "grade_C",
        "avg_sharpness", "avg_brightness", "status", "error"
    ]

    write_header = not extraction_log.exists()
    log_f = open(extraction_log, 'a', newline='', encoding='utf-8')
    writer = csv.DictWriter(log_f, fieldnames=ext_headers)
    if write_header:
        writer.writeheader()

    total_frames = 0
    clips_ok = 0
    clips_fail = 0
    grade_totals = {"A": 0, "B": 0, "C": 0}

    for i, (fname, info) in enumerate(sorted(valid_files.items())):
        clip_output = output_path / f"{info['farm']}_{info['cam']}"
        prefix = Path(fname).stem

        # Skip if already extracted
        existing = list(clip_output.glob(f"{prefix}_f*.jpg")) if clip_output.exists() else []
        if existing:
            print(f"  [{i+1}/{len(valid_files)}] {fname} — already done ({len(existing)} frames)")
            continue

        print(f"  [{i+1}/{len(valid_files)}] {fname}", end="", flush=True)

        full_path = mp4_paths.get(fname)
        if not full_path or not full_path.exists():
            print(f" — FILE NOT FOUND")
            entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "filename": fname, **{k: info[k] for k in ["farm","cam","day","clip","date","time"]},
                "frames_extracted": 0, "status": "file_not_found", "error": "Path not accessible"
            }
            writer.writerow(entry)
            log_f.flush()
            clips_fail += 1
            continue

        success, num_frames, error = extract_frames(full_path, clip_output, interval)

        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": fname, **{k: info[k] for k in ["farm","cam","day","clip","date","time"]},
        }

        if not success:
            entry.update({"frames_extracted": 0, "status": "failed", "error": error})
            writer.writerow(entry)
            log_f.flush()
            clips_fail += 1
            print(f" — FAILED: {error[:80]}")
            continue

        clips_ok += 1
        total_frames += num_frames
        entry.update({"frames_extracted": num_frames, "status": "ok", "error": ""})

        if do_grade and num_frames > 0:
            grades = {"A": 0, "B": 0, "C": 0}
            sharp_sum = bright_sum = 0

            for ff in sorted(clip_output.glob(f"{prefix}_f*.jpg")):
                g, s, b, c = grade_frame(ff)
                if g in grades:
                    grades[g] += 1
                sharp_sum += s
                bright_sum += b

            entry.update({
                "grade_A": grades["A"], "grade_B": grades["B"], "grade_C": grades["C"],
                "avg_sharpness": round(sharp_sum / max(num_frames, 1), 1),
                "avg_brightness": round(bright_sum / max(num_frames, 1), 1),
            })
            grade_totals["A"] += grades["A"]
            grade_totals["B"] += grades["B"]
            grade_totals["C"] += grades["C"]

            print(f" — {num_frames} frames [A:{grades['A']} B:{grades['B']} C:{grades['C']}]")
        else:
            print(f" — {num_frames} frames")

        writer.writerow(entry)
        log_f.flush()

    log_f.close()

    # Step 6: Summary
    print(f"\n{'='*65}")
    print(f"  EXTRACTION COMPLETE")
    print(f"{'='*65}")
    print(f"  Clips extracted:  {clips_ok}")
    print(f"  Clips failed:     {clips_fail}")
    print(f"  Clips skipped:    {len(all_skipped)}")
    print(f"  Total frames:     {total_frames}")

    if do_grade:
        gt = grade_totals
        total_graded = gt["A"] + gt["B"] + gt["C"]
        if total_graded > 0:
            print(f"\n  Quality Distribution:")
            print(f"    Grade A (clear):    {gt['A']:>6} ({gt['A']/total_graded*100:.1f}%)")
            print(f"    Grade B (usable):   {gt['B']:>6} ({gt['B']/total_graded*100:.1f}%)")
            print(f"    Grade C (unusable):  {gt['C']:>6} ({gt['C']/total_graded*100:.1f}%)")
            print(f"    Usable (A+B):       {gt['A']+gt['B']:>6} ({(gt['A']+gt['B'])/total_graded*100:.1f}%)")

    print(f"\n  Output:")
    if output_path.exists():
        for cam_dir in sorted(output_path.iterdir()):
            if cam_dir.is_dir():
                jpg_count = len(list(cam_dir.glob("*.jpg")))
                print(f"    {cam_dir.name}/  ({jpg_count} frames)")

    print(f"\n  Logs:")
    print(f"    {extraction_log}")
    print(f"    {skipped_csv}")
    print(f"{'='*65}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Poultry CCTV Frame Extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_frames.py \\
    --input-dir "G:/My Drive/poultry_project/raw_clips/saibabu_farm" \\
    --output-dir "G:/My Drive/poultry_project/extracted_frames" \\
    --log-dir "G:/My Drive/poultry_project/logs"

  # Faster (1 frame per 5 sec):
  python extract_frames.py --input-dir ... --output-dir ... --log-dir ... --interval 5

  # Skip grading:
  python extract_frames.py --input-dir ... --output-dir ... --log-dir ... --no-grade
        """
    )

    parser.add_argument("--input-dir", required=True, help="Folder with .mp4 clips")
    parser.add_argument("--output-dir", required=True, help="Folder for extracted frames")
    parser.add_argument("--log-dir", required=True, help="Folder for CSV logs")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL, help=f"Seconds between frames (default: {DEFAULT_INTERVAL})")
    parser.add_argument("--no-grade", action="store_true", help="Skip quality grading")

    args = parser.parse_args()

    if not check_ffmpeg():
        print("\nERROR: FFmpeg not found! Install and add to PATH.")
        print("  Windows:  winget install FFmpeg")
        print("  Mac:      brew install ffmpeg")
        print("  Linux:    sudo apt install ffmpeg")
        return

    run_extraction(args.input_dir, args.output_dir, args.log_dir, args.interval, not args.no_grade)


if __name__ == "__main__":
    main()
