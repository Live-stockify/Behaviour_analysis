# Migration Plan

How to move your existing `LiveStockify/` folder content into this clean repo structure.

## What Goes Where

| Existing File | New Location | Notes |
|---|---|---|
| `extract_frames.py` | `src/livestockify/data/extract_frames.py` | Add `def main()` if not present |
| `sample_frames.py` | `src/livestockify/data/sample_frames.py` | Add `def main()` if not present |
| `auto_organizer.py` | `src/livestockify/data/auto_organizer.py` | Already has `main()` |
| `prepare_and_train.py` | Split into `src/livestockify/training/prepare_dataset.py` + `src/livestockify/training/train.py` | Refactor: separate concerns |
| `tiled_inference.py` | `src/livestockify/inference/tiled_detector.py` | Optional, future use |
| `gdrive_downloader.py` | Keep in your local LiveStockify folder | Not part of the production repo |
| `runs/detect/.../best.pt` | `models/livestockify_v1.pt` | Rename and copy |
| `raw_clips/` | NOT in repo | Keep on disk only |
| `extracted_frames/` | NOT in repo | Keep on disk only |
| `sampled_frames/` | NOT in repo | Keep on disk only |
| `pre_annotations/` | NOT in repo | Keep on disk only |
| `roboflow_anuj/` | NOT in repo | Keep on disk only |
| `round2_dataset/` | NOT in repo | Keep on disk only |
| `comparison_results/` | NOT in repo | Keep on disk only |
| `logs/` | NOT in repo (excluded by .gitignore) | Auto-generated |
| `credentials.json` | NEVER in repo | Excluded by .gitignore |
| `live_env/` | NEVER in repo | Excluded by .gitignore |

## What NOT to Migrate

- The `runs/` folder (training artifacts, huge)
- Any `.mp4` files
- Any `extracted_frames` or `sampled_frames` data
- Any virtual environments
- `credentials.json`

## Migration Commands

Run these from inside your existing `~/Desktop/workspace/LiveStockify/` folder, after you've cloned the new repo somewhere.

```bash
# Set the variables
OLD=~/Desktop/workspace/LiveStockify
NEW=~/livestockify_clean   # wherever you cloned the new repo

# 1. Copy data scripts
cp $OLD/extract_frames.py $NEW/src/livestockify/data/
cp $OLD/sample_frames.py $NEW/src/livestockify/data/
cp $OLD/auto_organizer.py $NEW/src/livestockify/data/

# 2. Copy training script
cp $OLD/prepare_and_train.py $NEW/src/livestockify/training/

# 3. Copy production model (rename it)
cp $OLD/runs/detect/runs/detect/round2_v3/weights/best.pt $NEW/models/livestockify_v1.pt

# 4. Verify
ls -la $NEW/models/
ls -la $NEW/src/livestockify/data/
ls -la $NEW/src/livestockify/training/
```

## Post-Migration Refactoring

The old scripts work as-is when copied, but to make them properly importable as a Python package, they need small tweaks.

### Add a `main()` function and `if __name__ == "__main__"` block

If a script doesn't already have these, wrap its top-level code in a `main()` function. Example:

**Before:**
```python
parser = argparse.ArgumentParser(...)
args = parser.parse_args()
do_stuff(args)
```

**After:**
```python
def main():
    parser = argparse.ArgumentParser(...)
    args = parser.parse_args()
    do_stuff(args)

if __name__ == "__main__":
    main()
```

This lets you run them via `python -m livestockify.data.extract_frames` AND via direct execution.

### Update import paths

If any scripts import from each other, update the imports:

**Before:**
```python
from extract_frames import some_function
```

**After:**
```python
from livestockify.data.extract_frames import some_function
```

## Testing After Migration

```bash
cd $NEW
source venv/bin/activate
pip install -e .  # Install the package in dev mode

# Test that imports work
python -c "from livestockify.inference.detector import Detector; print('OK')"
python -c "from livestockify.data.extract_frames import main; print('OK')"

# Test the CLI commands
livestockify-inference --help
livestockify-extract --help
```

If imports fail, check that you ran `pip install -e .` and that the `__init__.py` files exist in all subpackages.

## What Stays in the Old Folder

Don't delete `~/Desktop/workspace/LiveStockify/` after migration. Keep it as your "data scratchpad" — it has the actual MP4s, extracted frames, and intermediate files that shouldn't go in git.

The new repo (`~/livestockify_clean/` or wherever) is your **code** repo.
The old folder is your **data** workspace.

## Setting Up Your Working Environment

After migration, your daily workflow becomes:

```bash
# When working on training/data:
cd ~/Desktop/workspace/LiveStockify   # data lives here
source live_env/bin/activate
# Use the installed package:
python -m livestockify.data.extract_frames --input-dir ./raw_clips ...

# When pushing code changes:
cd ~/livestockify_clean    # code lives here
git add .
git commit -m "..."
git push
```

You can also symlink the data folders into the clean repo if you prefer:

```bash
cd ~/livestockify_clean
ln -s ~/Desktop/workspace/LiveStockify/raw_clips ./raw_clips
ln -s ~/Desktop/workspace/LiveStockify/extracted_frames ./extracted_frames
ln -s ~/Desktop/workspace/LiveStockify/sampled_frames ./sampled_frames
```

The symlinks are excluded by .gitignore so they won't pollute git.
