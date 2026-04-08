# Model Weights

This directory holds trained YOLOv8 model weights.

## Current Production Model

| File | Size | mAP50 | Trained On | Date |
|------|------|-------|-----------|------|
| `livestockify_v1.pt` | ~22 MB | 0.454 | 730 Saibabu frames, ~39K labels | 2026-04 |

### Per-class performance (v1)

| Class | Precision | Recall | mAP50 |
|---|---|---|---|
| Drinking | 0.298 | 0.125 | 0.166 |
| Eating | 0.537 | 0.749 | **0.681** |
| Sitting | 0.315 | 0.644 | 0.380 |
| Standing | 0.496 | 0.671 | 0.588 |

**Known limitations:**
- Low recall on Drinking class (only 905 training labels — 2.3% of dataset)
- Trained only on Saibabu farm data — generalization to Naveen/Vamsi unknown
- Underpredicts in densely packed later-day frames (d18+)

## How to Add a New Model

1. Train using `python -m livestockify.training.train` (see `docs/training.md`)
2. Copy the resulting `best.pt` here with a versioned name: `livestockify_vN.pt`
3. Update this README with the new model's stats
4. Update `configs/inference.yaml` to point to the new weights
5. Commit the new model file (allowed by `.gitignore`)

## File Naming Convention

```
livestockify_v{MAJOR}.pt              # Production-ready model
livestockify_v{MAJOR}_{TAG}.pt        # Experimental variant
```

Example: `livestockify_v2.pt`, `livestockify_v2_tiled.pt`
