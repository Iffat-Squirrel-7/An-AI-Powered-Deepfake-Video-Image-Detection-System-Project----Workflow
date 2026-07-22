# Phase 2: Dataset Collection & Preparation

## Objectives
- Acquire high-quality labeled datasets of real and fake videos/images.
- Organize data into a consistent directory structure for downstream processing.

## Recommended Datasets
| Dataset | Content | Size | Notes |
|---|---|---|---|
| FaceForensics++ | Real + 4 manipulation methods (Deepfakes, Face2Face, FaceSwap, NeuralTextures) | ~1.8M frames | Requires signed access agreement |
| Celeb-DF (v2) | High-quality celebrity deepfakes | 590 real, 5639 fake videos | Good for generalization testing |
| DFDC (Deepfake Detection Challenge) | Facebook's large-scale dataset | 100k+ videos | Very large, diverse |
| WildDeepfake | Deepfakes collected from the internet | 7,314 face sequences | Good for real-world generalization |

## Key Tasks
1. Request/download datasets, verify checksums and licenses.
2. Set up folder structure:
```
data/
├── raw/
│   ├── real/
│   └── fake/
├── processed/
└── splits/
    ├── train.csv
    ├── val.csv
    └── test.csv
```
3. Build a metadata CSV (`video_id, label, source_dataset, split`).
4. Balance classes (real vs fake) — apply oversampling/undersampling if imbalanced.
5. Set up **DVC** for dataset version tracking.

## Sample Script: Metadata Builder
```python
import os
import pandas as pd

def build_metadata(root_dir):
    rows = []
    for label in ['real', 'fake']:
        folder = os.path.join(root_dir, label)
        for fname in os.listdir(folder):
            rows.append({'filepath': os.path.join(folder, fname),
                         'label': 0 if label == 'real' else 1})
    return pd.DataFrame(rows)

df = build_metadata('data/raw')
df.to_csv('data/metadata.csv', index=False)
```

## Deliverables
- Downloaded and verified datasets
- `metadata.csv` with all samples labeled
- Train/val/test split files
