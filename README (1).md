# AzSL Word Recognition

Azerbaijani Sign Language Word Recognition system using MediaPipe and temporal neural networks.

## Project Structure

```
azsl-word-recognition/
├── data/
│   ├── raw/AzSLD_Words_200/   ← original dataset (200 word classes)
│   ├── processed/              ← extracted frames
│   └── features/               ← MediaPipe landmark sequences
├── models/                     ← trained model checkpoints
├── notebooks/                  ← Jupyter notebooks for exploration
├── src/
│   ├── data/                   ← data loading & inspection
│   ├── features/               ← feature extraction (MediaPipe)
│   ├── models/                 ← model definitions
│   ├── evaluation/             ← metrics & error analysis
│   └── utils/                  ← shared helpers
├── outputs/
│   ├── figures/                ← plots & visualizations
│   └── reports/                ← inspection & evaluation reports
└── setup_project.py            ← dependency check & dir creation
```

## Quick Start

```bash
# 1. Setup
python setup_project.py

# 2. Inspect dataset
python src/data/inspect_dataset.py

# 3. (Next) Extract MediaPipe features
# python src/features/extract_landmarks.py
```

## Dataset

**AzSLD_Words_200** — 200 Azerbaijani sign language word classes, each with multiple video samples.

## Pipeline

```
Video → Frames → MediaPipe Hands → Landmark Sequences → ML Model → Azerbaijani Word
```
