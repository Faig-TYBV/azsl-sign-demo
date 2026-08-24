# Project Status — AzSL Word Recognition

**Status: PAUSED at feature-extraction stage (2026-08-23).**
Do not delete, overwrite, reset or refactor existing files. Do not restart full extraction blindly — it is resumable.

---

## 1. Current Project Goal

Word-level sign language recognition for Azerbaijani Sign Language using the **AzSLD_Words_200** video dataset. The pipeline extracts MediaPipe hand landmarks from videos, normalizes and temporally preprocesses them into fixed-length sequences, and will later train sequence classifiers (e.g., LSTM/GRU/Transformer) to recognize signed words.

## 2. Dataset Information

- Location: `data/raw/AzSLD_Words_200/<word_label>/<video_hash>.mp4`
- Total videos: **8,557**
- Organization: one folder per word class (62+ classes observed so far)
- Videos are hash-named `.mp4` files; original videos must remain untouched
- Dataset contains both **one-hand and two-hand** signs

## 3. Dataset Inspection Results

- Videos open correctly with OpenCV; typical lengths ~9–33 frames (P50/P75 ≈ 26 frames → chosen target sequence length)
- FPS read via `cv2.CAP_PROP_FPS`; some videos report unusual FPS values (handled)
- Class folders contain only video files; no corrupt folder structure found

## 4. Distribution Analysis Results

- Pilot sample of 16 classes: 3 one-hand / 13 two-hand videos → two-hand support is mandatory (`MAX_HANDS = 2`)
- Frame counts vary widely; uniform temporal resampling used so the full gesture is represented

## 5. MediaPipe Pipeline Architecture

Files under `src/features/`:

| File | Purpose |
|---|---|
| `extract_landmarks.py` | MediaPipe Tasks API `HandLandmarker` (VIDEO mode, ≤2 hands). Per frame: 21 landmarks × (x,y,z) = 63 features/hand → 126/frame. No-detection frames kept with validity flag + zero-filled landmarks. Normalization: wrist-relative translation + scale by ‖wrist−MCP9‖ distance. Deterministic left/right slot ordering by handedness label. |
| `preprocess_sequence.py` | Temporal resampling to 26 frames: longer sequences uniformly resampled across the FULL gesture; shorter end-padded; per-frame validity mask returned. |
| `run_pilot_extraction.py` | Pilot run on N videos across classes; saves `.npz` + metadata + summary report. |
| `run_full_extraction.py` | Full resumable extraction: per-video `.npz` under `data/features/full/<label>/`, append-only `extraction_metadata.jsonl`, skip-if-valid logic, incremental processing (never loads dataset into RAM), disk-space pre-check, progress logging every 100 videos. |
| `visualize_extraction.py` | Debug grid visualization of drawn landmarks → `outputs/figures/`. |
| `test_mediapipe_gpu.py` | Isolated GPU capability test (see §7). |

Feature file format (per video `.npz`, compressed):
- `features` — `(26, 126)` float32, resampled+padded
- `mask` — `(26,)` float32, 1.0 = real frame with hand detection
- `frame_features` — `(n_frames, 126)` raw normalized per-frame features
- `frame_valid` — `(n_frames,)` bool

Preprocessing version tag: `v1.0-wrist-scale-normalized`.

## 6. Pilot Extraction Results

- 16/16 videos successful, 0 failed
- Average detection rate: **96.8%** (min 89.3%, max 100%)
- One-hand: 3, two-hand: 13
- Feature shape verified: `(26, 126)`
- Avg feature file size ≈ 16 KB/video → full dataset ≈ **0.14 GB**
- Estimated full CPU processing time ≈ **3 hours** (~1.3 s/video)

## 7. GPU Investigation and Conclusion

- Installed MediaPipe 1.0.1 exposes `BaseOptions.Delegate.GPU`
- Test script `src/features/test_mediapipe_gpu.py`: CPU delegate works; GPU delegate fails with:
  `NotImplementedError: ValidatedGraphConfig Initialization failed. ImageCloneCalculator: GPU processing is disabled in build flags`
- **Conclusion:** the PyPI Windows wheel is compiled without GPU (OpenGL/EGL) support. Installing CUDA/cuDNN would NOT help (MediaPipe uses OpenGL ES, not CUDA). Keep CPU extraction. Do not change MediaPipe versions.

## 8. Existing Files and Purposes

```
models/hand_landmarker.task          MediaPipe HandLandmarker model bundle (verified)
data/raw/AzSLD_Words_200/            Original dataset (UNTOUCHABLE)
data/features/pilot/                 16 pilot .npz + pilot_metadata.json
data/features/full/<label>/*.npz     Extracted features (resumable output)
data/features/full/extraction_metadata.jsonl   Per-video metadata (append-only)
outputs/figures/                     Debug visualizations
outputs/reports/                     Logs and future reports
src/features/*.py                    Pipeline modules (see §5)
docs/PROJECT_STATUS.md               This document
```

## 9. Already Extracted Samples

- **2,003 / 8,557** feature files exist in `data/features/full/` (62 classes covered)
- Metadata lines in `extraction_metadata.jsonl`: 2,003
- Running average detection rate so far: **95.8%**

*(Earlier status snapshots mentioned 681 and 1847 files; current verified count is 2,003. All are valid and will be skipped on resume.)*

## 10. Remaining Work

1. Resume full extraction for remaining ~6,554 videos:
   `python src/features/run_full_extraction.py` (skips existing valid files automatically)
2. Run validation check over all outputs: expected count, shapes, corrupt files, label distribution, detection-rate statistics
3. Write final report to `outputs/reports/full_extraction_report.json` / `.txt`

## 11. Planned Next ML Stages

1. Train/val/test split stratified by word class (per-sample split; consider speaker-level splits if speaker IDs become available)
2. Baseline sequence model: bidirectional LSTM/GRU on `(26, 126)` sequences with mask-aware masked loss/metrics
3. Later: lightweight Transformer / temporal convolution; class balancing if distribution is skewed
4. Evaluation: top-1/top-5 accuracy, confusion analysis, per-class F1
5. No model training has been performed yet

## 12. Important Decisions and Reasons

| Decision | Reason |
|---|---|
| Target length 26 frames | Matches dataset P50/P75 frame counts |
| Uniform temporal resampling (not truncation) | Preserves the whole gesture |
| Keep no-detection frames + validity masks | Missing detections carry temporal info; loss can ignore them |
| Wrist-relative + scale normalization | Invariance to position and camera distance |
| Fixed left/right slot ordering | Consistent feature layout across frames/videos |
| Per-video .npz storage (no single huge file) | Resumability, avoids RAM issues, scalable (~0.14 GB total anyway) |
| Cumulative timestamp offset across videos | MediaPipe requires monotonically increasing timestamps per landmarker instance |
| `max(1, ...)` timestamp increment | Some videos report odd FPS making increment round to 0 |
| CPU-only extraction | Windows MediaPipe wheel compiled without GPU support (verified) |
| Do not install CUDA/cuDNN or downgrade MediaPipe | Would not fix a wheel build-flag limitation |

---

## Reproducibility — How to Continue

```powershell
# Environment: project venv at .venv (Python, mediapipe==1.0.1, opencv 5.0.0, numpy)

# 1. Resume full extraction (safe & resumable; skips existing valid files)
python src/features/run_full_extraction.py

# 2. Re-run pilot sanity check any time
python src/features/run_pilot_extraction.py --num-videos 16 --seed 42

# 3. Visual debug for one video
python src/features/visualize_extraction.py "data/raw/AzSLD_Words_200/<LABEL>/<HASH>.mp4"

# 4. GPU capability re-check (expected: NOT available)
python src/features/test_mediapipe_gpu.py

# Progress log during extraction: outputs/reports/full_extraction_log.txt
# Per-video metadata: data/features/full/extraction_metadata.jsonl