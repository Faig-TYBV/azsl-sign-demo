# Azerbaijani Sign Language Recognition

Real-time recognition of Azerbaijani Sign Language (AzSL) words from a video feed, with a small post-recognition workflow that lets the user build a sentence out of the words that were detected.

The pipeline runs end-to-end on a local machine: the webcam (or any video device) is captured in the browser, frames are streamed to a Python backend over WebSocket, MediaPipe extracts hand landmarks, a 26-frame sequence is classified by a trained GRU model, prediction smoothing and a temporal segmenter convert the per-window output into discrete detected words, and the frontend lets the user pick words in any order to assemble a final sentence.

## Current pipeline

```
Camera (browser)
  -> JPEG frame (480x360, ~10 fps)
  -> WebSocket ({"type":"frame","data":"<base64 JPEG>"})
  -> MediaPipe HandLandmarker (VIDEO mode, strictly increasing timestamps)
  -> 126 hand-landmark features per frame
        (2 hands x 21 landmarks x 3 coordinates)
  -> 26-frame rolling sequence buffer
  -> normalize_frame() / preprocess_sequence()
  -> GRU temporal-pooling classifier
  -> per-window softmax (raw_prediction, raw_confidence)
  -> prediction smoothing (5-frame weighted vote)
  -> temporal segmentation
        (hand-presence debounce, motion energy,
         confidence / stability gating, cooldown)
  -> segment_event
  -> frontend Detected Words history
  -> user taps to select words in any order
  -> Final Sentence = selectedWords.join(" ")
```

The browser is intentionally thin: it captures video, encodes frames, and renders UI. All landmark extraction, normalization, model inference, smoothing, and segmentation happen in the Python backend.

## Model

- **Architecture:** GRU-based temporal classifier with temporal mean + max pooling, defined in `src/models/gru_classifier.GRUClassifier`.
- **Input:** a 26-frame sequence of 126-dim feature vectors (one vector per frame).
- **Output:** softmax over 200 classes (one per word in the vocabulary).
- **Device:** runs on CUDA when available, falls back to CPU otherwise.
- **Checkpoint:** `outputs/checkpoints/gru_temporal_pool_best.pt`. **The checkpoint file is intentionally NOT stored in this repository.** See the *Model setup* section below.

## Repository structure

```
.
+- README.md                          # this file
+- .gitignore
+- src/
|   +- features/
|   |   +- extract_landmarks.py       # MediaPipe HandLandmarker, normalize_frame, 126-dim features
|   |   +- preprocess_sequence.py     # TARGET_SEQ_LEN=26, padding/masking for the model
|   |   +- normalization.py           # feature normalization utilities
|   +- models/
|   |   +- gru_classifier.py          # GRUClassifier architecture
|   +- training/                      # training scripts (see paper / history)
|   +- inference/
|   |   +- __init__.py
|   |   +- predict.py                 # load_model, get_device, EXPECTED_CONFIG, CheckpointVerificationError
|   |   +- realtime_inference.py      # realtime_predict
|   |   +- analyze_predictions.py     # inference analysis utility
|   |   +- evaluate_dataset.py        # offline evaluation utility
|   |   +- evaluate_dataset_test.py   # self-test for evaluate_dataset
|   +- web_demo/
|       +- backend.py                 # FastAPI WebSocket server, MediaPipe, segmentation, smoothing
|       +- frontend/
|       |   +- index.html             # single-page UI (camera, history, selection, sentence)
|       +- requirements.txt           # Python dependencies for the web demo
|       +- README.md                  # web-demo specific notes
+- tests/
|   +- test_evaluate_dataset.py
|   +- test_inference.py
+- outputs/                           # generated at runtime (gitignored)
    +- checkpoints/                   # where the .pt checkpoint should be placed
```

## Installation

Recommended: a fresh virtual environment and the web demo's requirements file.

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r src\web_demo\requirements.txt
```

**macOS / Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r src/web_demo/requirements.txt
```

The `requirements.txt` includes: `fastapi`, `uvicorn`, `opencv-python`, `mediapipe`, `torch`, `numpy`, `mkcert` (for optional local HTTPS).

## Model setup

The web demo and the inference scripts both expect the trained checkpoint to live at:

```
outputs/checkpoints/gru_temporal_pool_best.pt
```

**The `.pt` file is NOT included in this repository** (`.gitignore` blocks it on purpose to keep the repo small). A teammate should:

1. Obtain the checkpoint file by whatever out-of-band channel the team uses (shared drive, internal artifact store, training pipeline, etc.). The expected filename is `gru_temporal_pool_best.pt`.
2. Create the destination directory if it does not exist:
   ```bash
   mkdir -p outputs/checkpoints
   ```
3. Place the file at the path above. The web demo (`backend.py`) loads it on startup from `PROJECT_ROOT / "outputs/checkpoints/gru_temporal_pool_best.pt"`.

If the file is missing, the backend will refuse to start and will print `Failed to load model: ...` to the console.

## Local inference

After the checkpoint is in place, the inference utilities can be run from the project root with the virtual environment active:

```bash
# Evaluate the trained model on the offline dataset
python -m src.inference.evaluate_dataset

# Run a quick self-test for the evaluation utility
python -m src.inference.evaluate_dataset_test

# Analyze predictions from a previous run
python -m src.inference.analyze_predictions
```

The realtime inference helper `src.inference.realtime_predict` is also available and is used internally by the web demo for end-to-end prediction.

## Web demo

Start the backend (which also serves the frontend at `/`):

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.web_demo.backend:app --host 0.0.0.0 --port 8000
```

**macOS / Linux:**

```bash
.venv/bin/python -m uvicorn src.web_demo.backend:app --host 0.0.0.0 --port 8000
```

Once the server prints `WebSocket backend started.` and `Uvicorn running on http://0.0.0.0:8000`, open `http://localhost:8000` in a Chromium-based browser (Chrome / Edge) or Safari. The page will:

1. Connect to the WebSocket automatically.
2. Wait for the user to press **Start Camera** and grant camera permission.
3. Stream JPEG frames to the backend at ~10 fps (every 100 ms).
4. Display the live smoothed prediction and confidence.
5. Add each completed sign (as reported by the temporal segmenter via `segment_event`) to a **Detected Words** list.
6. Let the user tap detected words in any order to build a **Final Sentence**.

Use the existing **Reset Buffer** button to clear the 26-frame sequence and segmentation state in the backend. **Clear Selection** empties the sentence without touching the detected-word history. **Clear History** clears both.

The web demo requires the model checkpoint described in the *Model setup* section above. Without it, the backend will not start.

## Testing

The `tests/` directory contains pytest-style tests. With the virtual environment active and the checkpoint in place, run them from the project root:

```bash
pytest tests/
```

Or individually:

```bash
pytest tests/test_evaluate_dataset.py
pytest tests/test_inference.py
```

## Important notes

- **The dataset is NOT included** in this repository. The `data/` directory is git-ignored.
- **Trained model checkpoints are NOT included.** Obtain `gru_temporal_pool_best.pt` out-of-band and place it at `outputs/checkpoints/gru_temporal_pool_best.pt` as described in the *Model setup* section.
- **Generated training outputs are NOT included.** Anything under `outputs/` other than the `checkpoints/` directory you populate is git-ignored (experiment JSON, training history CSVs, analysis reports, figures, etc.).
- **iPhone / Cloudflare testing was a personal remote-test workflow used by the original author and is NOT required for teammates.** Teammates should run the project locally first; only after local testing is working should they consider a remote-tunnel setup if needed.
- **No Cloudflare tunnel URLs, no temporary development files, no run logs, no virtual environments, and no scratch debug scripts are part of this repository.** The `.gitignore` is configured to keep them out.


