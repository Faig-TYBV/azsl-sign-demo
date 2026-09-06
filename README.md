# Azerbaijani Sign Language (AzSL) Recognition

Production-ready, real-time Azerbaijani Sign Language (AzSL) recognition platform featuring both **Word-level continuous gesture recognition** (24 curated vocabulary signs) and **Alphabet fingerspelling recognition** with sentence assembly workflows.

The system runs end-to-end with high responsiveness: browser camera input is streamed via WebSocket to a FastAPI backend, MediaPipe extracts 3D hand landmarks, sequences are normalized and classified using a deep 2-layer GRU with temporal mean+max pooling, and predictions are displayed in a responsive, sticky interaction HUD.

---

## Key Features

1. **Dual Recognition Modes**:
   - **Word Mode**: Trial-based 26-frame sequence recognition for 24 high-frequency AzSL vocabulary words with automated 3-second countdown (`HAZIRLAŞIN... 3 -> 2 -> 1 -> BAŞLA!`) and result freezing.
   - **Alphabet Mode**: Real-time static fingerspelling with 84-dimensional geometric angle feature extraction and gesture stabilization.
2. **Interactive Sentence Builder**:
   - Detected words and fingerspelled terms can be selected and combined into complete sentences.
3. **Mobile & Desktop Optimized**:
   - Sticky Trial HUD positioned at the top of the viewport with safe-area support, thumb-friendly START/SKIP buttons, visual/ASCII progress bars, and zero required scrolling.
4. **Offline & Real-time Parity**:
   - Verified exact feature calculation and preprocessing parity between offline PyTorch models, MediaPipe pipelines, and browser runtime.

---

## System Architecture

```text
Camera (Browser / Mobile)
  │
  ├── [Word Mode] ───────────────────────────────────────────────┐
  │   → Base64 JPEG frame (~10-15 fps)                           │
  │   → WebSocket (/ws)                                          │
  │   → MediaPipe HandLandmarker (126 features: 2 hands x 21 x 3)│
  │   → 3-second Countdown (HAZIRLAŞIN... 3 -> 2 -> 1 -> BAŞLA!) │
  │   → 26-frame Rolling Capture Buffer                          │
  │   → Feature Normalization (Train-fitted 126-dim statistics) │
  │   → 2-Layer GRU Classifier (Hidden 128, Temporal Mean+Max)   │
  │   → Linear Head (256 -> 24 classes)                          │
  │   → Softmax Prediction + Confidence                          │
  │   → Result State & History Assembly                          │
  │                                                              │
  └── [Alphabet Mode] ───────────────────────────────────────────┘
      → Normalized 21 Hand Landmarks
      → 84-dim Geometric Feature Vector (Flexion, Spread, Direction)
      → Hierarchical MLP / Classifier
      → Temporal Gesture Stabilizer (Debounce & State Retention)
      → Live Spelled Word Buffer -> Add to Sentence
```

---

## Production Vocabulary (24 Classes)

Experiment 8 curates the following 24 essential sign language words (capped at 50 training samples per class to maintain class balance):

| Index | Word | Index | Word | Index | Word |
| :---: | :--- | :---: | :--- | :---: | :--- |
| **0** | MƏN | **8** | GƏLMƏK | **16** | HARDA |
| **1** | SƏN | **9** | YEMƏK | **17** | NECƏ |
| **2** | BİZ | **10** | ALMAQ | **18** | BURDA |
| **3** | SİZ | **11** | OLMAQ | **19** | BU GÜN |
| **4** | SALAM | **12** | EV | **20** | SABAH |
| **5** | SAĞLAM | **13** | BAKI | **21** | VAR |
| **6** | İSTƏMƏK | **14** | AZƏRBAYCAN | **22** | YOX |
| **7** | GETMƏK | **15** | TELEFON | **23** | BU |

---

## Production Model & Benchmark Metrics

The production model is trained under **Experiment 8 (Vocabulary 24, Cap 50)**:

* **Architecture**: 2-layer unidirectional GRU (hidden_size=128, dropout=0.3)
* **Temporal Pooling**: Concatenated Mean Pooling (128) + Max Pooling (128) = 256-dim embedding
* **Head**: Linear(256 $\to$ 24)
* **Parameters**: 203,544 trainable parameters
* **Sequence Length**: 26 frames $\times$ 126 features
* **Loss**: Class-weighted CrossEntropyLoss

### Official Held-out Test Set Results (676 Samples)

| Metric | Measured Result |
| :--- | :---: |
| **Test Accuracy** | **85.21%** (576 / 676 correct) |
| **Macro F1-Score** | **74.21%** |
| **Weighted F1-Score** | **86.03%** |
| **Macro Recall** | **80.26%** |
| **Macro Precision** | **72.10%** |
| **Zero-F1 Classes** | **0** (All 24 classes successfully recognized) |

---

## Model Artifacts & Files

| Component | Path | Description | Size |
| :--- | :--- | :--- | :---: |
| **Word Model Checkpoint** | `outputs/vocabulary_24_cap50/checkpoints/gru_24_cap50_best.pt` | Production PyTorch model weights | 2.45 MB |
| **Feature Normalizer** | `outputs/vocabulary_24_cap50/metadata/feature_normalization_stats_24.json` | 126-dim per-dimension mean & std | 6.6 KB |
| **Dataset Split** | `outputs/vocabulary_24_cap50/dataset_split_24_cap50.json` | Official 24-class partition (865 train / 676 val / 676 test) | 160 KB |
| **Alphabet Model** | `src/web_demo/frontend/models/azsl_hierarchical_model.json` | Weights and scalers for fingerspelling | 476 KB |
| **MediaPipe Task** | `models/hand_landmarker.task` | Google MediaPipe Hand Landmarker model | 7.46 MB |

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/NurlanAslnzade/Holberton-azsl-word-recognition.git
cd Holberton-azsl-word-recognition
```

### 2. Create Virtual Environment & Install Dependencies

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r src/web_demo/requirements.txt
pip install scikit-learn pytest
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/web_demo/requirements.txt
pip install scikit-learn pytest
```

### 3. Setup MediaPipe Model (if not present)
If `models/hand_landmarker.task` is not present, download it from Google's official MediaPipe repository:
```bash
mkdir -p models
curl -o models/hand_landmarker.task -L https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

---

## Running the Web Demo

Launch the production FastAPI backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.web_demo.backend:app --host 0.0.0.0 --port 8000
```

Once the console displays:
```text
Loading Experiment 8 word model and normalizer...
Experiment 8 model (24 classes) and normalizer loaded successfully on cuda
Loading alphabet classifier...
Alphabet classifier loaded successfully.
WebSocket backend started.
Uvicorn running on http://0.0.0.0:8000
```

Open `http://localhost:8000` in your web browser (Chrome, Edge, or Safari).

### Usage:
* **Word Mode**:
  1. Click **Start Camera**.
  2. Press **START TRIAL [SPACE]** (or tap spacebar).
  3. Prepare during the 3-second countdown (`HAZIRLAŞIN... 3 -> 2 -> 1 -> BAŞLA!`).
  4. Perform the gesture as the HUD records 26 frames (`FRAME 1/26 -> 26/26`).
  5. The result is classified, frozen, and displayed.
  6. Tap **NEXT TRIAL [SPACE]** to proceed to the next word.
* **Alphabet Mode**:
  1. Switch mode toggle to **Alphabet Mode**.
  2. Present hand gestures to spell letters. Hold each gesture for ~700 ms to accept.
  3. Use **Add to Sentence**, **Backspace**, or **Clear Word** to construct phrases.

---

## Running the Test Suite

Run the full automated test suite (all 91 tests must pass):

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q
```

Expected output:
```text
91 passed, 3 warnings in ~8-10s
```

---

## Training & Reproducibility

To retrain the 24-class Experiment 8 model from features:

```powershell
python scripts/run_experiment_8_training.py
```

The script:
1. Loads the 24-class split from `outputs/vocabulary_24_cap50/dataset_split_24_cap50.json`.
2. Applies normalization statistics.
3. Trains the 2-layer GRU with AdamW ($lr=1e-3, weight\_decay=1e-4$) and AMP on CUDA.
4. Performs early stopping (patience=7) on validation Macro F1.
5. Evaluates the best checkpoint on the held-out 676 test samples.

---

## License & Acknowledgements

* **AzSL Dataset**: Azerbaijani Sign Language Dataset (AzSLD).
* **MediaPipe**: Hand landmark detection by Google (Apache 2.0).
* **Fingerspelling Pipeline**: Murad's hierarchical AzSL fingerspelling neural classifier.


