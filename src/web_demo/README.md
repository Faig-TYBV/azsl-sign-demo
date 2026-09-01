# AzSLD Web Demo

This is a web-based real-time sign language recognition demo that uses the iPhone's camera via Safari (or any mobile browser) to recognize AzSL words in real time.

## Features

- Uses the iPhone (or Android) camera through the browser.
- Sends video frames via WebSocket to a Python backend.
- Backend processes frames with MediaPipe HandLandmarker, normalizes, and feeds into the canonical Experiment 2 GRU model.
- Displays real-time prediction, confidence, and buffer progress.
- Works on local Wi-Fi network.

## Model Used

- Canonical checkpoint: `outputs/checkpoints/gru_temporal_pool_best.pt`
- GRUClassifier with input_size=126, hidden_size=128, num_layers=2, dropout=0.3, bidirectional=False, pooling="mean_max", 200 classes.

## Prerequisites

- Python 3.8+ (tested with 3.12)
- Required Python packages (install via `pip install -r requirements.txt` or see below)
- The hand landmarker model: `models/hand_landmarker.task` (already present in the repository)

## Installation

1. Clone the repository (if you haven't already).
2. Ensure you are in the repository root: `c:\Users\ASUS\Desktop\azsl-word-recognition`
3. (Optional) Create a virtual environment: `python -m venv .venv`
4. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
5. Install required packages:
   ```bash
   pip install fastapi uvicorn opencv-python mediapipe torch numpy
   ```
   Note: The repository may already have these installed from the original setup.

## How to Run

1. Start the backend server:
   ```bash
   uvicorn src.web_demo.backend:app --host 0.0.0.0 --port 8000
   ```
   This will start the server listening on all network interfaces (important for other devices to connect).

2. Find your computer's local IP address:
   - Open Command Prompt and run `ipconfig`.
   - Look for "IPv4 Address" under your Wi-Fi adapter (e.g., `192.168.1.100`).

3. On your iPhone, ensure it is connected to the same Wi-Fi network as your computer.

4. Open Safari and navigate to:
   ```
   http://<YOUR_COMPUTER_IP>:8000
   ```
   Example: `http://192.168.1.100:8000`

5. Grant camera permission when prompted.

6. Press "Start Camera" to begin the demo.

7. The video feed will appear, and predictions will be shown below.
   - Buffer progress shows how many frames have been collected (0/26 to 26/26).
   - Raw prediction shows the model's immediate output.
   - Smoothed prediction shows a temporally smoothed result (majority voting over recent predictions).
   - Confidence values are shown as percentages.

8. Press "Stop Camera" to stop the video feed and disconnect the WebSocket.
9. Press "Reset Buffer" to clear the frame buffer and prediction history.

## Notes

- The demo uses the exact same preprocessing, normalization, and model logic as the original Experiment 2 implementation. No changes were made to the canonical pipeline.
- The backend serves the frontend HTML file at the root (`/`), so opening the base URL loads the interface.
- WebSocket is used for real-time frame transmission; fallback to HTTP polling is not implemented.
- For best performance, use a modern iPhone with Safari and ensure good lighting.
- If you encounter issues, check the backend console for logs.

## Troubleshooting

- **Cannot connect**: Ensure your iPhone and computer are on the same Wi-Fi network. Verify the IP address and port (default 8000). Check that the server is running and not blocked by a firewall.
- **Camera not working**: Make sure you granted camera permission in Safari. Reload the page and try again. Some older iOS versions may have limitations.
- **Model not loading**: Ensure the checkpoint file `outputs/checkpoints/gru_temporal_pool_best.pt` exists and matches the expected configuration.
- **Performance issues**: The demo runs at the camera's frame rate; you may experience lower FPS on older devices. Consider closing other apps.

## Stopping the Server

Press `Ctrl+C` in the command prompt where the server is running.

## Acknowledgments

This demo builds upon the original AzSLD codebase and leverages the Experiment 2 GRU model.