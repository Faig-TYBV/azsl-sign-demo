"""
Real-time webcam inference for Experiment 2 (GRU + mean_max pooling) model.

Pipeline:
  Webcam -> MediaPipe HandLandmarker -> 126D features -> normalization ->
  Rolling buffer of 26 frames -> model inference -> softmax -> top-k ->
  Temporal smoothing (majority voting) -> OpenCV UI.

Uses the exact same preprocessing as the dataset and the predict.py script.
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from dataclasses import make_dataclass

# Make project-root imports work both as `python -m src.inference.realtime_inference`
# and when run from other working directories.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.extract_landmarks import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    create_hand_landmarker,
    normalize_frame,
)
from src.features.preprocess_sequence import TARGET_SEQ_LEN
from src.models.gru_classifier import GRUClassifier  # noqa: E402

# Reuse the model loading and device functions from predict.py
from src.inference.predict import get_device, load_model, EXPECTED_CONFIG, CheckpointVerificationError

# ---------------------------------------------------------------------------


from src.inference.ambiguity_gate import is_ambiguous_prediction

# Real-time specific constants
WINDOW_SIZE = 5  # for prediction smoothing history
CONFIDENCE_THRESHOLD = 0.70  # minimum confidence to consider a raw prediction for smoothing
DRAW_LANDMARKS = True  # set to False to skip drawing landmarks for performance
FPS_UPDATE_INTERVAL = 1.0  # seconds over which to compute FPS
def main() -> int:
    """Run the real-time webcam inference loop."""
    # Initialize MediaPipe HandLandmarker
    try:
        landmarker = create_hand_landmarker()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        print("Download the hand_landmarker.task model first (see README).")
        return 1

    # Initialize webcam
    cap = cv2.VideoCapture(0)  # 0 is usually the default webcam
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return 1

    # Load the Experiment 2 model
    checkpoint_path = Path("outputs/checkpoints/gru_temporal_pool_best.pt")
    try:
        model, class_to_idx, idx_to_class, device = load_model(checkpoint_path)
    except (CheckpointVerificationError, FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: Failed to load model: {exc}")
        cap.release()
        return 1
# Prepare data buffers for the rolling window
    # We'll store the last TARGET_SEQ_LEN frames (each frame is 126D) and their validity
    feature_buffer = collections.deque(maxlen=TARGET_SEQ_LEN)
    validity_buffer = collections.deque(maxlen=TARGET_SEQ_LEN)

    # For prediction smoothing: history of raw predictions (class indices)
    # We'll store tuples (class_index, confidence) for the last WINDOW_SIZE frames
    # but only if confidence >= CONFIDENCE_THRESHOLD
    pred_history = collections.deque(maxlen=WINDOW_SIZE)
    smoothed_prediction = None  # idx_to_class string or None
    smoothed_confidence = 0.0

    # FPS measurement
    fps_counter = 0
    fps_timer = cv2.getTickCount()
    fps = 0.0

    print("Starting real-time inference. Press 'Q' or 'ESC' to quit, 'R' to reset buffer.")
    while True:
            ret, frame = cap.read()
            if not ret:
                print("WARNING: Failed to grab frame from webcam.")
                break
    
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
            # MediaPipe requires a timestamp in milliseconds
            timestamp_ms = int(cv2.getTickCount() * 1000 / cv2.getTickFrequency())
    
            # Detect hand landmarks
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame),
                timestamp_ms,
            )
    
            # Convert MediaPipe result to our FrameResult-like object
            if result.hand_landmarks:
                num_hands = len(result.hand_landmarks)
                handedness = [h.handedness[0].category_name for h in result.handedness]
                landmarks = np.zeros((2, 21, 3), dtype=np.float32)  # (MAX_HANDS, 21, 3)
                for i, hand_landmarks in enumerate(result.hand_landmarks):
                    if i >= 2:
                        break
                    landmarks[i] = np.array(
                        [[lmk.x, lmk.y, lmk.z] for lmk in hand_landmarks], dtype=np.float32
                    )
            else:
                num_hands = 0
                handedness = []
                landmarks = np.zeros((2, 21, 3), dtype=np.float32)
    
            # Create a FrameResult instance for normalize_frame
            FrameResult = make_dataclass(
                "FrameResult",
                ["num_hands", "landmarks"],
            )
            frame_result = FrameResult(num_hands=num_hands, landmarks=landmarks)
            features = normalize_frame(frame_result)  # shape (126,)
            validity = 1.0 if num_hands > 0 else 0.0
    
            # Add to buffers
            feature_buffer.append(features)
            validity_buffer.append(validity)
    
            # Prepare model input if we have a full buffer
            if len(feature_buffer) == TARGET_SEQ_LEN:
                # Convert buffers to numpy arrays
                features_seq = np.array(feature_buffer, dtype=np.float32)  # (26, 126)
                validity_seq = np.array(validity_buffer, dtype=np.float32)  # (26,)
    
                # Convert to torch tensor and add batch dimension
                input_tensor = torch.from_numpy(features_seq).unsqueeze(0).to(device)  # (1, 26, 126)
    
                # Inference
                model.eval()
                with torch.inference_mode():
                    logits = model(input_tensor)  # (1, 200)
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()  # (200,)
    
                # Get top-1 and top-2 predictions
                top_2_indices = np.argsort(-probs)[:2]
                pred_idx = int(top_2_indices[0])
                pred_confidence = float(probs[pred_idx])
                pred_class = idx_to_class[pred_idx]

                if len(top_2_indices) >= 2:
                    top2_idx = int(top_2_indices[1])
                    top2_class = idx_to_class[top2_idx]
                    top2_confidence = float(probs[top2_idx])
                    is_ambig = is_ambiguous_prediction(
                        pred_class, top2_class, pred_confidence, top2_confidence, margin_threshold=0.15
                    )
                else:
                    is_ambig = False

                # Update prediction history for smoothing (only if confidence >= threshold and not ambiguous)
                if pred_confidence >= CONFIDENCE_THRESHOLD and not is_ambig:
                    pred_history.append((pred_idx, pred_confidence))
    
                # Compute smoothed prediction from history (majority voting)
                if pred_history:
                    # Get the list of class indices from history
                    history_indices = [item[0] for item in pred_history]
                    # Find the most common class index
                    smoothed_idx = max(set(history_indices), key=history_indices.count)
                    # Compute average confidence for that class in the history
                    confidences = [
                        item[1]
                        for item in pred_history
                        if item[0] == smoothed_idx
                    ]
                    smoothed_confidence = float(np.mean(confidences)) if confidences else 0.0
                    smoothed_prediction = idx_to_class[smoothed_idx]
                else:
                    # If no predictions meet the confidence threshold, keep the previous smoothed prediction
                    # (or set to None if we don't have one yet)
                    pass
    
                # For display, we'll show both raw and smoothed
                display_pred_class = pred_class
                display_pred_confidence = pred_confidence
                display_smooth_class = smoothed_prediction if smoothed_prediction else "-"
                display_smooth_confidence = smoothed_confidence
            else:
                # Not enough frames yet
                display_pred_class = "-"
                display_pred_confidence = 0.0
                display_smooth_class = "-"
                display_smooth_confidence = 0.0
    # Draw landmarks if enabled
            if DRAW_LANDMARKS and result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    for lmk in hand_landmarks:
                        x = int(lmk.x * frame.shape[1])
                        y = int(lmk.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    
            # Measure FPS
            fps_counter += 1
            if fps_counter >= 10:  # update every 10 frames to reduce overhead
                now = cv2.getTickCount()
                elapsed = (now - fps_timer) / cv2.getTickFrequency()
                if elapsed > 0:
                    fps = fps_counter / elapsed
                fps_counter = 0
                fps_timer = now
    
            # Prepare UI text
            lines = [
                f"FPS: {fps:.1f}",
                f"Raw:   {display_pred_class:<20} {display_pred_confidence:.2%}",
                f"Smooth:{display_smooth_class:<20} {display_smooth_confidence:.2%}",
                f"Buffer: {len(feature_buffer)}/{TARGET_SEQ_LEN} frames",
                "",
                "Controls:",
                "  Q/ESC: Quit",
                "  R: Reset buffer",
            ]
    
            # Draw semi-transparent background for text
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (300, 140), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
            # Draw text
            y0 = 30
            dy = 20
            for i, line in enumerate(lines):
                y = y0 + i * dy
                cv2.putText(
                    frame,
                    line,
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
    
            # Show the frame
            cv2.imshow("AzSLD Real-time Inference", frame)
    
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # Q or ESC
                break
            elif key in (ord("r"), ord("R")):  # R to reset buffer
                feature_buffer.clear()
                validity_buffer.clear()
                pred_history.clear()
                smoothed_prediction = None
                smoothed_confidence = 0.0
                print("Buffer reset.")
    cap.release()
    cv2.destroyAllWindows()
    return 0
if __name__ == "__main__":
    sys.exit(main())
