"""
Isolated GPU capability test for MediaPipe 1.0.1 HandLandmarker on Windows.

Checks:
  1. Whether the installed MediaPipe Tasks API exposes a GPU delegate.
  2. Whether HandLandmarker can be initialized with the GPU delegate.
  3. Whether GPU inference actually works on a synthetic test frame.
  4. Reports CPU vs GPU mode clearly.

Does NOT modify the existing CPU pipeline and does NOT install anything.

Usage:
    python src/features/test_mediapipe_gpu.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = Path("models/hand_landmarker.task")


def gpu_delegate_available() -> bool:
    try:
        delegates = list(mp_python.BaseOptions.Delegate)
        print(f"Available delegates: {delegates}")
        return mp_python.BaseOptions.Delegate.GPU in delegates
    except AttributeError:
        return False


def nvidia_gpu_present() -> bool:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            print(f"NVIDIA GPU detected: {out.stdout.strip()}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    print("nvidia-smi not available or no NVIDIA GPU detected.")
    return False


def make_options(delegate) -> mp_vision.HandLandmarkerOptions:
    return mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(MODEL_PATH),
            delegate=delegate,
        ),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def test_inference(landmarker, label: str) -> bool:
    """Run one synthetic frame through VIDEO-mode inference."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (40, 40, 40)
    cv2.circle(frame, (320, 240), 80, (200, 180, 160), -1)  # skin-ish blob
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    try:
        result = landmarker.detect_for_video(mp_image, 0)
        print(f"[{label}] inference OK (hands detected: {len(result.hand_landmarks)})")
        return True
    except Exception as exc:
        print(f"[{label}] inference FAILED: {type(exc).__name__}: {exc}")
        return False


def main() -> None:
    print("=" * 60)
    print("  MediaPipe GPU Capability Test")
    print("=" * 60)
    print(f"MediaPipe version: {mp.__version__}")
    print(f"Model: {MODEL_PATH} (exists: {MODEL_PATH.is_file()})")

    if not MODEL_PATH.is_file():
        print("Model file missing — aborting.")
        sys.exit(1)

    nvidia_gpu_present()

    if not gpu_delegate_available():
        print("\nRESULT: GPU delegate NOT exposed by installed MediaPipe API.")
        print("Reason: mediapipe.tasks.python.core.base_options.BaseOptions has no")
        print("Delegate.GPU member. The Windows pip wheel of MediaPipe 1.0.1 is")
        print("built CPU-only; GPU delegate requires a build with OpenGL/Metal")
        print("support which is not shipped in the standard PyPI Windows wheel.")
        print("No packages will be installed. Stopping.")
        sys.exit(0)

    print("\nGPU delegate enum is available. Attempting GPU initialization...")

    # --- CPU baseline ---
    try:
        cpu_lm = mp_vision.HandLandmarker.create_from_options(make_options(mp_python.BaseOptions.Delegate.CPU))
        cpu_ok = test_inference(cpu_lm, "CPU")
        cpu_lm.close()
    except Exception as exc:
        cpu_ok = False
        print(f"[CPU] initialization FAILED: {exc}")

    # --- GPU attempt ---
    gpu_ok = False
    try:
        gpu_lm = mp_vision.HandLandmarker.create_from_options(make_options(mp_python.BaseOptions.Delegate.GPU))
        print("[GPU] HandLandmarker initialized with GPU delegate.")
        gpu_ok = test_inference(gpu_lm, "GPU")
        gpu_lm.close()
    except Exception as exc:
        print(f"[GPU] initialization/inference FAILED: {type(exc).__name__}: {exc}")
        print("Likely cause: the PyPI Windows wheel of MediaPipe is compiled")
        print("without GPU (OpenGL ES / EGL) support, so the GPU delegate falls")
        print("back or errors out at graph initialization.")

    print("\n" + "=" * 60)
    print("  RESULT")
    print("=" * 60)
    print(f"  CPU mode works : {cpu_ok}")
    print(f"  GPU mode works : {gpu_ok}")
    if gpu_ok:
        print("  -> GPU acceleration IS available. Pipeline can be updated to")
        print("     use delegate=BaseOptions.Delegate.GPU.")
    else:
        print("  -> GPU acceleration is NOT available in this MediaPipe")
        print("     Windows build. Keep the CPU pipeline. Do not install")
        print("     CUDA/cuDNN packages — they will not help; the limitation")
        print("     is in the MediaPipe wheel itself, not the driver.")
    print("=" * 60)


if __name__ == "__main__":
    main()