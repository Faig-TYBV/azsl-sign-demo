"""
Visualization / debug utility for the landmark extraction pipeline.

For a given video, renders a grid of sampled frames showing:
  - the original frame
  - detected hand landmarks (drawn on the frame)
  - detection result text (number of hands, handedness)

Saves the grid to outputs/figures/ and (optionally) shows it.

Usage:
    python src/features/visualize_extraction.py <video_path> [--num-frames 6] [--show]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.extract_landmarks import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    create_hand_landmarker,
    extract_video_landmarks,
)

# MediaPipe hand connections (landmark index pairs)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
]

COLORS = [(0, 255, 0), (0, 128, 255)]  # slot 0 green, slot 1 orange


def draw_landmarks(frame_bgr: np.ndarray, frame_result) -> np.ndarray:
    """Draw detected landmarks + connections onto a copy of the frame."""
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    for slot in range(frame_result.num_hands):
        color = COLORS[slot % len(COLORS)]
        lm = frame_result.landmarks[slot]
        pts = [(int(p[0] * w), int(p[1] * h)) for p in lm]
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, pts[a], pts[b], color, 2)
        for p in pts:
            cv2.circle(img, p, 3, color, -1)
        label = frame_result.handedness[slot] if slot < len(frame_result.handedness) else "?"
        cv2.putText(img, label, (pts[0][0] - 10, pts[0][1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize landmark extraction")
    parser.add_argument("video_path", type=Path)
    parser.add_argument("--num-frames", type=int, default=6)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--show", action="store_true", help="Display with cv2.imshow")
    args = parser.parse_args()

    if not args.video_path.is_file():
        print(f"Video not found: {args.video_path}")
        sys.exit(1)

    landmarker = create_hand_landmarker(args.model_path)
    try:
        result = extract_video_landmarks(args.video_path, landmarker)
    finally:
        landmarker.close()

    if result.error:
        print(f"Extraction error: {result.error}")
        sys.exit(1)

    n = result.original_frame_count
    print(f"Frames: {n} | with hands: {result.frames_with_hands} | "
          f"no hand: {result.frames_no_hand} | max hands: {result.max_hands_seen}")

    # Sample frames evenly across the video
    idxs = np.round(np.linspace(0, n - 1, min(args.num_frames, n))).astype(int)

    cap = cv2.VideoCapture(str(args.video_path))
    tiles = []
    for fi, target in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(target))
        ret, frame = cap.read()
        if not ret:
            continue
        fr = result.raw_frames[target]
        vis = draw_landmarks(frame, fr)
        text = f"f{target} hands={fr.num_hands}"
        cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        vis = cv2.resize(vis, (480, 360))
        tiles.append(vis)
    cap.release()

    if not tiles:
        print("No frames could be read.")
        sys.exit(1)

    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    while len(tiles) < rows * cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid_rows = [np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)]
    grid = np.vstack(grid_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"landmarks_{args.video_path.stem}.jpg"
    cv2.imwrite(str(out_path), grid)
    print(f"Saved visualization -> {out_path}")

    if args.show:
        cv2.imshow("Landmark extraction debug", grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()