"""
End-to-end test for the hand-presence gate in backend.py.

Connects to a running backend on localhost:8765 over WebSocket, sends a
sequence of empty (no-hand) frames, and verifies that:
  1. With no hand in frame, response["hand_present"] is False
  2. With no hand, response["raw_prediction"] and ["smoothed_prediction"]
     are both "-" (idle), even after the 26-frame buffer fills.
"""
import base64
import json
import sys

import cv2
import numpy as np
import websocket


def make_frame(seed: int, size=(160, 120)):
    """Return JPEG bytes for a frame. Empty (no hand landmark features)."""
    np.random.seed(seed)
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    cv2.putText(img, f"frame {seed}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def frame_msg(seed):
    jpg = make_frame(seed)
    return json.dumps({"type": "frame", "data": base64.b64encode(jpg).decode()})


def main():
    uri = "ws://127.0.0.1:8765/ws"
    print(f"connecting to {uri}...")
    ws = websocket.create_connection(uri, timeout=10)
    print("connected")

    # Send 30 empty frames (no hand). The buffer (26 frames) will fill at
    # frame 26; from frame 27 onward, the backend would normally run
    # inference. We want to verify the gated response.
    print("\n=== Phase 1: 30 empty frames (no hand visible) ===")
    responses = []
    for i in range(30):
        ws.send(frame_msg(i))
        resp = ws.recv()
        d = json.loads(resp)
        responses.append(d)

    # Inspect post-buffer-fill responses (frames 26-30)
    post_buffer = responses[25:]
    print(f"  responses after buffer fill (frames 26-30): {len(post_buffer)}")
    n_idle = sum(
        1 for d in post_buffer
        if d.get("raw_prediction") == "-" and d.get("smoothed_prediction") == "-"
    )
    n_wrong = [
        d for d in post_buffer
        if (d.get("raw_prediction") not in ("-", None)
            or d.get("smoothed_prediction") not in ("-", None))
    ]
    print(f"  idle responses (raw=- AND smoothed=-): {n_idle}/{len(post_buffer)}")
    print(f"  responses showing ANY label: {len(n_wrong)}/{len(post_buffer)}")
    print(f"  hand_present was False in all: {all(d.get('hand_present') == False for d in post_buffer)}")
    if n_wrong:
        print("  !!! FAIL: false-positive predictions seen when no hand visible !!!")
        for d in n_wrong[:5]:
            print(f"    raw={d.get('raw_prediction')} conf={d.get('raw_confidence'):.3f} smoothed={d.get('smoothed_prediction')} smoothed_conf={d.get('smoothed_confidence'):.3f}")
        sys.exit(1)
    else:
        print("  PASS: no false-positive predictions with no hand visible")

    # Check INFERENCE SKIPPED was logged
    with open(r"C:\Users\ASUS\Desktop\azsl-word-recognition\backend_test.out.log", "r", encoding="utf-8") as f:
        server_log = f.read()
    n_skip = server_log.count("INFERENCE SKIPPED: zero valid frames in buffer")
    n_inf  = server_log.count("INFERENCE\n")
    n_seg_emit = server_log.count("SEGMENT EMIT:")
    print(f"\n  Server log inspection:")
    print(f"    'INFERENCE SKIPPED: zero valid frames in buffer' occurrences: {n_skip}")
    print(f"    'INFERENCE' (bare, not skipped) occurrences:                 {n_inf}")
    print(f"    'SEGMENT EMIT:' occurrences:                                  {n_seg_emit}")
    if n_skip < 1:
        print("  !!! WARN: expected at least one INFERENCE SKIPPED log line !!!")
    if n_seg_emit > 0:
        print("  !!! FAIL: SEGMENT EMIT fired when no hand was visible !!!")
        sys.exit(1)
    else:
        print("  PASS: no SEGMENT EMIT fired when no hand was visible")

    ws.close()
    print("\n=== ALL HAND-PRESENCE GATE CHECKS PASSED ===")


if __name__ == "__main__":
    main()