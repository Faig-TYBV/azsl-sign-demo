/* =============================================================================
 * azsl_features.js - browser-side 126-dim word-mode feature extraction.
 *
 * Exact JS counterpart of the Python training/inference pipeline:
 *     src/features/extract_landmarks.py :: normalize_hand / normalize_frame
 *     src/features/extract_landmarks.py :: _mp_result_to_frame_result (slot order)
 *
 * WHY THIS EXISTS
 *   Recognition used to mean shipping 480x360 JPEGs to the server at ~10 fps
 *   (~1.2-2.4 Mbps per signer, and a MediaPipe instance per stream on a
 *   0.1-CPU box). Running MediaPipe in the browser instead means only the
 *   126 floats per frame need to travel, and the server is left with just the
 *   GRU - 203k parameters, microseconds.
 *
 * PARITY IS THE WHOLE POINT
 *   These vectors feed a model trained on the Python implementation's output.
 *   Any divergence - a different slot order, a different scale reference -
 *   degrades accuracy silently, with no error to notice. tests/test_js_feature_parity.py
 *   runs this file under node against the Python functions on shared inputs and
 *   requires the vectors to be identical. Do not change anything here without
 *   that test passing.
 *
 * No DOM and no MediaPipe dependency: pure compute, so it runs under node.
 * ============================================================================= */
(function (global) {
  'use strict';

  var NUM_LANDMARKS = 21;
  var NUM_COORDS = 3;
  var FEATURES_PER_HAND = NUM_LANDMARKS * NUM_COORDS; // 63
  var MAX_HANDS = 2;
  var FEATURE_DIM = MAX_HANDS * FEATURES_PER_HAND;    // 126

  // Landmark 9 is the middle-finger MCP. The distance from the wrist to it is
  // the scale reference, which is what makes the features invariant to how far
  // the hand is from the camera.
  var WRIST = 0;
  var MIDDLE_MCP = 9;

  // Matches the Python `if scale < 1e-6: return zeros`.
  var SCALE_EPS = 1e-6;

  /**
   * Wrist-relative, scale-normalised landmarks for a single hand.
   * @param {Array<Array<number>>} landmarks - 21 x [x, y, z]
   * @returns {Array<number>} 63 values, row-major (point-major, then x,y,z)
   */
  function normalizeHand(landmarks) {
    var out = new Array(FEATURES_PER_HAND);

    var wx = landmarks[WRIST][0];
    var wy = landmarks[WRIST][1];
    var wz = landmarks[WRIST][2];

    var rx = landmarks[MIDDLE_MCP][0] - wx;
    var ry = landmarks[MIDDLE_MCP][1] - wy;
    var rz = landmarks[MIDDLE_MCP][2] - wz;
    var scale = Math.sqrt(rx * rx + ry * ry + rz * rz);

    // Degenerate detection: the Python returns an all-zero hand rather than
    // dividing by ~0 and producing NaNs the model was never trained on.
    if (!(scale >= SCALE_EPS)) {
      for (var z = 0; z < FEATURES_PER_HAND; z++) out[z] = 0;
      return out;
    }

    for (var i = 0; i < NUM_LANDMARKS; i++) {
      out[i * 3] = (landmarks[i][0] - wx) / scale;
      out[i * 3 + 1] = (landmarks[i][1] - wy) / scale;
      out[i * 3 + 2] = (landmarks[i][2] - wz) / scale;
    }
    return out;
  }

  /**
   * Order hands into their fixed feature slots.
   *
   * The Python sorts by the handedness label string, so "Left" always occupies
   * slot 0 when both hands are present. A stable ordering matters more than
   * which hand wins: the model learned one layout, and swapping it mid-sequence
   * would look like both hands teleporting.
   *
   * @param {Array<{label: string, landmarks: Array}>} hands
   * @returns {Array} the same objects, ordered
   */
  function orderHands(hands) {
    var indexed = hands.map(function (h, i) {
      return { hand: h, i: i, label: (h && h.label) || 'hand' + i };
    });
    // Python's sorted() is stable and compares strings lexicographically.
    indexed.sort(function (a, b) {
      if (a.label < b.label) return -1;
      if (a.label > b.label) return 1;
      return a.i - b.i;
    });
    return indexed.map(function (e) { return e.hand; });
  }

  /**
   * The 126-dim frame vector. Hands beyond MAX_HANDS are dropped; missing
   * hands leave their slot zero-filled, exactly as the Python does.
   *
   * @param {Array<{label: string, landmarks: Array}>} hands - 0, 1 or 2 hands
   * @returns {Array<number>} 126 values
   */
  function frameFeatures126(hands) {
    var feats = new Array(FEATURE_DIM);
    for (var z = 0; z < FEATURE_DIM; z++) feats[z] = 0;

    if (!hands || !hands.length) return feats;

    var ordered = orderHands(hands);
    var n = Math.min(ordered.length, MAX_HANDS);
    for (var slot = 0; slot < n; slot++) {
      var lm = ordered[slot] && ordered[slot].landmarks;
      if (!lm || lm.length < NUM_LANDMARKS) continue;
      var norm = normalizeHand(lm);
      var base = slot * FEATURES_PER_HAND;
      for (var k = 0; k < FEATURES_PER_HAND; k++) feats[base + k] = norm[k];
    }
    return feats;
  }

  /** MediaPipe Tasks result -> the shape the functions above expect. */
  function handsFromMediaPipe(result) {
    var out = [];
    if (!result || !result.landmarks) return out;
    for (var i = 0; i < result.landmarks.length; i++) {
      var label = 'hand' + i;
      try {
        label = result.handedness[i][0].categoryName;
      } catch (e) { /* keep the positional fallback, as the Python does */ }
      out.push({
        label: label,
        landmarks: result.landmarks[i].map(function (p) { return [p.x, p.y, p.z]; }),
      });
    }
    return out;
  }

  var api = {
    NUM_LANDMARKS: NUM_LANDMARKS,
    FEATURES_PER_HAND: FEATURES_PER_HAND,
    MAX_HANDS: MAX_HANDS,
    FEATURE_DIM: FEATURE_DIM,
    normalizeHand: normalizeHand,
    orderHands: orderHands,
    frameFeatures126: frameFeatures126,
    handsFromMediaPipe: handsFromMediaPipe,
  };

  global.AzslFeatures = api;
  // So the parity test can require() this file under node.
  if (typeof module !== 'undefined' && module.exports) module.exports = api;

})(typeof window !== 'undefined' ? window : globalThis);
