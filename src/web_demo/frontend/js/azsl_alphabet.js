/* =============================================================================
 * azsl_alphabet.js - client-side AzSL alphabet (fingerspelling) inference.
 *
 * Single-file port of the relevant subset of alphabet-repo/js/gestures.js.
 * Preserves every algorithm and every numeric constant.
 *
 * Differences from the original ES-module file:
 *   - No ES export/import (browser <script> tag only).
 *   - Attached to window.AzslAlphabet.
 *   - No DOM or MediaPipe dependency - pure compute module.
 *
 * Trained input contract (must NOT change without retraining the model):
 *   - 84-dim feature vector per call: [0..62] normalized landmarks (xyz),
 *     [63..77] 15 finger joint angles, [78..81] 4 fingertip-gap distances,
 *     [82..83] 2 wrist velocity components.
 *
 * Public API (on window.AzslAlphabet): see bottom of file.
 * ============================================================================= */
(function (global) {
  'use strict';

  var LM = {
    WRIST: 0,
    THUMB_MCP: 2, THUMB_IP: 3, THUMB_TIP: 4,
    INDEX_MCP: 5, INDEX_PIP: 6, INDEX_TIP: 8,
    MIDDLE_MCP: 9, MIDDLE_PIP: 10, MIDDLE_TIP: 12,
    RING_MCP: 13, RING_PIP: 14, RING_TIP: 16,
    PINKY_MCP: 17, PINKY_PIP: 18, PINKY_TIP: 20
  };

  var FINGERS = [
    { base: LM.WRIST,   j1: LM.THUMB_MCP,  j2: LM.THUMB_IP,  tip: LM.THUMB_TIP  },
    { base: LM.WRIST,   j1: LM.INDEX_MCP,  j2: LM.INDEX_PIP, tip: LM.INDEX_TIP  },
    { base: LM.WRIST,   j1: LM.MIDDLE_MCP, j2: LM.MIDDLE_PIP, tip: LM.MIDDLE_TIP },
    { base: LM.WRIST,   j1: LM.RING_MCP,   j2: LM.RING_PIP,   tip: LM.RING_TIP   },
    { base: LM.WRIST,   j1: LM.PINKY_MCP,  j2: LM.PINKY_PIP,  tip: LM.PINKY_TIP  }
  ];
  var TIP_PAIRS = [
    [LM.THUMB_TIP, LM.INDEX_TIP],
    [LM.INDEX_TIP, LM.MIDDLE_TIP],
    [LM.MIDDLE_TIP, LM.RING_TIP],
    [LM.RING_TIP,  LM.PINKY_TIP]
  ];

  // Canonical Azerbaijani Latin alphabet (32 letters) - real UTF-8 chars.
  var AZ_ALPHABET = [
    'A', 'B', 'C', 'Ç', 'D', 'E', 'Ə', 'F', 'G', 'Ğ',
    'H', 'X', 'I', 'İ', 'J', 'K', 'Q', 'L', 'M', 'N',
    'O', 'Ö', 'P', 'R', 'S', 'Ş', 'T', 'U', 'Ü', 'V',
    'Y', 'Z'
  ];

  var LABELS = { SPACE: 'SPACE', DEL: 'DEL' };
  var CONTROL_LABEL_META = {
    SPACE: { friendly: 'SPACE',  symbol: '␣' },
    DEL:   { friendly: 'DELETE', symbol: '⌫' }
  };
  function labelMeta(label) {
    return CONTROL_LABEL_META[label] || { friendly: label, symbol: label };
  }

  var MIN_CONFIDENCE = 0.30;
  var HEURISTIC_CONF  = 0.92;
  var ANGLE_STRAIGHT  = 155;
  var ANGLE_FIST      = 100;

  var TRAJECTORY_WINDOW                = 14;
  var DYNAMIC_MOTION_THRESHOLD         = 0.045;
  var DYNAMIC_HYSTERESIS_MS            = 220;
  var TRAJECTORY_CONFIDENCE_BOOST      = 0.18;
  var TRAJECTORY_CONFIDENCE_PENALTY    = 0.25;

  var STATIC_DYNAMIC_PAIRS = {
    O: { dynamic: 'Ö', rule: 'downward' },
    U: { dynamic: 'Ü', rule: 'oscillation' },
    C: { dynamic: 'Ç', rule: 'downward' }
  };
  var KNOWN_DYNAMIC_LETTERS = {
    'Ç': 1, 'D': 1, 'G': 1, 'K': 1,
    'Ö': 1, 'Ü': 1, 'Y': 1, 'Z': 1, 'İ': 1
  };
  var CONTROL_OVERRIDE_CONFIDENCE = 0.75;

  // --- Geometry helpers ---
  function vecSub(a, b) { return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z }; }
  function vecMag(v)    { return Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }
  function vecDist(a, b){ return vecMag(vecSub(a, b)); }
  function angleBetween(v1, v2) {
    var mags = vecMag(v1) * vecMag(v2) || 1e-6;
    var cos  = Math.min(1, Math.max(-1, (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z) / mags));
    return Math.acos(cos) * (180 / Math.PI);
  }
  function jointAngleDeg(coords, mcpIdx, pipIdx, tipIdx) {
    var toMcp = vecSub(coords[mcpIdx], coords[pipIdx]);
    var toTip = vecSub(coords[tipIdx],  coords[pipIdx]);
    return angleBetween(toMcp, toTip);
  }

  // --- Normalization (must match scripts/extract_azsl_model.py) ---
  function normalizeLandmarks(landmarks, mirrorX) {
    var wrist = landmarks[LM.WRIST];
    var shifted = landmarks.map(function (p) { return vecSub(p, wrist); });
    var scale = vecMag(shifted[LM.MIDDLE_MCP]);
    if (scale < 1e-6) scale = 1e-6;
    var normalized = shifted.map(function (p) {
      return { x: p.x / scale, y: p.y / scale, z: p.z / scale };
    });
    if (mirrorX) normalized.forEach(function (p) { p.x = -p.x; });
    return normalized;
  }

  // ==========================================================================
  // 84-DIM FEATURE BUILDER (input contract - must match training exactly)
  // ==========================================================================
  function buildFeatureVector84(coords, velocity) {
    var v = velocity || { x: 0, y: 0 };
    var features = new Array(84);

    // [0..62] 21 landmarks x (x,y,z).
    for (var i = 0; i < 21; i++) {
      features[i * 3    ] = coords[i].x;
      features[i * 3 + 1] = coords[i].y;
      features[i * 3 + 2] = coords[i].z;
    }

    // [63..77] 15 finger joint angles - 5 fingers x 3 angles each:
    //   base-flex = angle at MCP between (base -> MCP) and (MCP -> PIP)
    //   tip-flex   = angle at PIP between (MCP -> PIP) and (PIP -> TIP)
    //   spread     = angle at MCP between (MCP -> PIP) and (MCP -> MIDDLE_PIP)
    var midPip = coords[LM.MIDDLE_PIP];
    for (var f = 0; f < FINGERS.length; f++) {
      var F = FINGERS[f];
      var mcp   = coords[F.j1];
      var pip   = coords[F.j2];
      var tip   = coords[F.tip];
      var baseP = coords[F.base];
      var vMcpBase = vecSub(mcp, baseP);
      var vMcpPip  = vecSub(pip, mcp);
      var vPipTip  = vecSub(tip, pip);
      var vSpread  = vecSub(midPip, mcp);
      features[63 + f * 3    ] = angleBetween(vMcpBase, vMcpPip);
      features[63 + f * 3 + 1] = angleBetween(vMcpPip,  vPipTip);
      features[63 + f * 3 + 2] = angleBetween(vMcpPip,  vSpread);
    }

    // [78..81] 4 fingertip-gap distances.
    for (var t = 0; t < TIP_PAIRS.length; t++) {
      var pair = TIP_PAIRS[t];
      features[78 + t] = vecDist(coords[pair[0]], coords[pair[1]]);
    }

    // [82..83] wrist (dX, dY) velocity.
    features[82] = v.x;
    features[83] = v.y;
    return features;
  }

  // ==========================================================================
  // SCALER + MLP FORWARD (sklearn-exported coefficients)
  // ==========================================================================
  function applyScaler(inputVec, scaler) {
    var mean = scaler.mean, std = scaler.std;
    var out = new Array(inputVec.length);
    for (var i = 0; i < inputVec.length; i++) {
      out[i] = (inputVec[i] - mean[i]) / (std[i] || 1);
    }
    return out;
  }

  function mlpForward(model, inputVec) {
    var activation = inputVec;
    for (var li = 0; li < model.layers.length; li++) {
      var layer = model.layers[li];
      var isOutputLayer = (li === model.layers.length - 1);
      var out = new Array(layer.biases.length);
      for (var j = 0; j < layer.biases.length; j++) {
        var sum = layer.biases[j];
        var weights = layer.weights;
        for (var k = 0; k < activation.length; k++) {
          sum += activation[k] * weights[k][j];
        }
        out[j] = sum;
      }
      if (!isOutputLayer) {
        for (var r = 0; r < out.length; r++) out[r] = Math.max(0, out[r]); // ReLU
      }
      activation = out;
    }

    var candidates;
    if (model.outputActivation === 'sigmoid_binary') {
      var p1 = 1 / (1 + Math.exp(-activation[0]));
      candidates = [
        { label: model.classes[0], confidence: 1 - p1 },
        { label: model.classes[1], confidence: p1     }
      ];
    } else {
      var maxLogit = Math.max.apply(null, activation);
      var exps = activation.map(function (v) { return Math.exp(v - maxLogit); });
      var sumExp = exps.reduce(function (a, b) { return a + b; }, 0) || 1e-9;
      candidates = model.classes.map(function (label, i) {
        return { label: label, confidence: exps[i] / sumExp };
      });
    }
    candidates.sort(function (a, b) { return b.confidence - a.confidence; });
    return candidates;
  }

  // ==========================================================================
  // MODEL REGISTRY
  // ==========================================================================
  var AZSL_MODEL = null;
  function setAzslModel(modelJson) { AZSL_MODEL = modelJson; }
  function getAzslModel()         { return AZSL_MODEL; }

  function classifyHierarchical(coords, velocity) {
    if (!AZSL_MODEL) return { label: null, confidence: 0, candidates: [] };
    var full84 = buildFeatureVector84(coords, velocity);
    var clusterCandidates = mlpForward(
      AZSL_MODEL.level1.model,
      applyScaler(full84, AZSL_MODEL.level1.scaler)
    );
    var clusterEntry = AZSL_MODEL.clusters[String(clusterCandidates[0].label)];
    if (!clusterEntry) return { label: null, confidence: 0, candidates: [] };
    var subInput = (clusterEntry.featureIndices.length === full84.length)
      ? full84
      : clusterEntry.featureIndices.map(function (idx) { return full84[idx]; });
    var letterCandidates = mlpForward(
      clusterEntry.model,
      applyScaler(subInput, clusterEntry.scaler)
    );
    return {
      label: letterCandidates[0].label,
      confidence: letterCandidates[0].confidence,
      candidates: letterCandidates.slice(0, 2)
    };
  }

  // ==========================================================================
  // CONTROL (SPACE / DEL) HEURISTICS - verbatim from Murad's code
  // ==========================================================================
  function fingerState(coords, mcpIdx, pipIdx, tipIdx) {
    var angle = jointAngleDeg(coords, mcpIdx, pipIdx, tipIdx);
    if (angle >= ANGLE_STRAIGHT) return 'extended';
    if (angle <= ANGLE_FIST)     return 'curled';
    return 'mid';
  }

  function thumbExtended(coords, margin) {
    margin = margin || 1.15;
    var tipToPinky = vecDist(coords[LM.THUMB_TIP], coords[LM.PINKY_MCP]);
    var mcpToPinky = vecDist(coords[LM.THUMB_MCP], coords[LM.PINKY_MCP]);
    return tipToPinky > mcpToPinky * margin;
  }

  function thumbPointingUp(coords, threshold) {
    threshold = threshold || 0.15;
    return coords[LM.THUMB_TIP].y < coords[LM.THUMB_MCP].y - threshold &&
           coords[LM.THUMB_TIP].y < -threshold;
  }

  function gestureResult(label) { return { label: label, confidence: HEURISTIC_CONF }; }

  function detectControlGesture(coords) {
    var fs = {
      index:  fingerState(coords, LM.INDEX_MCP,  LM.INDEX_PIP,  LM.INDEX_TIP),
      middle: fingerState(coords, LM.MIDDLE_MCP, LM.MIDDLE_PIP, LM.MIDDLE_TIP),
      ring:   fingerState(coords, LM.RING_MCP,   LM.RING_PIP,   LM.RING_TIP),
      pinky:  fingerState(coords, LM.PINKY_MCP,  LM.PINKY_PIP,  LM.PINKY_TIP)
    };
    var thumbOut = thumbExtended(coords);
    var thumbUp  = thumbPointingUp(coords);

    // SPACE: all fingers curled + thumb tucked across palm, pointing up.
    if (fs.index === 'curled' && fs.middle === 'curled' &&
        fs.ring  === 'curled' && fs.pinky  === 'curled' &&
        !thumbOut && thumbUp) {
      return gestureResult(LABELS.SPACE);
    }
    // DEL: index extended, others curled, thumb tucked across palm (not up).
    if (fs.index === 'extended' && fs.middle === 'curled' &&
        fs.ring  === 'curled'  && fs.pinky  === 'curled' &&
        !thumbOut && !thumbUp) {
      return gestureResult(LABELS.DEL);
    }
    return null;
  }

  // ==========================================================================
  // TEMPORAL / TRAJECTORY ENGINE - verbatim port of the original
  // ==========================================================================
  function LandmarkBuffer(capacity) {
    this.cap = capacity || TRAJECTORY_WINDOW;
    this.wristX   = new Float32Array(this.cap);
    this.wristY   = new Float32Array(this.cap);
    this.wristZ   = new Float32Array(this.cap);
    this.thumbX   = new Float32Array(this.cap);
    this.thumbY   = new Float32Array(this.cap);
    this.indexX   = new Float32Array(this.cap);
    this.indexY   = new Float32Array(this.cap);
    this.midMcpX  = new Float32Array(this.cap);
    this.midMcpY  = new Float32Array(this.cap);
    this.midMcpZ  = new Float32Array(this.cap);
    this.timestamps = new Float32Array(this.cap);
    this.start = 0;
    this.count = 0;
  }
  LandmarkBuffer.prototype.reset = function () { this.start = 0; this.count = 0; };
  LandmarkBuffer.prototype.push = function (landmarks, t) {
    var head = (this.start + this.count) % this.cap;
    this.wristX[head]   = landmarks[LM.WRIST].x;
    this.wristY[head]   = landmarks[LM.WRIST].y;
    this.wristZ[head]   = landmarks[LM.WRIST].z;
    this.thumbX[head]   = landmarks[LM.THUMB_TIP].x;
    this.thumbY[head]   = landmarks[LM.THUMB_TIP].y;
    this.indexX[head]   = landmarks[LM.INDEX_TIP].x;
    this.indexY[head]   = landmarks[LM.INDEX_TIP].y;
    this.midMcpX[head]  = landmarks[LM.MIDDLE_MCP].x;
    this.midMcpY[head]  = landmarks[LM.MIDDLE_MCP].y;
    this.midMcpZ[head]  = landmarks[LM.MIDDLE_MCP].z;
    this.timestamps[head] = t;
    if (this.count < this.cap) {
      this.count++;
    } else {
      this.start = (this.start + 1) % this.cap;
    }
  };
  LandmarkBuffer.prototype.at = function (logicalIdx) {
    var i = (this.start + logicalIdx) % this.cap;
    return {
      wristX: this.wristX[i], wristY: this.wristY[i], wristZ: this.wristZ[i],
      thumbX: this.thumbX[i], thumbY: this.thumbY[i],
      indexX: this.indexX[i], indexY: this.indexY[i],
      midMcpX: this.midMcpX[i], midMcpY: this.midMcpY[i], midMcpZ: this.midMcpZ[i],
      timestamp: this.timestamps[i]
    };
  };

  function computeTrajectoryFeatures(buffer, mirrorX) {
    var n = buffer.count;
    if (n < 2) return {
      isDynamic: false, speed: 0,
      netDx: 0, netDy: 0, netDz: 0,
      netIndexDx: 0, netIndexDy: 0,
      directionDeg: 0, verticalSignFlips: 0
    };
    var oldest = buffer.at(0);
    var newest = buffer.at(n - 1);
    var scale = Math.sqrt(
      (newest.midMcpX - oldest.wristX) * (newest.midMcpX - oldest.wristX) +
      (newest.midMcpY - oldest.wristY) * (newest.midMcpY - oldest.wristY) +
      (newest.midMcpZ - oldest.wristZ) * (newest.midMcpZ - oldest.wristZ)
    );
    if (scale < 1e-6) scale = 1e-6;
    var pathLength = 0, verticalSignFlips = 0, prevDy = null;
    for (var i = 1; i < n; i++) {
      var a = buffer.at(i - 1), b = buffer.at(i);
      var dx = b.wristX - a.wristX, dy = b.wristY - a.wristY, dz = b.wristZ - a.wristZ;
      pathLength += Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (Math.abs(dy) > 1e-5) {
        if (prevDy !== null && Math.sign(dy) !== Math.sign(prevDy)) verticalSignFlips++;
        prevDy = dy;
      }
    }
    var speed = (pathLength / scale) / (n - 1);
    var netDx = (newest.wristX - oldest.wristX) / scale;
    var netDy = (newest.wristY - oldest.wristY) / scale;
    var netDz = (newest.wristZ - oldest.wristZ) / scale;
    var netIndexDx = (newest.indexX - oldest.indexX) / scale;
    var netIndexDy = (newest.indexY - oldest.indexY) / scale;
    if (mirrorX) { netDx = -netDx; netIndexDx = -netIndexDx; }
    return {
      isDynamic: speed >= DYNAMIC_MOTION_THRESHOLD,
      speed: speed,
      netDx: netDx, netDy: netDy, netDz: netDz,
      netIndexDx: netIndexDx, netIndexDy: netIndexDy,
      directionDeg: Math.atan2(netDy, netDx) * (180 / Math.PI),
      verticalSignFlips: verticalSignFlips
    };
  }

  function DynamicHysteresis(holdMs) {
    this.holdMs = holdMs || DYNAMIC_HYSTERESIS_MS;
    this.pendingLabel = undefined;
    this.pendingSince = 0;
    this.confirmedLabel = null;
  }
  DynamicHysteresis.prototype.reset = function () {
    this.pendingLabel = undefined;
    this.pendingSince = 0;
    this.confirmedLabel = null;
  };
  DynamicHysteresis.prototype.update = function (suggestedLabel, now) {
    if (suggestedLabel !== this.pendingLabel) {
      this.pendingLabel = suggestedLabel;
      this.pendingSince = now;
    }
    if (now - this.pendingSince >= this.holdMs) {
      this.confirmedLabel = this.pendingLabel;
    }
    return this.confirmedLabel;
  };

  function rescoreWithTrajectory(letterResult, trajectory, hysteresis, now) {
    if (!letterResult || !letterResult.label) {
      if (hysteresis) hysteresis.update(null, now);
      return letterResult;
    }
    var label = letterResult.label;
    var confidence = letterResult.confidence;
    var pair = STATIC_DYNAMIC_PAIRS[label];
    var suggestedOverride = null;
    if (pair && trajectory.isDynamic) {
      if (pair.rule === 'downward' &&
          trajectory.netDy > 0 && Math.abs(trajectory.netDy) >= Math.abs(trajectory.netDx)) {
        suggestedOverride = pair.dynamic;
      } else if (pair.rule === 'oscillation' && trajectory.verticalSignFlips >= 2) {
        suggestedOverride = pair.dynamic;
      }
    }
    var confirmedOverride = hysteresis
      ? hysteresis.update(suggestedOverride, now)
      : suggestedOverride;
    if (pair && confirmedOverride === pair.dynamic) {
      label = pair.dynamic;
      confidence = Math.max(confidence, HEURISTIC_CONF);
    }
    if (KNOWN_DYNAMIC_LETTERS[label]) {
      confidence = trajectory.isDynamic
        ? Math.min(0.99, confidence + TRAJECTORY_CONFIDENCE_BOOST)
        : Math.max(0, confidence - TRAJECTORY_CONFIDENCE_PENALTY);
    }
    var candidates = (letterResult.candidates || []).map(function (c) {
      return (c.label === letterResult.label)
        ? { label: label, confidence: confidence }
        : c;
    });
    if (!candidates.length) candidates.push({ label: label, confidence: confidence });
    return { label: label, confidence: confidence, candidates: candidates };
  }

  // ==========================================================================
  // PUBLIC ENTRY POINT
  // ==========================================================================
  function predictGesture(landmarksArray, mirrorX, velocity, trajectory, hysteresis, now) {
    var coords = normalizeLandmarks(landmarksArray, mirrorX);
    var v = mirrorX
      ? { x: -velocity.x, y: velocity.y }
      : { x:  velocity.x, y: velocity.y };
    var letterResult = classifyHierarchical(coords, v);

    if (trajectory) {
      letterResult = rescoreWithTrajectory(
        letterResult, trajectory, hysteresis, (now == null ? 0 : now)
      );
    }

    if (letterResult.confidence >= CONTROL_OVERRIDE_CONFIDENCE) return letterResult;

    var control = detectControlGesture(coords);
    if (control) {
      return { label: control.label, confidence: control.confidence, candidates: [control] };
    }

    if (letterResult.confidence >= MIN_CONFIDENCE) return letterResult;
    return { label: null, confidence: letterResult.confidence, candidates: [] };
  }

  // ==========================================================================
  // EXPOSE
  // ==========================================================================
  global.AzslAlphabet = {
    LM: LM,
    AZ_ALPHABET: AZ_ALPHABET,
    LABELS: LABELS,
    MIN_CONFIDENCE: MIN_CONFIDENCE,
    DYNAMIC_HYSTERESIS_MS: DYNAMIC_HYSTERESIS_MS,
    labelMeta: labelMeta,
    setAzslModel: setAzslModel,
    getAzslModel: getAzslModel,
    normalizeLandmarks: normalizeLandmarks,
    buildFeatureVector84: buildFeatureVector84,
    applyScaler: applyScaler,
    mlpForward: mlpForward,
    classifyHierarchical: classifyHierarchical,
    detectControlGesture: detectControlGesture,
    LandmarkBuffer: LandmarkBuffer,
    computeTrajectoryFeatures: computeTrajectoryFeatures,
    DynamicHysteresis: DynamicHysteresis,
    rescoreWithTrajectory: rescoreWithTrajectory,
    predictGesture: predictGesture
  };

})(typeof window !== 'undefined' ? window : globalThis);
