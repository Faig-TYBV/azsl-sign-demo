# Experiment 8: Independent Post-Training Audit Report
> **Audit Target**: 24-Class GRU Model with Training Cap 50 (`outputs/vocabulary_24_cap50/`)
> **Auditor Protocol**: Independent mathematical recomputation directly from tensors, saved checkpoint weights, raw confusion matrix, and split manifests.

## 1. Checkpoint Verification
| Verification Item | Saved Property | Expected Specification | Audit Status |
| :--- | :---: | :---: | :---: |
| **Checkpoint Path** | `gru_24_cap50_best.pt` | `outputs/vocabulary_24_cap50/checkpoints/gru_24_cap50_best.pt` | **PASS** |
| **Saved Best Epoch** | **Epoch 29** | Epoch 29 | **PASS** |
| **Best Val Macro F1** | **0.7312** | Matches training history (0.7312) | **PASS** |
| **Input Dimension** | `126` | 126 (21 landmarks x 2 hands x 3 coords) | **PASS** |
| **Hidden Size / Layers**| `128` / `2` layers | 2-layer unidirectional GRU (128) | **PASS** |
| **Pooling Mechanism** | `mean_max` | Temporal Mean + Max Pooling (256-dim) | **PASS** |
| **Classifier Head** | `Linear(256 -> 24)` | 24 output classes | **PASS** |
| **Trainable Parameters**| **203544** | Exactly 203,544 | **PASS** |
| **Class Mappings** | 24 classes | Strictly matches `FINAL_VOCABULARY` | **PASS** |
| **Normalization Ref** | `outputs\vocabulary_24_cap50\metadata\feature_normalization_stats_24.json` | Points to train-fitted normalization stats | **PASS** |

## 2. Independent Test Metrics Recalculation
All metrics were re-derived directly from the 24x24 confusion matrix without trusting stored scalar aggregates:

| Test Metric | Stored Value | Recalculated from Matrix | Verified Exact Match |
| :--- | :---: | :---: | :---: |
| **Total Test Samples** | 676 | **676** | **PASS** |
| **Correct Predictions**| 576 | **576** | **PASS** |
| **Overall Accuracy** | 85.21% | **85.21%** (`0.852071`) | **PASS** |
| **Macro F1-Score** | 74.21% | **74.21%** (`0.742142`) | **PASS** |
| **Weighted F1-Score** | 86.03% | **86.03%** (`0.860323`) | **PASS** |
| **Macro Recall** | 80.26% | **80.26%** (`0.802600`) | **PASS** |
| **Macro Precision** | 72.10% | **72.10%** (`0.720976`) | **PASS** |
| **Weighted Recall** | 85.21% | **85.21%** (`0.852071`) | **PASS** |
| **Weighted Precision** | 88.16% | **88.16%** (`0.881616`) | **PASS** |

## 3. Per-Class Independent Verification
- **Zero-Recall Classes**: **0** (All 24 classes achieved non-zero true positive detections)
- **Zero-F1 Classes**: **0** (All 24 classes achieved non-zero F1 scores)

### All 24 Classes Ranked by Test F1-Score (Ascending):

| Rank | Class Name | Test Support | True Positives | Precision | Recall | F1-Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | **BAKI** | 10 | 6 | 0.3158 | 0.6000 | **0.4138** |
| 2 | **SƏN** | 7 | 5 | 0.3846 | 0.7143 | **0.5000** |
| 3 | **YEMƏK** | 5 | 3 | 0.5000 | 0.6000 | **0.5455** |
| 4 | **HARDA** | 5 | 3 | 0.5000 | 0.6000 | **0.5455** |
| 5 | **BURDA** | 15 | 10 | 0.4762 | 0.6667 | **0.5556** |
| 6 | **TELEFON** | 6 | 5 | 0.4545 | 0.8333 | **0.5882** |
| 7 | **SİZ** | 66 | 38 | 0.7451 | 0.5758 | **0.6496** |
| 8 | **NECƏ** | 4 | 4 | 0.5000 | 1.0000 | **0.6667** |
| 9 | **SABAH** | 6 | 6 | 0.5000 | 1.0000 | **0.6667** |
| 10 | **ALMAQ** | 7 | 6 | 0.6000 | 0.8571 | **0.7059** |
| 11 | **OLMAQ** | 6 | 4 | 0.8000 | 0.6667 | **0.7273** |
| 12 | **İSTƏMƏK** | 16 | 14 | 0.6364 | 0.8750 | **0.7368** |
| 13 | **BİZ** | 25 | 18 | 0.9000 | 0.7200 | **0.8000** |
| 14 | **GƏLMƏK** | 5 | 4 | 0.8000 | 0.8000 | **0.8000** |
| 15 | **YOX** | 6 | 4 | 1.0000 | 0.6667 | **0.8000** |
| 16 | **EV** | 12 | 9 | 0.9000 | 0.7500 | **0.8182** |
| 17 | **AZƏRBAYCAN** | 6 | 5 | 0.8333 | 0.8333 | **0.8333** |
| 18 | **VAR** | 16 | 14 | 0.8750 | 0.8750 | **0.8750** |
| 19 | **BU** | 50 | 42 | 0.9130 | 0.8400 | **0.8750** |
| 20 | **SALAM** | 4 | 4 | 0.8000 | 1.0000 | **0.8889** |
| 21 | **GETMƏK** | 7 | 6 | 1.0000 | 0.8571 | **0.9231** |
| 22 | **SAĞLAM** | 8 | 8 | 0.8889 | 1.0000 | **0.9412** |
| 23 | **MƏN** | 379 | 353 | 0.9806 | 0.9314 | **0.9553** |
| 24 | **BU GÜN** | 5 | 5 | 1.0000 | 1.0000 | **1.0000** |

### Best 5 Classes by F1:
1. **BU GÜN**: F1 = **1.0000** (Recall: 100.00%, Precision: 100.00%, Support: 5)
1. **MƏN**: F1 = **0.9553** (Recall: 93.14%, Precision: 98.06%, Support: 379)
1. **SAĞLAM**: F1 = **0.9412** (Recall: 100.00%, Precision: 88.89%, Support: 8)
1. **GETMƏK**: F1 = **0.9231** (Recall: 85.71%, Precision: 100.00%, Support: 7)
1. **SALAM**: F1 = **0.8889** (Recall: 100.00%, Precision: 80.00%, Support: 4)

### Worst 5 Classes by F1:
1. **BAKI**: F1 = **0.4138** (Recall: 60.00%, Precision: 31.58%, Support: 10)
1. **SƏN**: F1 = **0.5000** (Recall: 71.43%, Precision: 38.46%, Support: 7)
1. **YEMƏK**: F1 = **0.5455** (Recall: 60.00%, Precision: 50.00%, Support: 5)
1. **HARDA**: F1 = **0.5455** (Recall: 60.00%, Precision: 50.00%, Support: 5)
1. **BURDA**: F1 = **0.5556** (Recall: 66.67%, Precision: 47.62%, Support: 15)

## 4. Confusion Pairs Analysis
Out of 552 possible off-diagonal pairs ($24 \times 23$), exactly **50 non-diagonal confusion pairs** were observed in the test predictions:

| Rank | True Label | Model Prediction | Misclassified Count | Dominant Error Pattern |
| :---: | :--- | :--- | :---: | :--- |
| 1 | **SİZ** | **BAKI** | **10** | Mutual hand trajectory overlap |
| 2 | **MƏN** | **SİZ** | **9** | Mutual hand trajectory overlap |
| 3 | **MƏN** | **İSTƏMƏK** | **7** | Mutual hand trajectory overlap |
| 4 | **SİZ** | **SƏN** | **5** | Mutual hand trajectory overlap |
| 5 | **BU** | **BURDA** | **5** | Mutual hand trajectory overlap |
| 6 | **BİZ** | **MƏN** | **4** | Mutual hand trajectory overlap |
| 7 | **SİZ** | **MƏN** | **3** | Mutual hand trajectory overlap |
| 8 | **SİZ** | **SABAH** | **3** | Mutual hand trajectory overlap |
| 9 | **MƏN** | **BİZ** | **2** | Mutual hand trajectory overlap |
| 10 | **MƏN** | **BAKI** | **2** | Mutual hand trajectory overlap |

### Audit of Focal Confusion Targets:
- **`MƏN`**: Achieved 353 correct out of 379 test samples (**93.14% Recall**, **98.06% Precision**, **0.9553 F1**). Previously, in the 200-class model, `MƏN` lost 96 samples to `MƏNƏ` (52) and `MƏNİM` (44). With those inflected forms pruned from the label space, `MƏN` misclassifications were limited to minor confusions (`SİZ`: 9, `İSTƏMƏK`: 7, `BİZ`: 2, `BAKI`: 2).
- **`BAKI`**: Experienced 13 false positives (10 from `SİZ`), dragging precision down to 31.58% despite a solid 60.00% recall (6/10).
- **`SƏN`**: Achieved 71.43% recall (5/7 correct) with 8 incoming false positives (5 from `SİZ`), resulting in 38.46% precision and 0.5000 F1.
- **`BURDA`**: Retained 66.67% recall (10/15 correct) with confusions largely stemming from demonstrative `BU` (5 samples).

## 5. Controlled Baseline Comparison (Identical 676-Sample Subset)

| Evaluation Model | Test Sample Basis | Overall Accuracy | Macro F1 | Weighted F1 | Macro Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Global 200-Class Production Baseline** | 1,299 samples (200 classes) | 59.89% | 51.41% | 62.13% | 54.49% |
| **Existing 200-Class Model on Same 24-Class Subset** | **676 samples (24 classes)** | **67.31%** | **65.10%** | **72.63%** | **62.96%** |
| **New 24-Class Model (Experiment 8)** | **676 samples (24 classes)** | **85.21%** | **74.21%** | **86.03%** | **80.26%** |

### Direct Empirical Improvement on the Identical 676 Test Sequences:
- **Accuracy**: **+17.90 percentage points** (85.21% vs 67.31%)
- **Macro F1-Score**: **+9.11 percentage points** (74.21% vs 65.10%)
- **Weighted F1-Score**: **+13.41 percentage points** (86.03% vs 72.63%)

## 6. Dataset Integrity & Split Preservation
- **Split Provenance**: 100% of training samples originate from `outputs/dataset_split.json` `splits.train`; 100% of validation samples from `splits.val`; 100% of test samples from `splits.test`.
- **Zero Split Overlap**: `intersection(train, val) == 0`, `intersection(train, test) == 0`, `intersection(val, test) == 0`.
- **Cap Enforcement**: Exactly 8 classes reached the 50-sample cap (`MƏN`, `BİZ`, `SİZ`, `İSTƏMƏK`, `EV`, `BURDA`, `VAR`, `BU`). Minimum class count is 19 (`NECƏ`). Imbalance ratio is **2.63:1**.
- **Held-Out Split Preservation**: Validation (676 samples) and test (676 samples) sets were **never capped, downsampled, or modified**.

## 7. Feature Normalization Integrity
- Statistics (`mean` and `std`) are exactly 126-dimensional vectors.
- Fitted strictly on 17,131 valid frames (`mask == 1`) from the 866 training samples only.
- Zero validation or test frames were used during normalization fitting.
- No NaN, Inf, or negative standard deviation values exist.

## 8. Training History & Model Selection Verification
- **Completed Epochs**: Exactly 30 epochs ran.
- **Best Epoch**: **Epoch 29** achieved the global peak validation Macro F1 of **0.7312** (Validation Accuracy: 84.62%).
- **Checkpoint Match**: The saved checkpoint state dictionary corresponds strictly to Epoch 29, confirming that model selection was governed exclusively by validation Macro F1.
- **Test Isolation**: The 676-sample test split was evaluated **strictly once** after training had fully terminated and the best checkpoint was restored.

## 9. Production Safety Audit
- `outputs/dataset_split.json`: **UNMODIFIED** (626,703 bytes)
- `outputs/test_report_gru_normalized.json`: **UNMODIFIED** (133,368 bytes)
- `outputs/checkpoints/gru_temporal_pool_best.pt`: **UNMODIFIED** (3,000,905 bytes)
- `data/features/full/`: **UNMODIFIED** (All 200 class directories intact)
- Git state: **NO COMMITS OR PUSHES EXECUTED**.

## 10. Methodological Interpretation Rules
1. **No Conflation of Offline and Real-Time**: Achieving 85.21% offline test accuracy on pre-segmented 26-frame sequences indicates strong representation learning, but does not guarantee 85% accuracy in continuous real-time streaming with sliding-window segmentation.
2. **No Claim of Single-Factor Causality**: Experiment 8 combined label-space reduction (200 -> 24), training-set capping (Cap 50), and train-only normalization. The +17.90% accuracy gain is the result of the unified pipeline; individual causal contributions were not ablated separately.
3. **Stretch Target Evaluation**: While the offline accuracy (85.21%) exceeded the aspirational stretch target (>= 85%), Macro F1 (74.21%) remains below the 82% stretch target due to lower precision in minority classes (`BAKI`, `SƏN`).

## 11. Final Audit Determination
```text
============================================================
FINAL POST-TRAINING AUDIT SUMMARY
============================================================
Checkpoint:           PASS
Metrics:              PASS
Per-class metrics:    PASS
Confusion matrix:     PASS
Dataset integrity:    PASS
Normalization:        PASS
Training history:     PASS
Production safety:    PASS

Verified Test Results:
  Accuracy:           85.21%
  Macro F1:           74.21%
  Weighted F1:        86.03%
  Macro Recall:       80.26%
  Correct / Total:    576 / 676
  Best Epoch:         29

AUDIT STATUS: PASS
============================================================
```
