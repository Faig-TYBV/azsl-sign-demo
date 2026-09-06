# Dataset Subset Verification Report: 24 Classes, Cap 50
> **Experiment**: Isolated Dataset Subset for 24-Class GRU Training
> **Subset Location**: `outputs/vocabulary_24_cap50/`
> **Audit Timestamp**: Fully verified on local repository files

## 1. Executive Summary
An isolated dataset subset of the Azerbaijani Sign Language dataset (AzSLD) was constructed and rigorously audited for the upcoming 24-class training experiment. All 2,218 sample files (866 train, 676 val, 676 test) exist on disk and were verified against strict split isolation, provenance, and data-balancing rules. Zero production models, code, or checkpoints were touched, and zero model training was conducted.

## 2. Verified Class Sample Allocations

| Index | Class Name | Orig Train | Selected Train | Cap Applied | Orig Val | Subset Val | Orig Test | Subset Test |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | **MƏN** | 1769 | **50** | YES (Cap 50) | 379 | 379 | 379 | 379 |
| 1 | **SƏN** | 34 | **34** | NO (<50) | 7 | 7 | 7 | 7 |
| 2 | **BİZ** | 117 | **50** | YES (Cap 50) | 25 | 25 | 25 | 25 |
| 3 | **SİZ** | 308 | **50** | YES (Cap 50) | 66 | 66 | 66 | 66 |
| 4 | **SALAM** | 22 | **22** | NO (<50) | 4 | 4 | 4 | 4 |
| 5 | **SAĞLAM** | 39 | **39** | NO (<50) | 8 | 8 | 8 | 8 |
| 6 | **İSTƏMƏK** | 73 | **50** | YES (Cap 50) | 16 | 16 | 16 | 16 |
| 7 | **GETMƏK** | 34 | **34** | NO (<50) | 7 | 7 | 7 | 7 |
| 8 | **GƏLMƏK** | 24 | **24** | NO (<50) | 5 | 5 | 5 | 5 |
| 9 | **YEMƏK** | 26 | **26** | NO (<50) | 5 | 5 | 5 | 5 |
| 10 | **ALMAQ** | 31 | **31** | NO (<50) | 7 | 7 | 7 | 7 |
| 11 | **OLMAQ** | 28 | **28** | NO (<50) | 6 | 6 | 6 | 6 |
| 12 | **EV** | 53 | **50** | YES (Cap 50) | 12 | 12 | 12 | 12 |
| 13 | **BAKI** | 44 | **44** | NO (<50) | 10 | 10 | 10 | 10 |
| 14 | **AZƏRBAYCAN** | 25 | **25** | NO (<50) | 6 | 6 | 6 | 6 |
| 15 | **TELEFON** | 28 | **28** | NO (<50) | 6 | 6 | 6 | 6 |
| 16 | **HARDA** | 26 | **26** | NO (<50) | 5 | 5 | 5 | 5 |
| 17 | **NECƏ** | 19 | **19** | NO (<50) | 4 | 4 | 4 | 4 |
| 18 | **BURDA** | 72 | **50** | YES (Cap 50) | 15 | 15 | 15 | 15 |
| 19 | **BU GÜN** | 25 | **25** | NO (<50) | 5 | 5 | 5 | 5 |
| 20 | **SABAH** | 31 | **31** | NO (<50) | 6 | 6 | 6 | 6 |
| 21 | **VAR** | 78 | **50** | YES (Cap 50) | 16 | 16 | 16 | 16 |
| 22 | **YOX** | 30 | **30** | NO (<50) | 6 | 6 | 6 | 6 |
| 23 | **BU** | 234 | **50** | YES (Cap 50) | 50 | 50 | 50 | 50 |

| Total | **24 Classes** | 3,170 | **866** | 8 Capped Classes | 676 | **676** | 676 | **676** |

## 3. Mathematical & Integrity Checks

1. **Split Totals**: Exactly 866 train, 676 validation, and 676 test sequences (**PASS**).
2. **Duplicate Checks**:
   - Training intra-split duplicates: 0 (**PASS**)
   - Validation intra-split duplicates: 0 (**PASS**)
   - Test intra-split duplicates: 0 (**PASS**)
3. **Cross-Split Isolation**:
   - `intersection(train, val)` = 0 (**PASS**)
   - `intersection(train, test)` = 0 (**PASS**)
   - `intersection(val, test)` = 0 (**PASS**)
4. **Provenance & Zero Leakage**:
   - 100% of subset training samples originate from original `splits.train` (**PASS**)
   - 100% of subset validation samples originate from original `splits.val` (**PASS**)
   - 100% of subset test samples originate from original `splits.test` (**PASS**)
5. **File Existence**: All 2,218 feature files (`.npz`) verified present on disk (**PASS**).
6. **Training Cap Integrity**: Maximum samples per class is strictly 50; minimum is 19 (`NECƏ`); max/min ratio is 2.63:1 (**PASS**).
7. **Label Mapping**: Exact bijection between integers 0..23 and `FINAL_VOCABULARY` (**PASS**).

## 4. Deterministic Reproducibility Protocol

For classes exceeding the 50-sample cap, samples were selected using the deterministic per-class strategy:
```python
sorted_class_train_paths = sorted(original_class_train_paths)
rng = random.Random(42 + class_index)
selected_samples = sorted(rng.sample(sorted_class_train_paths, 50))
```
- Lexicographical sorting guarantees platform-independent input ordering.
- Distinct per-class seeds (`42 + class_index`) ensure selection is invariant to class iteration order.
- Selection never observes test, validation, or model prediction data.

## 5. Production Artifact Preservation

- `outputs/dataset_split.json`: Unmodified (Original 200-class split preserved)
- `outputs/test_report_gru_normalized.json`: Unmodified (Baseline metrics intact)
- `outputs/checkpoints/gru_temporal_pool_best.pt`: Unmodified (Production checkpoint intact)
- Model Training: **NO TRAINING CONDUCTED**.
