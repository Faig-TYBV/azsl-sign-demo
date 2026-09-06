# Final Rigorous Audit & Evaluation: AzSL Vocabulary Selection & Balancing Strategy

> **Document Status**: Second-Level Expert Review & Verification Complete

> **Audit Constraints**: No model training, no production code modification, no checkpoint alteration, no dataset mutation. Repository files remain the ground truth.

## Executive Summary & Final Determination

Following an exhaustive audit of the 200-class baseline metrics (`outputs/test_report_gru_normalized.json`) and dataset splits (`outputs/dataset_split.json`), we conclude that **the previously proposed 28-class vocabulary is suboptimal** for a live real-time demo.
Specifically, retaining classes with severely insufficient data (`GÖRMƏK` with only 13 training samples, `BİLMƏK` with only 16 training samples) or broken baseline performance (`O` with 11.1% recall, `İŞ` with 33.3% recall) introduces unacceptable statistical instability and inflates the training imbalance ratio.


We formally recommend trimming the vocabulary to **24 classes** and applying **Cap = 50** (primary) or **Cap = 75** (alternative) strictly to the training split:

- **Vocabulary Size**: **24 classes** (eliminates `GÖRMƏK`, `BİLMƏK`, `O`, `İŞ`)

- **Subset Test Performance of 200-class Model**: **67.31% Accuracy**, **65.10% Macro F1** (compared to 63.44% Acc / 60.66% F1 on 28 classes)

- **Training Cap**: **Cap = 50** yielding **866 training sequences** with a tight **2.63:1** imbalance ratio (vs 5.77:1 on 28 classes)

- **Data Integrity**: `subset_val` (676 samples) and `subset_test` (676 samples) remain 100% untouched and leak-free.


## 1. Audit of the Previously Proposed 28 Classes

All 28 proposed classes were audited against the 200-class ground-truth files. All 28 exist in the original label space. Special audit of controversial classes:


| Class | Total | Train | Val | Test | Baseline F1 | Recall | Precision | Top Confusions (200-class) | Audit Verdict |

| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |

| **BİLMƏK** | 22 | 16 | 3 | 3 | 0.333 | 0.333 | 0.333 | SRAĞAGÜN (1), BİLƏR (1) | **REJECT**: Only 1/3 test correct; low data. |

| **GÖRMƏK** | 19 | 13 | 3 | 3 | 0.444 | 0.667 | 0.333 | RƏSİM (1) | **REJECT**: Critically low train data (<15 samples). |

| **SİZ** | 440 | 308 | 66 | 66 | 0.580 | 0.439 | 0.853 | SİZİN (13), BAKI (6) | **RETAIN**: Solid train data (308); high precision (0.853). |

| **O** | 303 | 213 | 45 | 45 | 0.185 | 0.111 | 0.556 | ONUN (26), SƏN (3), ORA (3) | **REJECT**: Dismal recall (11.1%); BU covers 3rd person. |

| **TELEFON** | 40 | 28 | 6 | 6 | 0.600 | 0.500 | 0.750 | ZƏNG (2), 1 (1) | **RETAIN**: Clean object noun; high precision (0.750). |

| **BU** | 334 | 234 | 50 | 50 | 0.822 | 0.740 | 0.925 | GÖRƏ (4), ÜÇÜN (3) | **RETAIN**: High F1 (0.822); robust demonstrative. |


## 2. Audit of Low-Support Classes & Statistical Instability

In small datasets, metrics derived from 3–5 test samples are statistically unstable (a single misclassification causes a 20–33% swing in recall):

- **Classes with < 20 Total Samples**: `GÖRMƏK` (19 total, 13 train, 3 test). With only 3 test samples, its measured recall of 66.7% reflects exactly 2 correct predictions. Training a deep GRU on only 13 samples risks severe memorization/overfitting.

- **Classes with < 25 Total Samples**: `BİLMƏK` (22 total, 16 train, 3 test). Test recall was 33.3% (only 1 out of 3 correct).

- **Classes with < 15 Training Samples**: `GÖRMƏK` (13 train). This violates standard minimum thresholds for temporal landmark modeling.


> **Reviewer Rule**: Semantic desirability cannot override sample insufficiency. Both `GÖRMƏK` and `BİLMƏK` must be removed from the primary demo vocabulary.


## 3 & 4. Multi-Size Vocabulary & Multi-Cap Balancing Evaluation

We evaluated 5 candidate vocabulary sizes (20, 22, 24, 26, 28) across 6 balancing configurations on `subset_train`:


### Candidate Vocabulary: 20 Classes

*Fixed Split Membership*: Validation = 656 samples, Test = 656 samples.

*200-Class Model on this Subset*: **67.38% Accuracy**, **66.27% Macro F1**


| Balancing Cap | Total Train | Min Samples | Median | Max Samples | Max/Min Ratio | % Classes Capped | % Train Data Retained |

| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |

| **Cap 25** | 497 | 22 | 25.0 | 25 | **1.14:1** | 90.0% | 16.2% |

| **Cap 35** | 639 | 22 | 34.5 | 35 | **1.59:1** | 50.0% | 20.8% |

| **Cap 50** | 772 | 22 | 36.5 | 50 | **2.27:1** | 40.0% | 25.1% |

| **Cap 75** | 945 | 22 | 36.5 | 75 | **3.41:1** | 25.0% | 30.7% |

| **Cap 100** | 1048 | 22 | 36.5 | 100 | **4.55:1** | 20.0% | 34.1% |

| **Uncapped** | 3076 | 22 | 36.5 | 1769 | **80.41:1** | 0.0% | 100.0% |



### Candidate Vocabulary: 22 Classes

*Fixed Split Membership*: Validation = 667 samples, Test = 667 samples.

*200-Class Model on this Subset*: **67.47% Accuracy**, **66.01% Macro F1**


| Balancing Cap | Total Train | Min Samples | Median | Max Samples | Max/Min Ratio | % Classes Capped | % Train Data Retained |

| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |

| **Cap 25** | 547 | 22 | 25.0 | 25 | **1.14:1** | 86.4% | 17.5% |

| **Cap 35** | 690 | 22 | 34.0 | 35 | **1.59:1** | 45.5% | 22.1% |

| **Cap 50** | 823 | 22 | 34.0 | 50 | **2.27:1** | 36.4% | 26.3% |

| **Cap 75** | 996 | 22 | 34.0 | 75 | **3.41:1** | 22.7% | 31.9% |

| **Cap 100** | 1099 | 22 | 34.0 | 100 | **4.55:1** | 18.2% | 35.1% |

| **Uncapped** | 3127 | 22 | 34.0 | 1769 | **80.41:1** | 0.0% | 100.0% |



### Candidate Vocabulary: 24 Classes

*Fixed Split Membership*: Validation = 676 samples, Test = 676 samples.

*200-Class Model on this Subset*: **67.31% Accuracy**, **65.10% Macro F1**


| Balancing Cap | Total Train | Min Samples | Median | Max Samples | Max/Min Ratio | % Classes Capped | % Train Data Retained |

| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |

| **Cap 25** | 590 | 19 | 25.0 | 25 | **1.32:1** | 79.2% | 18.6% |

| **Cap 35** | 733 | 19 | 32.5 | 35 | **1.84:1** | 41.7% | 23.1% |

| **Cap 50** | 866 | 19 | 32.5 | 50 | **2.63:1** | 33.3% | 27.3% |

| **Cap 75** | 1039 | 19 | 32.5 | 75 | **3.95:1** | 20.8% | 32.8% |

| **Cap 100** | 1142 | 19 | 32.5 | 100 | **5.26:1** | 16.7% | 36.0% |

| **Uncapped** | 3170 | 19 | 32.5 | 1769 | **93.11:1** | 0.0% | 100.0% |



### Candidate Vocabulary: 26 Classes

*Fixed Split Membership*: Validation = 727 samples, Test = 727 samples.

*200-Class Model on this Subset*: **63.55% Accuracy**, **62.34% Macro F1**


| Balancing Cap | Total Train | Min Samples | Median | Max Samples | Max/Min Ratio | % Classes Capped | % Train Data Retained |

| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |

| **Cap 25** | 640 | 19 | 25.0 | 25 | **1.32:1** | 80.8% | 18.8% |

| **Cap 35** | 795 | 19 | 32.5 | 35 | **1.84:1** | 42.3% | 23.3% |

| **Cap 50** | 943 | 19 | 32.5 | 50 | **2.63:1** | 34.6% | 27.7% |

| **Cap 75** | 1141 | 19 | 32.5 | 75 | **3.95:1** | 23.1% | 33.5% |

| **Cap 100** | 1269 | 19 | 32.5 | 100 | **5.26:1** | 19.2% | 37.2% |

| **Uncapped** | 3410 | 19 | 32.5 | 1769 | **93.11:1** | 0.0% | 100.0% |



### Candidate Vocabulary: 28 Classes

*Fixed Split Membership*: Validation = 733 samples, Test = 733 samples.

*200-Class Model on this Subset*: **63.44% Accuracy**, **60.66% Macro F1**


| Balancing Cap | Total Train | Min Samples | Median | Max Samples | Max/Min Ratio | % Classes Capped | % Train Data Retained |

| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |

| **Cap 25** | 669 | 13 | 25.0 | 25 | **1.92:1** | 75.0% | 19.5% |

| **Cap 35** | 824 | 13 | 31.0 | 35 | **2.69:1** | 39.3% | 24.0% |

| **Cap 50** | 972 | 13 | 31.0 | 50 | **3.85:1** | 32.1% | 28.3% |

| **Cap 75** | 1170 | 13 | 31.0 | 75 | **5.77:1** | 21.4% | 34.0% |

| **Cap 100** | 1298 | 13 | 31.0 | 100 | **7.69:1** | 17.9% | 37.7% |

| **Uncapped** | 3439 | 13 | 31.0 | 1769 | **136.08:1** | 0.0% | 100.0% |



### Comparative Synthesis across Vocabulary Sizes:

1. **28 Classes**: Suffers from the lowest subset accuracy (63.44%), lowest subset macro F1 (60.66%), and highest imbalance ratio (5.77:1 at Cap 75) due to the presence of `GÖRMƏK` (13 tr), `BİLMƏK` (16 tr), and `O` (11.1% recall).

2. **26 Classes** (removes `GÖRMƏK`, `BİLMƏK`): Accuracy improves to 63.55%, macro F1 to 62.34%, min train count rises to 19.

3. **24 Classes** (removes `GÖRMƏK`, `BİLMƏK`, `O`, `İŞ`): **Optimal sweet spot**. Subset accuracy jumps to **67.31%** (+3.87%), macro F1 jumps to **65.10%** (+4.44%). At Cap 50, the imbalance ratio drops to an exceptionally balanced **2.63:1** while preserving 866 training sequences.

4. **22 / 20 Classes**: While offering slightly higher metrics (67.47% acc), dropping `NECƏ` breaks the essential greeting question *'SALAM SƏN NECƏ'*, and dropping `GƏLMƏK` eliminates the arrival/departure semantic pair.


## 5. Verification of Dataset Split Integrity

We verified that:

- `subset_train` is composed **strictly** of training samples from `outputs/dataset_split.json`.

- `subset_val` is composed **strictly** of validation samples from `outputs/dataset_split.json`.

- `subset_test` is composed **strictly** of test samples from `outputs/dataset_split.json`.

- **Zero data leakage**: No samples are transferred between splits. Balancing caps are applied **exclusively to `subset_train`**.


## 6. Disambiguation of Baseline Metrics

To prevent confusion between global and localized performance, we explicitly delineate:


| Baseline Metric Level | Overall Accuracy | Macro F1 | Context & Evaluation Basis |

| :--- | :---: | :---: | :--- |

| **A. Global 200-Class Production Baseline** | **59.89%** | **51.41%** | Evaluated on all 1,299 test sequences across all 200 classes. |

| **B. 200-Class Model on 24-Class Subset-Test** | **67.31%** | **65.10%** | Evaluated on the 676 test sequences belonging to the 24 classes. |

| **B2. 200-Class Model on 28-Class Subset-Test** | **63.44%** | **60.66%** | Evaluated on the 733 test sequences belonging to the 28 classes. |

| **C. Future 24-Class Model (Target)** | *Pending Train* | *Target $\ge 85\%$* | To be evaluated on the identical 676 subset-test sequences. |


## 7. Clarification of Confusion Claims

We strictly adhere to scientific terminology regarding confusion matrix effects:

> *Methodological Statement*: Removing an inflected variant (e.g. `ONUN`, `MƏNİM`, `İSTƏYİRƏM`) removes that class from the future label space, eliminating mutual competition at the softmax layer. However, whether this improves the recognition rate of the retained root class (`O`, `MƏN`, `İSTƏMƏK`) is an empirical hypothesis that must be validated after retraining.


## 8. Word-Sequence Audit & Sentence Builder Specification

All demo sequences are strictly composed of tokens from the recommended 24-class vocabulary. Because the GRU model performs temporal word classification (not auto-regressive natural language generation), these sequences represent **ordered gesture recognition outputs** in an AzSL dialogue context:


| ID | Output Word Sequence | English Translation | Sequence Classification |

| :---: | :--- | :--- | :--- |

| 1 | `SALAM` → `SƏN` → `NECƏ` | Hello, how are you? | **A. Natural AzSL greeting sentence** |

| 2 | `MƏN` → `EV` → `GETMƏK` → `İSTƏMƏK` | I want to go home. | **A. Natural AzSL sentence** |

| 3 | `SƏN` → `SABAH` → `BAKI` → `GƏLMƏK` | Are you coming to Baku tomorrow? | **A. Natural AzSL sentence** |

| 4 | `BURDA` → `TELEFON` → `VAR` | There is a phone here. / Is there a phone here? | **A. Natural AzSL sentence** |

| 5 | `BU` → `EV` → `HARDA` | Where is this house? | **A. Natural AzSL sentence** |

| 6 | `MƏN` → `SAĞLAM` → `OLMAQ` → `İSTƏMƏK` | I want to be healthy. | **A. Natural AzSL sentence** |

| 7 | `BİZ` → `AZƏRBAYCAN` → `GƏLMƏK` | We come to Azerbaijan. | **A. Natural AzSL sentence** |

| 8 | `MƏN` → `BU GÜN` → `YEMƏK` → `ALMAQ` | I buy food today. | **A. Natural AzSL sentence** |

| 9 | `BU` → `TELEFON` → `YOX` | There is no phone here / This phone is missing. | **B. Simplified recognition sequence** |

| 10 | `SİZ` → `BAKI` → `GETMƏK` → `İSTƏMƏK` | Do you want to go to Baku? | **A. Natural AzSL sentence** |


## 9. Independent Reviewer Assessment (Reviewer Perspective)

### Reviewer Analysis:

When asked to optimize specifically for a robust real-time demonstration, the critical review diverges from a naive 28-class selection:

1. **Prune `GÖRMƏK` (13 train) and `BİLMƏK` (16 train)**: Classes with <20 samples are statistically hazardous. In real-time inference with a sliding window buffer, low-data classes with noisy cluster centers frequently trigger false positives or fail to fire, embarrassing a live demonstrator.

2. **Prune `O` (11.1% recall)**: `O` has 303 samples, but only 5 out of 45 test sequences were recognized by the baseline GRU. Relying on `BU` (which has 82.2% baseline F1 and 234 train samples) provides a dependable indexical pointing gesture without carrying the failure mode of `O`.

3. **Prune `İŞ` (33.3% recall)**: `İŞ` achieved only 2 correct predictions out of 6 test samples. Its deletion prevents false positive interference with `EV` and verb phrases.

4. **Synthesis Verdict**: The reviewer recommendation converges decisively on **24 classes**.


## 10. Final Recommended Vocabulary (24 Classes)


| Class | Category | English Gloss | Baseline F1 | Recall | Precision | Train | Val | Test | Total | Selection Rationale |

| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |

| **MƏN** | Pronoun | *I / me* | 0.801 | 0.710 | 0.918 | 1769 | 379 | 379 | 2527 | Base 1st-person singular pronoun. Retaining MƏN while excluding MƏNƏ/MƏNİM isolates base form cleanly. |

| **SƏN** | Pronoun | *you (singular)* | 0.435 | 0.714 | 0.312 | 34 | 7 | 7 | 48 | Base 2nd-person singular pronoun. Essential for interactive demo dialogues. |

| **BİZ** | Pronoun | *we / us* | 0.762 | 0.640 | 0.941 | 117 | 25 | 25 | 167 | 1st-person plural pronoun. High baseline F1 (0.762) and strong train support (117). |

| **SİZ** | Pronoun | *you (plural / polite)* | 0.580 | 0.439 | 0.853 | 308 | 66 | 66 | 440 | 2nd-person plural/polite pronoun. High train data (308). Excluding SİZİN removes possessive confusion. |

| **SALAM** | Social / Greeting | *Hello* | 0.889 | 1.000 | 0.800 | 22 | 4 | 4 | 30 | Standard AzSL greeting gesture. High baseline F1 (0.889) and perfect recall (1.000). |

| **SAĞLAM** | Social / State | *Healthy / Well* | 0.933 | 0.875 | 1.000 | 39 | 8 | 8 | 55 | High quality state descriptor (F1: 0.933, Precision: 1.000). |

| **İSTƏMƏK** | Verb | *To want / desire* | 0.333 | 0.250 | 0.500 | 73 | 16 | 16 | 105 | Core modal verb for expressing intentions. Excluding İSTƏYİRƏM removes 50% of test errors. |

| **GETMƏK** | Verb | *To go* | 0.571 | 0.571 | 0.571 | 34 | 7 | 7 | 48 | Core directional motion verb. Clean baseline F1 (0.571). |

| **GƏLMƏK** | Verb | *To come* | 0.500 | 0.400 | 0.667 | 24 | 5 | 5 | 34 | Reciprocal motion verb to GETMƏK. Enables departure and arrival sentences. |

| **YEMƏK** | Verb / Noun | *To eat / Food* | 0.727 | 0.800 | 0.667 | 26 | 5 | 5 | 36 | High-frequency daily verb/object. High baseline F1 (0.727). |

| **ALMAQ** | Verb | *To buy / take* | 0.667 | 0.857 | 0.545 | 31 | 7 | 7 | 45 | Core transaction verb. High baseline recall (0.857). |

| **OLMAQ** | Verb | *To be / become* | 0.667 | 0.667 | 0.667 | 28 | 6 | 6 | 40 | Fundamental copula/state verb. Balanced baseline metrics (F1: 0.667). |

| **EV** | Noun | *House / Home* | 0.696 | 0.667 | 0.727 | 53 | 12 | 12 | 77 | Core locative entity. Strong baseline F1 (0.696) and solid sample count (77 total). |

| **BAKI** | Noun / Location | *Baku* | 0.518 | 0.700 | 0.412 | 44 | 10 | 10 | 64 | Capital city and primary geographic landmark in AzSL. High real-time demo value. |

| **AZƏRBAYCAN** | Noun / Location | *Azerbaijan* | 0.769 | 0.833 | 0.714 | 25 | 6 | 6 | 37 | National identifier gesture. High baseline F1 (0.769) and strong recall (0.833). |

| **TELEFON** | Noun / Object | *Phone* | 0.600 | 0.500 | 0.750 | 28 | 6 | 6 | 40 | Everyday modern object. Balanced metrics (F1: 0.600, Precision: 0.750). |

| **HARDA** | Question / Locative | *Where* | 0.500 | 0.600 | 0.429 | 26 | 5 | 5 | 36 | Essential spatial interrogative particle for question sentences. |

| **NECƏ** | Question / Manner | *How* | 0.600 | 0.750 | 0.500 | 19 | 4 | 4 | 27 | Critical interrogative particle for standard greeting 'SALAM SƏN NECƏ'. |

| **BURDA** | Locative | *Here* | 0.562 | 0.600 | 0.529 | 72 | 15 | 15 | 102 | Core spatial demonstrative locative. Strong sample volume (102 total). |

| **BU GÜN** | Time / Adverb | *Today* | 0.667 | 0.600 | 0.750 | 25 | 5 | 5 | 35 | Core present temporal marker. High precision (0.750) and F1 (0.667). |

| **SABAH** | Time / Adverb | *Tomorrow* | 0.471 | 0.667 | 0.364 | 31 | 6 | 6 | 43 | Core future temporal marker. Forms temporal contrast with BU GÜN. |

| **VAR** | Existential / State | *There is / have* | 0.938 | 0.938 | 0.938 | 78 | 16 | 16 | 110 | Top performing class in dataset (F1: 0.938, Prec: 0.938, Rec: 0.938, Total: 110). |

| **YOX** | Negative / Polarity | *No / There is not* | 0.615 | 0.667 | 0.571 | 30 | 6 | 6 | 42 | Core negative polarity particle. Strong baseline F1 (0.615). |

| **BU** | Demonstrative | *This* | 0.822 | 0.740 | 0.925 | 234 | 50 | 50 | 334 | Primary indexical demonstrative. Robust baseline F1 (0.822) and high data (334 total). |



### Rejected Classes & Rationale:

| Rejected Class | Total Samples | Train Samples | Baseline F1 | Primary Reason for Rejection |

| :--- | :---: | :---: | :---: | :--- |

| **GÖRMƏK** | 19 | 13 | 0.444 | Critically low sample volume (<15 training samples, only 3 test samples). High statistical instability; test metric based on only 2 correct predictions out of 3. |

| **BİLMƏK** | 22 | 16 | 0.333 | Only 1 out of 3 test samples recognized. Low training volume (16 samples). Severe confusion with modal and temporal tokens. |

| **O** | 303 | 213 | 0.185 | Dismal baseline recall (11.1%, only 5/45 correct). 26 errors went to ONUN, but also confused with SƏN (3), ORA (3), and BAZAR (2). Functionally covered by indexical demonstrative BU (F1: 0.822). |

| **İŞ** | 39 | 27 | 0.400 | Low recall (33.3%, only 2/6 correct). Confused with multiple sub-tokens (İŞSİZ, İSTİ). Removing İŞ eliminates false positive draw on EV and other motion sentences. |

| **MƏNİM** | 168 | 118 | 0.281 | Major confusion attractor with base pronoun MƏN (55 mutual misclassifications in 200-class model). Excluded to cleanly isolate MƏN. |

| **MƏNƏ** | 115 | 81 | 0.174 | Major confusion attractor with base pronoun MƏN (55 mutual misclassifications). Excluded to isolate MƏN. |

| **ONUN** | 118 | 83 | 0.354 | Massive false positive sink (61 incoming false positives, including 26 from O). Excluded from demo vocabulary. |

| **SİZİN** | 79 | 55 | 0.364 | Morphological variant of SİZ causing 15 mutual misclassifications. Excluded to isolate SİZ. |

| **İSTƏYİRƏM** | 40 | 28 | 0.286 | Conjugated form of İSTƏMƏK causing 11 mutual misclassifications. Excluded to isolate infinitive İSTƏMƏK. |



## 11. Final Balancing Recommendation

### Primary Recommendation: Cap = 50 Samples/Class

- **Vocabulary**: **24 classes**

- **Training Cap**: **50 samples/class**

- **Training Samples Retained**: **866 sequences** (27.3% of original subset train)

- **Minimum Class Count**: 19 samples (`NECƏ`)

- **Maximum Class Count**: 50 samples (8 classes capped)

- **Max/Min Imbalance Ratio**: **2.63:1**

- **Validation Set**: Untouched existing split (676 sequences)

- **Test Set**: Untouched existing split (676 sequences)

- **Justification**: A 2.63:1 ratio provides near-uniform class priors, preventing high-frequency gestures from dominating the sliding-window threshold in live streaming.


### Alternative Recommendation: Cap = 75 Samples/Class

- **Training Samples Retained**: **1039 sequences**

- **Max/Min Imbalance Ratio**: **3.95:1** (5 classes capped)

- **Justification**: Suitable if training loss curves indicate that additional sample variance is required for GRU regularization.


## 12. Future Training Configuration (Ablation Control)

To ensure that future performance gains are strictly attributable to vocabulary curation and class balancing, the training setup must strictly preserve production specifications:


```yaml

Model Architecture (Strictly Preserved):

  Input Shape: [Batch, 26, 126] (MediaPipe HandLandmarker, 2 hands x 21 landmarks x 3 coordinates)

  Backbone: 2-layer unidirectional GRU

  Hidden Size: 128

  Dropout: 0.3

  Pooling: Temporal Mean Pooling (128) + Temporal Max Pooling (128) = 256-dim feature representation

  Classifier: Linear(256 -> 24)


Training Setup (Strictly Preserved):

  Optimizer: AdamW

  Learning Rate: 1e-3

  Weight Decay: 1e-4

  Batch Size: 32

  Random Seed: 42

  Precision: Mixed Precision (torch.cuda.amp)

  Max Epochs: 30

  Early Stopping: Patience 7 on validation macro F1

  Data Sourcing: Exact split subsets from outputs/dataset_split.json (NO new random splits)

```
