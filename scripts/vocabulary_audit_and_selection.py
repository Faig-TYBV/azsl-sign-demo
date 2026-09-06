import os
import sys
import json
import csv
import math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

def main():
    print("=== RUNNING COMPLETE 200-CLASS AUDIT & VOCABULARY SELECTION ===")

    # 1. Load dataset_split.json
    with open("outputs/dataset_split.json", "r", encoding="utf-8") as f:
        split_data = json.load(f)

    class_to_idx = split_data["class_to_idx"]
    idx_to_class = split_data["idx_to_class"]
    splits = split_data["splits"]

    train_paths = splits["train"]
    val_paths = splits["val"]
    test_paths = splits["test"]

    train_counts = Counter(Path(p).parent.name for p in train_paths)
    val_counts = Counter(Path(p).parent.name for p in val_paths)
    test_counts = Counter(Path(p).parent.name for p in test_paths)

    total_counts = {}
    for c in idx_to_class:
        total_counts[c] = train_counts[c] + val_counts[c] + test_counts[c]

    # 2. Load test_report_gru_normalized.json
    with open("outputs/test_report_gru_normalized.json", "r", encoding="utf-8") as f:
        test_report = json.load(f)

    test_metrics = test_report["test"]
    cm = np.array(test_report["confusion_matrix"])
    per_class_f1 = test_metrics["per_class_f1"]
    per_class_precision = test_metrics["per_class_precision"]
    per_class_recall = test_metrics["per_class_recall"]
    support = test_metrics["support"]

    # Print baseline
    acc = test_metrics["accuracy"]
    macro_f1 = test_metrics["macro_f1"]
    weighted_f1 = test_metrics["weighted_f1"]
    macro_recall = test_metrics["macro_recall"]
    macro_precision = float(np.mean(per_class_precision))
    loss = test_metrics["loss"]

    print(f"Verified Baseline Metrics:")
    print(f"  Accuracy:         {acc*100:.2f}% ({acc:.6f})")
    print(f"  Macro F1:         {macro_f1*100:.2f}% ({macro_f1:.6f})")
    print(f"  Weighted F1:      {weighted_f1*100:.2f}% ({weighted_f1:.6f})")
    print(f"  Macro Recall:     {macro_recall*100:.2f}% ({macro_recall:.6f})")
    print(f"  Macro Precision:  {macro_precision*100:.2f}% ({macro_precision:.6f})")
    print(f"  Test Loss:        {loss:.6f}")
    print(f"  Total Samples:    {len(train_paths) + len(val_paths) + len(test_paths)} (Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)})")
    print(f"  Total Classes:    {len(idx_to_class)}")

    # 3. Class Quality Scoring
    class_stats = []
    zero_f1_classes = []

    for i, cname in enumerate(idx_to_class):
        prec = per_class_precision[i]
        rec = per_class_recall[i]
        f1 = per_class_f1[i]
        sup = support[i]
        tot = total_counts[cname]
        tr = train_counts[cname]
        vl = val_counts[cname]
        ts = test_counts[cname]

        tp = int(cm[i, i])
        fn = int(np.sum(cm[i, :]) - tp)
        fp = int(np.sum(cm[:, i]) - tp)

        # Sample score (log-scaled up to 100 samples)
        sample_score = min(1.0, math.log10(max(1, tot)) / math.log10(100))

        # Confusion score: penalizes both incoming false positives and outgoing false negatives
        confusion_rate = (fp + fn) / max(1, 2 * sup)
        confusion_score = max(0.0, 1.0 - min(1.0, confusion_rate))

        # Quality score formula
        quality_score = (
            0.35 * f1 +
            0.25 * rec +
            0.15 * prec +
            0.15 * sample_score +
            0.10 * confusion_score
        )

        # Tier assignment
        # Tier 1: Strong performance, solid data, high isolation
        # Tier 2: Viable performance, adequate data
        # Tier 3: Marginal
        # Tier 4: Zero F1
        if f1 == 0.0:
            tier = "Tier 4 (Zero Performance)"
            zero_f1_classes.append({
                "class_name": cname,
                "index": i,
                "total_samples": tot,
                "train": tr,
                "val": vl,
                "test": ts,
                "support": sup,
                "top_outgoing": [],
                "incoming_fp": fp
            })
        elif f1 >= 0.65 and tot >= 25 and quality_score >= 0.60:
            tier = "Tier 1 (High Quality)"
        elif f1 >= 0.40 and tot >= 18 and quality_score >= 0.45:
            tier = "Tier 2 (Viable Candidate)"
        else:
            tier = "Tier 3 (Marginal Quality)"

        # Top 3 outgoing misclassifications
        row = cm[i, :].copy()
        row[i] = 0
        top_out_idx = np.argsort(row)[::-1][:3]
        top_outgoing = [
            {"target_class": idx_to_class[idx], "count": int(row[idx])}
            for idx in top_out_idx if row[idx] > 0
        ]

        # Top 3 incoming false positives
        col = cm[:, i].copy()
        col[i] = 0
        top_in_idx = np.argsort(col)[::-1][:3]
        top_incoming = [
            {"source_class": idx_to_class[idx], "count": int(col[idx])}
            for idx in top_in_idx if col[idx] > 0
        ]

        stat = {
            "index": i,
            "class_name": cname,
            "precision": float(round(prec, 4)),
            "recall": float(round(rec, 4)),
            "f1_score": float(round(f1, 4)),
            "quality_score": float(round(quality_score, 4)),
            "sample_score": float(round(sample_score, 4)),
            "confusion_score": float(round(confusion_score, 4)),
            "tier": tier,
            "support": sup,
            "train_samples": tr,
            "val_samples": vl,
            "test_samples": ts,
            "total_samples": tot,
            "true_positives": tp,
            "false_negatives": fn,
            "false_positives": fp,
            "top_outgoing_confusions": top_outgoing,
            "top_incoming_confusions": top_incoming
        }
        class_stats.append(stat)

    print(f"\nClass quality summary:")
    t_counts = Counter(s["tier"] for s in class_stats)
    for t, cnt in t_counts.items():
        print(f"  {t}: {cnt} classes")

    # Populate zero-f1 details
    for z in zero_f1_classes:
        idx = z["index"]
        row = cm[idx, :].copy()
        top_idx = np.argsort(row)[::-1][:3]
        z["top_outgoing"] = [
            {"predicted_as": idx_to_class[j], "count": int(row[j])}
            for j in top_idx if row[j] > 0
        ]

    print(f"\nTotal Zero-F1 classes: {len(zero_f1_classes)}")

    # Sort by quality score descending
    class_stats_sorted = sorted(class_stats, key=lambda x: x["quality_score"], reverse=True)

    # Let's see top 25 classes by quality score
    print("\nTop 25 Classes by Quality Score:")
    for rank, s in enumerate(class_stats_sorted[:25], 1):
        print(f"{rank:2d}. {s['class_name']:<16} | F1: {s['f1_score']:.3f} | Rec: {s['recall']:.3f} | Prec: {s['precision']:.3f} | Tot: {s['total_samples']:4d} | QS: {s['quality_score']:.3f} | {s['tier']}")

    # Check specific words of semantic interest
    semantic_check = [
        "MƏN", "SƏN", "BİZ", "SİZ", "O", "SALAM", "SAĞLAM", "İSTƏMƏK", "GETMƏK", "GƏLMƏK",
        "BİLMƏK", "GÖRMƏK", "YEMƏK", "ALMAQ", "OLMAQ", "EV", "İŞ", "BAKI", "AZƏRBAYCAN",
        "TELEFON", "HARDA", "NECƏ", "BURDA", "BU GÜN", "SABAH", "VAR", "YOX", "KÖMƏK", "DOST"
    ]
    print("\nSemantic Word Candidates Audit:")
    for w in semantic_check:
        match = [s for s in class_stats if s["class_name"] == w]
        if match:
            s = match[0]
            print(f"{s['class_name']:<15} | F1: {s['f1_score']:.3f} | Rec: {s['recall']:.3f} | Prec: {s['precision']:.3f} | Train: {s['train_samples']:3d} | Tot: {s['total_samples']:4d} | QS: {s['quality_score']:.3f} | {s['tier']}")
        else:
            print(f"{w:<15} | NOT IN DATASET!")

if __name__ == "__main__":
    main()
