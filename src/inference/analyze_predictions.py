"""
Analysis of official test set predictions.
"""

import json
import csv
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import pandas as pd

def main():
    # Paths
    pred_csv = Path(r"c:\Users\ASUS\Desktop\azsl-word-recognition\outputs\inference\dataset_predictions.csv")
    eval_json = Path(r"c:\Users\ASUS\Desktop\azsl-word-recognition\outputs\inference\dataset_evaluation_report.json")
    saved_report = Path(r"c:\Users\ASUS\Desktop\azsl-word-recognition\outputs\test_report_gru_temporal_pooling.json")
    
    # Load predictions
    df = pd.read_csv(pred_csv)
    print(f"Loaded {len(df)} predictions")
    
    # Extract true and predicted class names
    true_classes = df["true_class"].tolist()
    pred_classes = df["predicted_class"].tolist()
    confidences = df["confidence"].tolist()
    
    # Load class mapping from eval JSON (or we can infer from unique classes)
    with open(eval_json, "r") as f:
        eval_data = json.load(f)
    
    # Get class ordering from the report (should be same as used in training)
    # The report contains per_class_precision etc in the order of idx_to_class
    # We'll need the idx_to_class list; it is not in the eval JSON? Actually it might be in the metadata.
    # Let's load the dataset split JSON to get idx_to_class.
    meta_path = Path(r"c:\Users\ASUS\Desktop\azsl-word-recognition\outputs\dataset_split.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    idx_to_class = meta["idx_to_class"]  # list where index corresponds to class label
    class_to_idx = meta["class_to_idx"]
    
    # Convert class names to indices for metric computation
    true_indices = [class_to_idx[c] for c in true_classes]
    pred_indices = [class_to_idx[c] for c in pred_classes]
    
    # Compute metrics
    accuracy = np.mean(np.array(true_indices) == np.array(pred_indices))
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(true_indices, pred_indices, average="macro", zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(true_indices, pred_indices, average="weighted", zero_division=0)
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(true_indices, pred_indices, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(true_indices, pred_indices)
    
    # Load saved report for comparison
    with open(saved_report, "r") as f:
        saved = json.load(f)
    saved_metrics = saved["test"]
    
    print("\n=== Benchmark Reproduction ===")
    print(f"{'Metric':<25} {'Saved Exp2':<12} {'New Evaluation':<15} {'Difference'}")
    print("-" * 60)
    for metric_name, saved_val in [("accuracy", "accuracy"), ("macro_f1", "macro_f1"), 
                                   ("macro_recall", "macro_recall"), 
                                   ("weighted_f1", "weighted_f1")]:
        if metric_name == "accuracy":
            new_val = accuracy
        elif metric_name == "macro_f1":
            new_val = f1_macro
        elif metric_name == "macro_recall":
            new_val = recall_macro
        elif metric_name == "weighted_f1":
            new_val = f1_weighted
        else:
            continue
        diff = new_val - saved_metrics[saved_val]
        print(f"{metric_name:<25} {saved_metrics[saved_val]:<12.4f} {new_val:<15.4f} {diff:+.4f}")
    
    # Also show precision and recall if needed
    print("\n=== Per-Class Metrics (showing extremes) ===")
    # Get indices sorted by f1
    f1_order = np.argsort(f1_per_class)  # ascending (worst first)
    print("\nTop 20 classes with lowest F1:")
    for i in range(min(20, len(f1_order))):
        idx = f1_order[i]
        class_name = idx_to_class[idx]
        print(f"  {class_name:<20} F1={f1_per_class[idx]:.3f}  P={precision_per_class[idx]:.3f}  R={recall_per_class[idx]:.3f}  support={support[idx]}")
    
    # Zero-F1 classes
    zero_f1_mask = (f1_per_class == 0)
    zero_f1_indices = np.where(zero_f1_mask)[0]
    print(f"\nZero-F1 classes ({len(zero_f1_indices)}):")
    if len(zero_f1_indices) > 0:
        for idx in zero_f1_indices[:20]:  # limit to 20
            print(f"  {idx_to_class[idx]}")
    else:
        print("  None")
    
    # Zero-recall classes
    zero_recall_mask = (recall_per_class == 0)
    zero_recall_indices = np.where(zero_recall_mask)[0]
    print(f"\nZero-recall classes ({len(zero_recall_indices)}):")
    if len(zero_recall_indices) > 0:
        for idx in zero_recall_indices[:20]:
            print(f"  {idx_to_class[idx]}")
    else:
        print("  None")
    
    # Confusion pairs (true -> predicted) excluding correct
    # Flatten confusion matrix and get top pairs
    # We'll create a list of (true_class, pred_class, count) for off-diagonal
    confusion_pairs = []
    for i in range(len(idx_to_class)):
        for j in range(len(idx_to_class)):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append((idx_to_class[i], idx_to_class[j], int(cm[i, j])))
    # Sort by count descending
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    print("\nTop 20 confusion pairs (true -> predicted):")
    for true_cls, pred_cls, cnt in confusion_pairs[:20]:
        print(f"  {true_cls} -> {pred_cls}: {cnt}")
    
    # Top error classes (classes with most errors as true label)
    # Errors per true class = sum of row minus diagonal
    errors_per_true = np.sum(cm, axis=1) - np.diag(cm)
    error_order = np.argsort(errors_per_true)[::-1]  # descending
    print("\nTop 20 classes with most errors (as true label):")
    for idx in error_order[:20]:
        class_name = idx_to_class[idx]
        print(f"  {class_name:<20} errors={errors_per_true[idx]}  (total={np.sum(cm[idx])})")
    
    # Confidence analysis
    correct_mask = np.array(true_indices) == np.array(pred_indices)
    conf_correct = np.array(confidences)[correct_mask]
    conf_incorrect = np.array(confidences)[~correct_mask]
    
    print("\n=== Confidence Analysis ===")
    print(f"Mean confidence correct: {np.mean(conf_correct):.4f}" if len(conf_correct) > 0 else "No correct predictions")
    print(f"Median confidence correct: {np.median(conf_correct):.4f}" if len(conf_correct) > 0 else "No correct predictions")
    print(f"Mean confidence incorrect: {np.mean(conf_incorrect):.4f}" if len(conf_incorrect) > 0 else "No incorrect predictions")
    print(f"Median confidence incorrect: {np.median(conf_incorrect):.4f}" if len(conf_incorrect) > 0 else "No incorrect predictions")
    
    # Accuracy at confidence thresholds
    thresholds = [0.5, 0.7, 0.9]
    for thresh in thresholds:
        mask = np.array(confidences) >= thresh
        if np.sum(mask) > 0:
            acc_thresh = np.mean(np.array(true_indices)[mask] == np.array(pred_indices)[mask])
            print(f"Accuracy for confidence >= {thresh}: {acc_thresh:.4f} (over {np.sum(mask)} samples)")
        else:
            print(f"Accuracy for confidence >= {thresh}: no samples")
    
    # Percentage of incorrect predictions with confidence >= threshold
    for thresh in [0.8, 0.9]:
        mask_incorrect = np.array(confidences) >= thresh
        incorrect_mask = ~correct_mask
        if np.sum(incorrect_mask) > 0:
            pct = np.sum(mask_incorrect & incorrect_mask) / np.sum(incorrect_mask) * 100
            print(f"Percentage of incorrect predictions with confidence >= {thresh}: {pct:.2f}%")
        else:
            print(f"Percentage of incorrect predictions with confidence >= {thresh}: N/A")
    
    # Top 20 highest-confidence incorrect predictions
    incorrect_indices = np.where(~correct_mask)[0]
    if len(incorrect_indices) > 0:
        incorrect_confidences = np.array(confidences)[incorrect_indices]
        # Sort by confidence descending
        sorted_idx = np.argsort(incorrect_confidences)[::-1]
        top20_incorrect = incorrect_indices[sorted_idx[:20]]
        print("\nTop 20 highest-confidence incorrect predictions:")
        for rank, idx in enumerate(top20_incorrect, start=1):
            true_cls = true_classes[idx]
            pred_cls = pred_classes[idx]
            conf = confidences[idx]
            video_path = df.iloc[idx]["video_path"]
            print(f"{rank:2d}. {true_cls} -> {pred_cls} (conf={conf:.4f}) [{video_path}]")
    else:
        print("\nNo incorrect predictions.")
    
    # Pronoun analysis: specific families
    print("\n=== Pronoun Analysis ===")
    pronouns = ["MƏN", "MƏNƏ", "MƏNİM", "SİZ", "SİZİN", "O", "ONUN", "ONLAR"]
    for p in pronouns:
        if p in class_to_idx:
            idx = class_to_idx[p]
            true_count = np.sum(np.array(true_indices) == idx)
            correct_count = np.sum((np.array(true_indices) == idx) & correct_mask)
            acc = correct_count / true_count if true_count > 0 else 0
            print(f"{p:>8}: true={true_count:4d}, correct={correct_count:4d}, accuracy={acc:.3f}")
        else:
            print(f"{p:>8}: class not found")
    
    # Confusion within pronoun families
    print("\nPronoun confusion pairs:")
    for true_p in ["MƏN", "MƏNƏ", "MƏNİM"]:
        for pred_p in ["MƏN", "MƏNƏ", "MƏNİM"]:
            if true_p != pred_p and true_p in class_to_idx and pred_p in class_to_idx:
                ti = class_to_idx[true_p]
                pi = class_to_idx[pred_p]
                cnt = cm[ti, pi]
                if cnt > 0:
                    print(f"  {true_p} -> {pred_p}: {cnt}")
    for true_p in ["SİZ", "SİZİN"]:
        for pred_p in ["SİZ", "SİZİN"]:
            if true_p != pred_p and true_p in class_to_idx and pred_p in class_to_idx:
                ti = class_to_idx[true_p]
                pi = class_to_idx[pred_p]
                cnt = cm[ti, pi]
                if cnt > 0:
                    print(f"  {true_p} -> {pred_p}: {cnt}")
    for true_p in ["O", "ONUN", "ONLAR"]:
        for pred_p in ["O", "ONUN", "ONLAR"]:
            if true_p != pred_p and true_p in class_to_idx and pred_p in class_to_idx:
                ti = class_to_idx[true_p]
                pi = class_to_idx[pred_p]
                cnt = cm[ti, pi]
                if cnt > 0:
                    print(f"  {true_p} -> {pred_p}: {cnt}")

if __name__ == "__main__":
    main()



