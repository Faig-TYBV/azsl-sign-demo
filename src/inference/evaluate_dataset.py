"""
Batch evaluation of the official test set using the Experiment 2 checkpoint.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Import the inference pipeline functions
from src.inference import predict


# Default paths (can be overridden for testing)
DEFAULT_METADATA_PATH = Path(r"c:\Users\ASUS\Desktop\azsl-word-recognition\outputs\dataset_split.json")
DEFAULT_CHECKPOINT_PATH = Path(r"c:\Users\ASUS\Desktop\azsl-word-recognition\outputs\checkpoints\gru_temporal_pool_best.pt")
DEFAULT_OUTPUT_DIR = Path(r"c:\Users\ASUS\Desktop\azsl-word-recognition\outputs\inference")


def main(metadata_path=DEFAULT_METADATA_PATH,
         checkpoint_path=DEFAULT_CHECKPOINT_PATH,
         output_dir=DEFAULT_OUTPUT_DIR):
    # Load split metadata
    with open(metadata_path, 'r') as f:
        data = json.load(f)
    
    test_feature_paths = data['splits']['test']  # list of strings (POSIX paths)
    
    # Load model and class mapping from checkpoint
    model, class_to_idx, idx_to_class, device = predict.load_model(checkpoint_path)
    model.to(device)
    model.eval()
    
    # Verify that the class mapping in the checkpoint matches the metadata (optional)
    metadata_class_to_idx = data['class_to_idx']
    if class_to_idx != metadata_class_to_idx:
        print("Warning: class mapping in checkpoint differs from metadata.")
    
    # Prepare to collect results
    results = []  # list of dicts for CSV
    all_true = []
    all_pred = []
    all_confidences = []
    
    print(f"Evaluating {len(test_feature_paths)} test videos...")
    for i, feature_path_str in enumerate(test_feature_paths):
        # Convert feature path to raw video path
        feature_path = Path(feature_path_str)
        class_name = feature_path.parent.name  # true class from the feature directory
        video_stem = feature_path.stem
        raw_video_path = Path(r"c:\Users\ASUS\Desktop\azsl-word-recognition\data\raw\AzSLD_Words_200") / class_name / f"{video_stem}.mp4"
        
        # Run inference on the raw video
        try:
            pred_result = predict.predict_video(
                video_path=str(raw_video_path),
                checkpoint_path=str(checkpoint_path),
                top_k=5
            )
        except Exception as e:
            print(f"Error processing {raw_video_path}: {e}")
            # Skip this video? or assign a dummy prediction? We'll skip and continue.
            continue
        
        true_class = class_name
        predicted_class = pred_result['predicted_class']
        confidence = pred_result['confidence']
        top5 = pred_result['top_k']  # list of dicts with keys 'class' and 'probability'
        
        # Build top5 string for CSV: we'll use a JSON string for deterministic representation
        top5_json = json.dumps([(item['class'], item['probability']) for item in top5])
        
        # Determine if correct
        correct = (true_class == predicted_class)
        
        # Get number of frames and valid frames from preprocessing info
        preproc = pred_result['preprocessing']
        num_frames = preproc['num_original_frames']
        valid_frames = preproc['valid_frames']
        
        results.append({
            'video_path': str(raw_video_path),
            'true_class': true_class,
            'predicted_class': predicted_class,
            'correct': correct,
            'confidence': confidence,
            'top5_predictions': top5_json,
            'num_frames': num_frames,
            'valid_frames': valid_frames
        })
        
        all_true.append(class_to_idx[true_class])
        all_pred.append(class_to_idx[predicted_class])
        all_confidences.append(confidence)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(test_feature_paths)} videos")
    
    # Write CSV
    csv_path = output_dir / "dataset_predictions.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['video_path', 'true_class', 'predicted_class', 'correct', 'confidence', 'top5_predictions', 'num_frames', 'valid_frames'])
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved predictions to {csv_path}")
    
    # Compute metrics
    accuracy = accuracy_score(all_true, all_pred)
    # Precision, recall, F1
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(all_true, all_pred, average='macro', zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(all_true, all_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(all_true, all_pred, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(all_true, all_pred)
    
    # Confidence statistics
    confidences_np = np.array(all_confidences)
    correct_mask = np.array(all_true) == np.array(all_pred)
    conf_correct = confidences_np[correct_mask]
    conf_incorrect = confidences_np[~correct_mask]
    
    # Build report dictionary
    report = {
        "checkpoint": str(checkpoint_path),
        "model_configuration": {
            "input_size": 126,
            "hidden_size": 128,
            "num_layers": 2,
            "num_classes": 200,
            "dropout": 0.3,
            "bidirectional": False,
            "pooling": "mean_max"
        },
        "number_of_evaluated_videos": len(results),
        "number_correct": int(sum(t == p for t, p in zip(all_true, all_pred))),
        "number_incorrect": int(len(all_true) - sum(t == p for t, p in zip(all_true, all_pred))),
        "accuracy": float(accuracy),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(precision_weighted),
        "weighted_recall": float(recall_weighted),
        "weighted_f1": float(f1_weighted),
        "per_class_precision": [float(p) for p in precision_per_class],
        "per_class_recall": [float(r) for r in recall_per_class],
        "per_class_f1": [float(f) for f in f1_per_class],
        "confusion_matrix": cm.tolist(),  # 2D list
        "confidence_statistics": {
            "mean_confidence_correct": float(np.mean(conf_correct)) if len(conf_correct) > 0 else 0.0,
            "median_confidence_correct": float(np.median(conf_correct)) if len(conf_correct) > 0 else 0.0,
            "mean_confidence_incorrect": float(np.mean(conf_incorrect)) if len(conf_incorrect) > 0 else 0.0,
            "median_confidence_incorrect": float(np.median(conf_incorrect)) if len(conf_incorrect) > 0 else 0.0,
        }
    }
    
    # Write JSON report
    json_path = output_dir / "dataset_evaluation_report.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved evaluation report to {json_path}")
    
    # Print summary
    print(f"\nEvaluation Summary:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Macro F1: {f1_macro:.4f}")
    print(f"  Weighted F1: {f1_weighted:.4f}")

if __name__ == "__main__":
    main()
