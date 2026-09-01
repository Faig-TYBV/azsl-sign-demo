"""
Unit tests for the dataset evaluation script.
"""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import csv

import pytest

# We'll test the functions by importing the module
from src.inference import evaluate_dataset


def test_import():
    """Test that the module can be imported."""
    assert hasattr(evaluate_dataset, 'main')


@patch('src.inference.evaluate_dataset.predict')
def test_main_structure(mock_predict):
    """Test that main function runs without error (with mocks)."""
    # Mock the predict.load_model to return dummy objects matching our metadata
    dummy_model = MagicMock()
    dummy_model.eval.return_value = None
    dummy_model.to.return_value = dummy_model
    # class_to_idx and idx_to_class must match our metadata
    dummy_class_to_idx = {"class0": 0, "class1": 1}
    dummy_idx_to_class = ["class0", "class1"]
    mock_predict.load_model.return_value = (dummy_model, dummy_class_to_idx, dummy_idx_to_class, "cpu")
    
    # Mock predict_video to return dummy results
    mock_predict.predict_video.return_value = {
        "video": "dummy.mp4",
        "predicted_class": "class0",
        "confidence": 0.9,
        "top_k": [{"class": "class0", "probability": 0.9}, {"class": "class1", "probability": 0.1}],
        "preprocessing": {
            "num_original_frames": 10,
            "original_fps": 30.0,
            "valid_frames": 10,
            "valid_ratio": 1.0,
            "feature_shape": [26, 126]
        }
    }
    
    # Create a temporary directory for outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        metadata_path = tmpdir / "dataset_split.json"
        checkpoint_path = tmpdir / "checkpoint.pt"
        output_dir = tmpdir / "outputs" / "inference"
        
        # Create dummy metadata
        metadata = {
            "splits": {
                "test": ["data/features/full/class0/video1.npz"]
            },
            "class_to_idx": {"class0": 0, "class1": 1},
            "idx_to_class": ["class0", "class1"]
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        # Create dummy checkpoint file
        checkpoint_path.touch()
        
        # Call main with our temporary paths
        evaluate_dataset.main(metadata_path=metadata_path,
                              checkpoint_path=checkpoint_path,
                              output_dir=output_dir)
        
        # Check that output files were created
        assert (output_dir / "dataset_predictions.csv").exists()
        assert (output_dir / "dataset_evaluation_report.json").exists()
        
        # Check CSV content using csv module
        csv_path = output_dir / "dataset_predictions.csv"
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["true_class"] == "class0"
        assert rows[0]["predicted_class"] == "class0"
        assert rows[0]["correct"] == "True"  # CSV writes bool as string "True"/"False"
        
        # Check JSON content
        with open(output_dir / "dataset_evaluation_report.json") as f:
            report = json.load(f)
        assert report["number_of_evaluated_videos"] == 1
        assert report["number_correct"] == 1
        assert report["accuracy"] == 1.0


if __name__ == "__main__":
    test_import()
    test_main_structure()
    print("All tests passed!")
