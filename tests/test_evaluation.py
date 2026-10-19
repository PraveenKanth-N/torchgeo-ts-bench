"""Unit tests for predictive and system evaluation metrics."""

import numpy as np
import torch
from torchgeo_ts_bench.evaluation import BenchmarkEvaluator


def test_classification_metrics():
    """Verify accuracy, precision, recall, and F1 calculations."""
    y_true = np.array([0, 1, 2, 1, 0, 2])
    y_pred = np.array([0, 1, 2, 0, 0, 2])

    metrics = BenchmarkEvaluator.evaluate_classification(y_true, y_pred)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    assert metrics["accuracy"] == 5.0 / 6.0


def test_segmentation_metrics():
    """Verify mean IoU, Dice, and pixel accuracy."""
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    metrics = BenchmarkEvaluator.evaluate_segmentation(y_true, y_pred, num_classes=2)
    assert "mean_iou" in metrics
    assert "dice_score" in metrics
    assert "pixel_accuracy" in metrics
    assert 0.0 <= metrics["mean_iou"] <= 1.0


def test_regression_metrics():
    """Verify MAE, RMSE, and R2 calculation."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])

    metrics = BenchmarkEvaluator.evaluate_regression(y_true, y_pred)
    assert metrics["mae"] > 0.0
    assert metrics["rmse"] > 0.0
    assert metrics["r2"] > 0.8


def test_efficiency_telemetry():
    """Verify process memory measurement and inference benchmarking."""
    mem_mb = BenchmarkEvaluator.get_process_memory_mb()
    assert mem_mb > 10.0  # Python process uses at least 10 MB

    # Quick dummy model benchmarking
    model = torch.nn.Linear(10, 2)
    dummy_input = torch.randn(4, 10)
    eff = BenchmarkEvaluator.benchmark_inference_efficiency(
        model=model,
        sample_batch=dummy_input,
        num_warmup=2,
        num_timed_runs=5,
    )
    assert eff["mean_inference_latency_ms"] >= 0.0
    assert eff["throughput_samples_per_sec"] > 0.0
    assert eff["peak_process_memory_mb"] > 0.0
