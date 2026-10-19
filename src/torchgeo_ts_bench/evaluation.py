"""Evaluation engine for predictive performance and system resource efficiency.

Calculates task-specific metrics (Accuracy, F1, Precision, Recall, IoU, MAE, RMSE, R²)
alongside training time, inference latency, throughput, peak RAM, and GPU VRAM.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import psutil
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from torch import Tensor, nn


@dataclass
class EvaluationResults:
    """Encapsulates predictive metrics, efficiency measurements, and metadata."""

    model_name: str
    task: str
    dataset_name: str
    predictive_metrics: Dict[str, float]
    efficiency_metrics: Dict[str, float]
    parameter_metrics: Dict[str, int]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to a flat dictionary suitable for CSV / JSON logging."""
        out: Dict[str, Any] = {
            "model_name": self.model_name,
            "task": self.task,
            "dataset_name": self.dataset_name,
        }
        for k, v in self.predictive_metrics.items():
            out[f"metric_{k}"] = round(float(v), 5)
        for k, v in self.efficiency_metrics.items():
            out[f"efficiency_{k}"] = round(float(v), 4)
        for k, v in self.parameter_metrics.items():
            out[f"param_{k}"] = int(v)
        for k, v in self.metadata.items():
            if isinstance(v, (str, int, float, bool)):
                out[f"meta_{k}"] = v
        return out


class BenchmarkEvaluator:
    """Standardized evaluation engine for Earth Observation foundation models."""

    @staticmethod
    def get_process_memory_mb() -> float:
        """Measure current process RSS memory in megabytes (MB)."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024.0 * 1024.0)

    @staticmethod
    def get_peak_gpu_memory_mb() -> Optional[float]:
        """Measure peak allocated CUDA GPU memory in MB if available."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        return None

    @classmethod
    def evaluate_classification(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Compute standard multi-class remote sensing classification metrics."""
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        acc = float(accuracy_score(y_true, y_pred))
        p_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        r_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        p_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        r_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        return {
            "accuracy": acc,
            "precision_macro": p_macro,
            "recall_macro": r_macro,
            "f1_macro": f1_macro,
            "precision_weighted": p_weighted,
            "recall_weighted": r_weighted,
            "f1_weighted": f1_weighted,
        }

    @classmethod
    def evaluate_segmentation(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        num_classes: int,
    ) -> Dict[str, float]:
        """Compute IoU, mean IoU, Dice, and pixel accuracy for segmentation."""
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        pixel_acc = float(np.mean(y_true == y_pred))
        ious: List[float] = []
        dices: List[float] = []

        for c in range(num_classes):
            intersection = np.sum((y_true == c) & (y_pred == c))
            union = np.sum((y_true == c) | (y_pred == c))
            if union > 0:
                ious.append(float(intersection / union))
                dice = float((2.0 * intersection) / (np.sum(y_true == c) + np.sum(y_pred == c)))
                dices.append(dice)

        miou = float(np.mean(ious)) if len(ious) > 0 else 0.0
        mean_dice = float(np.mean(dices)) if len(dices) > 0 else 0.0

        return {
            "pixel_accuracy": pixel_acc,
            "mean_iou": miou,
            "dice_score": mean_dice,
        }

    @classmethod
    def evaluate_regression(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Compute MAE, RMSE, and R² for continuous biophysical regression."""
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0

        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }

    @classmethod
    def evaluate_change_detection(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Compute binary Precision, Recall, F1, and IoU for change detection."""
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        p = float(precision_score(y_true, y_pred, zero_division=0))
        r = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        intersection = np.sum((y_true == 1) & (y_pred == 1))
        union = np.sum((y_true == 1) | (y_pred == 1))
        iou = float(intersection / union) if union > 0 else 1.0

        return {
            "precision": p,
            "recall": r,
            "f1_score": f1,
            "iou": iou,
        }

    @classmethod
    def benchmark_inference_efficiency(
        cls,
        model: nn.Module,
        sample_batch: Tensor,
        num_warmup: int = 5,
        num_timed_runs: int = 20,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, float]:
        """Benchmark inference latency, throughput, and memory consumption.

        Args:
            model: PyTorch model.
            sample_batch: Tensor [B, T, C, H, W].
            num_warmup: Number of un-timed warmup passes.
            num_timed_runs: Number of timed iterations.
            device: Computation device (CPU or CUDA).

        Returns:
            Dictionary with mean latency (ms), p50, p95, throughput (samples/s), peak RAM / VRAM.
        """
        model.eval()
        sample_batch = sample_batch.to(device)
        batch_size = sample_batch.shape[0]

        # Reset GPU peak stats if available
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        mem_start_rss = cls.get_process_memory_mb()

        # Warmup passes
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(sample_batch)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)

        # Timed passes
        latencies: List[float] = []
        with torch.no_grad():
            for _ in range(num_timed_runs):
                t_start = time.perf_counter()
                _ = model(sample_batch)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t_end = time.perf_counter()
                latencies.append((t_end - t_start) * 1000.0)  # in milliseconds

        mem_end_rss = cls.get_process_memory_mb()
        peak_gpu_mem = cls.get_peak_gpu_memory_mb()

        latencies_arr = np.array(latencies)
        mean_latency = float(np.mean(latencies_arr))
        p50_latency = float(np.percentile(latencies_arr, 50))
        p95_latency = float(np.percentile(latencies_arr, 95))
        throughput = float(batch_size / (mean_latency / 1000.0))

        results = {
            "mean_inference_latency_ms": mean_latency,
            "p50_inference_latency_ms": p50_latency,
            "p95_inference_latency_ms": p95_latency,
            "throughput_samples_per_sec": throughput,
            "peak_process_memory_mb": max(mem_start_rss, mem_end_rss),
        }
        if peak_gpu_mem is not None:
            results["peak_gpu_memory_mb"] = peak_gpu_mem

        return results
