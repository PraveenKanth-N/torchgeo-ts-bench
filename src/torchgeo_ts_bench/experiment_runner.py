"""Experiment runner orchestrating end-to-end spatiotemporal benchmarks.

Coordinates data loading, leakage audits, training, inference profiling,
metric calculation, persistence, and automated report generation.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from .data_loader import (
    PASTISSpatiotemporalDataset,
    SyntheticSpatiotemporalDataset,
    create_spatiotemporal_dataloader,
)
from .evaluation import BenchmarkEvaluator, EvaluationResults
from .leakage_checks import LeakageDetector
from .model_registry import get_model, list_available_models
from .preprocessing import SpatiotemporalPreprocessor
from .reporting import BenchmarkReporter
from .reproducibility import ReproducibilityManager
from .result_storage import BenchmarkResultStorage
from .tasks import SpatiotemporalClassificationTask


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class BenchmarkExperimentRunner:
    """End-to-end orchestrator for standardized geospatial foundation model benchmarks."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None) -> None:
        self.config = config_dict or self.get_default_config()
        self.storage = BenchmarkResultStorage(output_dir=self.config.get("output_dir", "outputs"))
        self.reporter = BenchmarkReporter(output_dir=self.config.get("output_dir", "outputs"))

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Default standardized experiment configuration."""
        return {
            "experiment_name": "Standardized-Spatiotemporal-Foundation-Benchmark",
            "seed": 42,
            "dataset_type": "pastis",  # "pastis" or "synthetic"
            "data_root": "data/pastis",
            "fixture_dir": r"D:\torchgeo\torchgeo-0.10.1\tests\data\pastis",
            "output_dir": "outputs",
            "task": "classification",
            "num_classes": 5,
            "seq_len": 6,
            "in_channels": 3,
            "height": 32,
            "width": 32,
            "batch_size": 4,
            "epochs": 3,
            "lr": 0.001,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "models": ["convlstm", "ltae", "st_3dcnn", "temporal_transformer"],
            "leakage_checks": {
                "spatial_iou_threshold": 0.0,
                "enforce_chronological_split": False,
            },
            "preprocessing": {
                "normalization": "min_max",
                "scale_factor": 1.0,
            },
        }

    def run(self) -> Dict[str, Any]:
        """Execute the complete benchmarking pipeline."""
        print("=" * 70)
        print("[*] STARTING TORCHGEO-TS-BENCH STANDARDIZED BENCHMARK RUN")
        print("=" * 70)

        # 1. Reproducibility & Environment Setup
        seed = self.config.get("seed", 42)
        ReproducibilityManager.set_seed(seed)
        sys_env = ReproducibilityManager.capture_environment()
        config_hash = ReproducibilityManager.compute_config_hash(self.config)
        self.config["config_hash"] = config_hash

        print(f"[*] Configuration Hash: {config_hash}")
        print(f"[*] Platform: {sys_env.os_platform} ({sys_env.os_release}) | Python: {sys_env.python_version}")
        print(f"[*] PyTorch: {sys_env.torch_version} | TorchGeo: {sys_env.torchgeo_version}")
        print(f"[*] Compute Target: {self.config.get('device', 'cpu')}")

        device = torch.device(self.config.get("device", "cpu"))

        # 2. Dataset Initialization
        ds_type = self.config.get("dataset_type", "pastis").lower()
        print(f"[*] Initializing dataset: {ds_type.upper()}...")

        if ds_type == "pastis":
            try:
                dataset = PASTISSpatiotemporalDataset(
                    root=self.config.get("data_root", "data/pastis"),
                    folds=(1, 2),
                    bands=("B04", "B03", "B02"),
                    target_seq_len=self.config.get("seq_len", 6),
                    num_classes=self.config.get("num_classes", 5),
                    source_fixture_dir=self.config.get("fixture_dir", None),
                )
                print(f"    Loaded real PASTIS spatiotemporal dataset: {len(dataset)} sequences.")
            except Exception as e:
                print(f"    [!] Error loading PASTIS dataset ({e}). Falling back to synthetic test fixture.")
                dataset = SyntheticSpatiotemporalDataset(
                    num_samples=32,
                    seq_len=self.config.get("seq_len", 6),
                    in_channels=self.config.get("in_channels", 3),
                    num_classes=self.config.get("num_classes", 5),
                    seed=seed,
                )
        else:
            dataset = SyntheticSpatiotemporalDataset(
                num_samples=32,
                seq_len=self.config.get("seq_len", 6),
                in_channels=self.config.get("in_channels", 3),
                num_classes=self.config.get("num_classes", 5),
                seed=seed,
            )
            print(f"    Loaded synthetic spatiotemporal test fixture: {len(dataset)} sequences.")

        # 3. Create Standardized Splits
        n_total = len(dataset)
        n_train = max(1, int(0.6 * n_total))
        n_val = max(1, int(0.2 * n_total))
        n_test = max(1, n_total - n_train - n_val)

        train_indices = list(range(0, n_train))
        val_indices = list(range(n_train, n_train + n_val))
        test_indices = list(range(n_train + n_val, n_total))

        # If tiny dataset, ensure test split has samples
        if not test_indices and val_indices:
            test_indices = val_indices

        train_set = Subset(dataset, train_indices)
        val_set = Subset(dataset, val_indices)
        test_set = Subset(dataset, test_indices)

        print(f"[*] Dataset Splits -> Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

        # 4. Mandatory Leakage-Aware Split Audit
        print("[*] Running automated Leakage Detection Audit...")
        train_meta = [dataset.get_sample_metadata(i) for i in train_indices]
        val_meta = [dataset.get_sample_metadata(i) for i in val_indices]
        test_meta = [dataset.get_sample_metadata(i) for i in test_indices]

        leak_cfg = self.config.get("leakage_checks", {})
        leakage_report = LeakageDetector.audit_splits(
            train_metadata=train_meta,
            test_metadata=test_meta,
            val_metadata=val_meta,
            spatial_iou_threshold=leak_cfg.get("spatial_iou_threshold", 0.0),
            enforce_chronological_split=leak_cfg.get("enforce_chronological_split", False),
        )
        print(f"    Leakage Audit Result: {leakage_report.summary}")

        # 5. Preprocessing & Dataloaders
        preprocessor = SpatiotemporalPreprocessor(
            normalization_method=self.config.get("preprocessing", {}).get("normalization", "min_max"),
            scale_factor=self.config.get("preprocessing", {}).get("scale_factor", 1.0),
        )

        batch_size = self.config.get("batch_size", 4)
        train_loader = create_spatiotemporal_dataloader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = create_spatiotemporal_dataloader(val_set, batch_size=batch_size, shuffle=False)
        test_loader = create_spatiotemporal_dataloader(test_set, batch_size=batch_size, shuffle=False)

        task_engine = SpatiotemporalClassificationTask(num_classes=self.config.get("num_classes", 5))

        # 6. Model Training & Evaluation Loop
        model_names: List[str] = self.config.get("models", ["convlstm", "ltae"])
        evaluation_results_list: List[EvaluationResults] = []
        tabular_records: List[Dict[str, Any]] = []

        epochs = self.config.get("epochs", 3)
        lr = self.config.get("lr", 0.001)

        for m_name in model_names:
            print("-" * 50)
            print(f"[*] Benchmarking Model: {m_name.upper()}...")

            # Initialize Model Adapter
            model = get_model(
                name=m_name,
                in_channels=self.config.get("in_channels", 3),
                seq_len=self.config.get("seq_len", 6),
                num_classes=self.config.get("num_classes", 5),
                task=self.config.get("task", "classification"),
            ).to(device)

            param_counts = model.count_parameters()
            print(f"    Total Parameters: {param_counts['total_parameters']:,}")

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # Training Phase
            t_train_start = time.perf_counter()
            model.train()
            for ep in range(epochs):
                for batch in train_loader:
                    x = preprocessor.transform(batch["image"]).to(device)
                    y = batch["label"].to(device)
                    optimizer.zero_grad()
                    logits = model(x)
                    loss = task_engine.compute_loss(logits, y)
                    loss.backward()
                    optimizer.step()
            train_duration = time.perf_counter() - t_train_start
            print(f"    Training completed in {train_duration:.2f} seconds.")

            # Evaluation on Test Split
            model.eval()
            all_preds: List[int] = []
            all_targets: List[int] = []
            test_batch_sample: Optional[torch.Tensor] = None

            with torch.no_grad():
                for batch in test_loader:
                    x = preprocessor.transform(batch["image"]).to(device)
                    y = batch["label"].to(device)
                    if test_batch_sample is None:
                        test_batch_sample = x.clone()
                    logits = model(x)
                    preds = task_engine.predict(logits)
                    all_preds.extend(preds.cpu().numpy().tolist())
                    all_targets.extend(y.cpu().numpy().tolist())

            # Compute Predictive Metrics
            pred_metrics = BenchmarkEvaluator.evaluate_classification(
                y_true=all_targets,
                y_pred=all_preds,
            )
            print(f"    Accuracy: {pred_metrics['accuracy']:.4f} | F1 (Macro): {pred_metrics['f1_macro']:.4f}")

            # Profile Computational Efficiency
            if test_batch_sample is None:
                test_batch_sample = torch.randn(
                    batch_size,
                    self.config.get("seq_len", 6),
                    self.config.get("in_channels", 3),
                    self.config.get("height", 32),
                    self.config.get("width", 32),
                )

            eff_metrics = BenchmarkEvaluator.benchmark_inference_efficiency(
                model=model,
                sample_batch=test_batch_sample,
                num_warmup=3,
                num_timed_runs=15,
                device=device,
            )
            eff_metrics["training_duration_sec"] = train_duration
            print(f"    Latency: {eff_metrics['mean_inference_latency_ms']:.2f} ms | Peak RAM: {eff_metrics['peak_process_memory_mb']:.1f} MB")

            eval_res = EvaluationResults(
                model_name=model.model_name,
                task=self.config.get("task", "classification"),
                dataset_name=dataset.dataset_name,
                predictive_metrics=pred_metrics,
                efficiency_metrics=eff_metrics,
                parameter_metrics=param_counts,
                metadata={
                    "model_family": model.model_family,
                    "is_synthetic": dataset.is_synthetic,
                    "config_hash": config_hash,
                },
            )
            evaluation_results_list.append(eval_res)
            tabular_records.append(eval_res.to_dict())

        # 7. Persistence and Outputs
        results_df = pd.DataFrame(tabular_records)
        self.storage.save_experiment_config(self.config)
        self.storage.append_or_write_results_csv(tabular_records, overwrite=True)

        full_metrics = {
            "experiment_config": self.config,
            "system_environment": sys_env.to_dict(),
            "leakage_audit": leakage_report.to_dict(),
            "results": tabular_records,
        }
        self.storage.save_metrics_json(full_metrics)

        # 8. Report & Visualizations Generation
        print("[*] Generating visualizations and benchmark report...")
        self.reporter.generate_model_comparison_plot(results_df)
        self.reporter.generate_runtime_comparison_plot(results_df)
        self.reporter.generate_benchmark_report_markdown(
            experiment_config=self.config,
            system_env=sys_env.to_dict(),
            leakage_report=leakage_report.to_dict(),
            results_df=results_df,
        )

        print("=" * 70)
        print("[+] BENCHMARK RUN COMPLETED SUCCESSFULLY!")
        print(f"   Artifacts saved in: {self.storage.output_dir}")
        print("=" * 70)

        return full_metrics


def main() -> None:
    """CLI entrypoint for running the benchmark suite."""
    parser = argparse.ArgumentParser(description="TorchGeo-TS-Bench Benchmark Runner")
    parser.add_argument("--config", type=str, default=None, help="Path to benchmark.yaml config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--device", type=str, default=None, help="Override compute device (cpu/cuda)")
    args = parser.parse_args()

    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    if config is None:
        config = BenchmarkExperimentRunner.get_default_config()

    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.device is not None:
        config["device"] = args.device

    runner = BenchmarkExperimentRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
