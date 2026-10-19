"""End-to-end integration test for the benchmark pipeline."""

import os
import shutil
from torchgeo_ts_bench.experiment_runner import BenchmarkExperimentRunner


def test_end_to_end_pipeline_execution(tmp_path):
    """Verify end-to-end execution of the benchmark suite on a synthetic dataset."""
    test_out_dir = str(tmp_path / "test_outputs")

    config = {
        "experiment_name": "Test-E2E-Run",
        "seed": 99,
        "dataset_type": "synthetic",
        "output_dir": test_out_dir,
        "task": "classification",
        "num_classes": 3,
        "seq_len": 4,
        "in_channels": 2,
        "height": 16,
        "width": 16,
        "batch_size": 2,
        "epochs": 1,
        "lr": 0.01,
        "device": "cpu",
        "models": ["convlstm", "st_3dcnn"],
        "leakage_checks": {
            "spatial_iou_threshold": 0.0,
            "enforce_chronological_split": False,
        },
        "preprocessing": {
            "normalization": "min_max",
            "scale_factor": 1.0,
        },
    }

    runner = BenchmarkExperimentRunner(config)
    results = runner.run()

    assert "results" in results
    assert len(results["results"]) == 2

    # Verify generated artifact files exist
    assert os.path.exists(os.path.join(test_out_dir, "results.csv"))
    assert os.path.exists(os.path.join(test_out_dir, "metrics.json"))
    assert os.path.exists(os.path.join(test_out_dir, "experiment_config.yaml"))
    assert os.path.exists(os.path.join(test_out_dir, "model_comparison.png"))
    assert os.path.exists(os.path.join(test_out_dir, "runtime_comparison.png"))
    assert os.path.exists(os.path.join(test_out_dir, "benchmark_report.md"))
