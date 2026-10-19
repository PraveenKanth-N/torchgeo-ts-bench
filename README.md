# TorchGeo-TS-Bench: A Standardized Benchmark Suite for Spatiotemporal Geospatial Foundation Models

**Advanced Python Programming Course Project**  
**VIT University**

### Project Team Members:
- **Praveen Kanth N** — `23MIC0078`
- **Om Aravindhashan S** — `23MID0222`
- **Monish R** — `23MID0318`

---

## 1. Overview and Core Research Contribution

Geospatial artificial intelligence (GeoAI) has witnessed rapid proliferation of foundation models (FMs) and deep learning architectures tailored for Earth Observation (EO). However, comparing these models fairly remains problematic due to disparate data samplers, ad-hoc preprocessing pipelines, inconsistent splits, and insidious **spatial and temporal data leakage**.

**TorchGeo-TS-Bench** is an original benchmarking framework built on top of **TorchGeo 0.10.1**. It introduces a standardized, reproducible, leakage-aware spatiotemporal benchmarking layer specifically designed to evaluate foundation models on multi-temporal satellite imagery under rigorously identical experimental conditions.

### Key Contributions (Section 3A)
1. **Unified Benchmark Protocol:** Standardizes dataset loading, spatial/temporal windowing, preprocessing, and task evaluation across spatiotemporal foundation models.
2. **Leakage-Aware Evaluation Engine:** Automates geographic bounding-box overlap checks, chronological sequence audits, and duplicate footprint detection.
3. **Joint Quality & Computational Efficiency Comparison:** Compares predictive metrics (Accuracy, F1-score, Precision, Recall) against latency, memory footprint, throughput, and parameter counts (Pareto analysis).
4. **Configuration-Based Reproducibility:** Serializes random seeds, hardware telemetry, software package versions, and SHA-256 configuration hashes.
5. **Automated Reporting and Interactive UI:** Automatically produces publication-grade Markdown reports, comparison plots, and an interactive Streamlit dashboard.

---

## 2. Feature-to-Evidence Matrix (Section 3A Mandatory Verification)

| Proposed Contribution | Implemented Module | Validation Test / Experiment | Actual Output Artifact | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Unified Benchmark Protocol** | `data_loader.py`, `sampling.py`, `preprocessing.py` | Multi-temporal batch collation & windowing | `configs/benchmark.yaml`, `outputs/experiment_config.yaml` | **Implemented & Tested** |
| **Leakage-Aware Evaluation** | `leakage_checks.py` | Spatial overlap, chronological sequence & duplicate test | `outputs/metrics.json` (`leakage_audit`), `tests/test_leakage_checks.py` | **Implemented & Tested** |
| **Joint Quality & Efficiency Telemetry** | `evaluation.py`, `reporting.py` | Multi-model benchmarking on real Sentinel-2 sequences | `outputs/results.csv`, `outputs/runtime_comparison.png` | **Implemented & Tested** |
| **Configuration-Based Reproducibility** | `reproducibility.py`, `result_storage.py` | Seed locking, system telemetry, config SHA-256 fingerprint | `outputs/metrics.json` (`system_environment`) | **Implemented & Tested** |
| **Automated Benchmark Reporting & UI** | `reporting.py`, `app.py` | Auto-generated Markdown report & Streamlit dashboard | `outputs/benchmark_report.md`, `app.py` | **Implemented & Tested** |

---

## 3. Architecture and Pipeline

The framework implements a modular, extensible pipeline:

```
Satellite Earth Observation Data (Sentinel-2 / PASTIS)
                        │
                        ▼
         TorchGeo Dataset Loader (data_loader.py)
                        │
                        ▼
     Spatial & Temporal Sampling (sampling.py)
                        │
                        ▼
      Standardized Preprocessing (preprocessing.py)
                        │
                        ▼
      Leakage Detection Audit (leakage_checks.py)  ──► [Passed / Warnings Logged]
                        │
                        ▼
     Model Registry & Adapters (model_adapters.py)
         ├── TorchGeo-ConvLSTM
         ├── TorchGeo-LTAE
         ├── ST-3DCNN Baseline
         └── Temporal Transformer Baseline
                        │
                        ▼
       Downstream Tasks & Loss (tasks.py)
                        │
                        ▼
      Evaluation & Profiling Engine (evaluation.py)
         ├── Predictive Metrics (Accuracy, F1, Precision, Recall)
         └── Efficiency Telemetry (Latency, Peak RAM, Throughput, Params)
                        │
                        ▼
     Artifact Storage & Reporting (reporting.py)
         ├── outputs/results.csv
         ├── outputs/metrics.json
         ├── outputs/model_comparison.png
         ├── outputs/runtime_comparison.png
         └── outputs/benchmark_report.md
                        │
                        ▼
       Interactive Dashboard UI (app.py)
```

---

## 4. Setup and Installation on Windows

### Prerequisites
- Windows 10/11 x64
- Python 3.12+ (Tested on Python 3.14.3)
- PowerShell

### Step-by-Step Installation Commands:
```powershell
# 1. Open PowerShell and navigate to workspace
cd D:\torchgeo

# 2. Activate the virtual environment
& "D:\torchgeo\.venv\Scripts\Activate.ps1"

# 3. Ensure torchgeo is installed in editable mode
& "D:\torchgeo\.venv\Scripts\python.exe" -m pip install -e "D:\torchgeo\torchgeo-0.10.1" --no-deps

# 4. Install benchmarking suite in editable mode
cd D:\torchgeo\torchgeo-ts-bench
& "D:\torchgeo\.venv\Scripts\python.exe" -m pip install -e . --no-deps
```

---

## 5. Execution Commands

### Running the Automated Test Suite:
```powershell
cd D:\torchgeo\torchgeo-ts-bench
& "D:\torchgeo\.venv\Scripts\python.exe" scripts/run_tests.py
```

### Running the End-to-End Benchmark Experiment:
```powershell
cd D:\torchgeo\torchgeo-ts-bench
& "D:\torchgeo\.venv\Scripts\python.exe" scripts/run_benchmark.py --config configs/benchmark.yaml
```

### Launching the Interactive Streamlit Dashboard:
```powershell
cd D:\torchgeo\torchgeo-ts-bench
& "D:\torchgeo\.venv\Scripts\python.exe" -m streamlit run app.py
```

---

## 6. Output Artifacts

Following completion of a benchmark run, the following artifacts are generated in `outputs/`:
- `outputs/results.csv`: Flat tabular summary of all model evaluations.
- `outputs/metrics.json`: Full hierarchical metrics, leakage audit logs, and environment snapshot.
- `outputs/experiment_config.yaml`: Exact YAML configuration with deterministic hash.
- `outputs/model_comparison.png`: High-resolution comparison plot of predictive metrics.
- `outputs/runtime_comparison.png`: Pareto frontier visualization (Latency vs Accuracy vs Memory).
- `outputs/benchmark_report.md`: Formal academic benchmark report.
