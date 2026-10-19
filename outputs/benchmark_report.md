# TorchGeo-TS-Bench: A Standardized Benchmark Suite for Spatiotemporal Geospatial Foundation Models

**Project Title:** A Standardized Benchmark Suite for Spatiotemporal Geospatial Foundation Models  
**Course:** Advanced Python Programming  
**Authors:**
- **Praveen Kanth N** (23MIC0078)
- **Om Aravindhashan S** (23MID0222)
- **Monish R** (23MID0318)

---

## 1. Executive Summary

This report presents experimental evaluation results obtained using **TorchGeo-TS-Bench**, our standardized benchmarking framework built on top of the open-source **TorchGeo 0.10.1** library. 

TorchGeo-TS-Bench addresses critical research gaps in geospatial machine learning:
1. **Lack of standardized evaluation protocols** for multi-temporal satellite sequences.
2. **Hidden data leakage** caused by overlapping geographic tiles, temporal window contamination, and repeated sampling.
3. **Absence of joint predictive-quality and computational efficiency benchmarking** (Pareto trade-offs).

---

## 2. Experimental Setup and Hardware Telemetry

The experiment was executed under strictly recorded hardware and software environment specifications:

- **Operating System:** Windows (11)
- **Python Version:** 3.14.3
- **PyTorch Version:** 2.14.1+cpu
- **TorchGeo Version:** 0.10.1
- **CPU:** Intel64 Family 6 Model 140 Stepping 1, GenuineIntel (4 physical cores, 8 logical cores)
- **System RAM:** 15.77 GB
- **CUDA Device:** N/A (CPU execution mode)
- **Random Seed:** 42
- **Configuration Fingerprint (SHA-256):** `cdf6a3a8abf8b5d1`

---

## 3. Mandatory Research Novelty and Feature-to-Evidence Matrix


| Proposed Contribution | Implemented Module | Validation Test / Experiment | Actual Output Artifact | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Unified Benchmark Protocol** | `data_loader.py`, `sampling.py`, `preprocessing.py` | Standardized spatiotemporal batching & sampling | `outputs/experiment_config.yaml` | **Implemented & Tested** |
| **Leakage-Aware Evaluation** | `leakage_checks.py` | Automated spatial, temporal & duplicate audit | `outputs/metrics.json` (`leakage_audit`) | **Implemented & Tested** |
| **Joint Quality & Efficiency Comparison** | `evaluation.py`, `reporting.py` | Multi-model Pareto frontier & telemetry benchmark | `outputs/results.csv`, `outputs/runtime_comparison.png` | **Implemented & Tested** |
| **Configuration-Based Reproducibility** | `reproducibility.py`, `result_storage.py` | Seed locking, hardware fingerprinting, config hashing | `outputs/metrics.json` (`system_environment`) | **Implemented & Tested** |
| **Automated Benchmark Reporting & UI** | `reporting.py`, `app.py` | Automated markdown report & interactive Streamlit app | `outputs/benchmark_report.md`, `app.py` | **Implemented & Tested** |


---

## 4. Leakage-Aware Split Audit

**Overall Audit Status:** ✅ PASSED (Leakage-Free)

### Audit Summary
- **Spatial Overlap Check:** 0 bounding-box intersection violations detected.
- **Chronological Sequence Check:** 0 temporal contamination issues detected.
- **Duplicate Sample Check:** 0 duplicate sample identity reuses detected.
- **Detailed Finding:** PASSED: All splits are verified leakage-free across space, time, and sample identities.

> **Controlled Leakage Software Test Proof:**  
> Our controlled unit test in `tests/test_leakage_checks.py` deliberately injects overlapping coordinates (IoU > 0.1) and out-of-order temporal timestamps. The audit successfully flagged 100% of injected anomalies, proving that the verification engine prevents false benchmark claims.

---

## 5. Model Leaderboard (Predictive Performance)

Models were evaluated under identical experimental conditions (dataset: `PASTIS-SITS`, sequence length: `6`, batch size: `4`):

| Rank | Model Name | Accuracy | F1 (Macro) | Precision (Macro) | Recall (Macro) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | **TorchGeo-ConvLSTM** | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2 | **TorchGeo-LTAE** | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 3 | **ST-3DCNN-Baseline** | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 4 | **Temporal-Transformer** | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

---

## 6. Computational Efficiency and Resource Profiling

| Model Name | Training Time | Latency (mean) | Throughput | Peak Process RAM | Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TorchGeo-ConvLSTM** | 7.22s | 114.79 ms | 34.8 smp/s | 658.2 MB | 270,918 |
| **TorchGeo-LTAE** | 0.22s | 2.13 ms | 1876.5 smp/s | 655.1 MB | 35,083 |
| **ST-3DCNN-Baseline** | 0.40s | 7.54 ms | 530.6 smp/s | 658.7 MB | 62,661 |
| **Temporal-Transformer** | 0.50s | 1.99 ms | 2012.0 smp/s | 682.4 MB | 72,069 |

---

## 7. Comparative Analysis and Pareto Discussion

1. **Recurrent Spatiotemporal Models (`TorchGeo-ConvLSTM`):**
   - Retains 2D spatial feature representations across temporal transitions via convolutional gates.
   - Provides strong spatial-temporal inductive biases for pixelwise and sequence tasks, with moderate computational footprint.

2. **Temporal Attention Models (`TorchGeo-LTAE`):**
   - Utilizes lightweight temporal multi-head attention over time-series feature embeddings.
   - Extremely lightweight parameter footprint with low inference latency, ideal for resource-constrained Earth Observation pipelines.

3. **3D Spatiotemporal Baseline (`ST-3DCNN`):**
   - Applies 3D convolutions jointly across time and spatial dimensions.
   - Fast inference on small patch sizes with balanced memory utilization.

4. **Temporal Transformer Baseline:**
   - Evaluates full multi-head self-attention across the temporal dimension.
   - Scalable to long observation periods with flexible temporal positional embeddings.

---

## 8. Artifact Locations and Reproduction Instructions

All artifacts are saved locally in `outputs/`:
- **Results CSV:** `outputs/results.csv`
- **Metrics JSON:** `outputs/metrics.json`
- **Configuration YAML:** `outputs/experiment_config.yaml`
- **Model Comparison Plot:** `outputs/model_comparison.png`
- **Runtime Trade-Off Plot:** `outputs/runtime_comparison.png`

### Reproduction Command:
```powershell
# Navigate to project root
cd D:\torchgeo\torchgeo-ts-bench

# Execute standardized benchmark run
& "D:\torchgeo\.venv\Scripts\python.exe" scripts/run_benchmark.py --config configs/benchmark.yaml

# Launch interactive dashboard
& "D:\torchgeo\.venv\Scripts\python.exe" -m streamlit run app.py
```
