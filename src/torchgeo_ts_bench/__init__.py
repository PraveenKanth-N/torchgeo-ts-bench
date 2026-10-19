"""TorchGeo-TS-Bench: A Standardized Benchmark Suite for Spatiotemporal Geospatial Foundation Models.

Advanced Python Programming Course Project.
Authors:
    - Praveen Kanth N (23MIC0078)
    - Om Aravindhashan S (23MID0222)
    - Monish R (23MID0318)
"""

__version__ = "0.1.0"
__authors__ = [
    "Praveen Kanth N (23MIC0078)",
    "Om Aravindhashan S (23MID0222)",
    "Monish R (23MID0318)",
]

from .data_loader import (
    BaseSpatiotemporalDataset,
    PASTISSpatiotemporalDataset,
    GeoTIFFTimeSeriesDataset,
    SyntheticSpatiotemporalDataset,
    create_spatiotemporal_dataloader,
)
from .sampling import (
    SpatialSamplingConfig,
    TemporalSamplingConfig,
    SpatioTemporalSamplingManager,
)
from .preprocessing import SpatiotemporalPreprocessor
from .model_registry import (
    register_model,
    get_model,
    list_available_models,
    get_model_info,
)
from .model_adapters import (
    SpatiotemporalModelAdapter,
    ConvLSTMAdapter,
    LTAEAdapter,
    ST3DCNNAdapter,
    TemporalTransformerAdapter,
)
from .tasks import (
    BaseDownstreamTask,
    SpatiotemporalClassificationTask,
    SpatiotemporalSegmentationTask,
    SpatiotemporalRegressionTask,
    ChangeDetectionTask,
)
from .evaluation import BenchmarkEvaluator, EvaluationResults
from .leakage_checks import LeakageDetector, LeakageAuditReport
from .reproducibility import ReproducibilityManager, SystemEnvironment
from .result_storage import BenchmarkResultStorage
from .reporting import BenchmarkReporter
from .experiment_runner import BenchmarkExperimentRunner

__all__ = [
    "__version__",
    "__authors__",
    "BaseSpatiotemporalDataset",
    "PASTISSpatiotemporalDataset",
    "GeoTIFFTimeSeriesDataset",
    "SyntheticSpatiotemporalDataset",
    "create_spatiotemporal_dataloader",
    "SpatialSamplingConfig",
    "TemporalSamplingConfig",
    "SpatioTemporalSamplingManager",
    "SpatiotemporalPreprocessor",
    "register_model",
    "get_model",
    "list_available_models",
    "get_model_info",
    "SpatiotemporalModelAdapter",
    "ConvLSTMAdapter",
    "LTAEAdapter",
    "ST3DCNNAdapter",
    "TemporalTransformerAdapter",
    "BaseDownstreamTask",
    "SpatiotemporalClassificationTask",
    "SpatiotemporalSegmentationTask",
    "SpatiotemporalRegressionTask",
    "ChangeDetectionTask",
    "BenchmarkEvaluator",
    "EvaluationResults",
    "LeakageDetector",
    "LeakageAuditReport",
    "ReproducibilityManager",
    "SystemEnvironment",
    "BenchmarkResultStorage",
    "BenchmarkReporter",
    "BenchmarkExperimentRunner",
]
