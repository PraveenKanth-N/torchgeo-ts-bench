"""Model registry for spatiotemporal foundation models and baselines.

Provides discovery, dynamic instantiation, and architectural metadata catalog.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Type

from .model_adapters import (
    ConvLSTMAdapter,
    LTAEAdapter,
    ST3DCNNAdapter,
    SpatiotemporalModelAdapter,
    TemporalTransformerAdapter,
)

_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_model(
    name: str,
    family: str,
    description: str,
    supported_tasks: Optional[List[str]] = None,
    is_foundation_model: bool = False,
    requirements: Optional[str] = None,
) -> Callable[[Type[SpatiotemporalModelAdapter]], Type[SpatiotemporalModelAdapter]]:
    """Decorator to register a spatiotemporal model adapter."""

    def decorator(cls: Type[SpatiotemporalModelAdapter]) -> Type[SpatiotemporalModelAdapter]:
        key = name.lower()
        _MODEL_REGISTRY[key] = {
            "name": name,
            "class": cls,
            "family": family,
            "description": description,
            "supported_tasks": supported_tasks or ["classification"],
            "is_foundation_model": is_foundation_model,
            "requirements": requirements or "Standard PyTorch / TorchGeo environment",
        }
        return cls

    return decorator


# Register built-in models
register_model(
    name="convlstm",
    family="Recurrent Spatiotemporal",
    description="TorchGeo ConvLSTM: Convolutional LSTM recurrent cells processing 2D spatial features across time.",
    supported_tasks=["classification", "segmentation", "regression"],
    is_foundation_model=False,
)(ConvLSTMAdapter)

register_model(
    name="ltae",
    family="Temporal Attention",
    description="TorchGeo L-TAE: Lightweight Temporal Attention Encoder for multi-temporal remote sensing sequences.",
    supported_tasks=["classification", "regression"],
    is_foundation_model=False,
)(LTAEAdapter)

register_model(
    name="st_3dcnn",
    family="3D Spatiotemporal Convolution",
    description="3D Convolutional Network performing joint spatiotemporal feature learning.",
    supported_tasks=["classification"],
    is_foundation_model=False,
)(ST3DCNNAdapter)

register_model(
    name="temporal_transformer",
    family="Temporal Self-Attention Transformer",
    description="Multihead Self-Attention temporal transformer encoding sequence dynamics.",
    supported_tasks=["classification"],
    is_foundation_model=False,
)(TemporalTransformerAdapter)


# Documented Foundation Models catalog (Presto, CROMA, Prithvi-EO)
FOUNDATION_MODELS_CATALOG: Dict[str, Dict[str, Any]] = {
    "presto": {
        "name": "Presto (Pretrained Remote Sensing Transformer)",
        "paper": "Tseng et al., 2024 (NASA Harvest)",
        "architecture": "Masked Autoencoder (ViT backbone) with channel group embeddings",
        "input_bands": 17,
        "supported_sensors": ["Sentinel-1", "Sentinel-2", "ERA5", "SRTM", "Dynamic World"],
        "download_size": "~350 MB pretrained checkpoint",
        "access_requirements": "Requires downloading pretrained weights from NASA Harvest Hugging Face / TorchGeo hub.",
        "hardware_recommendation": "GPU recommended (minimum 4 GB VRAM) for batch inference.",
        "torchgeo_support": "Included in torchgeo.models.Presto",
    },
    "croma": {
        "name": "CROMA (Cross-Modal Alignment)",
        "paper": "Fuller et al., 2023",
        "architecture": "Cross-attention joint MSI (optical) and SAR (radar) foundation model",
        "input_bands": 14,
        "supported_sensors": ["Sentinel-1 SAR (VV, VH)", "Sentinel-2 MSI (12 bands)"],
        "download_size": "~1.2 GB pretrained checkpoint",
        "access_requirements": "Requires matched SAR + MSI paired acquisitions and pretrained weights.",
        "hardware_recommendation": "NVIDIA GPU with >= 8 GB VRAM recommended.",
        "torchgeo_support": "Included in torchgeo.models.CROMA",
    },
    "prithvi_eo": {
        "name": "Prithvi-EO (NASA-IBM Geospatial Foundation Model)",
        "paper": "Jakubik et al., 2023",
        "architecture": "Temporal ViT (MAE) trained on Harmonized Landsat-Sentinel (HLS)",
        "input_bands": 6,
        "supported_sensors": ["HLS (B02, B03, B04, B8A, B11, B12)"],
        "download_size": "~1.1 GB (Prithvi-100M)",
        "access_requirements": "Hugging Face NASA-IMB repo (requires terratorch or custom ViT adapter).",
        "hardware_recommendation": "Dedicated GPU (>= 8 GB VRAM) required for fine-tuning.",
        "torchgeo_support": "External / TerraTorch ecosystem",
    },
}


def list_available_models() -> List[str]:
    """Return list of all registered model identifiers."""
    return list(_MODEL_REGISTRY.keys())


def get_model_info(name: str) -> Dict[str, Any]:
    """Retrieve metadata description for a registered model."""
    key = name.lower()
    if key in _MODEL_REGISTRY:
        info = dict(_MODEL_REGISTRY[key])
        info.pop("class", None)
        return info
    if key in FOUNDATION_MODELS_CATALOG:
        return FOUNDATION_MODELS_CATALOG[key]
    raise KeyError(f"Model '{name}' not found in registry. Available: {list_available_models()}")


def get_model(name: str, **kwargs: Any) -> SpatiotemporalModelAdapter:
    """Instantiate a registered spatiotemporal model adapter."""
    key = name.lower()
    if key not in _MODEL_REGISTRY:
        raise KeyError(
            f"Model '{name}' not registered. Available models: {list_available_models()}"
        )
    cls = _MODEL_REGISTRY[key]["class"]
    return cls(**kwargs)
