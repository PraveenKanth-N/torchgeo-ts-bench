"""Unit tests for spatiotemporal model adapters and parameter counting."""

import pytest
import torch
from torchgeo_ts_bench.model_adapters import (
    ConvLSTMAdapter,
    LTAEAdapter,
    ST3DCNNAdapter,
    TemporalTransformerAdapter,
)
from torchgeo_ts_bench.model_registry import get_model, list_available_models


@pytest.mark.parametrize("model_key", ["convlstm", "ltae", "st_3dcnn", "temporal_transformer"])
def test_model_forward_shape(model_key: str):
    """Verify each registered spatiotemporal model takes [B, T, C, H, W] and outputs [B, num_classes]."""
    b, t, c, h, w = 2, 6, 3, 32, 32
    num_classes = 5

    model = get_model(
        name=model_key,
        in_channels=c,
        seq_len=t,
        num_classes=num_classes,
        task="classification",
    )

    x = torch.randn(b, t, c, h, w)
    out = model(x)

    assert out.shape == (b, num_classes)
    param_counts = model.count_parameters()
    assert param_counts["total_parameters"] > 0
    assert param_counts["trainable_parameters"] > 0


def test_model_registry_discovery():
    """Verify registry lists available models correctly."""
    models = list_available_models()
    assert "convlstm" in models
    assert "ltae" in models
    assert "st_3dcnn" in models
    assert "temporal_transformer" in models
