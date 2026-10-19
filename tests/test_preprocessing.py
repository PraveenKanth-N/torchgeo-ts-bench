"""Unit tests for spatiotemporal preprocessing and normalization."""

import torch
from torchgeo_ts_bench.preprocessing import SpatiotemporalPreprocessor


def test_preprocessing_nan_cleaning_and_min_max():
    """Verify NaN/Inf removal and min-max normalization."""
    preprocessor = SpatiotemporalPreprocessor(
        normalization_method="min_max",
        clip_range=(0.0, 1000.0),
        scale_factor=1.0 / 1000.0,
        fill_value=0.0,
    )

    x = torch.tensor([
        [0.0, 500.0, 1000.0],
        [float("nan"), 2000.0, float("inf")],
    ]).view(1, 1, 2, 3)

    out = preprocessor.transform(x)

    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_preprocessing_band_selection():
    """Verify band selection pruning along channel dimension."""
    preprocessor = SpatiotemporalPreprocessor(
        band_indices=[0, 2],
        normalization_method="none",
    )

    x = torch.randn(4, 5, 16, 16)  # [T=4, C=5, H=16, W=16]
    out = preprocessor.transform(x)

    assert out.shape == (4, 2, 16, 16)
