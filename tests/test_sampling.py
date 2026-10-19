"""Unit tests for spatial and temporal sampling."""

import torch
from torchgeo_ts_bench.sampling import (
    SpatialSamplingConfig,
    SpatioTemporalSamplingManager,
    TemporalSamplingConfig,
)


def test_spatiotemporal_sampling_manager():
    """Verify temporal sequence alignment and spatial patch cropping."""
    manager = SpatioTemporalSamplingManager(
        spatial_cfg=SpatialSamplingConfig(patch_size=(32, 32), random_crop=False),
        temporal_cfg=TemporalSamplingConfig(target_seq_len=6, temporal_strategy="uniform"),
    )

    # Input has 10 frames and larger 48x48 spatial dimensions
    input_tensor = torch.randn(10, 3, 48, 48)
    sampled = manager.sample_tensor(input_tensor, seed=42)

    assert sampled.shape == (6, 3, 32, 32)


def test_temporal_padding_when_input_is_short():
    """Verify temporal manager properly repeats frames when input has fewer frames."""
    manager = SpatioTemporalSamplingManager(
        spatial_cfg=SpatialSamplingConfig(patch_size=(16, 16)),
        temporal_cfg=TemporalSamplingConfig(target_seq_len=8),
    )

    short_tensor = torch.randn(3, 4, 16, 16)
    sampled = manager.sample_tensor(short_tensor)

    assert sampled.shape == (8, 4, 16, 16)
