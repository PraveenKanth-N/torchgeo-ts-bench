"""Unit tests for spatiotemporal data loading and batching."""

import os
import torch
from torchgeo_ts_bench.data_loader import (
    PASTISSpatiotemporalDataset,
    SyntheticSpatiotemporalDataset,
    create_spatiotemporal_dataloader,
)


def test_synthetic_dataset_shapes_and_metadata():
    """Verify synthetic dataset shapes, timestamps, and sample metadata."""
    num_samples = 10
    seq_len = 6
    in_channels = 3
    height = 32
    width = 32
    num_classes = 4

    ds = SyntheticSpatiotemporalDataset(
        num_samples=num_samples,
        seq_len=seq_len,
        in_channels=in_channels,
        height=height,
        width=width,
        num_classes=num_classes,
        seed=123,
    )

    assert len(ds) == num_samples
    assert ds.is_synthetic is True

    sample = ds[0]
    assert "image" in sample
    assert "label" in sample
    assert "timestamps" in sample
    assert "bbox" in sample
    assert "sample_id" in sample

    assert sample["image"].shape == (seq_len, in_channels, height, width)
    assert len(sample["timestamps"]) == seq_len
    assert len(sample["bbox"]) == 4

    meta = ds.get_sample_metadata(0)
    assert meta["sample_id"] == sample["sample_id"]
    assert meta["is_synthetic"] is True


def test_spatiotemporal_dataloader_batching():
    """Verify DataLoader collation of spatiotemporal batches."""
    ds = SyntheticSpatiotemporalDataset(num_samples=8, seq_len=4, in_channels=2, height=16, width=16)
    loader = create_spatiotemporal_dataloader(ds, batch_size=4, shuffle=False)

    batch = next(iter(loader))
    assert batch["image"].shape == (4, 4, 2, 16, 16)  # [B, T, C, H, W]
    assert batch["label"].shape == (4,)
    assert len(batch["timestamps"]) == 4
    assert len(batch["bbox"]) == 4
    assert len(batch["sample_id"]) == 4
    assert batch["is_synthetic"] is True


def test_pastis_dataset_fixture_loading():
    """Verify PASTIS spatiotemporal dataset loader using the TorchGeo test fixture."""
    fixture_dir = r"D:\torchgeo\torchgeo-0.10.1\tests\data\pastis"
    if os.path.exists(fixture_dir):
        ds = PASTISSpatiotemporalDataset(
            root="data/test_pastis",
            folds=(1, 2),
            bands=("B04", "B03", "B02"),
            target_seq_len=6,
            source_fixture_dir=fixture_dir,
        )
        assert len(ds) >= 1
        sample = ds[0]
        assert sample["image"].shape[0] == 6  # T
        assert sample["image"].shape[1] == 3  # C
        assert sample["image"].ndim == 4      # [T, C, H, W]
        assert sample["metadata"]["is_synthetic"] is False
