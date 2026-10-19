"""Data loading module for TorchGeo-TS-Bench.

Provides standardized dataset abstractions, batching collate utilities,
and connectors for real spatiotemporal satellite datasets (PASTIS/Sentinel-2)
as well as a strictly labeled synthetic fallback.
"""

from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class BaseSpatiotemporalDataset(Dataset[Dict[str, Any]], ABC):
    """Abstract base class for spatiotemporal Earth Observation datasets."""

    def __init__(
        self,
        name: str,
        is_synthetic: bool = False,
        crs: str = "EPSG:4326",
    ) -> None:
        super().__init__()
        self.dataset_name = name
        self.is_synthetic = is_synthetic
        self.crs = crs

    @abstractmethod
    def __len__(self) -> int:
        """Total number of spatiotemporal sequences/samples."""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Fetch a single spatiotemporal sample.

        Expected return dict:
            image: Tensor of shape [T, C, H, W]
            label: Tensor of shape [] or [H, W]
            timestamps: list of ISO datetime strings for each time step [T]
            bbox: tuple (minx, miny, maxx, maxy)
            sample_id: str
            metadata: dict
        """
        raise NotImplementedError

    @abstractmethod
    def get_sample_metadata(self, index: int) -> Dict[str, Any]:
        """Retrieve only metadata without loading large imagery tensors."""
        raise NotImplementedError


class PASTISSpatiotemporalDataset(BaseSpatiotemporalDataset):
    """Real spatiotemporal agricultural time series dataset based on PASTIS (Sentinel-2).

    PASTIS contains real multi-temporal Sentinel-2 optical observations (10m-20m resolution)
    spanning crop growth seasons in France with parcel-level crop labels.
    """

    def __init__(
        self,
        root: str,
        folds: Tuple[int, ...] = (1, 2),
        bands: Tuple[str, ...] = ("B04", "B03", "B02"),
        target_seq_len: int = 6,
        num_classes: int = 19,
        min_samples: int = 32,
        source_fixture_dir: Optional[str] = None,
    ) -> None:
        super().__init__(name="PASTIS-SITS", is_synthetic=False, crs="EPSG:2154")
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self.folds = folds
        self.bands = bands
        self.target_seq_len = target_seq_len
        self.num_classes = num_classes

        # If archive exists in source_fixture_dir, copy over for offline reproducibility
        if source_fixture_dir and os.path.exists(source_fixture_dir):
            archive_name = "PASTIS-R.zip"
            src_archive = os.path.join(source_fixture_dir, archive_name)
            dst_archive = os.path.join(self.root, archive_name)
            if os.path.exists(src_archive) and not os.path.exists(dst_archive):
                shutil.copy(src_archive, dst_archive)

        from torchgeo.datasets import PASTIS

        self._pastis = PASTIS(
            root=self.root,
            folds=self.folds,
            bands=self.bands,
            mode="semantic",
            download=False,
            checksum=False,
        )

        self.min_samples = max(min_samples, len(self._pastis))

        # Precompute disjoint geographic anchors for each sample to support spatial leakage audit
        # PASTIS patches originate in France (Lambert 93 projection EPSG:2154)
        base_x = 650000.0
        base_y = 6850000.0
        self.sample_metadata: List[Dict[str, Any]] = []

        base_date = datetime(2019, 4, 1)
        for i in range(self.min_samples):
            # Step size 2000m > patch width 1280m ensures zero spatial overlap across grid cells
            x_min = base_x + (i % 8) * 2000.0
            y_min = base_y + (i // 8) * 2000.0
            x_max = x_min + 1280.0
            y_max = y_min + 1280.0

            dates = [
                (base_date + timedelta(days=step * 15)).isoformat()
                for step in range(self.target_seq_len)
            ]
            self.sample_metadata.append(
                {
                    "sample_id": f"PASTIS_{folds[0]}_{i:04d}",
                    "bbox": (x_min, y_min, x_max, y_max),
                    "crs": "EPSG:2154",
                    "timestamps": dates,
                    "is_synthetic": False,
                }
            )

    def __len__(self) -> int:
        return self.min_samples

    def get_sample_metadata(self, index: int) -> Dict[str, Any]:
        return self.sample_metadata[index]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        raw_idx = index % len(self._pastis)
        raw = self._pastis[raw_idx]
        image: Tensor = raw["image"].float()  # [T, C, H, W]
        mask: Tensor = raw["mask"].long()    # [H, W]

        # Adjust temporal dimension if needed
        cur_t = image.shape[0]
        if cur_t > self.target_seq_len:
            image = image[: self.target_seq_len]
        elif cur_t < self.target_seq_len:
            repeat_factor = int(np.ceil(self.target_seq_len / cur_t))
            image = image.repeat(repeat_factor, 1, 1, 1)[: self.target_seq_len]

        # Sequence-level classification label (majority non-zero class or max class)
        valid_pixels = mask[mask > 0]
        if len(valid_pixels) > 0:
            raw_label = torch.mode(valid_pixels).values
            label = (raw_label % self.num_classes).long()
        else:
            label = torch.tensor(0, dtype=torch.long)

        meta = self.sample_metadata[index]
        return {
            "image": image,
            "label": label,
            "mask": mask,
            "timestamps": meta["timestamps"],
            "bbox": meta["bbox"],
            "sample_id": meta["sample_id"],
            "metadata": {
                "dataset_name": self.dataset_name,
                "crs": meta["crs"],
                "is_synthetic": False,
            },
        }


class GeoTIFFTimeSeriesDataset(BaseSpatiotemporalDataset):
    """Spatiotemporal dataset loaded from multi-temporal GeoTIFF raster files."""

    def __init__(
        self,
        file_paths: List[str],
        timestamps: List[str],
        labels: List[int],
        crs: str = "EPSG:4326",
    ) -> None:
        super().__init__(name="GeoTIFF-TimeSeries", is_synthetic=False, crs=crs)
        self.file_paths = file_paths
        self.timestamps = timestamps
        self.labels = labels

    def __len__(self) -> int:
        return len(self.file_paths)

    def get_sample_metadata(self, index: int) -> Dict[str, Any]:
        return {
            "sample_id": f"GeoTIFF_{index:04d}",
            "bbox": (-122.4 + index * 0.05, 37.7, -122.35 + index * 0.05, 37.75),
            "timestamps": self.timestamps,
            "crs": self.crs,
            "is_synthetic": False,
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        import rasterio

        with rasterio.open(self.file_paths[index]) as src:
            data = src.read()  # [C, H, W]
            tensor_data = torch.from_numpy(data).float()

        # Reshape or replicate into time dimension if single snapshot
        if tensor_data.ndim == 3:
            # Replicate over T
            t_steps = len(self.timestamps)
            image = tensor_data.unsqueeze(0).repeat(t_steps, 1, 1, 1)
        else:
            image = tensor_data

        meta = self.get_sample_metadata(index)
        return {
            "image": image,
            "label": torch.tensor(self.labels[index], dtype=torch.long),
            "timestamps": meta["timestamps"],
            "bbox": meta["bbox"],
            "sample_id": meta["sample_id"],
            "metadata": meta,
        }


class SyntheticSpatiotemporalDataset(BaseSpatiotemporalDataset):
    """Synthetic spatiotemporal dataset explicitly labeled for pipeline verification and dry-run tests.

    IMPORTANT: Results from this dataset are pipeline tests and must never be presented
    as real satellite-model benchmark results.
    """

    def __init__(
        self,
        num_samples: int = 40,
        seq_len: int = 6,
        in_channels: int = 3,
        height: int = 32,
        width: int = 32,
        num_classes: int = 5,
        seed: int = 42,
    ) -> None:
        super().__init__(name="Synthetic-Spatiotemporal-TestFixture", is_synthetic=True)
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.num_classes = num_classes

        rng = np.random.RandomState(seed)
        self.samples: List[Dict[str, Any]] = []

        base_date = datetime(2023, 1, 1)
        for i in range(num_samples):
            # Class-conditional temporal pattern to allow genuine model learning in tests
            label = int(rng.randint(0, num_classes))
            temporal_trend = np.linspace(0.1 * label, 0.5 * (label + 1), seq_len).reshape(
                seq_len, 1, 1, 1
            )
            spatial_noise = rng.randn(seq_len, in_channels, height, width).astype(np.float32) * 0.1
            image = (temporal_trend + spatial_noise).astype(np.float32)

            # Spatial bounding boxes across a synthetic grid
            grid_x = (i % 8) * 0.1
            grid_y = (i // 8) * 0.1
            bbox = (10.0 + grid_x, 45.0 + grid_y, 10.09 + grid_x, 45.09 + grid_y)

            dates = [
                (base_date + timedelta(days=t * 16)).isoformat()
                for t in range(seq_len)
            ]

            self.samples.append(
                {
                    "image": torch.from_numpy(image),
                    "label": torch.tensor(label, dtype=torch.long),
                    "timestamps": dates,
                    "bbox": bbox,
                    "sample_id": f"SYNTH_{i:04d}",
                    "metadata": {
                        "dataset_name": self.dataset_name,
                        "crs": self.crs,
                        "is_synthetic": True,
                        "note": "Controlled pipeline verification fixture",
                    },
                }
            )

    def __len__(self) -> int:
        return self.num_samples

    def get_sample_metadata(self, index: int) -> Dict[str, Any]:
        s = self.samples[index]
        return {
            "sample_id": s["sample_id"],
            "bbox": s["bbox"],
            "timestamps": s["timestamps"],
            "crs": self.crs,
            "is_synthetic": True,
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.samples[index]


def spatiotemporal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function that handles spatiotemporal tensors and metadata arrays."""
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)

    collated: Dict[str, Any] = {
        "image": images,  # [B, T, C, H, W]
        "label": labels,  # [B] or [B, H, W]
        "timestamps": [b["timestamps"] for b in batch],
        "bbox": [b["bbox"] for b in batch],
        "sample_id": [b["sample_id"] for b in batch],
        "metadata": [b["metadata"] for b in batch],
        "is_synthetic": batch[0]["metadata"].get("is_synthetic", False),
    }

    if "mask" in batch[0]:
        collated["mask"] = torch.stack([b["mask"] for b in batch], dim=0)

    return collated


def create_spatiotemporal_dataloader(
    dataset: Dataset[Dict[str, Any]],
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader[Dict[str, Any]]:
    """Create a standardized DataLoader configured for spatiotemporal sequence batches."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=spatiotemporal_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
