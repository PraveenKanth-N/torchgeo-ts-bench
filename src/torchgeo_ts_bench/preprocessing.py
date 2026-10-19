"""Preprocessing module for TorchGeo-TS-Bench.

Handles missing values, nodata handling, normalization (z-score, min-max),
band selection, and spatiotemporal sequence standardization.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor


# Standard Sentinel-2 L2A BOA Reflectance statistics (scaled 0-10000 -> 0-1)
S2_MEANS: Dict[str, float] = {
    "B02": 0.1084,  # Blue
    "B03": 0.1248,  # Green
    "B04": 0.1254,  # Red
    "B08": 0.2820,  # NIR
}

S2_STDS: Dict[str, float] = {
    "B02": 0.0652,
    "B03": 0.0543,
    "B04": 0.0645,
    "B08": 0.0812,
}


class SpatiotemporalPreprocessor:
    """Standardized preprocessor for multi-temporal satellite imagery."""

    def __init__(
        self,
        normalization_method: str = "min_max",  # "min_max", "z_score", "none"
        clip_range: Optional[Tuple[float, float]] = (0.0, 10000.0),
        scale_factor: float = 1.0 / 10000.0,
        band_indices: Optional[List[int]] = None,
        fill_value: float = 0.0,
        means: Optional[List[float]] = None,
        stds: Optional[List[float]] = None,
    ) -> None:
        self.normalization_method = normalization_method
        self.clip_range = clip_range
        self.scale_factor = scale_factor
        self.band_indices = band_indices
        self.fill_value = fill_value
        self.means = means
        self.stds = stds

    def transform(self, tensor: Tensor) -> Tensor:
        """Preprocess a spatiotemporal tensor of shape [..., C, H, W] or [T, C, H, W].

        Args:
            tensor: Input float tensor.

        Returns:
            Preprocessed tensor with cleaned values and normalization applied.
        """
        x = tensor.clone().float()

        # 1. Clean NaNs and Infs
        nan_mask = torch.isnan(x) | torch.isinf(x)
        if nan_mask.any():
            x = torch.where(nan_mask, torch.tensor(self.fill_value, dtype=x.dtype, device=x.device), x)

        # 2. Band selection if specified
        if self.band_indices is not None:
            # Assumes channel dimension is either dim 1 (in [T, C, H, W]) or dim -3
            if x.ndim == 4:
                x = x[:, self.band_indices, :, :]
            elif x.ndim == 5:
                x = x[:, :, self.band_indices, :, :]

        # 3. Clip range if specified
        if self.clip_range is not None:
            min_val, max_val = self.clip_range
            x = torch.clamp(x, min=min_val, max=max_val)

        # 4. Normalization
        if self.normalization_method == "min_max":
            if self.scale_factor != 1.0:
                x = x * self.scale_factor
            else:
                x_min = x.min()
                x_max = x.max()
                if (x_max - x_min) > 1e-6:
                    x = (x - x_min) / (x_max - x_min)
        elif self.normalization_method == "z_score":
            if self.means is not None and self.stds is not None:
                # Shape adjustment for broadcasting across channel dimension
                c = x.shape[1] if x.ndim == 4 else x.shape[2]
                mean_t = torch.tensor(self.means, dtype=x.dtype, device=x.device).view(
                    1, -1, 1, 1
                )
                std_t = torch.tensor(self.stds, dtype=x.dtype, device=x.device).view(
                    1, -1, 1, 1
                )
                if x.ndim == 5:
                    mean_t = mean_t.unsqueeze(0)
                    std_t = std_t.unsqueeze(0)
                x = (x - mean_t) / (std_t + 1e-6)
            else:
                mean = x.mean()
                std = x.std()
                x = (x - mean) / (std + 1e-6)

        return x

    def get_config(self) -> Dict[str, Any]:
        """Return preprocessing settings dictionary for reproducible experiment logs."""
        return {
            "normalization_method": self.normalization_method,
            "clip_range": self.clip_range,
            "scale_factor": self.scale_factor,
            "band_indices": self.band_indices,
            "fill_value": self.fill_value,
            "means": self.means,
            "stds": self.stds,
        }
