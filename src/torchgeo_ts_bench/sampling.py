"""Spatial and temporal sampling module for TorchGeo-TS-Bench.

Wraps and standardizes spatial and temporal sampling mechanisms compatible
with TorchGeo's sampler paradigm for spatiotemporal foundation models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor


@dataclass
class SpatialSamplingConfig:
    """Configuration for spatial patch extraction."""

    patch_size: Tuple[int, int] = (32, 32)
    stride: Optional[Tuple[int, int]] = None
    padding_mode: str = "reflect"
    random_crop: bool = False


@dataclass
class TemporalSamplingConfig:
    """Configuration for temporal window sampling."""

    target_seq_len: int = 6
    sampling_rate: int = 1  # 1 = every frame, 2 = every second frame
    temporal_strategy: str = "uniform"  # "uniform", "random", "first_k", "last_k"


class SpatioTemporalSamplingManager:
    """Manages spatial patch extraction and temporal sub-sampling for sequence tensors."""

    def __init__(
        self,
        spatial_cfg: Optional[SpatialSamplingConfig] = None,
        temporal_cfg: Optional[TemporalSamplingConfig] = None,
    ) -> None:
        self.spatial_cfg = spatial_cfg or SpatialSamplingConfig()
        self.temporal_cfg = temporal_cfg or TemporalSamplingConfig()

    def sample_tensor(
        self, tensor: Tensor, seed: Optional[int] = None
    ) -> Tensor:
        """Sample a spatiotemporal tensor of shape [T, C, H, W] -> [T_target, C, H_target, W_target].

        Args:
            tensor: Input tensor [T, C, H, W].
            seed: Optional random seed for reproducible sampling.

        Returns:
            Sampled spatiotemporal tensor.
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        t_out = self._sample_temporal(tensor)
        st_out = self._sample_spatial(t_out)
        return st_out

    def _sample_temporal(self, tensor: Tensor) -> Tensor:
        """Sub-sample or pad along the temporal dimension (axis 0)."""
        t, c, h, w = tensor.shape
        target_t = self.temporal_cfg.target_seq_len
        strategy = self.temporal_cfg.temporal_strategy

        if t == target_t:
            return tensor

        if t > target_t:
            if strategy == "uniform":
                indices = np.linspace(0, t - 1, target_t, dtype=int)
            elif strategy == "random":
                indices = np.sort(np.random.choice(t, target_t, replace=False))
            elif strategy == "first_k":
                indices = np.arange(target_t)
            elif strategy == "last_k":
                indices = np.arange(t - target_t, t)
            else:
                indices = np.linspace(0, t - 1, target_t, dtype=int)
            return tensor[indices]

        # If t < target_t, repeat/pad temporal frames
        repeat_factor = int(np.ceil(target_t / t))
        repeated = tensor.repeat(repeat_factor, 1, 1, 1)
        return repeated[:target_t]

    def _sample_spatial(self, tensor: Tensor) -> Tensor:
        """Crop or pad along the spatial dimensions (H, W)."""
        t, c, h, w = tensor.shape
        target_h, target_w = self.spatial_cfg.patch_size

        if h == target_h and w == target_w:
            return tensor

        # Handle crop when input is larger than target
        if h >= target_h and w >= target_w:
            if self.spatial_cfg.random_crop:
                start_h = np.random.randint(0, h - target_h + 1)
                start_w = np.random.randint(0, w - target_w + 1)
            else:
                # Center crop
                start_h = (h - target_h) // 2
                start_w = (w - target_w) // 2
            return tensor[:, :, start_h : start_h + target_h, start_w : start_w + target_w]

        # Handle pad when input is smaller
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        return torch.nn.functional.pad(
            tensor,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0.0,
        )
