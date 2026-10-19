"""Model adapter interfaces for spatiotemporal Earth Observation models.

Harmonizes inputs [B, T, C, H, W], output shapes, and efficiency profiling
for TorchGeo foundation models and baselines.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn


class SpatiotemporalModelAdapter(nn.Module, ABC):
    """Abstract base adapter for spatiotemporal foundation models and baselines."""

    def __init__(
        self,
        model_name: str,
        model_family: str,
        in_channels: int,
        seq_len: int,
        num_classes: int,
        task: str = "classification",
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.model_family = model_family
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.task = task

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass accepting input of shape [B, T, C, H, W].

        Returns:
            Logits of shape:
                [B, num_classes] for classification
                [B, num_classes, H, W] for segmentation
                [B, 1] for regression
        """
        raise NotImplementedError

    def count_parameters(self) -> Dict[str, int]:
        """Count total and trainable parameters in the model."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_parameters": total, "trainable_parameters": trainable}


class ConvLSTMAdapter(SpatiotemporalModelAdapter):
    """Adapter wrapping TorchGeo's ConvLSTM spatiotemporal model."""

    def __init__(
        self,
        in_channels: int = 3,
        seq_len: int = 6,
        num_classes: int = 5,
        hidden_dims: Optional[List[int]] = None,
        kernel_size: Tuple[int, int] = (3, 3),
        task: str = "classification",
    ) -> None:
        super().__init__(
            model_name="TorchGeo-ConvLSTM",
            model_family="Recurrent Spatiotemporal",
            in_channels=in_channels,
            seq_len=seq_len,
            num_classes=num_classes,
            task=task,
        )
        from torchgeo.models import ConvLSTM

        if hidden_dims is None:
            hidden_dims = [32, 64]
        self.hidden_dims = hidden_dims

        self.backbone = ConvLSTM(
            input_dim=in_channels,
            hidden_dim=hidden_dims,
            kernel_size=kernel_size,
            num_layers=len(hidden_dims),
            return_all_layers=False,
        )

        last_hidden = hidden_dims[-1]
        if task == "classification":
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(last_hidden, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes),
            )
        elif task == "segmentation":
            self.head = nn.Sequential(
                nn.Conv2d(last_hidden, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, num_classes, kernel_size=1),
            )
        else:  # regression
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(last_hidden, 1),
            )

    def forward(self, x: Tensor) -> Tensor:
        # Input shape: [B, T, C, H, W]
        # TorchGeo ConvLSTM forward_features returns (layer_output_list, last_state_list)
        _, last_state_list = self.backbone.forward_features(x)
        # last_state_list[-1] is tuple (h_t, c_t) of final layer
        final_hidden = last_state_list[-1][0]  # [B, hidden_dims[-1], H, W]
        return self.head(final_hidden)


class LTAEAdapter(SpatiotemporalModelAdapter):
    """Adapter wrapping TorchGeo's LTAE (Lightweight Temporal Attention Encoder)."""

    def __init__(
        self,
        in_channels: int = 3,
        seq_len: int = 6,
        num_classes: int = 5,
        d_model: Optional[int] = 64,
        task: str = "classification",
    ) -> None:
        super().__init__(
            model_name="TorchGeo-LTAE",
            model_family="Temporal Attention",
            in_channels=in_channels,
            seq_len=seq_len,
            num_classes=num_classes,
            task=task,
        )
        from torchgeo.models import LTAE

        # Pass len_max_seq=seq_len to ensure proper layer normalization
        self.backbone = LTAE(
            in_channels=in_channels,
            d_model=d_model,
            n_head=4,
            len_max_seq=seq_len,
            n_neurons=[d_model, 128],
        )

        out_dim = 128
        if task == "classification":
            self.head = nn.Sequential(
                nn.Linear(out_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes),
            )
        else:
            self.head = nn.Linear(out_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        # Input shape: [B, T, C, H, W]
        # Pool spatial dimensions to obtain temporal sequence of spectral representations
        b, t, c, h, w = x.shape
        x_pooled = x.mean(dim=(-2, -1))  # [B, T, C]
        features = self.backbone(x_pooled)  # [B, 128]
        return self.head(features)


class ST3DCNNAdapter(SpatiotemporalModelAdapter):
    """3D Spatiotemporal Convolutional Baseline."""

    def __init__(
        self,
        in_channels: int = 3,
        seq_len: int = 6,
        num_classes: int = 5,
        task: str = "classification",
    ) -> None:
        super().__init__(
            model_name="ST-3DCNN-Baseline",
            model_family="3D Spatiotemporal Convolution",
            in_channels=in_channels,
            seq_len=seq_len,
            num_classes=num_classes,
            task=task,
        )
        # Conv3D operates on (B, C, D=T, H, W)
        self.conv_net = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),  # (T//2, H//2, W//2)
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Input shape: [B, T, C, H, W] -> permute to [B, C, T, H, W] for Conv3D
        x_3d = x.permute(0, 2, 1, 3, 4)
        feat = self.conv_net(x_3d)
        return self.head(feat)


class TemporalTransformerAdapter(SpatiotemporalModelAdapter):
    """Temporal Transformer Self-Attention Baseline."""

    def __init__(
        self,
        in_channels: int = 3,
        seq_len: int = 6,
        num_classes: int = 5,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        task: str = "classification",
    ) -> None:
        super().__init__(
            model_name="Temporal-Transformer",
            model_family="Temporal Self-Attention Transformer",
            in_channels=in_channels,
            seq_len=seq_len,
            num_classes=num_classes,
            task=task,
        )
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Input shape: [B, T, C, H, W] -> Spatial pool to [B, T, C]
        x_pooled = x.mean(dim=(-2, -1))
        tokens = self.input_proj(x_pooled) + self.pos_embedding[:, : x.shape[1], :]
        encoded = self.transformer(tokens)  # [B, T, d_model]
        cls_token = encoded.mean(dim=1)  # Temporal average pooling
        return self.head(cls_token)
