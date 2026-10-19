"""Downstream tasks module for spatiotemporal Earth Observation benchmarking.

Defines loss functions, training steps, and task-specific predictions for:
- Classification
- Semantic Segmentation
- Regression
- Change Detection
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor, nn


class BaseDownstreamTask(ABC):
    """Abstract base class for downstream geospatial benchmarking tasks."""

    def __init__(self, task_name: str) -> None:
        self.task_name = task_name

    @abstractmethod
    def compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Calculate loss for backpropagation."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, logits: Tensor) -> Tensor:
        """Convert raw model logits to task predictions."""
        raise NotImplementedError


class SpatiotemporalClassificationTask(BaseDownstreamTask):
    """Spatiotemporal sequence classification task (e.g. crop type, land use)."""

    def __init__(
        self,
        num_classes: int = 5,
        class_weights: Optional[Tensor] = None,
    ) -> None:
        super().__init__(task_name="classification")
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

    def compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.criterion(logits, targets)

    def predict(self, logits: Tensor) -> Tensor:
        return torch.argmax(logits, dim=-1)

    def predict_probabilities(self, logits: Tensor) -> Tensor:
        return torch.softmax(logits, dim=-1)


class SpatiotemporalSegmentationTask(BaseDownstreamTask):
    """Pixelwise spatiotemporal parcel or semantic segmentation."""

    def __init__(
        self,
        num_classes: int = 5,
        ignore_index: int = -100,
    ) -> None:
        super().__init__(task_name="segmentation")
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        # logits: [B, C, H, W], targets: [B, H, W]
        return self.criterion(logits, targets)

    def predict(self, logits: Tensor) -> Tensor:
        return torch.argmax(logits, dim=1)


class SpatiotemporalRegressionTask(BaseDownstreamTask):
    """Continuous biophysical / spatiotemporal parameter regression."""

    def __init__(self, loss_type: str = "mse") -> None:
        super().__init__(task_name="regression")
        self.loss_type = loss_type
        if loss_type == "l1":
            self.criterion = nn.L1Loss()
        else:
            self.criterion = nn.MSELoss()

    def compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.criterion(logits.squeeze(-1), targets.float())

    def predict(self, logits: Tensor) -> Tensor:
        return logits.squeeze(-1)


class ChangeDetectionTask(BaseDownstreamTask):
    """Bi-temporal or multi-temporal surface change detection."""

    def __init__(self, pos_weight: Optional[float] = None) -> None:
        super().__init__(task_name="change_detection")
        weight = torch.tensor([pos_weight]) if pos_weight else None
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=weight)

    def compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.criterion(logits.squeeze(-1), targets.float())

    def predict(self, logits: Tensor) -> Tensor:
        probs = torch.sigmoid(logits.squeeze(-1))
        return (probs > 0.5).long()
