"""Reproducibility module for TorchGeo-TS-Bench.

Enforces deterministic seed management, environment telemetry capture,
and configuration fingerprinting.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
import psutil
import torch


@dataclass
class SystemEnvironment:
    """Snapshot of hardware, operating system, and deep learning framework versions."""

    os_platform: str
    os_release: str
    python_version: str
    torch_version: str
    torchgeo_version: str
    cpu_model: str
    physical_cores: int
    logical_cores: int
    total_ram_gb: float
    cuda_available: bool
    cuda_device_name: Optional[str] = None
    cuda_version: Optional[str] = None
    gpu_memory_gb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ReproducibilityManager:
    """Manages experiment determinism, environment auditing, and config hashing."""

    @staticmethod
    def set_seed(seed: int = 42) -> None:
        """Lock random seeds across all libraries for deterministic execution."""
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @staticmethod
    def capture_environment() -> SystemEnvironment:
        """Capture comprehensive system telemetry."""
        import torchgeo

        total_ram = round(psutil.virtual_memory().total / (1024.0**3), 2)
        cpu_name = platform.processor() or "Unknown CPU"

        cuda_avail = torch.cuda.is_available()
        cuda_dev_name = None
        cuda_ver = None
        gpu_ram = None

        if cuda_avail:
            cuda_dev_name = torch.cuda.get_device_name(0)
            cuda_ver = torch.version.cuda
            props = torch.cuda.get_device_properties(0)
            gpu_ram = round(props.total_memory / (1024.0**3), 2)

        return SystemEnvironment(
            os_platform=platform.system(),
            os_release=platform.release(),
            python_version=platform.python_version(),
            torch_version=torch.__version__,
            torchgeo_version=getattr(torchgeo, "__version__", "0.10.1"),
            cpu_model=cpu_name,
            physical_cores=psutil.cpu_count(logical=False) or 4,
            logical_cores=psutil.cpu_count(logical=True) or 8,
            total_ram_gb=total_ram,
            cuda_available=cuda_avail,
            cuda_device_name=cuda_dev_name,
            cuda_version=cuda_ver,
            gpu_memory_gb=gpu_ram,
        )

    @staticmethod
    def compute_config_hash(config_dict: Dict[str, Any]) -> str:
        """Generate a deterministic SHA-256 hash of the configuration dictionary."""
        serialized = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
