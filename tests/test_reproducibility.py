"""Unit tests for reproducibility, seed management, and environment telemetry."""

import torch
import numpy as np
from torchgeo_ts_bench.reproducibility import ReproducibilityManager


def test_seed_locking():
    """Verify that set_seed ensures deterministic random number generation."""
    ReproducibilityManager.set_seed(42)
    a1 = torch.randn(5)
    n1 = np.random.rand(5)

    ReproducibilityManager.set_seed(42)
    a2 = torch.randn(5)
    n2 = np.random.rand(5)

    assert torch.equal(a1, a2)
    assert np.array_equal(n1, n2)


def test_environment_telemetry_capture():
    """Verify system telemetry capture returns valid OS and library metadata."""
    env = ReproducibilityManager.capture_environment()
    d = env.to_dict()

    assert "os_platform" in d
    assert "python_version" in d
    assert "torch_version" in d
    assert "torchgeo_version" in d
    assert d["total_ram_gb"] > 0.0


def test_config_hash_consistency():
    """Verify configuration hash generation is deterministic."""
    cfg1 = {"model": "convlstm", "seed": 42, "lr": 0.001}
    cfg2 = {"lr": 0.001, "model": "convlstm", "seed": 42}  # Different order

    h1 = ReproducibilityManager.compute_config_hash(cfg1)
    h2 = ReproducibilityManager.compute_config_hash(cfg2)

    assert h1 == h2
