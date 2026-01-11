"""Tests for configuration loading and validation."""

import tempfile
from pathlib import Path

import pytest
import yaml

from asvspoof5_domain_invariant_cm.utils.config import (
    load_config,
    merge_configs,
    set_seed,
    get_device,
    flatten_config,
    unflatten_config,
)


class TestLoadConfig:
    """Test YAML config loading."""

    def test_load_simple_config(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"key": "value", "number": 42}, f)
            f.flush()

            config = load_config(f.name)

            assert config["key"] == "value"
            assert config["number"] == 42

    def test_load_nested_config(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({
                "model": {
                    "backbone": "wavlm",
                    "hidden_dim": 768,
                },
                "training": {
                    "lr": 1e-4,
                    "epochs": 10,
                },
            }, f)
            f.flush()

            config = load_config(f.name)

            assert config["model"]["backbone"] == "wavlm"
            assert config["training"]["lr"] == 1e-4

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")


class TestMergeConfigs:
    """Test config merging."""

    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}

        merged = merge_configs(base, override)

        assert merged["a"] == 1
        assert merged["b"] == 3  # Overridden
        assert merged["c"] == 4  # New key

    def test_nested_merge(self):
        base = {
            "model": {"backbone": "wavlm", "hidden": 768},
            "training": {"lr": 1e-4},
        }
        override = {
            "model": {"hidden": 512},  # Override nested value
            "training": {"epochs": 20},  # Add nested value
        }

        merged = merge_configs(base, override)

        assert merged["model"]["backbone"] == "wavlm"  # Preserved
        assert merged["model"]["hidden"] == 512  # Overridden
        assert merged["training"]["lr"] == 1e-4  # Preserved
        assert merged["training"]["epochs"] == 20  # Added

    def test_base_not_modified(self):
        base = {"a": 1, "b": {"c": 2}}
        override = {"a": 10, "b": {"c": 20}}

        merged = merge_configs(base, override)

        # Original base should be unchanged
        assert base["a"] == 1
        assert base["b"]["c"] == 2


class TestFlattenConfig:
    """Test config flattening for logging."""

    def test_flatten_nested(self):
        config = {
            "model": {
                "backbone": "wavlm",
                "hidden": 768,
            },
            "lr": 1e-4,
        }

        flat = flatten_config(config)

        assert flat["model.backbone"] == "wavlm"
        assert flat["model.hidden"] == 768
        assert flat["lr"] == 1e-4

    def test_flatten_deeply_nested(self):
        config = {
            "a": {
                "b": {
                    "c": {
                        "d": 42
                    }
                }
            }
        }

        flat = flatten_config(config)

        assert flat["a.b.c.d"] == 42

    def test_unflatten(self):
        flat = {
            "model.backbone": "wavlm",
            "model.hidden": 768,
            "lr": 1e-4,
        }

        config = unflatten_config(flat)

        assert config["model"]["backbone"] == "wavlm"
        assert config["model"]["hidden"] == 768
        assert config["lr"] == 1e-4


class TestSetSeed:
    """Test reproducibility seed setting."""

    def test_set_seed_deterministic(self):
        import torch
        import numpy as np

        set_seed(42)
        a1 = torch.randn(10)
        b1 = np.random.rand(10)

        set_seed(42)
        a2 = torch.randn(10)
        b2 = np.random.rand(10)

        assert torch.allclose(a1, a2)
        assert np.allclose(b1, b2)

    def test_different_seeds_different_results(self):
        import torch

        set_seed(42)
        a1 = torch.randn(10)

        set_seed(123)
        a2 = torch.randn(10)

        assert not torch.allclose(a1, a2)


class TestGetDevice:
    """Test device selection."""

    def test_get_device_cpu(self):
        device = get_device("cpu")
        assert str(device) == "cpu"

    def test_get_device_auto(self):
        import torch

        # Pass None for auto detection
        device = get_device(None)

        # Should be valid device
        if torch.cuda.is_available():
            assert "cuda" in str(device)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            assert str(device) == "mps"
        else:
            assert str(device) == "cpu"

    def test_get_device_explicit_cuda(self):
        import torch

        if torch.cuda.is_available():
            device = get_device("cuda")
            assert "cuda" in str(device)
        else:
            # Skip if no CUDA
            pass
