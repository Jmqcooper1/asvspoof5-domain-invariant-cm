"""Tests for loss functions."""

import pytest
import torch
import torch.nn.functional as F

from asvspoof5_domain_invariant_cm.training.losses import (
    CombinedDANNLoss,
    CombinedERMLoss,
    TaskLoss,
    compute_class_weights,
    build_loss,
)


class TestTaskLoss:
    """Test task (ERM) loss computation."""

    def test_basic_loss(self):
        loss_fn = TaskLoss()

        task_logits = torch.randn(8, 2)
        y_task = torch.randint(0, 2, (8,))

        loss = loss_fn(task_logits, y_task)

        assert loss.ndim == 0  # Scalar
        assert loss >= 0

    def test_with_class_weights(self):
        class_weights = torch.tensor([0.3, 0.7])
        loss_fn = TaskLoss(class_weights=class_weights)

        task_logits = torch.randn(8, 2)
        y_task = torch.randint(0, 2, (8,))

        loss = loss_fn(task_logits, y_task)

        assert loss.ndim == 0
        assert loss >= 0

    def test_perfect_prediction_low_loss(self):
        loss_fn = TaskLoss()

        # Perfect predictions: all zeros predicted as class 0
        task_logits = torch.tensor([
            [10.0, -10.0],
            [10.0, -10.0],
            [-10.0, 10.0],
            [-10.0, 10.0],
        ])
        y_task = torch.tensor([0, 0, 1, 1])

        loss = loss_fn(task_logits, y_task)

        assert loss < 0.01


class TestCombinedERMLoss:
    """Test combined ERM loss."""

    def test_basic_loss(self):
        loss_fn = CombinedERMLoss()

        task_logits = torch.randn(8, 2)
        y_task = torch.randint(0, 2, (8,))

        result = loss_fn(task_logits, y_task)

        assert "total_loss" in result
        assert "task_loss" in result
        assert result["total_loss"].ndim == 0


class TestCombinedDANNLoss:
    """Test DANN loss computation."""

    @pytest.fixture
    def batch(self):
        return {
            "task_logits": torch.randn(8, 2),
            "codec_logits": torch.randn(8, 10),
            "codec_q_logits": torch.randn(8, 5),
            "y_task": torch.randint(0, 2, (8,)),
            "y_codec": torch.randint(0, 10, (8,)),
            "y_codec_q": torch.randint(0, 5, (8,)),
        }

    def test_basic_dann_loss(self, batch):
        loss_fn = CombinedDANNLoss(lambda_domain=1.0)

        result = loss_fn(
            batch["task_logits"],
            batch["codec_logits"],
            batch["codec_q_logits"],
            batch["y_task"],
            batch["y_codec"],
            batch["y_codec_q"],
        )

        assert "task_loss" in result
        assert "codec_loss" in result
        assert "codec_q_loss" in result
        assert "total_loss" in result
        assert result["total_loss"].ndim == 0

    def test_lambda_zero_equals_erm(self, batch):
        """With lambda=0, DANN loss should equal ERM loss."""
        dann_loss_fn = CombinedDANNLoss(lambda_domain=0.0)
        erm_loss_fn = CombinedERMLoss()

        dann_result = dann_loss_fn(
            batch["task_logits"],
            batch["codec_logits"],
            batch["codec_q_logits"],
            batch["y_task"],
            batch["y_codec"],
            batch["y_codec_q"],
        )

        erm_result = erm_loss_fn(batch["task_logits"], batch["y_task"])

        assert torch.allclose(dann_result["total_loss"], erm_result["total_loss"])

    def test_set_lambda(self, batch):
        """Test lambda can be updated."""
        loss_fn = CombinedDANNLoss(lambda_domain=0.5)
        loss_fn.set_lambda(1.0)

        assert loss_fn.lambda_domain == 1.0


class TestBuildLoss:
    """Test loss factory function."""

    def test_build_erm_loss(self):
        loss_fn = build_loss(method="erm")

        assert isinstance(loss_fn, CombinedERMLoss)

    def test_build_dann_loss(self):
        loss_fn = build_loss(method="dann", lambda_domain=0.5)

        assert isinstance(loss_fn, CombinedDANNLoss)
        assert loss_fn.lambda_domain == 0.5


class TestClassWeights:
    """Test class weight computation."""

    def test_balanced_weights(self):
        # 80% class 0, 20% class 1
        labels = torch.tensor([0, 0, 0, 0, 1]).numpy()
        weights = compute_class_weights(labels, num_classes=2)

        # Class 1 should have higher weight
        assert weights[1] > weights[0]

    def test_equal_distribution(self):
        labels = torch.tensor([0, 0, 1, 1]).numpy()
        weights = compute_class_weights(labels, num_classes=2)

        assert torch.allclose(weights[0], weights[1])

    def test_weights_sum(self):
        labels = torch.randint(0, 3, (100,)).numpy()
        weights = compute_class_weights(labels, num_classes=3)

        # Weights should sum to num_classes (balanced)
        assert torch.allclose(weights.sum(), torch.tensor(3.0), atol=0.1)
