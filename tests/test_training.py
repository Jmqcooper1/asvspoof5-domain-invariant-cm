"""Tests for training components."""

import pytest
import torch
import torch.nn as nn

from asvspoof5_domain_invariant_cm.training.sched import (
    build_lr_scheduler,
    build_optimizer,
    LambdaScheduler,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


class TestLRScheduler:
    """Test learning rate scheduler builders."""

    @pytest.fixture
    def model_and_optimizer(self):
        model = nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        return model, optimizer

    def test_cosine_scheduler(self, model_and_optimizer):
        _, optimizer = model_and_optimizer

        scheduler = build_lr_scheduler(
            optimizer,
            name="cosine",
            num_warmup_steps=100,
            num_training_steps=1000,
        )

        assert scheduler is not None

        # Step through warmup - need to call optimizer.step() first
        initial_lr = optimizer.param_groups[0]["lr"]
        for _ in range(100):
            optimizer.step()  # Dummy step
            scheduler.step()

        warmup_end_lr = optimizer.param_groups[0]["lr"]
        # After warmup, LR should be at peak
        assert warmup_end_lr >= initial_lr * 0.9  # Allow some tolerance

    def test_linear_scheduler(self, model_and_optimizer):
        _, optimizer = model_and_optimizer

        scheduler = build_lr_scheduler(
            optimizer,
            name="linear",
            num_warmup_steps=100,
            num_training_steps=1000,
        )

        assert scheduler is not None

    def test_constant_scheduler(self, model_and_optimizer):
        _, optimizer = model_and_optimizer

        scheduler = build_lr_scheduler(optimizer, name="constant")

        # Should return None for constant
        assert scheduler is None

    def test_cosine_with_warmup_function(self, model_and_optimizer):
        _, optimizer = model_and_optimizer

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=10,
            num_training_steps=100,
        )

        # Get LR values through schedule
        lrs = []
        for _ in range(100):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()  # Dummy step
            scheduler.step()

        # LR should increase during warmup then decrease
        assert lrs[9] >= lrs[0]  # End of warmup >= start
        assert lrs[-1] < lrs[10]  # Final < peak


class TestLambdaScheduler:
    """Test DANN lambda (GRL strength) scheduler."""

    def test_constant_lambda(self):
        scheduler = LambdaScheduler(
            schedule_type="constant",
            start_value=0.5,
            total_epochs=100,
        )

        # Should always return start_value
        for epoch in [0, 10, 50, 99]:
            assert scheduler.get_lambda(epoch) == 0.5

    def test_linear_lambda(self):
        scheduler = LambdaScheduler(
            schedule_type="linear",
            start_value=0.0,
            end_value=1.0,
            warmup_epochs=0,
            total_epochs=100,
        )

        # At epoch 0
        assert scheduler.get_lambda(0) == pytest.approx(0.0)

        # At epoch 50 (halfway)
        assert scheduler.get_lambda(50) == pytest.approx(0.5)

        # At epoch 100 (end)
        assert scheduler.get_lambda(100) == pytest.approx(1.0)

    def test_linear_lambda_with_warmup(self):
        scheduler = LambdaScheduler(
            schedule_type="linear",
            start_value=0.0,
            end_value=1.0,
            warmup_epochs=20,
            total_epochs=100,
        )

        # During warmup
        assert scheduler.get_lambda(0) == 0.0
        assert scheduler.get_lambda(10) == 0.0
        assert scheduler.get_lambda(19) == 0.0

        # After warmup starts ramping
        assert scheduler.get_lambda(20) == pytest.approx(0.0)  # Start of ramp
        assert scheduler.get_lambda(60) == pytest.approx(0.5, abs=0.05)  # Halfway through ramp

    def test_exponential_lambda(self):
        scheduler = LambdaScheduler(
            schedule_type="exponential",
            start_value=0.0,
            end_value=1.0,
            total_epochs=100,
        )

        # Should increase from 0 to end_value
        lambda_0 = scheduler.get_lambda(0)
        lambda_50 = scheduler.get_lambda(50)
        lambda_99 = scheduler.get_lambda(99)

        assert lambda_0 < lambda_50 < lambda_99
        assert lambda_99 == pytest.approx(1.0, abs=0.1)


class TestOptimizerBuilder:
    """Test optimizer building."""

    def test_build_adamw(self):
        model = nn.Linear(10, 2)

        optimizer = build_optimizer(
            model,
            name="adamw",
            lr=1e-4,
            weight_decay=0.01,
        )

        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.defaults["lr"] == 1e-4

    def test_build_adam(self):
        model = nn.Linear(10, 2)

        optimizer = build_optimizer(
            model,
            name="adam",
            lr=1e-3,
        )

        assert isinstance(optimizer, torch.optim.Adam)

    def test_build_sgd(self):
        model = nn.Linear(10, 2)

        optimizer = build_optimizer(
            model,
            name="sgd",
            lr=0.01,
        )

        assert isinstance(optimizer, torch.optim.SGD)

    def test_unknown_optimizer_raises(self):
        model = nn.Linear(10, 2)

        with pytest.raises(ValueError, match="Unknown optimizer"):
            build_optimizer(model, name="unknown")
