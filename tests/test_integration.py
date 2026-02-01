"""Integration tests - validate full pipeline before cloud training."""

import tempfile
from pathlib import Path

import pytest
import torch
import yaml

from asvspoof5_domain_invariant_cm.models.backbones import create_backbone
from asvspoof5_domain_invariant_cm.models.heads import (
    ClassifierHead,
    StatsPooling,
)
from asvspoof5_domain_invariant_cm.models.dann import (
    DANNModel,
    MultiHeadDomainDiscriminator,
)
from asvspoof5_domain_invariant_cm.models.erm import ERMModel
from asvspoof5_domain_invariant_cm.training.losses import CombinedDANNLoss, CombinedERMLoss
from asvspoof5_domain_invariant_cm.training.sched import (
    build_lr_scheduler,
    build_optimizer,
    LambdaScheduler,
)
from asvspoof5_domain_invariant_cm.utils.config import load_config, set_seed


class TestConfigParsing:
    """Test that all config files parse without error."""

    @pytest.fixture
    def config_dir(self) -> Path:
        return Path(__file__).parent.parent / "configs"

    def test_wavlm_erm_config(self, config_dir: Path):
        cfg = load_config(config_dir / "wavlm_erm.yaml")
        # Top-level keys, not nested under "model"
        assert "backbone" in cfg
        assert "training" in cfg
        assert cfg["backbone"]["name"] == "wavlm_base_plus"
        assert cfg["training"]["method"] == "erm"

    def test_wavlm_dann_config(self, config_dir: Path):
        cfg = load_config(config_dir / "wavlm_dann.yaml")
        assert cfg["training"]["method"] == "dann"
        # DANN-specific settings are under "dann" key
        assert "dann" in cfg
        assert "lambda_" in cfg["dann"]

    def test_w2v2_erm_config(self, config_dir: Path):
        cfg = load_config(config_dir / "w2v2_erm.yaml")
        assert cfg["backbone"]["name"] == "wav2vec2_base"
        assert cfg["training"]["method"] == "erm"

    def test_w2v2_dann_config(self, config_dir: Path):
        cfg = load_config(config_dir / "w2v2_dann.yaml")
        assert cfg["training"]["method"] == "dann"
        assert "dann" in cfg


class TestModelInstantiation:
    """Test model creation without loading pretrained weights."""

    @pytest.fixture
    def dummy_config(self) -> dict:
        return {
            "hidden_size": 768,
            "num_codecs": 10,
            "num_codec_qs": 5,
        }

    def test_erm_model_forward(self, dummy_config: dict):
        """Test ERM model forward pass with random weights."""
        set_seed(42)

        pooling = StatsPooling()
        classifier = ClassifierHead(
            input_dim=768 * 2,  # stats pooling doubles dim
            hidden_dim=256,
            num_classes=2,
        )

        # Create minimal ERM-like forward
        batch_size, seq_len, hidden = 2, 100, 768
        hidden_states = torch.randn(batch_size, seq_len, hidden)

        pooled = pooling(hidden_states)
        assert pooled.shape == (batch_size, hidden * 2)

        logits = classifier(pooled)
        assert logits.shape == (batch_size, 2)

    def test_dann_model_components(self, dummy_config: dict):
        """Test DANN components work together."""
        set_seed(42)

        batch_size, seq_len, hidden = 2, 100, 768
        hidden_states = torch.randn(batch_size, seq_len, hidden)

        pooling = StatsPooling()
        classifier = ClassifierHead(input_dim=hidden * 2, hidden_dim=256, num_classes=2)
        domain_disc = MultiHeadDomainDiscriminator(
            input_dim=hidden * 2,
            hidden_dim=512,
            num_codecs=dummy_config["num_codecs"],
            num_codec_qs=dummy_config["num_codec_qs"],
        )

        pooled = pooling(hidden_states)
        task_logits = classifier(pooled)
        # MultiHeadDomainDiscriminator.forward() takes only x, no lambda_val
        codec_logits, codec_q_logits = domain_disc(pooled)

        assert task_logits.shape == (batch_size, 2)
        assert codec_logits.shape == (batch_size, dummy_config["num_codecs"])
        assert codec_q_logits.shape == (batch_size, dummy_config["num_codec_qs"])


class TestTrainingStep:
    """Test single training step works."""

    def test_erm_training_step(self):
        """Simulate one ERM training step."""
        set_seed(42)
        device = torch.device("cpu")

        # Mini model
        model = torch.nn.Sequential(
            torch.nn.Linear(768, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 2),
        ).to(device)

        optimizer = build_optimizer(model, name="adamw", lr=1e-4)
        loss_fn = CombinedERMLoss()

        # Dummy batch
        features = torch.randn(4, 768, device=device)
        labels = torch.randint(0, 2, (4,), device=device)

        # Forward
        logits = model(features)
        loss_dict = loss_fn(logits, labels)

        # CombinedERMLoss returns a dict, extract total_loss
        loss = loss_dict["total_loss"]

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_dann_training_step(self):
        """Simulate one DANN training step."""
        set_seed(42)
        device = torch.device("cpu")

        # Mini model with domain heads
        feature_extractor = torch.nn.Linear(768, 256)
        task_head = torch.nn.Linear(256, 2)
        codec_head = torch.nn.Linear(256, 10)
        codec_q_head = torch.nn.Linear(256, 5)

        params = (
            list(feature_extractor.parameters())
            + list(task_head.parameters())
            + list(codec_head.parameters())
            + list(codec_q_head.parameters())
        )

        optimizer = torch.optim.AdamW(params, lr=1e-4)
        # CombinedDANNLoss doesn't take num_codecs/num_codec_qs
        loss_fn = CombinedDANNLoss(lambda_domain=0.5)

        # Dummy batch
        features = torch.randn(4, 768, device=device)
        task_labels = torch.randint(0, 2, (4,), device=device)
        codec_labels = torch.randint(0, 10, (4,), device=device)
        codec_q_labels = torch.randint(0, 5, (4,), device=device)

        # Forward
        hidden = feature_extractor(features)
        task_logits = task_head(hidden)
        codec_logits = codec_head(hidden)
        codec_q_logits = codec_q_head(hidden)

        loss_dict = loss_fn(
            task_logits=task_logits,
            codec_logits=codec_logits,
            codec_q_logits=codec_q_logits,
            task_labels=task_labels,
            codec_labels=codec_labels,
            codec_q_labels=codec_q_labels,
        )

        loss = loss_dict["total_loss"]

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        assert not torch.isnan(loss)


class TestCheckpointing:
    """Test checkpoint save/load roundtrip."""

    def test_checkpoint_roundtrip(self):
        """Save and load checkpoint, verify weights match."""
        set_seed(42)

        model = torch.nn.Linear(768, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Save original weights
        original_weight = model.weight.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test.pt"

            # Save
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": 5,
                    "best_metric": 0.95,
                },
                ckpt_path,
            )

            # Corrupt weights
            model.weight.data.fill_(0)
            assert not torch.allclose(model.weight, original_weight)

            # Load
            ckpt = torch.load(ckpt_path, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])

            # Verify
            assert torch.allclose(model.weight, original_weight)
            assert ckpt["epoch"] == 5
            assert ckpt["best_metric"] == 0.95


class TestSchedulers:
    """Test LR and lambda schedulers."""

    def test_lr_scheduler_warmup(self):
        """Verify warmup behavior."""
        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        scheduler = build_lr_scheduler(
            optimizer,
            name="cosine",
            num_warmup_steps=100,
            num_training_steps=1000,
        )

        # LR should increase during warmup
        initial_lr = optimizer.param_groups[0]["lr"]
        for _ in range(50):
            # Simulate a training step
            optimizer.step()
            scheduler.step()

        mid_warmup_lr = optimizer.param_groups[0]["lr"]
        assert mid_warmup_lr > initial_lr  # LR increased during warmup

    def test_lambda_scheduler(self):
        """Test DANN lambda scheduling."""
        sched = LambdaScheduler(
            schedule_type="linear",
            start_value=0.0,
            end_value=1.0,
            warmup_epochs=5,
            total_epochs=50,
        )

        # During warmup
        assert sched.get_lambda(0) == 0.0
        assert sched.get_lambda(4) == 0.0

        # After warmup, should ramp up
        assert sched.get_lambda(5) == 0.0  # start of ramp
        assert sched.get_lambda(50) == 1.0  # end


class TestBackboneLoading:
    """Test backbone can be loaded (downloads weights on first run)."""

    @pytest.mark.slow
    def test_wavlm_loads(self):
        """Test WavLM backbone loads correctly."""
        # This downloads ~360MB on first run
        backbone = create_backbone(
            name="wavlm",
            pretrained="microsoft/wavlm-base-plus",
            freeze=True,
        )
        assert backbone is not None

        # Test forward with dummy input
        dummy_wav = torch.randn(1, 16000)  # 1 second
        with torch.no_grad():
            mixed, all_hidden = backbone(dummy_wav)

        # Should return (mixed_output, all_hidden_states)
        assert mixed.ndim == 3  # (B, T', D)
        assert isinstance(all_hidden, list)

    @pytest.mark.slow
    def test_wav2vec2_loads(self):
        """Test Wav2Vec2 backbone loads correctly."""
        backbone = create_backbone(
            name="wav2vec2",
            pretrained="facebook/wav2vec2-base",
            freeze=True,
        )
        assert backbone is not None

        dummy_wav = torch.randn(1, 16000)
        with torch.no_grad():
            mixed, all_hidden = backbone(dummy_wav)

        assert mixed.ndim == 3
        assert isinstance(all_hidden, list)


class TestDummyDataPipeline:
    """Test data pipeline with synthetic data."""

    def test_collate_batch(self):
        """Test audio collation."""
        from asvspoof5_domain_invariant_cm.data.audio import collate_audio_batch

        samples = [
            {
                "waveform": torch.randn(1, 48000),
                "y_task": 0,
                "y_codec": 1,
                "y_codec_q": 2,
                "flac_file": "test1.flac",
            },
            {
                "waveform": torch.randn(1, 32000),
                "y_task": 1,
                "y_codec": 0,
                "y_codec_q": 1,
                "flac_file": "test2.flac",
            },
        ]

        batch = collate_audio_batch(samples, fixed_length=16000)

        assert batch["waveform"].shape == (2, 16000)
        assert batch["y_task"].shape == (2,)
        assert batch["attention_mask"].shape == (2, 16000)

    def test_full_forward_pass(self):
        """Test full model forward with collated batch."""
        from asvspoof5_domain_invariant_cm.data.audio import collate_audio_batch

        set_seed(42)

        # Create dummy batch
        samples = [
            {"waveform": torch.randn(1, 16000), "y_task": 0, "y_codec": 0, "y_codec_q": 0}
            for _ in range(2)
        ]
        batch = collate_audio_batch(samples, fixed_length=16000)

        # Create mini model (no pretrained backbone)
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv1d(1, 768, 400, stride=320)
                self.pool = torch.nn.AdaptiveAvgPool1d(1)
                self.classifier = torch.nn.Linear(768, 2)

            def forward(self, waveform):
                x = waveform.unsqueeze(1)  # [B, 1, T]
                x = self.conv(x)  # [B, 768, T']
                x = self.pool(x).squeeze(-1)  # [B, 768]
                return self.classifier(x)

        model = DummyModel()
        logits = model(batch["waveform"])

        assert logits.shape == (2, 2)
        assert not torch.isnan(logits).any()


class TestGradientReversal:
    """Test GRL behavior."""

    def test_grl_forward_is_identity(self):
        """Forward pass should be identity."""
        from asvspoof5_domain_invariant_cm.models.dann import GradientReversalLayer

        grl = GradientReversalLayer(lambda_=1.0)
        x = torch.randn(4, 256)
        y = grl(x)

        assert torch.allclose(x, y)

    def test_grl_backward_negates(self):
        """Backward pass should negate gradients."""
        from asvspoof5_domain_invariant_cm.models.dann import GradientReversalLayer

        grl = GradientReversalLayer(lambda_=1.0)

        x = torch.randn(4, 256, requires_grad=True)
        y = grl(x)
        loss = y.sum()
        loss.backward()

        # Gradient should be -1 for each element (since dy/dx = 1, but GRL negates)
        expected_grad = -torch.ones_like(x)
        assert torch.allclose(x.grad, expected_grad)
