"""Tests for model components."""

import pytest
import torch

from asvspoof5_domain_invariant_cm.models import (
    AttentionPooling,
    ClassifierHead,
    GradientReversalLayer,
    MeanPooling,
    MultiHeadDomainDiscriminator,
    ProjectionHead,
    StatsPooling,
    create_pooling,
)
from asvspoof5_domain_invariant_cm.models.backbones import LayerWeightedPooling


class TestGradientReversalLayer:
    """Test Gradient Reversal Layer."""

    def test_forward_pass_unchanged(self):
        """GRL should not change values in forward pass."""
        grl = GradientReversalLayer(lambda_=1.0)
        x = torch.randn(2, 10, requires_grad=True)
        y = grl(x)

        assert torch.allclose(x, y)

    def test_backward_negates_gradient(self):
        """GRL should negate gradients in backward pass."""
        grl = GradientReversalLayer(lambda_=1.0)
        x = torch.randn(2, 10, requires_grad=True)
        y = grl(x)
        loss = y.sum()
        loss.backward()

        # Gradient should be negated: d(sum)/dx = 1, so after GRL it's -1
        expected = -torch.ones_like(x)
        assert torch.allclose(x.grad, expected)

    def test_lambda_scaling(self):
        """GRL should scale gradients by lambda."""
        grl = GradientReversalLayer(lambda_=0.5)
        x = torch.randn(2, 10, requires_grad=True)
        y = grl(x)
        loss = y.sum()
        loss.backward()

        # Gradient should be -0.5
        expected = -0.5 * torch.ones_like(x)
        assert torch.allclose(x.grad, expected)

    def test_set_lambda(self):
        """Test lambda scheduling."""
        grl = GradientReversalLayer(lambda_=0.0)
        grl.set_lambda(0.75)

        x = torch.randn(2, 10, requires_grad=True)
        y = grl(x)
        loss = y.sum()
        loss.backward()

        expected = -0.75 * torch.ones_like(x)
        assert torch.allclose(x.grad, expected)


class TestPoolingLayers:
    """Test pooling layer implementations."""

    @pytest.fixture
    def hidden_states(self):
        """Sample hidden states: [B, T, D]."""
        return torch.randn(2, 100, 768)

    @pytest.fixture
    def lengths(self):
        """Sample lengths: [B]."""
        return torch.tensor([100, 50])  # Full and half length

    def test_mean_pooling_shape(self, hidden_states, lengths):
        pool = MeanPooling()
        out = pool(hidden_states, lengths)

        assert out.shape == (2, 768)

    def test_mean_pooling_with_lengths(self, hidden_states, lengths):
        pool = MeanPooling()
        out = pool(hidden_states, lengths)

        # Manually compute mean for first sample (full)
        expected_0 = hidden_states[0].mean(dim=0)
        assert torch.allclose(out[0], expected_0, atol=1e-5)

        # Manually compute mean for second sample (first 50 frames)
        expected_1 = hidden_states[1, :50].mean(dim=0)
        assert torch.allclose(out[1], expected_1, atol=1e-5)

    def test_mean_pooling_no_lengths(self, hidden_states):
        pool = MeanPooling()
        out = pool(hidden_states, lengths=None)

        assert out.shape == (2, 768)
        expected = hidden_states.mean(dim=1)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_stats_pooling_shape(self, hidden_states, lengths):
        pool = StatsPooling()
        out = pool(hidden_states, lengths)

        # Stats pooling: concat(mean, std) -> 2*D
        assert out.shape == (2, 768 * 2)

    def test_stats_pooling_no_lengths(self, hidden_states):
        pool = StatsPooling()
        out = pool(hidden_states, lengths=None)

        assert out.shape == (2, 768 * 2)

        # Check first half is mean, second half is std
        mean_out = out[:, :768]
        std_out = out[:, 768:]

        expected_mean = hidden_states.mean(dim=1)
        expected_std = hidden_states.std(dim=1)
        assert torch.allclose(mean_out, expected_mean, atol=1e-5)
        assert torch.allclose(std_out, expected_std, atol=1e-4)

    def test_attention_pooling_shape(self, hidden_states, lengths):
        pool = AttentionPooling(input_dim=768)
        out = pool(hidden_states, lengths)

        assert out.shape == (2, 768)

    def test_create_pooling_factory(self):
        pool_mean = create_pooling("mean")
        pool_stats = create_pooling("stats")
        pool_attn = create_pooling("attention", input_dim=768)

        assert isinstance(pool_mean, MeanPooling)
        assert isinstance(pool_stats, StatsPooling)
        assert isinstance(pool_attn, AttentionPooling)


class TestLayerWeightedPooling:
    """Test layer mixing.

    Note: LayerWeightedPooling takes num_layers and init_lower_bias.
    Layer selection is done separately via select_layers function.
    """

    @pytest.fixture
    def hidden_states_list(self):
        """List of hidden states from 12 layers."""
        return [torch.randn(2, 100, 768) for _ in range(12)]

    def test_all_layers(self, hidden_states_list):
        pool = LayerWeightedPooling(num_layers=12)
        out = pool(hidden_states_list)

        assert out.shape == (2, 100, 768)

    def test_subset_layers(self, hidden_states_list):
        # Use only 4 layers
        subset = hidden_states_list[-4:]
        pool = LayerWeightedPooling(num_layers=4)
        out = pool(subset)

        assert out.shape == (2, 100, 768)

    def test_weights_shape(self):
        pool = LayerWeightedPooling(num_layers=6)
        assert pool.weights.shape[0] == 6

    def test_weights_sum_to_one_after_softmax(self, hidden_states_list):
        pool = LayerWeightedPooling(num_layers=12)
        _ = pool(hidden_states_list)

        # Get softmax weights
        weights = torch.softmax(pool.weights, dim=0)
        assert torch.allclose(weights.sum(), torch.tensor(1.0))

    def test_init_lower_bias(self):
        pool = LayerWeightedPooling(num_layers=6, init_lower_bias=True)
        # Lower layers should have higher initial weights
        assert pool.weights[0] > pool.weights[-1]

    def test_init_uniform(self):
        pool = LayerWeightedPooling(num_layers=6, init_lower_bias=False)
        # All weights should be equal
        assert torch.allclose(pool.weights, torch.ones(6))


class TestHeads:
    """Test projection and classifier heads."""

    def test_projection_head_shape(self):
        head = ProjectionHead(input_dim=1536, hidden_dim=512, output_dim=256)
        x = torch.randn(4, 1536)
        out = head(x)

        assert out.shape == (4, 256)

    def test_classifier_head_shape(self):
        head = ClassifierHead(input_dim=256, num_classes=2)
        x = torch.randn(4, 256)
        out = head(x)

        assert out.shape == (4, 2)

    def test_classifier_head_multiclass(self):
        head = ClassifierHead(input_dim=256, num_classes=10)
        x = torch.randn(4, 256)
        out = head(x)

        assert out.shape == (4, 10)


class TestMultiHeadDomainDiscriminator:
    """Test domain discriminator."""

    def test_output_shapes(self):
        disc = MultiHeadDomainDiscriminator(
            input_dim=1536,
            num_codecs=10,
            num_codec_qs=5,
            hidden_dim=512,
        )
        x = torch.randn(4, 1536)
        codec_logits, codec_q_logits = disc(x)

        assert codec_logits.shape == (4, 10)
        assert codec_q_logits.shape == (4, 5)

    def test_shared_layers_exist(self):
        disc = MultiHeadDomainDiscriminator(
            input_dim=1536,
            num_codecs=10,
            num_codec_qs=5,
        )

        # Should have shared MLP + two heads
        assert hasattr(disc, "shared")
        assert hasattr(disc, "codec_head")
        assert hasattr(disc, "codec_q_head")
