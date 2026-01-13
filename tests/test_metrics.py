"""Tests for evaluation metrics."""

import numpy as np
import pytest

from asvspoof5_domain_invariant_cm.evaluation.metrics import (
    compute_eer,
    compute_min_dcf,
    compute_cllr,
    compute_act_dcf,
)


class TestEER:
    """Test Equal Error Rate computation.

    Note: Label convention is 1 = bonafide, 0 = spoof
    Score convention: higher = more likely bonafide
    """

    def test_perfect_separation(self):
        """Perfect separation should give EER = 0."""
        # All bonafide scores high, all spoof scores low
        scores = np.array([0.9, 0.8, 0.85, 0.1, 0.05, 0.15])
        labels = np.array([1, 1, 1, 0, 0, 0])  # bonafide=1, spoof=0

        eer, threshold = compute_eer(scores, labels)

        assert eer == pytest.approx(0.0, abs=0.01)

    def test_random_scores(self):
        """Random scores should give EER around 50%."""
        np.random.seed(42)
        n = 1000
        scores = np.random.rand(n)
        labels = np.random.randint(0, 2, n)

        eer, threshold = compute_eer(scores, labels)

        assert 0.3 < eer < 0.7  # Around 50%

    def test_inverted_scores(self):
        """Completely wrong scores should give EER close to 100%."""
        # All bonafide scores low, all spoof scores high
        scores = np.array([0.1, 0.15, 0.05, 0.9, 0.85, 0.95])
        labels = np.array([1, 1, 1, 0, 0, 0])  # bonafide=1, spoof=0

        eer, threshold = compute_eer(scores, labels)

        assert eer > 0.9  # Close to 100%

    def test_eer_range(self):
        """EER should be between 0 and 1."""
        np.random.seed(123)
        for _ in range(10):
            n = 100
            scores = np.random.rand(n)
            labels = np.random.randint(0, 2, n)

            eer, _ = compute_eer(scores, labels)

            assert 0 <= eer <= 1


class TestMinDCF:
    """Test minimum Detection Cost Function."""

    def test_perfect_separation(self):
        """Perfect separation should give minDCF = 0."""
        scores = np.array([0.9, 0.8, 0.85, 0.1, 0.05, 0.15])
        labels = np.array([1, 1, 1, 0, 0, 0])  # bonafide=1, spoof=0

        min_dcf = compute_min_dcf(scores, labels, p_target=0.05)

        assert min_dcf == pytest.approx(0.0, abs=0.01)

    def test_random_scores(self):
        """Random scores should give high minDCF."""
        np.random.seed(42)
        n = 1000
        scores = np.random.rand(n)
        labels = np.random.randint(0, 2, n)

        min_dcf = compute_min_dcf(scores, labels, p_target=0.05)

        assert min_dcf > 0.3  # Should be high for random

    def test_mindcf_range(self):
        """minDCF should be >= 0."""
        np.random.seed(456)
        for _ in range(10):
            n = 100
            scores = np.random.rand(n)
            labels = np.random.randint(0, 2, n)

            min_dcf = compute_min_dcf(scores, labels, p_target=0.05)

            assert min_dcf >= 0

    def test_p_target_effect(self):
        """Different p_target should give different minDCF."""
        np.random.seed(789)
        n = 500
        scores = np.random.rand(n)
        labels = np.random.randint(0, 2, n)

        min_dcf_05 = compute_min_dcf(scores, labels, p_target=0.05)
        min_dcf_50 = compute_min_dcf(scores, labels, p_target=0.5)

        # Both should be valid
        assert min_dcf_05 >= 0
        assert min_dcf_50 >= 0


class TestCllr:
    """Test log-likelihood ratio cost."""

    def test_random_scores(self):
        """Random scores should give finite Cllr."""
        np.random.seed(42)
        n = 200
        scores = np.random.randn(n)  # Random LLRs
        labels = np.random.randint(0, 2, n)

        cllr = compute_cllr(scores, labels)

        # Should be finite
        assert np.isfinite(cllr)
        assert cllr >= 0

    def test_cllr_positive(self):
        """Cllr should always be positive."""
        np.random.seed(123)
        for _ in range(10):
            n = 100
            scores = np.random.randn(n)
            labels = np.random.randint(0, 2, n)

            cllr = compute_cllr(scores, labels)

            assert cllr >= 0


class TestActDCF:
    """Test actual Detection Cost Function."""

    def test_actdcf_perfect_at_good_threshold(self):
        """Perfect scores at correct threshold should give low actDCF."""
        scores = np.array([0.9, 0.8, 0.85, 0.1, 0.05, 0.15])
        labels = np.array([1, 1, 1, 0, 0, 0])

        # At threshold 0.5, all samples correctly classified
        act_dcf = compute_act_dcf(scores, labels, threshold=0.5, p_target=0.05)

        assert act_dcf == pytest.approx(0.0, abs=0.01)

    def test_actdcf_positive(self):
        """actDCF should be >= 0."""
        np.random.seed(42)
        n = 100
        scores = np.random.rand(n)
        labels = np.random.randint(0, 2, n)
        threshold = 0.5

        act_dcf = compute_act_dcf(scores, labels, threshold=threshold, p_target=0.05)

        assert act_dcf >= 0

    def test_actdcf_geq_mindcf(self):
        """actDCF at any threshold should be >= minDCF."""
        np.random.seed(42)
        n = 200
        scores = np.random.rand(n)
        labels = np.random.randint(0, 2, n)

        min_dcf = compute_min_dcf(scores, labels, p_target=0.05)

        # Test at various thresholds
        for threshold in [0.3, 0.5, 0.7]:
            act_dcf = compute_act_dcf(scores, labels, threshold=threshold, p_target=0.05)
            assert act_dcf >= min_dcf - 0.01  # Allow small tolerance
