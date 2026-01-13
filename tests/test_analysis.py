"""Tests for analysis tools (probes, CKA, patching)."""

import numpy as np
import pytest

from asvspoof5_domain_invariant_cm.analysis.repr_similarity import (
    compute_linear_cka,
    compute_cka,
)
from asvspoof5_domain_invariant_cm.analysis.probes import (
    train_domain_probe,
)


class TestLinearCKA:
    """Test Centered Kernel Alignment computation."""

    def test_identical_representations(self):
        """CKA of identical representations should be 1."""
        np.random.seed(42)
        X = np.random.randn(100, 256)
        cka = compute_linear_cka(X, X)

        assert cka == pytest.approx(1.0, abs=0.01)

    def test_cka_range(self):
        """CKA should be between 0 and 1."""
        np.random.seed(42)
        for _ in range(10):
            X = np.random.randn(50, 64)
            Y = np.random.randn(50, 64)
            cka = compute_linear_cka(X, Y)

            assert 0 <= cka <= 1

    def test_cka_symmetry(self):
        """CKA should be symmetric: CKA(X,Y) = CKA(Y,X)."""
        np.random.seed(42)
        X = np.random.randn(50, 64)
        Y = np.random.randn(50, 128)  # Different dimensions

        cka_xy = compute_linear_cka(X, Y)
        cka_yx = compute_linear_cka(Y, X)

        assert cka_xy == pytest.approx(cka_yx)

    def test_cka_different_dimensions(self):
        """CKA should work with different feature dimensions."""
        np.random.seed(42)
        X = np.random.randn(100, 256)
        Y = np.random.randn(100, 512)

        cka = compute_linear_cka(X, Y)

        assert 0 <= cka <= 1


class TestRBFCKA:
    """Test CKA with RBF kernel."""

    def test_rbf_cka_identical(self):
        """RBF CKA of identical representations should be 1."""
        np.random.seed(42)
        X = np.random.randn(50, 64)
        cka = compute_cka(X, X, kernel="rbf")

        assert cka == pytest.approx(1.0, abs=0.01)

    def test_rbf_cka_range(self):
        """RBF CKA should be between 0 and 1."""
        np.random.seed(42)
        X = np.random.randn(50, 64)
        Y = np.random.randn(50, 64)
        cka = compute_cka(X, Y, kernel="rbf")

        assert 0 <= cka <= 1


class TestDomainProbes:
    """Test domain probe training."""

    def test_probe_training(self):
        """Test basic probe training."""
        # Generate synthetic data with some class structure
        n_samples = 200
        n_features = 128
        n_classes = 4

        # Create separable data
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)

        # Add class-dependent signal
        for c in range(n_classes):
            X[y == c] += np.random.randn(n_features) * 0.5

        result = train_domain_probe(X, y, classifier="logistic", cv_folds=3)

        # Should have some predictive power
        assert result["accuracy"] > 0.25  # Better than random for 4 classes
        assert "accuracy_std" in result
        assert "cv_scores" in result

    def test_probe_with_svm(self):
        """Test probe with SVM classifier."""
        n_samples = 100
        n_features = 64
        n_classes = 3

        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)

        # Add signal
        for c in range(n_classes):
            X[y == c] += np.random.randn(n_features) * 0.3

        result = train_domain_probe(X, y, classifier="svm", cv_folds=3)

        assert result["accuracy"] > 0.3
        assert "n_samples" in result
        assert "n_classes" in result

    def test_probe_perfect_separation(self):
        """Test probe with perfectly separable data."""
        n_per_class = 50
        n_classes = 2

        # Create perfectly separable data
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(n_per_class, 64) + 5,  # Class 0
            np.random.randn(n_per_class, 64) - 5,  # Class 1
        ])
        y = np.array([0] * n_per_class + [1] * n_per_class)

        result = train_domain_probe(X, y, classifier="logistic", cv_folds=3)

        # Should achieve near-perfect accuracy
        assert result["accuracy"] > 0.95
