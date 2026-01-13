"""MFCC baseline model for comparison with SSL models.

Provides both:
- Sklearn LogisticRegression baseline (fast, no GPU needed)
- Simple MLP baseline (trainable with PyTorch)
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class MFCCDataset(torch.utils.data.Dataset):
    """Dataset for pre-extracted MFCC features.

    Args:
        features_path: Path to .npy file with features.
        metadata_path: Path to metadata CSV.
    """

    def __init__(
        self,
        features_path: Path,
        metadata_path: Path,
    ):
        import pandas as pd

        self.features = np.load(features_path)
        self.metadata = pd.read_csv(metadata_path)

        assert len(self.features) == len(self.metadata), (
            f"Feature count mismatch: {len(self.features)} vs {len(self.metadata)}"
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict:
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        row = self.metadata.iloc[idx]

        return {
            "features": features,
            "y_task": int(row["y_task"]),
            "y_codec": int(row["y_codec"]),
            "y_codec_q": int(row["y_codec_q"]),
            "flac_file": row["flac_file"],
        }


class MFCCClassifierMLP(nn.Module):
    """MLP classifier for MFCC features.

    Args:
        input_dim: Input feature dimension (n_mfcc * 2 for mean+std).
        hidden_dim: Hidden layer dimension.
        num_classes: Number of output classes.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features [B, D].

        Returns:
            Logits [B, num_classes].
        """
        return self.classifier(x)


def train_logreg_baseline(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: Optional[np.ndarray] = None,
    val_labels: Optional[np.ndarray] = None,
    max_iter: int = 1000,
    C: float = 1.0,
) -> dict:
    """Train logistic regression baseline.

    Args:
        train_features: Training features [N, D].
        train_labels: Training labels [N].
        val_features: Validation features.
        val_labels: Validation labels.
        max_iter: Maximum iterations.
        C: Regularization parameter.

    Returns:
        Dictionary with trained model and metrics.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Standardize features
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)

    # Train logistic regression
    clf = LogisticRegression(
        max_iter=max_iter,
        C=C,
        solver="lbfgs",
        random_state=42,
    )
    clf.fit(train_features_scaled, train_labels)

    # Training accuracy
    train_acc = clf.score(train_features_scaled, train_labels)
    train_probs = clf.predict_proba(train_features_scaled)[:, 1]

    results = {
        "model": clf,
        "scaler": scaler,
        "train_acc": train_acc,
        "train_probs": train_probs,
    }

    # Validation
    if val_features is not None and val_labels is not None:
        val_features_scaled = scaler.transform(val_features)
        val_acc = clf.score(val_features_scaled, val_labels)
        val_probs = clf.predict_proba(val_features_scaled)[:, 1]

        results["val_acc"] = val_acc
        results["val_probs"] = val_probs

    return results


def compute_mfcc_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """Compute EER and minDCF for MFCC baseline.

    Args:
        scores: Prediction scores (probability of class 1 / spoof).
        labels: Ground truth labels (0=bonafide, 1=spoof).

    Returns:
        Dictionary with metrics.
    """
    from ..evaluation.metrics import compute_eer, compute_min_dcf

    # Convert to bonafide scores (higher = more bonafide)
    bonafide_scores = 1 - scores

    eer, threshold = compute_eer(bonafide_scores, labels)
    min_dcf = compute_min_dcf(bonafide_scores, labels)

    return {
        "eer": eer,
        "eer_threshold": threshold,
        "min_dcf": min_dcf,
    }
