"""Non-semantic baseline models for cached embeddings.

Provides classifiers for TRILLsson and other pre-extracted embeddings.
"""

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


class CachedEmbeddingDataset(torch.utils.data.Dataset):
    """Dataset for cached embeddings (e.g., TRILLsson).

    Args:
        embeddings_path: Path to embeddings .npy file.
        metadata_path: Path to metadata .csv file.
    """

    def __init__(
        self,
        embeddings_path: Path,
        metadata_path: Path,
    ):
        import pandas as pd

        self.embeddings = np.load(embeddings_path)
        self.metadata = pd.read_csv(metadata_path)

        assert len(self.embeddings) == len(self.metadata), (
            f"Embeddings ({len(self.embeddings)}) and metadata ({len(self.metadata)}) "
            "length mismatch"
        )

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> dict:
        row = self.metadata.iloc[idx]
        return {
            "embedding": torch.from_numpy(self.embeddings[idx].astype(np.float32)),
            "y_task": int(row["y_task"]),
            "y_codec": int(row["y_codec"]),
            "y_codec_q": int(row["y_codec_q"]),
            "flac_file": row["flac_file"],
        }


class MLPEmbeddingClassifier(nn.Module):
    """MLP classifier for cached embeddings.

    Args:
        input_dim: Input embedding dimension.
        hidden_dims: List of hidden layer dimensions.
        num_classes: Number of output classes.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dims: list[int] = [256, 128],
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input embeddings of shape (B, D).

        Returns:
            Logits of shape (B, num_classes).
        """
        return self.classifier(x)


def train_sklearn_classifier(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    classifier_type: Literal["logistic", "mlp"] = "logistic",
    **kwargs,
) -> tuple:
    """Train sklearn classifier on embeddings.

    Args:
        train_embeddings: Training embeddings of shape (N, D).
        train_labels: Training labels of shape (N,).
        classifier_type: Type of classifier.
        **kwargs: Additional classifier arguments.

    Returns:
        Tuple of (classifier, scaler).
    """
    # Standardize features
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_embeddings)

    if classifier_type == "logistic":
        clf = LogisticRegression(
            max_iter=kwargs.get("max_iter", 1000),
            random_state=kwargs.get("random_state", 42),
            n_jobs=-1,
        )
    elif classifier_type == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=kwargs.get("hidden_layer_sizes", (256, 128)),
            max_iter=kwargs.get("max_iter", 500),
            random_state=kwargs.get("random_state", 42),
            early_stopping=True,
            validation_fraction=0.1,
        )
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

    clf.fit(train_scaled, train_labels)

    return clf, scaler


def predict_sklearn(
    clf,
    scaler: StandardScaler,
    embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Get predictions and probabilities from sklearn classifier.

    Args:
        clf: Trained sklearn classifier.
        scaler: Feature scaler.
        embeddings: Embeddings to predict on.

    Returns:
        Tuple of (predictions, probabilities).
    """
    scaled = scaler.transform(embeddings)
    predictions = clf.predict(scaled)
    probabilities = clf.predict_proba(scaled)

    return predictions, probabilities
