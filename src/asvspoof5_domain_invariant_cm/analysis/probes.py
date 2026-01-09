"""Layer-wise domain probing for analyzing domain leakage."""

from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def train_domain_probe(
    embeddings: np.ndarray,
    domain_labels: np.ndarray,
    classifier: str = "logistic",
    cv_folds: int = 5,
    seed: Optional[int] = None,
) -> dict:
    """Train a linear probe to predict domain from embeddings.

    Lower accuracy indicates more domain-invariant representations.

    Args:
        embeddings: Feature embeddings of shape (N, D).
        domain_labels: Domain labels of shape (N,).
        classifier: Type of classifier ('logistic', 'svm').
        cv_folds: Number of cross-validation folds.
        seed: Random seed.

    Returns:
        Dictionary with accuracy and other metrics.
    """
    # Standardize features
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # Initialize classifier
    if classifier == "logistic":
        clf = LogisticRegression(
            max_iter=1000,
            random_state=seed,
            n_jobs=-1,
        )
    elif classifier == "svm":
        clf = SVC(
            kernel="linear",
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    # Cross-validation
    scores = cross_val_score(
        clf,
        embeddings_scaled,
        domain_labels,
        cv=cv_folds,
        scoring="accuracy",
    )

    return {
        "accuracy": float(np.mean(scores)),
        "accuracy_std": float(np.std(scores)),
        "cv_scores": scores.tolist(),
        "n_samples": len(embeddings),
        "n_classes": len(np.unique(domain_labels)),
    }


def layerwise_probing(
    layer_embeddings: dict[int, np.ndarray],
    domain_labels: np.ndarray,
    classifier: str = "logistic",
    cv_folds: int = 5,
    seed: Optional[int] = None,
) -> dict:
    """Run domain probes on embeddings from each layer.

    Args:
        layer_embeddings: Dictionary mapping layer index to embeddings.
        domain_labels: Domain labels.
        classifier: Classifier type.
        cv_folds: Number of CV folds.
        seed: Random seed.

    Returns:
        Dictionary with per-layer probe results.
    """
    results = {}

    for layer_idx, embeddings in layer_embeddings.items():
        result = train_domain_probe(
            embeddings,
            domain_labels,
            classifier=classifier,
            cv_folds=cv_folds,
            seed=seed,
        )
        results[layer_idx] = result

    # Find layer with max leakage
    max_layer = max(results.keys(), key=lambda k: results[k]["accuracy"])

    return {
        "per_layer": results,
        "max_leakage_layer": max_layer,
        "max_leakage_accuracy": results[max_layer]["accuracy"],
    }


def compare_probe_accuracies(
    erm_results: dict,
    dann_results: dict,
) -> dict:
    """Compare probe accuracies between ERM and DANN models.

    Args:
        erm_results: Probing results from ERM model.
        dann_results: Probing results from DANN model.

    Returns:
        Dictionary with comparison metrics.
    """
    comparison = {}

    for layer in erm_results["per_layer"].keys():
        erm_acc = erm_results["per_layer"][layer]["accuracy"]
        dann_acc = dann_results["per_layer"][layer]["accuracy"]

        comparison[layer] = {
            "erm_accuracy": erm_acc,
            "dann_accuracy": dann_acc,
            "reduction": erm_acc - dann_acc,
            "relative_reduction": (erm_acc - dann_acc) / max(erm_acc, 1e-6),
        }

    # Aggregate
    avg_erm = np.mean([c["erm_accuracy"] for c in comparison.values()])
    avg_dann = np.mean([c["dann_accuracy"] for c in comparison.values()])

    return {
        "per_layer": comparison,
        "avg_erm_accuracy": float(avg_erm),
        "avg_dann_accuracy": float(avg_dann),
        "avg_reduction": float(avg_erm - avg_dann),
    }
