"""Layer-wise domain probing for analyzing domain leakage."""

from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
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
    unique_labels, label_counts = np.unique(domain_labels, return_counts=True)
    n_classes = int(len(unique_labels))
    if n_classes < 2:
        return {
            "status": "skipped",
            "skip_reason": "only_one_class_in_labels",
            "accuracy": float("nan"),
            "accuracy_std": float("nan"),
            "cv_scores": [],
            "n_samples": int(len(embeddings)),
            "n_classes": n_classes,
            "cv_folds_requested": int(cv_folds),
            "cv_folds_used": 0,
            "class_counts": {int(k): int(v) for k, v in zip(unique_labels, label_counts)},
        }

    min_class_count = int(label_counts.min())
    cv_folds_used = int(min(cv_folds, min_class_count))
    if cv_folds_used < 2:
        return {
            "status": "skipped",
            "skip_reason": "insufficient_samples_per_class_for_cv",
            "accuracy": float("nan"),
            "accuracy_std": float("nan"),
            "cv_scores": [],
            "n_samples": int(len(embeddings)),
            "n_classes": n_classes,
            "cv_folds_requested": int(cv_folds),
            "cv_folds_used": cv_folds_used,
            "class_counts": {int(k): int(v) for k, v in zip(unique_labels, label_counts)},
        }

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
    cv = StratifiedKFold(n_splits=cv_folds_used, shuffle=True, random_state=seed)
    scores = cross_val_score(
        clf,
        embeddings_scaled,
        domain_labels,
        cv=cv,
        scoring="accuracy",
    )

    return {
        "status": "ok",
        "accuracy": float(np.mean(scores)),
        "accuracy_std": float(np.std(scores)),
        "cv_scores": scores.tolist(),
        "n_samples": len(embeddings),
        "n_classes": n_classes,
        "cv_folds_requested": int(cv_folds),
        "cv_folds_used": cv_folds_used,
        "class_counts": {int(k): int(v) for k, v in zip(unique_labels, label_counts)},
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

    # Find layer with max leakage (ignore skipped/NaN)
    valid_layers = [
        k
        for k, v in results.items()
        if v.get("status") == "ok" and np.isfinite(v.get("accuracy", float("nan")))
    ]
    if valid_layers:
        max_layer = max(valid_layers, key=lambda k: results[k]["accuracy"])
        max_acc = results[max_layer]["accuracy"]
    else:
        max_layer = None
        max_acc = float("nan")

    return {
        "per_layer": results,
        "max_leakage_layer": max_layer,
        "max_leakage_accuracy": max_acc,
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
