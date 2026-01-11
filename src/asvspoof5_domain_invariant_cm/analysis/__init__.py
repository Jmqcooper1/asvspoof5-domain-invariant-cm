"""Interpretability and analysis tools."""

from .patching import (
    ActivationCache,
    activation_patching,
    compute_patching_effect,
    register_hooks,
    remove_hooks,
)
from .probes import (
    compare_probe_accuracies,
    layerwise_probing,
    train_domain_probe,
)
from .repr_similarity import (
    compare_representations,
    compute_cka,
    compute_linear_cka,
    layerwise_cka_matrix,
)

__all__ = [
    # Probes
    "train_domain_probe",
    "layerwise_probing",
    "compare_probe_accuracies",
    # CKA
    "compute_linear_cka",
    "compute_cka",
    "compare_representations",
    "layerwise_cka_matrix",
    # Patching
    "ActivationCache",
    "register_hooks",
    "remove_hooks",
    "activation_patching",
    "compute_patching_effect",
]
