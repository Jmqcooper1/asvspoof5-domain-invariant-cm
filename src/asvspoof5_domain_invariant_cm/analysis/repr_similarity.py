"""Representation similarity analysis using CKA and related methods."""

import numpy as np


def _center_gram(gram: np.ndarray) -> np.ndarray:
    """Center a Gram matrix."""
    n = gram.shape[0]
    unit = np.ones((n, n)) / n
    return gram - gram @ unit - unit @ gram + unit @ gram @ unit


def _hsic(K: np.ndarray, L: np.ndarray) -> float:
    """Compute HSIC (Hilbert-Schmidt Independence Criterion)."""
    K_centered = _center_gram(K)
    L_centered = _center_gram(L)
    n = K.shape[0]
    return np.trace(K_centered @ L_centered) / ((n - 1) ** 2)


def compute_linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA (Centered Kernel Alignment).

    CKA measures similarity between representations invariant to
    orthogonal transformation and isotropic scaling.

    Args:
        X: First representation matrix of shape (N, D1).
        Y: Second representation matrix of shape (N, D2).

    Returns:
        CKA similarity score in [0, 1].
    """
    # Compute Gram matrices
    K = X @ X.T
    L = Y @ Y.T

    # Compute CKA
    hsic_xy = _hsic(K, L)
    hsic_xx = _hsic(K, K)
    hsic_yy = _hsic(L, L)

    cka = hsic_xy / np.sqrt(hsic_xx * hsic_yy + 1e-10)

    return float(cka)


def compute_cka(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: str = "linear",
    sigma: float = None,
) -> float:
    """Compute CKA with specified kernel.

    Args:
        X: First representation matrix of shape (N, D1).
        Y: Second representation matrix of shape (N, D2).
        kernel: Kernel type ('linear', 'rbf').
        sigma: RBF kernel bandwidth (estimated if None).

    Returns:
        CKA similarity score.
    """
    if kernel == "linear":
        return compute_linear_cka(X, Y)
    elif kernel == "rbf":
        # RBF kernel
        if sigma is None:
            # Median heuristic
            from scipy.spatial.distance import pdist

            sigma_x = np.median(pdist(X))
            sigma_y = np.median(pdist(Y))
            sigma = (sigma_x + sigma_y) / 2

        def rbf_kernel(Z, sigma):
            row_norms = np.sum(Z**2, axis=1, keepdims=True)
            sq_dists = row_norms + row_norms.T - 2 * Z @ Z.T
            return np.exp(-sq_dists / (2 * sigma**2))

        K = rbf_kernel(X, sigma)
        L = rbf_kernel(Y, sigma)

        hsic_xy = _hsic(K, L)
        hsic_xx = _hsic(K, K)
        hsic_yy = _hsic(L, L)

        return float(hsic_xy / np.sqrt(hsic_xx * hsic_yy + 1e-10))
    else:
        raise ValueError(f"Unknown kernel: {kernel}")


def compare_representations(
    erm_layers: dict[int, np.ndarray],
    dann_layers: dict[int, np.ndarray],
) -> dict:
    """Compare representations from ERM and DANN models layer-by-layer.

    Args:
        erm_layers: Dictionary mapping layer index to ERM embeddings.
        dann_layers: Dictionary mapping layer index to DANN embeddings.

    Returns:
        Dictionary with CKA similarities per layer.
    """
    results = {}

    common_layers = set(erm_layers.keys()) & set(dann_layers.keys())

    for layer in sorted(common_layers):
        cka = compute_linear_cka(erm_layers[layer], dann_layers[layer])
        results[layer] = {
            "cka": cka,
            "erm_shape": erm_layers[layer].shape,
            "dann_shape": dann_layers[layer].shape,
        }

    # Summary statistics
    cka_values = [r["cka"] for r in results.values()]

    return {
        "per_layer": results,
        "mean_cka": float(np.mean(cka_values)),
        "min_cka": float(np.min(cka_values)),
        "max_cka": float(np.max(cka_values)),
        "most_different_layer": min(results.keys(), key=lambda k: results[k]["cka"]),
    }


def layerwise_cka_matrix(
    layers: dict[int, np.ndarray],
) -> np.ndarray:
    """Compute CKA between all pairs of layers.

    Args:
        layers: Dictionary mapping layer index to embeddings.

    Returns:
        CKA matrix of shape (num_layers, num_layers).
    """
    layer_indices = sorted(layers.keys())
    n_layers = len(layer_indices)

    cka_matrix = np.zeros((n_layers, n_layers))

    for i, li in enumerate(layer_indices):
        for j, lj in enumerate(layer_indices):
            if i <= j:
                cka = compute_linear_cka(layers[li], layers[lj])
                cka_matrix[i, j] = cka
                cka_matrix[j, i] = cka

    return cka_matrix
