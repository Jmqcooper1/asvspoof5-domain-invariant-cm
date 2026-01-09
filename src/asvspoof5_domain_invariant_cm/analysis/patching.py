"""Activation patching for mechanistic analysis.

Patching involves replacing activations from one model with another
to test causal effects on predictions and domain leakage.
"""

from typing import Optional, Callable

import torch
import torch.nn as nn
import numpy as np


class ActivationCache:
    """Cache for storing activations during forward pass."""

    def __init__(self):
        self.activations = {}

    def hook_fn(self, name: str) -> Callable:
        """Create a hook function for a named module."""

        def hook(module, input, output):
            self.activations[name] = output.detach()

        return hook

    def clear(self):
        """Clear cached activations."""
        self.activations = {}


def register_hooks(
    model: nn.Module,
    layer_names: list[str],
) -> tuple[ActivationCache, list]:
    """Register forward hooks to cache activations.

    Args:
        model: PyTorch model.
        layer_names: Names of layers to hook.

    Returns:
        Tuple of (cache, list of hook handles).
    """
    cache = ActivationCache()
    handles = []

    for name, module in model.named_modules():
        if name in layer_names:
            handle = module.register_forward_hook(cache.hook_fn(name))
            handles.append(handle)

    return cache, handles


def remove_hooks(handles: list) -> None:
    """Remove registered hooks."""
    for handle in handles:
        handle.remove()


def activation_patching(
    source_model: nn.Module,
    target_model: nn.Module,
    inputs: torch.Tensor,
    patch_layers: list[str],
    metric_fn: Callable,
) -> dict:
    """Perform activation patching from source to target model.

    Args:
        source_model: Model to get activations from (e.g., DANN).
        target_model: Model to patch activations into (e.g., ERM).
        inputs: Input tensor.
        patch_layers: Names of layers to patch.
        metric_fn: Function to compute metric on model output.

    Returns:
        Dictionary with patching results.
    """
    # Get source activations
    source_cache, source_handles = register_hooks(source_model, patch_layers)

    with torch.no_grad():
        source_model(inputs)

    source_activations = {k: v.clone() for k, v in source_cache.activations.items()}
    remove_hooks(source_handles)

    # Get baseline target output
    with torch.no_grad():
        baseline_output = target_model(inputs)
        baseline_metric = metric_fn(baseline_output)

    results = {
        "baseline_metric": baseline_metric,
        "patched_results": {},
    }

    # Patch each layer and measure effect
    for layer_name in patch_layers:
        if layer_name not in source_activations:
            continue

        # Create patching hook
        source_act = source_activations[layer_name]

        def patch_hook(module, input, output):
            return source_act

        # Find target module
        target_module = None
        for name, module in target_model.named_modules():
            if name == layer_name:
                target_module = module
                break

        if target_module is None:
            continue

        # Register hook and run forward
        handle = target_module.register_forward_hook(patch_hook)

        with torch.no_grad():
            patched_output = target_model(inputs)
            patched_metric = metric_fn(patched_output)

        handle.remove()

        results["patched_results"][layer_name] = {
            "patched_metric": patched_metric,
            "metric_change": patched_metric - baseline_metric,
        }

    return results


def compute_patching_effect(
    source_model: nn.Module,
    target_model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    patch_layers: list[str],
    device: torch.device,
) -> dict:
    """Compute average patching effect over a dataset.

    Args:
        source_model: Source model (e.g., DANN).
        target_model: Target model (e.g., ERM).
        dataloader: DataLoader for evaluation.
        patch_layers: Layers to patch.
        device: Computation device.

    Returns:
        Dictionary with aggregated patching effects.
    """
    source_model.eval()
    target_model.eval()

    all_results = {layer: [] for layer in patch_layers}
    baseline_metrics = []

    for batch in dataloader:
        inputs = batch["waveform"].to(device)

        def task_accuracy(output):
            preds = output["task_logits"].argmax(dim=-1)
            labels = batch["label"].to(device)
            return (preds == labels).float().mean().item()

        result = activation_patching(
            source_model,
            target_model,
            inputs,
            patch_layers,
            task_accuracy,
        )

        baseline_metrics.append(result["baseline_metric"])

        for layer, layer_result in result["patched_results"].items():
            all_results[layer].append(layer_result["metric_change"])

    # Aggregate
    aggregated = {
        "baseline_accuracy": float(np.mean(baseline_metrics)),
        "per_layer": {},
    }

    for layer, changes in all_results.items():
        if changes:
            aggregated["per_layer"][layer] = {
                "mean_change": float(np.mean(changes)),
                "std_change": float(np.std(changes)),
            }

    return aggregated
