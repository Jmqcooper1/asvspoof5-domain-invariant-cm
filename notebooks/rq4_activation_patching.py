#!/usr/bin/env python
# coding: utf-8

# # RQ4: Activation Patching for Domain Invariance
# 
# **Research Question:** Can activation patching reduce domain leakage without full retraining?
# 
# **Hypothesis:** By identifying layers where DANN diverges most from ERM (via CKA), we can "transplant" domain-invariant representations into ERM at inference time.
# 
# ## Approach
# 
# 1. **CKA Analysis** — Identify which layers show largest representation difference between ERM and DANN
# 2. **Activation Patching** — During inference, replace ERM activations at layer L with DANN activations
# 3. **Evaluation** — Measure domain probe accuracy and EER on patched models
# 
# ## Expected Outcome
# 
# If successful, this provides a lightweight method to improve domain robustness without expensive DANN training.

# ## Setup

# In[5]:


import os
import sys
from pathlib import Path

project_root = Path("..").resolve()
src_root = project_root / "src"
for candidate_path in (src_root, project_root):
    candidate_path_str = str(candidate_path)
    if candidate_path_str not in sys.path:
        sys.path.insert(0, candidate_path_str)

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Core utilities
from asvspoof5_domain_invariant_cm.utils import (
    get_device,
    get_manifest_path,
    get_manifests_dir,
    get_runs_dir,
    set_seed,
)

# Data loading - use create_dataloader (not get_dataloader)
from asvspoof5_domain_invariant_cm.data import (
    ASVspoof5Dataset,
    AudioCollator,
    create_dataloader,
    load_vocab,
)

# Analysis tools
from asvspoof5_domain_invariant_cm.analysis import (
    compute_linear_cka,
    compare_representations,
    layerwise_probing,
    ActivationCache,
    register_hooks,
    remove_hooks,
)

# Evaluation metrics
from asvspoof5_domain_invariant_cm.evaluation import compute_eer, compute_min_dcf

# Model components
from asvspoof5_domain_invariant_cm.models import (
    ClassifierHead,
    DANNModel,
    ERMModel,
    MultiHeadDomainDiscriminator,
    ProjectionHead,
    create_backbone,
    create_pooling,
)

set_seed(42)
device = get_device()
print(f"Using device: {device}")


# ## 1. Load Checkpoints
# 
# Load the ERM and DANN models (WavLM backbone).
# 
# We use a robust model loading function that handles architecture reconstruction from config.

# In[6]:


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device):
    """Load model from checkpoint with proper architecture reconstruction.

    Adapted from scripts/evaluate.py - handles both ERM and DANN models,
    auto-detects discriminator dimensions from weights.

    Args:
        checkpoint_path: Path to the checkpoint file.
        device: Device to load the model onto.

    Returns:
        Tuple of (model, config, codec_vocab, codec_q_vocab).
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    state_dict = checkpoint.get("model_state_dict", {})

    run_dir = checkpoint_path.parent.parent
    run_codec_vocab_path = run_dir / "codec_vocab.json"
    run_codec_q_vocab_path = run_dir / "codec_q_vocab.json"
    fallback_codec_vocab_path = get_manifests_dir() / "codec_vocab.json"
    fallback_codec_q_vocab_path = get_manifests_dir() / "codec_q_vocab.json"

    codec_vocab_path = run_codec_vocab_path if run_codec_vocab_path.exists() else fallback_codec_vocab_path
    codec_q_vocab_path = run_codec_q_vocab_path if run_codec_q_vocab_path.exists() else fallback_codec_q_vocab_path

    if not codec_vocab_path.exists():
        raise FileNotFoundError(
            f"Missing codec_vocab.json. Checked {run_codec_vocab_path} and {fallback_codec_vocab_path}."
        )
    if not codec_q_vocab_path.exists():
        raise FileNotFoundError(
            f"Missing codec_q_vocab.json. Checked {run_codec_q_vocab_path} and {fallback_codec_q_vocab_path}."
        )

    codec_vocab = load_vocab(codec_vocab_path)
    codec_q_vocab = load_vocab(codec_q_vocab_path)

    # Build architecture from config
    backbone_cfg = config.get("backbone", {})
    projection_cfg = config.get("projection", {})
    classifier_cfg = config.get("classifier", {})
    pooling_cfg = config.get("pooling", {})
    training_cfg = config.get("training", {})

    layer_selection = backbone_cfg.get("layer_selection", {})
    backbone = create_backbone(
        name=backbone_cfg.get("name", "wavlm_base_plus"),
        pretrained=backbone_cfg.get("pretrained", "microsoft/wavlm-base-plus"),
        freeze=True,  # Always freeze for inference
        layer_selection=layer_selection.get("method", "weighted"),
        k=layer_selection.get("k", 6),
        layer_indices=layer_selection.get("layers"),
        init_lower_bias=layer_selection.get("init_lower_bias", True),
    )

    pooling_method = pooling_cfg.get("method", "stats")
    pooling = create_pooling(pooling_method, backbone.hidden_size)

    proj_input_dim = backbone.hidden_size * 2 if pooling_method == "stats" else backbone.hidden_size

    projection = ProjectionHead(
        input_dim=proj_input_dim,
        hidden_dim=projection_cfg.get("hidden_dim", 512),
        output_dim=projection_cfg.get("output_dim", 256),
        num_layers=projection_cfg.get("num_layers", 2),
        dropout=projection_cfg.get("dropout", 0.1),
    )

    repr_dim = projection_cfg.get("output_dim", 256)
    task_head = ClassifierHead(
        input_dim=repr_dim,
        num_classes=classifier_cfg.get("num_classes", 2),
        hidden_dim=classifier_cfg.get("hidden_dim", 0),
        dropout=classifier_cfg.get("dropout", 0.1),
    )

    method = training_cfg.get("method", "erm")

    if method == "dann":
        dann_cfg = config.get("dann", {})
        disc_cfg = dann_cfg.get("discriminator", {})

        # Auto-detect from weights if available
        disc_weight_key = "domain_discriminator.shared.0.weight"
        if disc_weight_key in state_dict:
            disc_input_dim = state_dict[disc_weight_key].shape[1]
        else:
            disc_input_dim = disc_cfg.get("input_dim", proj_input_dim)

        domain_discriminator = MultiHeadDomainDiscriminator(
            input_dim=disc_input_dim,
            num_codecs=len(codec_vocab),
            num_codec_qs=len(codec_q_vocab),
            hidden_dim=disc_cfg.get("hidden_dim", 512),
            dropout=disc_cfg.get("dropout", 0.1),
        )

        model = DANNModel(
            backbone=backbone,
            pooling=pooling,
            projection=projection,
            task_head=task_head,
            domain_discriminator=domain_discriminator,
            lambda_=0.0,  # No GRL during inference
        )
    else:
        model = ERMModel(
            backbone=backbone,
            pooling=pooling,
            projection=projection,
            task_head=task_head,
        )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, config, codec_vocab, codec_q_vocab


# In[7]:


def resolve_checkpoint_path(*relative_parts: str) -> Path:
    """Resolve checkpoints from RUNS_DIR env or project runs directory."""
    runs_dir = Path(os.environ["RUNS_DIR"]) if "RUNS_DIR" in os.environ else get_runs_dir()
    return runs_dir.joinpath(*relative_parts)


# Parameterized pairwise setup (defaults keep prior ERM vs DANN behavior).
model_a_run = os.environ.get("RQ4_MODEL_A_RUN", "wavlm_erm")
model_a_ckpt = os.environ.get("RQ4_MODEL_A_CKPT", "best.pt")
model_b_run = os.environ.get("RQ4_MODEL_B_RUN", "wavlm_dann")
model_b_ckpt = os.environ.get("RQ4_MODEL_B_CKPT", "epoch_5_patched.pt")

MODEL_A_CHECKPOINT = resolve_checkpoint_path(model_a_run, "checkpoints", model_a_ckpt)
MODEL_B_CHECKPOINT = resolve_checkpoint_path(model_b_run, "checkpoints", model_b_ckpt)

print(f"Model A checkpoint: {MODEL_A_CHECKPOINT}")
print(f"Model B checkpoint: {MODEL_B_CHECKPOINT}")
print(f"Model A checkpoint exists: {MODEL_A_CHECKPOINT.exists()}")
print(f"Model B checkpoint exists: {MODEL_B_CHECKPOINT.exists()}")

if not MODEL_A_CHECKPOINT.exists() or not MODEL_B_CHECKPOINT.exists():
    raise FileNotFoundError(
        "Missing model checkpoints. Configure RUNS_DIR and optionally "
        "RQ4_MODEL_A_RUN/RQ4_MODEL_A_CKPT/RQ4_MODEL_B_RUN/RQ4_MODEL_B_CKPT."
    )

# Load models
print("\nLoading model A...")
model_a_model, model_a_config, codec_vocab, codec_q_vocab = load_model_from_checkpoint(MODEL_A_CHECKPOINT, device)
model_a_method = model_a_config.get("training", {}).get("method", "erm")
print(f"Model A method: {model_a_method}")

print("\nLoading model B...")
model_b_model, model_b_config, _, _ = load_model_from_checkpoint(MODEL_B_CHECKPOINT, device)
model_b_method = model_b_config.get("training", {}).get("method", "dann")
print(f"Model B method: {model_b_method}")

supported_methods = {"erm", "dann"}
if model_a_method not in supported_methods or model_b_method not in supported_methods:
    raise NotImplementedError(
        "This notebook currently supports training.method in {'erm', 'dann'} only. "
        f"Got model_a={model_a_method}, model_b={model_b_method}."
    )

model_a_name = os.environ.get("RQ4_MODEL_A_LABEL", f"A:{model_a_method.upper()}")
model_b_name = os.environ.get("RQ4_MODEL_B_LABEL", f"B:{model_b_method.upper()}")
patched_model_name = f"Patched {model_a_name}"

print(f"\nModel labels: {model_a_name} vs {model_b_name}")
print("DANN domain objective is applied on pre-projection pooled features.")
print("With a frozen backbone, strongest differences are expected after backbone hidden states.")

# Backward-compatible aliases for existing downstream variable names.
erm_model, erm_config = model_a_model, model_a_config
dann_model, dann_config = model_b_model, model_b_config

print(f"\nCodec vocab size: {len(codec_vocab)}")
print(f"Codec Q vocab size: {len(codec_q_vocab)}")


# ## 2. CKA Analysis
# 
# Centered Kernel Alignment (CKA) measures representational similarity between layers.
# We compute CKA between ERM and DANN at each layer to identify where they diverge most.
# 
# The library's `compute_linear_cka` function already handles numerical stability with epsilon.

# In[ ]:


def create_eval_dataloader(
    split: str = "dev",
    codec_vocab: dict = None,
    codec_q_vocab: dict = None,
    config: dict = None,
    max_samples: int = None,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
) -> torch.utils.data.DataLoader:
    """Create dataloader for CKA analysis and evaluation.

    Args:
        split: Data split ('dev' or 'eval').
        codec_vocab: CODEC vocabulary.
        codec_q_vocab: CODEC_Q vocabulary.
        config: Model config dict.
        max_samples: Maximum samples to use (None for all).
        batch_size: Batch size.
        num_workers: Number of workers.
        seed: Random seed for subset selection.

    Returns:
        DataLoader instance.
    """
    audio_cfg = config.get("audio", {}) if config else {}
    sample_rate = audio_cfg.get("sample_rate", 16000)
    max_duration = audio_cfg.get("max_duration_sec", 6.0)

    manifest_path = get_manifest_path(split)
    print(f"Loading manifest from: {manifest_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing manifest: {manifest_path}. Run scripts/prepare_asvspoof5.py after setting ASVSPOOF5_ROOT."
        )

    dataset = ASVspoof5Dataset(
        manifest_path=manifest_path,
        codec_vocab=codec_vocab,
        codec_q_vocab=codec_q_vocab,
        max_duration_sec=max_duration,
        sample_rate=sample_rate,
        mode="eval",
    )

    print(f"Dataset size: {len(dataset)}")

    # Subsample if needed
    if max_samples and max_samples < len(dataset):
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(dataset), size=max_samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"Subsampled to {len(dataset)} samples")

    fixed_length = int(max_duration * sample_rate)
    collator = AudioCollator(fixed_length=fixed_length, mode="eval")

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )


# In[ ]:


def _get_selected_layer_indices(backbone: torch.nn.Module, total_layers: int) -> list[int]:
    """Resolve which hidden-state indices participate in backbone layer mixing."""
    selection = getattr(backbone, "layer_selection", "weighted")
    k = int(getattr(backbone, "k", total_layers))
    explicit_indices = getattr(backbone, "layer_indices", None)

    if selection == "first_k":
        return list(range(min(k, total_layers)))
    if selection == "last_k":
        start = max(0, total_layers - k)
        return list(range(start, total_layers))
    if selection == "specific" and explicit_indices:
        return [int(idx) for idx in explicit_indices if 0 <= int(idx) < total_layers]

    # weighted / fallback: use all transformer layers
    return list(range(total_layers))


@torch.no_grad()
def extract_layer_representations(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_batches: int = None,
    representation: str = "layer_contrib",
) -> dict:
    """Extract model representations for CKA.

    Supported representations: hidden_states, mixed, repr, layer_contrib.
    """
    model.eval()
    layer_reps = {}
    total_batches = num_batches if num_batches else len(dataloader)

    for batch_idx, batch in enumerate(tqdm(dataloader, total=total_batches, desc=f"Extracting ({representation})")):
        if num_batches and batch_idx >= num_batches:
            break

        waveform = batch["waveform"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        lengths = batch["lengths"].to(device)

        outputs = model(waveform, attention_mask, lengths)
        all_hidden_states = outputs.get("all_hidden_states", [])
        if not all_hidden_states:
            raise RuntimeError("Model did not return all_hidden_states. Check model forward method.")

        if representation == "hidden_states":
            for layer_idx, hidden_state in enumerate(all_hidden_states):
                pooled = hidden_state.mean(dim=1)
                layer_reps.setdefault(layer_idx, []).append(pooled.cpu())

        elif representation == "repr":
            if "repr" not in outputs:
                raise RuntimeError("Model did not return repr. Cannot compute CKA with representation='repr'.")
            layer_reps.setdefault("repr", []).append(outputs["repr"].cpu())

        elif representation == "mixed":
            mixed, _ = model.backbone(waveform, attention_mask)
            layer_reps.setdefault("mixed", []).append(mixed.mean(dim=1).cpu())

        elif representation == "layer_contrib":
            total_layers = len(all_hidden_states)
            selected_indices = _get_selected_layer_indices(model.backbone, total_layers)
            selected_states = [all_hidden_states[idx] for idx in selected_indices]

            layer_pooling = getattr(model.backbone, "layer_pooling", None)
            if layer_pooling is None or not hasattr(layer_pooling, "weights"):
                raise RuntimeError("Backbone missing layer_pooling.weights needed for layer_contrib extraction.")

            weights = torch.softmax(layer_pooling.weights.detach(), dim=0)
            if weights.numel() != len(selected_states):
                raise RuntimeError(
                    "Layer weight count does not match selected states: "
                    f"weights={weights.numel()} states={len(selected_states)}"
                )

            for local_idx, (layer_idx, hidden_state) in enumerate(zip(selected_indices, selected_states)):
                contribution = hidden_state * weights[local_idx]
                pooled = contribution.mean(dim=1)
                layer_reps.setdefault(int(layer_idx), []).append(pooled.cpu())

        else:
            raise ValueError(
                f"Unknown representation '{representation}'. "
                "Expected one of: hidden_states, mixed, repr, layer_contrib"
            )

    result = {}
    for key, rep_list in layer_reps.items():
        if rep_list:
            result[key] = torch.cat(rep_list, dim=0)

    print(f"Extracted representations for {len(result)} keys: {sorted(result.keys(), key=lambda x: str(x))}")
    return result


# In[ ]:


# Select representation used by CKA and downstream patch-layer selection.
# Use layer_contrib by default; hidden_states is usually trivial (CKA~1.0) with frozen backbones.
cka_representation = "layer_contrib"
valid_representations = {"hidden_states", "mixed", "repr", "layer_contrib"}
if cka_representation not in valid_representations:
    raise ValueError(f"Invalid cka_representation={cka_representation}. Choose from {sorted(valid_representations)}")

if cka_representation == "hidden_states":
    is_frozen_base = bool(erm_config.get("backbone", {}).get("freeze", True))
    is_frozen_donor = bool(dann_config.get("backbone", {}).get("freeze", True))
    if is_frozen_base and is_frozen_donor:
        print(
            "Warning: hidden_states with frozen SSL backbones typically yields CKA~1.0. "
            "Use 'layer_contrib', 'mixed', or 'repr' for a more informative comparison."
        )

# Create dataloader for CKA analysis (use subset for speed)
print(f"Creating evaluation dataloader for CKA analysis ({cka_representation})...")
eval_loader = create_eval_dataloader(
    split="dev",
    codec_vocab=codec_vocab,
    codec_q_vocab=codec_q_vocab,
    config=erm_config,
    max_samples=5000,  # Use subset for speed
    batch_size=32,
)


# In[ ]:


# Extract representations from model A
print(f"Extracting {model_a_name} representations ({cka_representation})...")
erm_reps = extract_layer_representations(
    erm_model,
    eval_loader,
    device,
    representation=cka_representation,
)
print(f"Extracted {len(erm_reps)} keys, shapes: {[(k, v.shape) for k, v in erm_reps.items()]}")


# In[ ]:


# Reset dataloader for model B
eval_loader = create_eval_dataloader(
    split="dev",
    codec_vocab=codec_vocab,
    codec_q_vocab=codec_q_vocab,
    config=dann_config,
    max_samples=5000,
    batch_size=32,
)

# Extract representations from model B
print(f"Extracting {model_b_name} representations ({cka_representation})...")
dann_reps = extract_layer_representations(
    dann_model,
    eval_loader,
    device,
    representation=cka_representation,
)
print(f"Extracted {len(dann_reps)} keys, shapes: {[(k, v.shape) for k, v in dann_reps.items()]}")


# In[ ]:


# Compute CKA per representation key
print(f"Computing CKA between {model_a_name} and {model_b_name} using '{cka_representation}'...")
cka_results = compare_representations(
    erm_layers={k: v.numpy() for k, v in erm_reps.items()},
    dann_layers={k: v.numpy() for k, v in dann_reps.items()},
)

print("\nCKA Results:")
print(f"Model pair: {model_a_name} vs {model_b_name}")
print(f"Representation mode: {cka_representation}")
print(f"Mean CKA: {cka_results['mean_cka']:.4f}")
print(f"Min CKA:  {cka_results['min_cka']:.4f}")
print(f"Max CKA:  {cka_results['max_cka']:.4f}")
print(f"Most different key: {cka_results['most_different_layer']}")

print("\nPer-key CKA:")
for key in sorted(cka_results['per_layer'].keys(), key=lambda x: str(x)):
    cka = cka_results['per_layer'][key]['cka']
    print(f"  Key {key}: CKA = {cka:.4f}")

cka_layer_keys = [k for k in cka_results['per_layer'].keys() if isinstance(k, int)]
if len(cka_layer_keys) < 3:
    raise RuntimeError(
        "Activation patching needs at least 3 integer transformer-layer keys from CKA. "
        f"Current keys: {sorted(cka_results['per_layer'].keys(), key=lambda x: str(x))}. "
        "Use cka_representation='hidden_states' or 'layer_contrib'."
    )


# In[ ]:


# Plot CKA scores for integer layer keys (needed for activation patching)
layers = sorted([k for k in cka_results['per_layer'].keys() if isinstance(k, int)])
if len(layers) < 3:
    raise RuntimeError(
        "Cannot plot patch-layer CKA bars because fewer than 3 integer layer keys were found. "
        "Use cka_representation='hidden_states' or 'layer_contrib'."
    )

cka_scores = [cka_results['per_layer'][l]['cka'] for l in layers]

plt.figure(figsize=(12, 5))
bars = plt.bar(layers, cka_scores, color='steelblue', edgecolor='black')

# Highlight the most divergent layers
sorted_by_cka = sorted(layers, key=lambda l: cka_results['per_layer'][l]['cka'])
divergent_layers = sorted_by_cka[:3]
for i, l in enumerate(layers):
    if l in divergent_layers:
        bars[i].set_color('tomato')

plt.xlabel('Layer Index', fontsize=12)
plt.ylabel('CKA Similarity', fontsize=12)
plt.title(
    f"{model_a_name} vs {model_b_name} Representation Similarity (CKA, {cka_representation})\nRed = Most Divergent Layers",
    fontsize=14,
)
plt.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='High similarity threshold')
plt.ylim(0, 1.05)
plt.xticks(layers)
plt.legend()
plt.tight_layout()
plt.show()

print(f"\nMost divergent layers (lowest CKA): {divergent_layers}")
cka_value_strings = [f"{cka_results['per_layer'][l]['cka']:.4f}" for l in divergent_layers]
print(f"CKA values: {cka_value_strings}")


# ## 3. Activation Patching
# 
# Replace ERM activations at specific layers with DANN activations during inference.
# 
# We use forward hooks to:
# 1. Capture DANN activations at target layers
# 2. Replace ERM activations with DANN activations at those layers

# In[ ]:


class PatchedModel(torch.nn.Module):
    """Model that patches activations from a donor model at specified layers.

    Hooks into the backbone's internal transformer layers to replace
    hidden states from model_a with those from model_b.
    """

    def __init__(
        self,
        base_model: torch.nn.Module,
        donor_model: torch.nn.Module,
        patch_layers: list[int],
    ):
        """
        Args:
            base_model: The model to run inference on.
            donor_model: The model to take activations from.
            patch_layers: List of layer indices to patch (0-indexed transformer layers).
        """
        super().__init__()
        self.base_model = base_model
        self.donor_model = donor_model
        self.patch_layers = set(patch_layers)

        # Keep deterministic execution while patching.
        self.base_model.eval()
        self.donor_model.eval()

        self._donor_activations = {}
        self._handles = []
        self._setup_hooks()

    def _get_layer_name(self, layer_idx: int) -> str:
        """Get the full module name for a transformer layer.

        For WavLM: backbone.model.encoder.layers.{layer_idx}
        Note: layer_idx 0 corresponds to the first transformer layer (after CNN encoder)
        """
        return f"backbone.model.encoder.layers.{layer_idx}"

    def _setup_hooks(self):
        """Setup forward hooks to capture and replace activations."""
        for layer_idx in self.patch_layers:
            layer_name = self._get_layer_name(layer_idx)

            # Find module in donor
            donor_module = dict(self.donor_model.named_modules()).get(layer_name)
            if donor_module is None:
                raise ValueError(f"Layer {layer_name} not found in donor model")

            def make_capture_hook(idx):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        self._donor_activations[idx] = output[0].detach()
                    else:
                        self._donor_activations[idx] = output.detach()

                return hook

            handle = donor_module.register_forward_hook(make_capture_hook(layer_idx))
            self._handles.append(handle)

        # Register patching hooks on base model
        for layer_idx in self.patch_layers:
            layer_name = self._get_layer_name(layer_idx)

            base_module = dict(self.base_model.named_modules()).get(layer_name)
            if base_module is None:
                raise ValueError(f"Layer {layer_name} not found in base model")

            def make_patch_hook(idx):
                def hook(module, input, output):
                    if idx not in self._donor_activations:
                        return output

                    donor_act = self._donor_activations[idx]
                    base_hidden = output[0] if isinstance(output, tuple) else output
                    if donor_act.shape != base_hidden.shape:
                        raise RuntimeError(
                            "Activation shape mismatch during patching at layer "
                            f"{idx}: donor={tuple(donor_act.shape)} vs base={tuple(base_hidden.shape)}"
                        )

                    if isinstance(output, tuple):
                        # Keep tuple tail (including position_bias) from the base forward.
                        # Recomputing position_bias in later layers can fail for WavLM blocks
                        # that do not own relative-bias embeddings.
                        return (donor_act, *output[1:])
                    return donor_act

                return hook

            handle = base_module.register_forward_hook(make_patch_hook(layer_idx))
            self._handles.append(handle)

        print(f"Registered {len(self._handles)} hooks for patching layers {sorted(self.patch_layers)}")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def forward(
        self,
        waveform: torch.Tensor,
        attention_mask: torch.Tensor = None,
        lengths: torch.Tensor = None,
    ) -> dict:
        """Forward pass with activation patching.

        1. Run donor model to capture activations
        2. Run base model with patched activations (via hooks)
        """
        self._donor_activations.clear()

        # Run donor to capture activations
        with torch.no_grad():
            _ = self.donor_model(waveform, attention_mask, lengths)

        missing_layers = sorted(self.patch_layers.difference(self._donor_activations.keys()))
        if missing_layers:
            raise RuntimeError(f"Missing donor activations for patch layers: {missing_layers}")

        # Run base model (hooks will patch activations)
        output = self.base_model(waveform, attention_mask, lengths)

        return output

    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()


# In[ ]:


# Identify most divergent layers (lowest CKA = most different representations)
cka_per_layer = cka_results['per_layer']
sorted_layers = sorted(
    [k for k in cka_per_layer.keys() if isinstance(k, int)],
    key=lambda k: cka_per_layer[k]['cka'],
)

# Patch the 3 most divergent transformer layers that exist in the model
available_transformer_layers = {
    int(name.split(".")[-1])
    for name, _ in erm_model.named_modules()
    if name.startswith("backbone.model.encoder.layers.") and name.split(".")[-1].isdigit()
}
divergent_layers = [layer_idx for layer_idx in sorted_layers if layer_idx in available_transformer_layers][:3]
if len(divergent_layers) < 3:
    raise RuntimeError(
        f"Expected at least 3 patchable transformer layers, found {len(divergent_layers)}. "
        f"Available layers: {sorted(available_transformer_layers)}"
    )
print(f"Most divergent transformer layers (lowest CKA): {divergent_layers}")
cka_value_strings = [f"{cka_per_layer[layer_idx]['cka']:.4f}" for layer_idx in divergent_layers]
print(f"CKA values: {cka_value_strings}")

# Create patched model (base=model A, donor=model B)
patched_model = PatchedModel(
    base_model=erm_model,
    donor_model=dann_model,
    patch_layers=divergent_layers,
)
patched_model.eval()
print(f"\n{patched_model_name} created successfully!")


# ## 4. Evaluation
# 
# Compare:
# 1. Original ERM
# 2. Original DANN
# 3. Patched ERM (with DANN activations at selected layers)

# In[ ]:


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """Compute EER and collect predictions for analysis.

    Args:
        model: Model to evaluate.
        dataloader: Evaluation dataloader.
        device: Computation device.

    Returns:
        dict with 'eer', 'min_dcf', 'scores', 'labels'
    """
    model.eval()
    all_scores = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        waveform = batch['waveform'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        lengths = batch['lengths'].to(device)
        # Defensive access - handle case where y_task might not be present
        labels = batch.get('y_task')
        if labels is None:
            raise ValueError(
                "Batch missing 'y_task' field. Ensure dataloader/collator "
                "returns task labels in eval mode."
            )

        outputs = model(waveform, attention_mask, lengths)

        # Score convention in this codebase: higher score = more likely bonafide
        # Labels: 0 = bonafide, 1 = spoof
        probs = torch.softmax(outputs['task_logits'], dim=-1)
        scores = probs[:, 0].cpu().numpy()  # P(bonafide)

        all_scores.extend(scores)
        all_labels.extend(labels.numpy())

    scores = np.array(all_scores)
    labels = np.array(all_labels)

    eer, threshold = compute_eer(scores, labels)
    min_dcf = compute_min_dcf(scores, labels)

    return {
        'eer': eer,
        'min_dcf': min_dcf,
        'eer_threshold': threshold,
        'scores': scores,
        'labels': labels,
    }


# In[ ]:


# Create fresh dataloader for evaluation (full dev set or subset)
print("Creating evaluation dataloader...")
eval_loader = create_eval_dataloader(
    split="dev",
    codec_vocab=codec_vocab,
    codec_q_vocab=codec_q_vocab,
    config=erm_config,
    max_samples=10000,  # Use subset for faster evaluation, None for full set
    batch_size=32,
)


# In[ ]:


# Evaluate model A
print(f"Evaluating {model_a_name}...")
erm_results = evaluate_model(erm_model, eval_loader, device)
print(f"{model_a_name} EER: {erm_results['eer']:.2%}, minDCF: {erm_results['min_dcf']:.4f}")


# In[ ]:


# Recreate dataloader for DANN
eval_loader = create_eval_dataloader(
    split="dev",
    codec_vocab=codec_vocab,
    codec_q_vocab=codec_q_vocab,
    config=dann_config,
    max_samples=10000,
    batch_size=32,
)

# Evaluate model B
print(f"Evaluating {model_b_name}...")
dann_results = evaluate_model(dann_model, eval_loader, device)
print(f"{model_b_name} EER: {dann_results['eer']:.2%}, minDCF: {dann_results['min_dcf']:.4f}")


# In[ ]:


# Recreate dataloader for patched model
eval_loader = create_eval_dataloader(
    split="dev",
    codec_vocab=codec_vocab,
    codec_q_vocab=codec_q_vocab,
    config=erm_config,
    max_samples=10000,
    batch_size=32,
)

# Evaluate patched model A
print(f"Evaluating {patched_model_name}...")
patched_results = evaluate_model(patched_model, eval_loader, device)
print(f"{patched_model_name} EER: {patched_results['eer']:.2%}, minDCF: {patched_results['min_dcf']:.4f}")


# In[ ]:


# Compare results
print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)
print(f"{'Model':<20} {'EER':>10} {'minDCF':>10}")
print("-"*60)
print(f"{model_a_name:<20} {erm_results['eer']:>9.2%} {erm_results['min_dcf']:>10.4f}")
print(f"{model_b_name:<20} {dann_results['eer']:>9.2%} {dann_results['min_dcf']:>10.4f}")
print(f"{patched_model_name:<20} {patched_results['eer']:>9.2%} {patched_results['min_dcf']:>10.4f}")
print("="*60)


# ## 5. Domain Probe on Patched Model
# 
# Verify that patching reduces domain leakage by running codec probes on the patched representations.
# 
# Lower probe accuracy = more domain-invariant representations.

# In[ ]:


def extract_representations_for_probing(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: int = 5000,
    representation: str = "hidden_states",
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Extract representations and domain labels for probing.

    Supported representations: hidden_states, mixed, repr, layer_contrib.
    """
    model.eval()
    layer_reps = {}
    all_codec = []
    all_codec_q = []
    n_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting for probing ({representation})"):
            if max_samples and n_samples >= max_samples:
                break

            waveform = batch["waveform"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            lengths = batch["lengths"].to(device)

            outputs = model(waveform, attention_mask, lengths)
            all_hidden_states = outputs.get("all_hidden_states", [])
            batch_size = waveform.shape[0]

            if representation == "hidden_states":
                for layer_idx, hidden_state in enumerate(all_hidden_states):
                    pooled = hidden_state.mean(dim=1).cpu().numpy()
                    layer_reps.setdefault(layer_idx, []).append(pooled)

            elif representation == "repr":
                if "repr" not in outputs:
                    raise RuntimeError("Model did not return repr. Cannot probe representation='repr'.")
                layer_reps.setdefault("repr", []).append(outputs["repr"].cpu().numpy())

            elif representation == "mixed":
                mixed, _ = model.backbone(waveform, attention_mask)
                layer_reps.setdefault("mixed", []).append(mixed.mean(dim=1).cpu().numpy())

            elif representation == "layer_contrib":
                total_layers = len(all_hidden_states)
                selected_indices = _get_selected_layer_indices(model.backbone, total_layers)
                selected_states = [all_hidden_states[idx] for idx in selected_indices]

                layer_pooling = getattr(model.backbone, "layer_pooling", None)
                if layer_pooling is None or not hasattr(layer_pooling, "weights"):
                    raise RuntimeError("Backbone missing layer_pooling.weights needed for layer_contrib probing.")

                weights = torch.softmax(layer_pooling.weights.detach(), dim=0)
                if weights.numel() != len(selected_states):
                    raise RuntimeError(
                        "Layer weight count does not match selected states: "
                        f"weights={weights.numel()} states={len(selected_states)}"
                    )

                for local_idx, (layer_idx, hidden_state) in enumerate(zip(selected_indices, selected_states)):
                    contribution = hidden_state * weights[local_idx]
                    pooled = contribution.mean(dim=1).cpu().numpy()
                    layer_reps.setdefault(int(layer_idx), []).append(pooled)

            else:
                raise ValueError(
                    f"Unknown representation '{representation}'. "
                    "Expected one of: hidden_states, mixed, repr, layer_contrib"
                )

            # Defensive access - y_codec/y_codec_q may not be present in all eval modes
            y_codec = batch.get("y_codec")
            y_codec_q = batch.get("y_codec_q")
            if y_codec is not None:
                all_codec.append(y_codec.numpy())
            if y_codec_q is not None:
                all_codec_q.append(y_codec_q.numpy())

            n_samples += batch_size

    for key in list(layer_reps.keys()):
        layer_reps[key] = np.concatenate(layer_reps[key], axis=0)
        if max_samples:
            layer_reps[key] = layer_reps[key][:max_samples]

    if not all_codec:
        raise ValueError("No y_codec labels were found in batches. Cannot run codec probing.")
    if not all_codec_q:
        raise ValueError("No y_codec_q labels were found in batches. Cannot run codec_q probing.")

    all_codec = np.concatenate(all_codec)
    all_codec_q = np.concatenate(all_codec_q)
    if max_samples:
        all_codec = all_codec[:max_samples]
        all_codec_q = all_codec_q[:max_samples]

    return layer_reps, all_codec, all_codec_q


# In[ ]:


# Select representation for domain probing.
# Keep hidden_states to preserve layer-wise leakage profile.
probe_representation = "hidden_states"

# Create dataloader for probing
probe_loader = create_eval_dataloader(
    split="dev",
    codec_vocab=codec_vocab,
    codec_q_vocab=codec_q_vocab,
    config=erm_config,
    max_samples=5000,
    batch_size=32,
)

# Extract model A representations for probing
print(f"Extracting {model_a_name} representations for probing ({probe_representation})...")
erm_reps_probe, codec_labels, codec_q_labels = extract_representations_for_probing(
    erm_model,
    probe_loader,
    device,
    max_samples=5000,
    representation=probe_representation,
)
print(f"Extracted representation keys: {list(erm_reps_probe.keys())[:10]}")
print(f"Codec labels shape: {codec_labels.shape}, unique values: {np.unique(codec_labels).shape[0]}")


# In[ ]:


# Run domain probes on model A
print(f"\nRunning CODEC probes on {model_a_name}...")
erm_probe_inputs = {k: v for k, v in erm_reps_probe.items() if isinstance(k, int)}
if not erm_probe_inputs:
    raise RuntimeError(
        "Layer-wise probing requires integer layer keys. "
        f"Current probe_representation={probe_representation} produced keys={list(erm_reps_probe.keys())}"
    )

erm_codec_probes = layerwise_probing(
    erm_probe_inputs,
    codec_labels,
    classifier="logistic",
    cv_folds=5,
    seed=42,
)

print(f"{model_a_name} max leakage layer: {erm_codec_probes['max_leakage_layer']}")
print(f"{model_a_name} max accuracy: {erm_codec_probes['max_leakage_accuracy']:.4f}")


# In[ ]:


# Recreate dataloader and extract patched model representations
probe_loader = create_eval_dataloader(
    split="dev",
    codec_vocab=codec_vocab,
    codec_q_vocab=codec_q_vocab,
    config=erm_config,
    max_samples=5000,
    batch_size=32,
)

print(f"Extracting patched model representations for probing ({probe_representation})...")
patched_reps_probe, _, _ = extract_representations_for_probing(
    patched_model,
    probe_loader,
    device,
    max_samples=5000,
    representation=probe_representation,
)


# In[ ]:


# Run domain probes on patched model
print("Running CODEC probes on patched model...")
patched_probe_inputs = {k: v for k, v in patched_reps_probe.items() if isinstance(k, int)}
if not patched_probe_inputs:
    raise RuntimeError(
        "Layer-wise probing requires integer layer keys. "
        f"Current probe_representation={probe_representation} produced keys={list(patched_reps_probe.keys())}"
    )

patched_codec_probes = layerwise_probing(
    patched_probe_inputs,
    codec_labels,
    classifier="logistic",
    cv_folds=5,
    seed=42,
)

print(f"Patched Max leakage layer: {patched_codec_probes['max_leakage_layer']}")
print(f"Patched Max accuracy: {patched_codec_probes['max_leakage_accuracy']:.4f}")


# In[ ]:


# Print comparison
print("\n" + "="*70)
print("DOMAIN PROBE COMPARISON (CODEC)")
print("="*70)
print(f"{'Layer':<10} {f'{model_a_name} Acc':<18} {f'{patched_model_name} Acc':<22} {'Reduction':<12} {'Patched?':<10}")
print("-"*70)

for layer in sorted([k for k in erm_codec_probes['per_layer'].keys()]):
    erm_result = erm_codec_probes['per_layer'][layer]
    patched_result = patched_codec_probes['per_layer'][layer]

    erm_acc = erm_result.get('accuracy', float('nan'))
    patched_acc = patched_result.get('accuracy', float('nan'))

    if np.isfinite(erm_acc) and np.isfinite(patched_acc):
        reduction = erm_acc - patched_acc
        is_patched = "✓" if layer in divergent_layers else ""
        print(f"{layer:<10} {erm_acc:<18.4f} {patched_acc:<22.4f} {reduction:+12.4f} {is_patched:<10}")
    else:
        print(f"{layer:<10} {'skipped':<18} {'skipped':<22} {'-':<12}")

print("="*70)


# In[ ]:


# Plot probe accuracy comparison
layers_for_plot = sorted([k for k in erm_codec_probes['per_layer'].keys() if isinstance(k, int)])
erm_accs = []
patched_accs = []

for layer in layers_for_plot:
    erm_acc = erm_codec_probes['per_layer'][layer].get('accuracy', float('nan'))
    patched_acc = patched_codec_probes['per_layer'][layer].get('accuracy', float('nan'))
    erm_accs.append(erm_acc if np.isfinite(erm_acc) else 0)
    patched_accs.append(patched_acc if np.isfinite(patched_acc) else 0)

x = np.arange(len(layers_for_plot))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 6))
bars1 = ax.bar(x - width/2, erm_accs, width, label=model_a_name, color='steelblue')
bars2 = ax.bar(x + width/2, patched_accs, width, label=patched_model_name, color='coral')

# Highlight patched layers
for i, layer in enumerate(layers_for_plot):
    if layer in divergent_layers:
        ax.axvspan(i - 0.5, i + 0.5, alpha=0.2, color='yellow')

ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel('Domain Probe Accuracy', fontsize=12)
ax.set_title(
    f'Domain Probe Accuracy: {model_a_name} vs {patched_model_name}\n'
    '(Yellow = Patched Layers, Lower = More Domain Invariant)',
    fontsize=14,
)
ax.set_xticks(x)
ax.set_xticklabels(layers_for_plot)
ax.legend()
ax.set_ylim(0, 1)
ax.axhline(y=1/len(np.unique(codec_labels)), color='gray', linestyle='--', alpha=0.5, label='Chance level')

plt.tight_layout()
plt.show()


# ## 6. Results Summary

# In[ ]:


# Create summary table
results_df = pd.DataFrame([
    {
        "Model": model_a_name,
        "Eval EER": f"{erm_results['eer']:.2%}",
        "Min DCF": f"{erm_results['min_dcf']:.4f}",
        "Max Codec Probe Acc": f"{erm_codec_probes['max_leakage_accuracy']:.2%}",
        "Max Leakage Layer": erm_codec_probes['max_leakage_layer'],
        "Notes": "Base model"
    },
    {
        "Model": model_b_name,
        "Eval EER": f"{dann_results['eer']:.2%}",
        "Min DCF": f"{dann_results['min_dcf']:.4f}",
        "Max Codec Probe Acc": "n/a",
        "Max Leakage Layer": "n/a",
        "Notes": "Donor model"
    },
    {
        "Model": patched_model_name,
        "Eval EER": f"{patched_results['eer']:.2%}",
        "Min DCF": f"{patched_results['min_dcf']:.4f}",
        "Max Codec Probe Acc": f"{patched_codec_probes['max_leakage_accuracy']:.2%}",
        "Max Leakage Layer": patched_codec_probes['max_leakage_layer'],
        "Notes": f"Layers {divergent_layers} patched from {model_b_name}"
    },
])

print("\n" + "="*100)
print("RESULTS SUMMARY")
print("="*100)
print(results_df.to_string(index=False))
print("="*100)


# In[ ]:


# Save results to file
results_df.to_csv("rq4_results_summary.csv", index=False)
print("Results saved to rq4_results_summary.csv")

# Save CKA results
cka_df = pd.DataFrame([
    {"Layer": layer, "CKA": cka_results['per_layer'][layer]['cka']}
    for layer in sorted(cka_results['per_layer'].keys())
])
cka_df.to_csv("rq4_cka_results.csv", index=False)
print("CKA results saved to rq4_cka_results.csv")


# ## Conclusions
# 
# ### Key Findings
# 
# 1. **CKA Analysis**
#    - Compared model pair: `model_a_name` vs `model_b_name`
#    - Representation mode for CKA: `cka_representation`
#    - Most divergent patchable layers: `divergent_layers`
# 
# 2. **Performance Impact**
#    - Base model metric: `erm_results`
#    - Donor model metric: `dann_results`
#    - Patched model metric: `patched_results`
# 
# 3. **Domain Invariance**
#    - Base model max probe leakage: `erm_codec_probes['max_leakage_accuracy']`
#    - Patched model max probe leakage: `patched_codec_probes['max_leakage_accuracy']`
# 
# ### Scope Notes
# 
# - This notebook currently supports checkpoints whose `training.method` is `erm` or `dann`.
# - In this codebase, DANN applies the domain objective to pre-projection pooled features.
# - With frozen backbones, differences are expected mostly in post-backbone representations.
# 
# ### Trade-offs
# 
# - **Computational cost:** Patching runs donor + base during inference (roughly 2x forward passes).
# - **Flexibility:** You can target specific layers without retraining.
# - **Simplicity:** Useful as a lightweight intervention once donor checkpoints exist.
# 
# ### Future Work
# 
# 1. **Selective patching:** Patch only samples likely to benefit.
# 2. **Partial patching:** Interpolate between base and donor activations.
# 3. **Distillation:** Train one model to mimic patched behavior.
# 4. **Multi-layer analysis:** Study interactions across patch sets.

# In[ ]:


# Cleanup
if 'patched_model' in globals() and patched_model is not None:
    patched_model.remove_hooks()
    print("Hooks cleaned up.")
else:
    print("No patched model found; skipping hook cleanup.")

