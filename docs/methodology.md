# Methodology

This document describes the core methods used in this project.

## Overview

We compare two training paradigms for speech deepfake detection:

1. **ERM (Empirical Risk Minimization):** Standard supervised training
2. **DANN (Domain-Adversarial Neural Network):** Training with domain invariance objective

## Model Architecture

```
                    ┌─────────────────────┐
                    │   SSL Backbone      │
                    │  (WavLM / W2V2)     │
                    │     [frozen]        │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Layer Selection   │
                    │   + Mixing Weights  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Projection Head   │
                    │      (MLP)          │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼─────────┐     │      ┌─────────▼─────────┐
    │  Task Classifier  │     │      │ Domain Discriminator│
    │  (bonafide/spoof) │     │      │  (CODEC, CODEC_Q)  │
    └───────────────────┘     │      └───────────────────┘
                              │               ▲
                              │               │
                              │      ┌────────┴────────┐
                              │      │ Gradient Reversal│
                              │      │     Layer (GRL)  │
                              │      └─────────────────┘
```

## SSL Backbones

### WavLM Base+

- 12 transformer layers
- 768-dim hidden size
- Pre-trained on 94k hours of speech

### Wav2Vec 2.0 Base

- 12 transformer layers
- 768-dim hidden size
- Pre-trained on LibriSpeech 960h

### Layer Selection Strategy

Based on recent findings that lower SSL layers are highly discriminative for deepfake detection:

1. Extract hidden states from all layers
2. Learn weighted combination (trainable scalar weights)
3. Optionally restrict to lower K layers

```python
class LayerWeightedPooling(nn.Module):
    def __init__(self, num_layers: int, init_lower: bool = True):
        super().__init__()
        # Initialize with higher weights for lower layers
        weights = torch.linspace(1.0, 0.1, num_layers) if init_lower else torch.ones(num_layers)
        self.weights = nn.Parameter(weights)
    
    def forward(self, hidden_states: list[torch.Tensor]) -> torch.Tensor:
        # hidden_states: list of [B, T, D] tensors
        weights = F.softmax(self.weights, dim=0)
        stacked = torch.stack(hidden_states, dim=0)  # [L, B, T, D]
        weighted = (stacked * weights.view(-1, 1, 1, 1)).sum(dim=0)
        return weighted
```

## Domain-Adversarial Training (DANN)

### Domain Labels: Training vs Evaluation

**Critical distinction:**

| Context | Domain Labels | Source |
|---------|---------------|--------|
| **Training** | Synthetic codec domains | Codec augmentation (MP3, AAC, OPUS, etc.) |
| **Evaluation** | Protocol codec domains | ASVspoof5 metadata (C01-C11) |

During DANN training, the domain discriminator learns to predict **synthetic** codec domains
created by our augmentation pipeline. This is necessary because train/dev sets have no native
codec diversity (all samples are uncoded).

During evaluation, per-domain metrics use **protocol** metadata (CODEC, CODEC_Q) from the
ASVspoof5 eval set. These are **different** domain taxonomies.

### Gradient Reversal Layer

During forward pass: identity function
During backward pass: negate gradients by factor λ

```python
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None
```

### Multi-Head Domain Discriminator

Separate classification heads for CODEC and CODEC_Q:

```python
class MultiHeadDomainDiscriminator(nn.Module):
    def __init__(self, input_dim: int, num_codecs: int, num_codec_qs: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.codec_head = nn.Linear(256, num_codecs)
        self.codec_q_head = nn.Linear(256, num_codec_qs)
    
    def forward(self, x):
        shared = self.shared(x)
        return self.codec_head(shared), self.codec_q_head(shared)
```

### Training Objective

```
L_total = L_task + λ * (L_codec + L_codec_q)
```

Where:
- `L_task`: Cross-entropy for bonafide/spoof classification
- `L_codec`: Cross-entropy for CODEC prediction (after GRL)
- `L_codec_q`: Cross-entropy for CODEC_Q prediction (after GRL)
- `λ`: Adversarial weight (hyperparameter to tune)

## Baselines

### ERM Baseline

Same architecture without domain discriminator:

```
L_total = L_task
```

### Non-Semantic Baseline (TRILL/TRILLsson)

- Extract paralinguistic embeddings (frozen)
- Train lightweight classifier (logistic regression or small MLP)
- No fine-tuning of embedding model

## Interpretability Methods

### Layer-wise Domain Probing

1. Extract embeddings from each transformer layer
2. Train linear probe to predict CODEC/CODEC_Q
3. Compare probe accuracy: ERM vs DANN
4. Lower accuracy after DANN = successful domain information removal

### Representation Similarity (CKA)

Compare layer representations between ERM and DANN models:

```python
def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between two representations."""
    ...
```

### Activation Patching

1. Identify layers/components with high domain probe accuracy
2. Replace activations from DANN model into ERM model
3. Measure change in:
   - Domain probe accuracy (should decrease)
   - Task performance (should be maintained)

## Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| Learning rate | 1e-4 | AdamW optimizer |
| Batch size | 32 | Adjust for GPU memory |
| λ (adversarial weight) | 0.1 | Sweep: [0.01, 0.1, 0.5, 1.0] |
| Projection dim | 256 | After layer pooling |
| Dropout | 0.1 | In projection head |
| Max epochs | 50 | With early stopping |

## Known Domain Mismatch

The synthetic codec augmentation does not perfectly replicate ASVspoof5 eval codecs.

### Synthetic vs ASVspoof5 Codec Mapping

| Synthetic Codec | ASVspoof5 Codec IDs | Coverage |
|-----------------|---------------------|----------|
| NONE | `-` (uncoded) | ✅ Exact match |
| MP3 | C05 (mp3_wb), C07 (partial) | ⚠️ Partial |
| AAC | C06 (m4a_wb) | ✅ Good match |
| OPUS | C01 (opus_wb), C08 (opus_nb) | ✅ Good match |
| SPEEX | C03 (speex_wb), C10 (speex_nb) | ✅ Good match (if ffmpeg supports) |
| AMR | C02 (amr_wb), C09 (amr_nb) | ✅ Good match (if ffmpeg supports) |

### Not Covered by Synthetic Augmentation

| Eval Codec | Description | Why Not Simulated |
|------------|-------------|-------------------|
| **C04** | Encodec (neural codec) | Fundamentally different artifacts; requires neural codec implementation |
| **C07** | MP3 + Encodec cascade | Compound degradation; only MP3 portion approximated |
| **C11** | Device/channel (BT, cable, MST) | Acoustic simulation, not codec; would require room impulse responses |

### Expected Behavior

DANN should show improvements for codecs covered by synthetic augmentation (C01, C02, C03, C05, C06, C08, C09, C10).
Limited or no improvement expected for C04, C07, C11.

## Reproducibility

### Random Seed Policy

Seeds are set for:
- `torch.manual_seed(seed)`
- `torch.cuda.manual_seed_all(seed)`
- `numpy.random.seed(seed)`
- `random.seed(seed)`

### DataLoader Worker Seeding

Augmentation uses `random.random()` which requires per-worker seeding:

```python
def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)
    numpy.random.seed(worker_seed + worker_id)
```

This ensures reproducible augmentation across runs.

### Deterministic Evaluation

- **Training mode:** Random crop for temporal augmentation
- **Evaluation mode:** Center crop for deterministic inference

### Cache Determinism

If augmentation caching is enabled:
- Cache key = hash(original_path, codec, quality)
- Same sample + codec + quality → same cached file
- Cache writes are atomic (temp file + `os.replace()`)

### Multi-Seed Experiments

For statistical reliability, run experiments with multiple seeds:

```bash
for seed in 42 123 456; do
    python scripts/train.py --config configs/wavlm_dann.yaml \
        --seed $seed --name wavlm_dann_seed${seed}
done
```
