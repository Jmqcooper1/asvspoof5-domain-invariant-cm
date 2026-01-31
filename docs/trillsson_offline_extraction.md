# TRILLsson Offline Extraction Workflow

TRILLsson is a non-semantic audio embedding model from Google that requires TensorFlow and TensorFlow Hub. Since these dependencies are heavy and may not be available on all compute environments (like Snellius), we recommend an offline extraction workflow.

## Overview

The workflow consists of three steps:
1. **Extract embeddings locally** with TensorFlow
2. **Upload to compute cluster** 
3. **Train classifier on cluster** (no TensorFlow required)

## Step 1: Local Extraction

On your local machine with TensorFlow installed:

```bash
# Install dependencies
pip install tensorflow tensorflow-hub torchaudio numpy pandas

# Extract embeddings for all splits
python scripts/extract_trillsson.py --split train --output-dir data/features/trillsson
python scripts/extract_trillsson.py --split dev --output-dir data/features/trillsson
python scripts/extract_trillsson.py --split eval --output-dir data/features/trillsson
```

This creates:
- `data/features/trillsson/train.npy` - Embeddings array (N, 1024)
- `data/features/trillsson/train_metadata.csv` - Sample metadata
- Similar files for dev and eval splits

## Step 2: Upload to Cluster

Transfer the extracted features to your compute environment:

```bash
# Upload to Snellius (adjust paths as needed)
rsync -avz data/features/trillsson/ snellius:$ASVSPOOF5_ROOT/../features/trillsson/

# Or for other clusters
scp -r data/features/trillsson/ user@cluster:/path/to/features/
```

## Step 3: Train Classifier

On the compute cluster (no TensorFlow needed):

```bash
# The training script only loads numpy embeddings
python scripts/train_trillsson.py --classifier logistic
python scripts/train_trillsson.py --classifier mlp --wandb
```

## Benefits

- **No TensorFlow on cluster**: Avoid heavy dependencies and potential version conflicts
- **Faster iteration**: Embedding extraction is one-time, classifier training is fast
- **Resource efficiency**: TensorFlow extraction can use local GPU, cluster focuses on training
- **Reproducibility**: Embeddings are fixed, ensuring consistent results across runs

## TRILLsson Models

Available model variants:
- `trillsson1` through `trillsson5`
- Default is `trillsson3`
- Use `--model` flag to specify different variants

## File Formats

- **Embeddings**: NumPy `.npy` files, shape `(n_samples, embedding_dim)`
- **Metadata**: CSV files with sample information matching the original manifests
- **Embedding dimension**: 1024 for all TRILLsson variants

## Troubleshooting

**TensorFlow not available locally?**
- Use Google Colab or another cloud environment for extraction
- Consider using pre-extracted embeddings if available

**Large file transfers?**
- Compress before upload: `tar -czf trillsson.tar.gz data/features/trillsson/`
- Use rsync with compression: `rsync -avz`

**Embedding dimension mismatch?**
- Check model variant - all should produce 1024-dim embeddings
- Verify files weren't corrupted during transfer