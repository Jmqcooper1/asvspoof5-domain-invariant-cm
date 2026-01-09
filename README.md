# ASVspoof 5 Domain-Invariant CM (Track 1)

This repository contains code and experiments for domain-invariant speech deepfake
detection on ASVspoof 5 Track 1 (stand-alone bonafide vs spoof). The core method
compares standard training (ERM) against Domain-Adversarial Neural Networks
(DANN) using a gradient reversal layer, with mechanistic analysis via domain probes
and limited activation patching.

## Project Goals

- Improve robustness under codec / codec-quality domain shift (CODEC, CODEC_Q).
- Quantify and reduce domain leakage in learned representations.
- Compare two SSL backbones (WavLM vs Wav2Vec 2.0).
- Include a non-semantic baseline (TRILL/TRILLsson embeddings + lightweight head).
- Provide representation-centric interpretability (layer-wise probes, representation
  similarity, limited patching).

## Dataset: ASVspoof 5 (Track 1)

- Official site: https://www.asvspoof.org/
- Zenodo record (ASVspoof 5): https://zenodo.org/records/14498691
- HF mirror (optional): https://huggingface.co/datasets/jungjee/asvspoof5

**Important parsing note:**
ASVspoof 5 protocol files are whitespace-separated even though they use a `.tsv`
extension. Parse by splitting on whitespace and do not hardcode column counts.

See [docs/dataset.md](docs/dataset.md) for exact setup instructions.

## Installation (install uv first)

```bash
uv sync --all-extras
```

For minimal install:

```bash
uv sync
```

## Quickstart

1. Download and prepare data:

```bash
bash scripts/download_asvspoof5.sh
bash scripts/unpack_asvspoof5.sh
uv run python scripts/make_manifest.py
```

2. Train ERM baseline:

```bash
uv run python scripts/train.py --config configs/train/erm.yaml --model configs/model/wavlm_base.yaml
```

3. Train DANN:

```bash
uv run python scripts/train.py --config configs/train/dann.yaml --model configs/model/wavlm_base.yaml
```

4. Evaluate:

```bash
uv run python scripts/evaluate.py --split dev --track 1
```

5. Run probes:

```bash
uv run python scripts/run_probes.py --split dev --track 1
```

## Repository Structure

```
asvspoof5-domain-invariant-cm/
├── configs/           # YAML configs for data, model, train, eval
├── docs/              # Documentation (proposal, dataset, methodology)
├── scripts/           # Entrypoints (train, evaluate, probes, patching)
└── src/               # Python package with data, models, evaluation, analysis
```

## Reproducibility

Each run should store:
- the full resolved config
- training logs
- predictions (per-utterance scores)
- metrics (minDCF, EER, optional calibration metrics)
- git commit hash

Do not commit datasets or generated artifacts to git.

## References

- ASVspoof 5 (workshop): https://www.isca-archive.org/asvspoof_2024/wang24_asvspoof.html
- ASVspoof 5 resources paper (arXiv): https://arxiv.org/abs/2502.08857
- ASVspoof 5 evaluation paper (arXiv): https://arxiv.org/abs/2601.03944
- DANN (JMLR): https://www.jmlr.org/papers/v17/15-239.html
- WavLM (arXiv): https://arxiv.org/abs/2110.13900
- Wav2Vec 2.0 (arXiv): https://arxiv.org/abs/2006.11477

## License

MIT
