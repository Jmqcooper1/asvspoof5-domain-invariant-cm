# Domain-Invariant Speech Deepfake Detection with Domain-Adversarial Training and Mechanistic Analysis

## Working Title

Domain-Invariant Speech Deepfake Detection with Domain-Adversarial Training and Mechanistic Analysis

## Problem and Motivation

Speech deepfake detectors often generalize poorly in real conditions because they learn shortcut cues tied to nuisance factors such as codec compression, recording channel/device, and dataset collection artifacts. ASVspoof 5 is a recent benchmark explicitly designed to reflect these modern challenges: crowdsourced audio with diverse recording conditions, adversarial attacks at scale, and codec/neural compression effects. This thesis treats speech deepfake detection as a domain generalization problem: a reliable detector should predict bonafide vs spoof while being robust to domain shifts such as codec and codec quality.

## Objective and Outcomes

The thesis aims to develop and analyze a domain-invariant deepfake detector for ASVspoof 5 Track 1 (stand-alone bonafide vs spoof detection). The intended outcomes are:

1. A detection pipeline whose out-of-domain performance under codec/quality shifts improves over standard training.
2. Quantitative evidence that the learned representations contain less codec/quality information ("domain leakage") while retaining deepfake-discriminative signal.
3. A mechanistic/interpretability analysis identifying where domain information is encoded and how domain-adversarial training changes it, including a limited causal intervention study (activation patching near the head and/or a small number of identified layers).

## Dataset and Protocol

Primary dataset: ASVspoof 5 (Track 1). Data will be obtained from the official Zenodo release (with Hugging Face as a mirror). Experiments will follow the ASVspoof 5 evaluation plan and use the official evaluation package for metrics and comparability.

### Parsing and Metadata (Track 1)

Although the files are named `.tsv`, the ASVspoof 5 README specifies that the protocol files are **space-separated**; therefore, they will be parsed using **whitespace splitting** (do not hardcode tab separation) and without hardcoding the number of columns. For Track 1, the protocol rows contain (as per the README):

- `SPEAKER_ID`
- `FLAC_FILE_NAME`
- `SPEAKER_GENDER`
- `CODEC`
- `CODEC_Q`
- `CODEC_SEED`
- `ATTACK_TAG`
- `ATTACK_LABEL`
- `KEY` (bonafide/spoof)
- `TMP`

### Leakage Control via CODEC_SEED

For any custom domain-held-out analyses (e.g., leave-one-codec-out within train/dev), we will prevent leakage between near-duplicate original/coded utterances by grouping samples by `CODEC_SEED` (and/or by the original utterance id) when constructing splits, so that coded variants of the same seed do not appear in both "train" and "test" subsets.

## Domain Definition (Option 1A)

Codec-related domains will be defined using the provided metadata:

- `CODEC` (codec type / codec family)
- `CODEC_Q` (codec quality configuration)

Domain handling will use a **multi-head discriminator** predicting `CODEC` and `CODEC_Q` as separate outputs, encouraging invariance to both.

## Method Overview

I will compare standard empirical risk minimization (ERM) training against Domain-Adversarial Neural Network (DANN) training (gradient reversal layer) in a controlled setting.

Model components:

- SSL backbone feature extractor
- Deepfake classification head (bonafide vs spoof)
- Multi-head domain discriminator predicting `CODEC` and `CODEC_Q` from the same intermediate representation (via gradient reversal)

## Backbones and Fine-tuning Strategy (Option 2B)

To ensure conclusions are not backbone-specific, the ERM vs DANN comparison will be performed for two SSL backbones:

- WavLM (Base / Base+)
- Wav2Vec 2.0 (Base)

Large variants are optional extensions if they provide a meaningful robustness gain.

Efficiency-first training plan:

- Start with frozen SSL backbones.
- Use selected hidden-layer outputs (with emphasis on lower layers, motivated by recent layer-wise findings that lower SSL layers can be highly discriminative for deepfake detection).
- Learn a trainable layer-mixing plus a small projection head; apply DANN to the projection output.

Optional extension: unfreeze only a small portion of the backbone (e.g., last N blocks or lightweight adapters) to test whether invariance improves further.

## Baselines

1. ERM baseline for each backbone (same architecture without domain adversary).
2. DANN model for each backbone (same architecture with multi-head domain discriminator).
3. Non-semantic baseline: TRILL/TRILLsson-style paralinguistic embeddings (frozen) with a lightweight classifier. This provides a contrast to semantic SSL backbones and tests whether robustness can be achieved with representations not primarily trained for linguistic content.
4. Official ASVspoof baselines (e.g., AASIST / RawNet-family baselines provided by ASVspoof resources) will be used as reference points and to validate the evaluation pipeline.

## Hypotheses

**H1:** Domain-adversarial training using codec and codec-quality domains reduces domain leakage in representations and improves out-of-domain performance under codec/quality shift compared to ERM, while maintaining competitive in-domain performance.

**H2:** Domain leakage is concentrated in specific layers/components; DANN suppresses or redistributes this information while preserving deepfake-relevant information.

**H3:** The robustness and leakage-reduction effects are consistent across WavLM and Wav2Vec 2.0, but the layer-wise localization of leakage differs between backbones.

## Research Questions

**RQ1:** Does DANN improve cross-domain generalization for bonafide vs spoof detection under `CODEC`/`CODEC_Q` shift on ASVspoof 5 Track 1 compared to ERM?

**RQ2:** What is the trade-off between deepfake detection performance and domain invariance as the adversarial loss weight is varied?

**RQ3:** Where is codec/quality domain information encoded across layers and components, and how does DANN change this distribution?

**RQ4:** Can targeted interventions (limited activation patching near the head and/or a small number of identified layers) reduce domain leakage and improve robustness without full retraining?

## Evaluation

Primary metrics will follow ASVspoof 5 Track 1: **minDCF** (primary) and **EER** (secondary), with **Cllr/actDCF** reported where feasible for calibration-aware analysis.

Domain invariance will be quantified using:

- domain probe accuracy (predicting `CODEC` and `CODEC_Q` from frozen representations at different layers),
- domain discriminator accuracy (during/after training),
- in-domain vs out-of-domain performance gap under codec/quality-held-out analyses.

Statistical reliability will be addressed with multiple random seeds where feasible and confidence intervals (e.g., bootstrap on evaluation scores).

## Interpretability / Mechanistic Analysis

The interpretability focus is representation-centric rather than saliency-only:

- Layer-wise probing to locate domain leakage
- Representation comparisons between ERM and DANN models (e.g., CKA)
- Limited activation patching experiments to test whether swapping domain-heavy activations from DANN into ERM reduces domain leakage and affects detection decisions

## Key References (with URLs)

### Benchmarks and Evaluation (ASVspoof 5)

- Wang, X., et al. (2024). *ASVspoof 5: crowdsourced speech data, deepfakes, and adversarial attacks at scale.* ASVspoof Workshop 2024.
  Paper (ISCA Archive): https://www.isca-archive.org/asvspoof_2024/wang24_asvspoof.html
  PDF: https://www.isca-archive.org/asvspoof_2024/wang24_asvspoof.pdf
  DOI: https://doi.org/10.21437/ASVspoof.2024-1

- Wang, X., et al. (2025). *ASVspoof 5: Design, Collection and Validation of Resources for Spoofing, Deepfake, and Adversarial Attack Detection Using Crowdsourced Speech.* arXiv:2502.08857
  arXiv: https://arxiv.org/abs/2502.08857
  PDF: https://arxiv.org/pdf/2502.08857
  Dataset (Zenodo): https://zenodo.org/records/14498691
  Dataset mirror (Hugging Face): https://huggingface.co/datasets/jungjee/asvspoof5

- Wang, X., et al. (2026). *ASVspoof 5: Evaluation of Spoofing, Deepfake, and Adversarial Attack Detection Using Crowdsourced Speech.* arXiv:2601.03944
  arXiv: https://arxiv.org/abs/2601.03944
  PDF: https://arxiv.org/pdf/2601.03944

### Robustness and Modern Failure Modes

- Zhang, Z., et al. (2024/2025). *I Can Hear You: Selective Robust Training for Deepfake Audio Detection.* (ICLR 2025; DeepFakeVox-HQ; F-SAT). arXiv:2411.00121
  arXiv: https://arxiv.org/abs/2411.00121
  PDF (authors' page): https://www.cs.columbia.edu/~junfeng/papers/fsat-iclr25.pdf
  OpenReview: https://openreview.net/forum?id=2GcR9bO620

### Surveys / Overviews

- Li, M., Ahmadiadli, Y., & Zhang, X. P. (2024). *A Survey on Speech Deepfake Detection.* arXiv:2404.13914
  https://arxiv.org/abs/2404.13914
- Delgado, H., et al. (2022). *ASVspoof 2021: accelerating progress in spoofed and deepfake speech detection.* arXiv:2210.02437
  https://arxiv.org/abs/2210.02437

### Domain-Adversarial Training

- Ganin, Y., et al. (2016). *Domain-Adversarial Training of Neural Networks.* JMLR 2016.
  arXiv: https://arxiv.org/abs/1505.07818
  JMLR: https://www.jmlr.org/papers/v17/15-239.html
- Kim, J. W., et al. (2025). *Domain adversarial training for mitigating gender bias in speech-based mental health detection.* arXiv:2505.03359
  https://arxiv.org/abs/2505.03359

### SSL Backbones

- Chen, S., et al. (2022). *WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing.* arXiv:2110.13900
  https://arxiv.org/abs/2110.13900
- Baevski, A., et al. (2020). *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.* NeurIPS 2020. arXiv:2006.11477
  https://arxiv.org/abs/2006.11477
- Hsu, W.-N., et al. (2021). *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units.* arXiv:2106.07447
  https://arxiv.org/abs/2106.07447

### SSL for Deepfake Detection

- Guo, Y., et al. (2024). *Audio deepfake detection with self-supervised WavLM and multi-fusion attentive classifier.* ICASSP 2024 (IEEE Xplore).
  https://ieeexplore.ieee.org/document/10447923/
- Combei, D., et al. (2024). *WavLM model ensemble for audio deepfake detection.* arXiv:2408.07414
  https://arxiv.org/abs/2408.07414
- El Kheir, Y., et al. (2025). *Comprehensive Layer-wise Analysis of SSL Models for Audio Deepfake Detection.* Findings of NAACL 2025 (ACL Anthology).
  PDF: https://aclanthology.org/2025.findings-naacl.227.pdf

### Non-semantic Baseline

- Das, A., et al. (2025). *Generalizable Audio Spoofing Detection using Non-Semantic Representations.* Interspeech 2025. arXiv:2509.00186
  arXiv: https://arxiv.org/abs/2509.00186
  ISCA Archive entry: https://www.isca-archive.org/interspeech_2025/das25_interspeech.html

### Probing / Interpretability in Speech Anti-spoofing

- Liu, X., et al. (2025). *Explaining Speaker and Spoof Embeddings via Probing.* (ICASSP 2025; also on arXiv). arXiv:2412.18191
  https://arxiv.org/abs/2412.18191

### Representation Comparison and Attribution

- Kornblith, S., et al. (2019). *Similarity of Neural Network Representations Revisited.* ICML 2019. arXiv:1905.00414
  https://arxiv.org/abs/1905.00414
- Sundararajan, M., et al. (2017). *Axiomatic Attribution for Deep Networks (Integrated Gradients).* ICML 2017. arXiv:1703.01365
  https://arxiv.org/abs/1703.01365

### Component-level Intervention Precedent

- Chintam, A., et al. (2023). *Identifying and adapting transformer-components responsible for gender bias in an English language model.* arXiv:2310.12611
  https://arxiv.org/abs/2310.12611

### Audio Deepfake Explainability (Optional)

- Salvi, D., Bestagini, P., & Tubaro, S. (2023). *Towards Frequency Band Explainability in Synthetic Speech Detection.* (IEEE Xplore).
  https://ieeexplore.ieee.org/document/10289804/
- Grinberg, P., et al. (2025). *What Does an Audio Deepfake Detector Focus on? A Relevancy-based XAI Method.* (IEEE Xplore).
  https://ieeexplore.ieee.org/document/10887568
- Lee, S., et al. (2025). *iWAX: interpretable Wav2vec-AASIST-XGBoost framework.* Scientific Reports 2025.
  https://www.nature.com/articles/s41598-025-24361-5
