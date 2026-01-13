You are an expert ML engineer/researcher. Build a working, reproducible codebase (Python)
for my MSc thesis project:

Goal
- Implement and evaluate Domain-Adversarial Training (DANN) for domain-invariant speech
  deepfake detection on ASVspoof 5 Track 1.
- Compare ERM vs DANN for two SSL backbones (Option 2B):
  1) WavLM Base/Base+ (HF: microsoft/wavlm-base-plus)
  2) Wav2Vec 2.0 Base (HF: facebook/wav2vec2-base)
- Use domains (Option 1A): CODEC and CODEC_Q (multi-head domain discriminator).
- Include a non-semantic baseline: TRILL/TRILLsson embeddings + lightweight classifier
  (offline feature extraction is OK). Provide a fallback classical baseline if TRILLsson
  is hard (LFCC-LCNN or LFCC-GMM from ASVspoof baselines).
- Include mechanistic/interpretability analyses:
  1) Layer-wise domain probes (CODEC, CODEC_Q decodability across SSL layers)
  2) Representation similarity (CKA) ERM vs DANN
  3) Limited activation patching near the head and optionally 1–2 identified layers

Constraints / Preferences
- Single GPU preferred (A100 40GB or H100 96GB available). Be efficient; base models as
  default. Large variants optional.
- No timeline in outputs.
- Reproducible experiments: configs, seeds, saved metrics/tables/plots.
- Use PyTorch + torchaudio + transformers; scikit-learn for probes; optional wandb.

Dataset: ASVspoof 5 Track 1
- I have the official Zenodo package structure like:
  data/asvspoof5/
    ASVspoof5_protocols/ (or extracted files)
    flac_T/ (train audio)
    flac_D/ (dev audio)
    flac_E_eval/ (eval audio)
- Protocol files (IMPORTANT): they are named .tsv but README says they are
  space-separated / whitespace-separated. Parse by splitting on any whitespace.
- Track 1 protocol row contains these fields in order:
  SPEAKER_ID FLAC_FILE_NAME SPEAKER_GENDER CODEC CODEC_Q CODEC_SEED ATTACK_TAG
  ATTACK_LABEL KEY TMP
  where KEY ∈ {bonafide, spoof}.
- Use CODEC and CODEC_Q as domain labels. Values may be "-" => treat as "NONE".
- Leakage control: for any custom splits inside train/dev (e.g., held-out codec), group
  by CODEC_SEED so coded/original variants don’t leak across splits.

Repo requirements (create these files/modules)
- Provide a clean project structure:
  src/
    data/
      asvspoof5.py          # protocol parsing, manifests, datasets
      audio.py              # audio loading, padding, batching, optional aug
    models/
      grl.py                # GradientReversalLayer
      heads.py              # pooling + projection + task/domain heads
      dann.py               # model wrapper: backbone + heads
      nonsemantic.py        # TRILLsson baseline wrapper (training uses cached feats)
    training/
      loop.py               # train/val loops, checkpointing
      losses.py             # task loss, domain losses, weighting
      sched.py              # lr schedulers
    evaluation/
      metrics.py            # EER, minDCF (and helpers)
      reports.py            # per-domain breakdown tables, bootstraps
    analysis/
      probes.py             # layer-wise probes CODEC/CODEC_Q
      cka.py                # representation similarity
      patching.py           # activation patching utilities/experiments
    utils/
      config.py             # YAML config loading + validation
      seed.py               # deterministic seeds
      io.py                 # saving/loading manifests, npy, parquet
  scripts/
    prepare_asvspoof5.py    # build manifest from protocol files
    train.py                # ERM vs DANN training entrypoint
    evaluate.py             # compute metrics overall + per-domain
    probe_domain.py         # layer-wise probes and plots
    run_cka.py              # CKA analysis
    run_patching.py         # limited activation patching experiment
    extract_trillsson.py    # offline embedding extraction to npy (TF/TFHub ok)
  configs/
    wavlm_erm.yaml
    wavlm_dann.yaml
    w2v2_erm.yaml
    w2v2_dann.yaml
    trillsson_baseline.yaml
  README.md with exact commands to reproduce.
- Add minimal unit tests:
  tests/test_protocol_parse.py
  tests/test_dataset_shapes.py

Implementation details (be concrete)
1) Data pipeline
- Implement protocol parser that:
  - reads whitespace-separated rows
  - assigns columns exactly as above
  - creates absolute paths to FLAC files based on split prefix:
    FLAC_FILE_NAME begins with T_/D_/E_ ; map to flac_T/, flac_D/, flac_E_eval/
  - maps KEY: bonafide->0, spoof->1
  - encodes CODEC and CODEC_Q to integer class ids (store vocab dicts)
  - stores SPEAKER_ID, ATTACK_LABEL, ATTACK_TAG, CODEC_SEED for analysis
- Dataloader:
  - load FLAC with torchaudio (16 kHz)
  - chunk or pad to fixed length (configurable, e.g., 4s or 6s); implement:
    - random crop for train, center crop for eval
    - if shorter, pad with zeros
  - return: waveform [1, T], y_task, y_codec, y_codec_q, metadata (ids)

2) Model: SSL backbone + layer mixing + pooling + heads
- Use HF Transformers with output_hidden_states=True.
- Backbone frozen by default.
- Layer selection:
  - choose lower K transformer layers (config: layers=[1..K] or indices)
  - implement trainable scalar weights over selected layers:
    w = softmax(alpha); h = sum_i w_i * hidden_states[i]
- Pooling:
  - statistics pooling over time: concat(mean(h), std(h)) -> [2*D]
- Projection head:
  - small MLP: [2D] -> 256 (ReLU, dropout) -> 256
- Task head:
  - linear 256 -> 2 logits (bonafide/spoof)
- Domain discriminator (multi-head):
  - shared MLP 256 -> 256 -> ReLU -> dropout
  - head_codec: 256 -> n_codec
  - head_codec_q: 256 -> n_codec_q
- Gradient Reversal Layer (GRL):
  - apply GRL to the representation fed into discriminator
  - GRL strength lambda is configurable and schedulable (e.g., ramp up)

3) Training objectives
- ERM: task_loss = CrossEntropy(task_logits, y_task)
- DANN:
  task_loss + λ_dom * (loss_codec + loss_codec_q)
  where discriminator receives GRL(repr) so feature extractor + head learn to confuse it.
- Provide config toggles:
  - freeze_backbone: true/false
  - unfreeze_last_n_blocks: optional extension
  - lambda_dom schedule: constant or linear warmup
- Handle class imbalance if present:
  - optional class weights for task loss
  - report class priors

4) Evaluation
- Primary metrics: minDCF (primary), EER (secondary).
  - Implement EER from scores and labels.
  - Implement minDCF with configurable costs/priors OR integrate official ASVspoof5
    scoring scripts if available; at minimum provide working minDCF implementation and
    document it.
- Produce:
  - overall metrics on dev.track_1 and eval.track_1 (eval if labels available)
  - per-domain breakdown tables:
    - metrics by CODEC
    - metrics by CODEC_Q
    - optionally by ATTACK_LABEL
- Domain shift analyses:
  - “held-out codec” within train/dev as an extra experiment:
    - remove one CODEC from training manifest (grouped by CODEC_SEED)
    - train on remaining; evaluate only on held-out CODEC subset (dev)
    - repeat for top N codecs by count (configurable)

5) Probing (interpretability)
- After training ERM and DANN models:
  - extract representations per layer (same layer indices used for mixing AND optionally
    a few other layers) on a subset of dev
  - train linear probes (sklearn LogisticRegression) to predict:
    - CODEC
    - CODEC_Q
  - report probe accuracy vs layer index for ERM vs DANN (plot + CSV)
  - also probe on projection head output (should be less decodable for DANN)

6) Representation similarity (CKA)
- Compute linear CKA between ERM and DANN representations per layer on the same inputs.
- Output a heatmap + CSV matrix.

7) Limited activation patching (mechanistic)
- Keep scope limited and causal:
  - Identify “domain-heavy units” using probe weights on the projection representation
    (or discriminator gradients).
  - For a batch, run ERM forward pass; replace (patch) selected units in ERM repr with
    the corresponding units from DANN repr; re-run heads.
  - Measure:
    - change in domain probe accuracy (or discriminator confidence)
    - change in task score/logits and metrics on a small evaluation subset
- Optional: patch one selected transformer layer output (only 1–2 layers, not all).

8) Non-semantic baseline (TRILL/TRILLsson)
- Provide script extract_trillsson.py that:
  - reads manifest (paths) and writes embeddings to
    data/features/trillsson/{split}.npy and a metadata CSV (file_id -> row index)
  - uses TF/TFHub or an equivalent reliable source (document install)
- Training for TRILLsson:
  - simple classifier (logreg or small MLP)
  - evaluate same metrics and per-domain breakdown (CODEC/CODEC_Q)

9) Reproducibility and usability
- All experiments driven by YAML configs (configs/*.yaml).
- Save outputs under runs/{exp_name}/:
  - config copy, checkpoints, metrics.json, tables/*.csv, plots/*.png
- Provide CLI examples in README:
  - prepare dataset
  - train ERM/DANN for each backbone
  - run evaluation + per-domain reports
  - run probes, CKA, patching
- Ensure code runs without editing paths by using a single env var:
  ASVSPOOF5_ROOT=/path/to/data/asvspoof5

Acceptance criteria (definition of done)
- I can run:
  1) prepare_asvspoof5.py to build manifests for train/dev/eval track 1
  2) train.py for wavlm_erm.yaml and wavlm_dann.yaml (and w2v2 equivalents)
  3) evaluate.py to produce overall + per-CODEC + per-CODEC_Q tables
  4) probe_domain.py to produce layer-wise probe plots ERM vs DANN
  5) run_cka.py to produce a CKA heatmap ERM vs DANN
  6) run_patching.py to produce a small report showing patching changes domain
     leakage and (ideally) affects robustness in the expected direction
  7) (optional) extract_trillsson.py + trillsson_baseline.yaml to run the non-semantic
     baseline end-to-end

Now generate the codebase skeleton + key modules with working implementations and
reasonable defaults. Prioritize correctness, clarity, and reproducibility over exotic tricks.