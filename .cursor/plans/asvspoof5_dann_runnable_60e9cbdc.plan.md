---
name: ASVspoof5 DANN runnable
overview: Make the current repo fully runnable/reproducible for ASVspoof5 Track1 ERM vs DANN (WavLM Base+, W2V2 Base) with CODEC/CODEC_Q domains, plus probes/CKA/patching + TRILLsson baseline, using ASVSPOOF5_ROOT and your bonafide=0 spoof=1 convention.
todos:
  - id: audit_and_align_data
    content: Replace manifest+dataset handling with ASVSPOOF5_ROOT-based absolute paths, '-'→'NONE', bonafide=0 spoof=1 labels, domain vocab persistence, and torchaudio crop/pad + collate with attention masks.
    status: completed
  - id: model_pipeline_alignment
    content: "Refactor backbone layer selection and DANN/ERM forward path to: layer_mix -> stats_pool -> projection -> repr; domain heads consume GRL(repr); return repr + all_hidden_states for analysis."
    status: completed
    dependencies:
      - audit_and_align_data
  - id: training_loop_and_cli
    content: Add training package (loop/sched/losses) and implement scripts/train.py to run ERM and DANN end-to-end with checkpoints/logging/reproducibility artifacts under runs/.
    status: completed
    dependencies:
      - model_pipeline_alignment
  - id: evaluation_and_reports
    content: Implement scripts/evaluate.py plus evaluation/reports.py; ensure metrics support bonafide=0 spoof=1 convention; output predictions.tsv, metrics.json, and per-domain tables.
    status: completed
    dependencies:
      - training_loop_and_cli
  - id: analysis_probes_cka_patching
    content: Implement probe_domain.py, run_cka.py, and run_patching.py using returned hidden states/repr; produce CSVs/plots/JSON reports comparing ERM vs DANN.
    status: completed
    dependencies:
      - evaluation_and_reports
  - id: nonsemantic_and_fallback
    content: Add TRILL/TRILLsson extraction + baseline training/eval on cached features; add minimal LFCC+GMM fallback scripts.
    status: completed
    dependencies:
      - evaluation_and_reports
  - id: configs_and_tests_and_readme
    content: Add top-level runnable configs (wavlm_erm/dann, w2v2_erm/dann, trillsson_baseline), minimal unit tests, and README commands using ASVSPOOF5_ROOT.
    status: completed
    dependencies:
      - analysis_probes_cka_patching
      - nonsemantic_and_fallback
---

# Updated implementation plan

## What’s already correct in this repo (we will reuse)

- **Whitespace protocol parsing** exists in `scripts/make_manifest.py` and `src/asvspoof5_domain_invariant_cm/data/asvspoof5.py` via `sep=r"\s+"`.
- **HF backbones + hidden states** exist in `src/asvspoof5_domain_invariant_cm/models/backbones.py`.
- **GRL + multi-head discriminator** exists in `src/asvspoof5_domain_invariant_cm/models/dann.py`.
- **EER/minDCF + per-domain eval helpers** exist in `src/asvspoof5_domain_invariant_cm/evaluation/metrics.py` and `src/asvspoof5_domain_invariant_cm/evaluation/domain_eval.py`.

## Mismatches/gaps vs your thesis spec (must fix)

- **Label convention is wrong**: current dataset uses `KEY_TO_LABEL = {"bonafide": 1, "spoof": 0}` in `src/asvspoof5_domain_invariant_cm/data/asvspoof5.py`, but you want **bonafide=0, spoof=1**.
- **Audio pathing is wrong for Track1 filenames**: current manifest code assumes `audio_dir / flac_file`, but Track1 `FLAC_FILE_NAME` begins with `T_/D_/E_` and you want absolute paths based on **`ASVSPOOF5_ROOT`**.
- **No “- → NONE” normalization** for `CODEC` / `CODEC_Q`.
- **No torchaudio crop/pad batching** (current dataset uses `soundfile`, and collate pads to batch max length; you want fixed chunk length with random/center crop and attention masks).
- **Model pipeline ordering differs**: current `DANNModel` does `backbone -> projection -> pooling`, but you want `hidden_states -> layer_mix -> stats_pool -> projection -> repr(256)` and domain head uses `GRL(repr)`.
- **Entrypoints are stubs**: `scripts/train.py`, `scripts/evaluate.py`, `scripts/run_probes.py`, `scripts/run_patching.py` all raise `NotImplementedError`.
- **No training package**: `src/asvspoof5_domain_invariant_cm/training/` does not exist.
- **No TRILLsson extraction pipeline**: only a config exists (`configs/model/trillsson.yaml`).
- **No unit tests**: `tests/` missing.

## Target repo structure (keep package layout, add missing modules)

We will implement your requested modules under the existing package root `src/asvspoof5_domain_invariant_cm/` (so imports stay clean), and add the requested scripts/config aliases at repo root.

## Dataflow (intended)

```mermaid
flowchart TD
  Proto[ASVspoof5_protocols/*.tsv] --> Prepare[scripts/prepare_asvspoof5.py]
  Audio[ASVSPOOF5_ROOT/flac_T|flac_D|flac_E_eval] --> Prepare
  Prepare --> Manifest[data/manifests/{train,dev,eval}.parquet]

  Manifest --> Train[scripts/train.py]
  Train --> RunDir[runs/exp_name/*]
  RunDir --> Eval[scripts/evaluate.py]
  Eval --> Tables[runs/.../tables/*.csv]
  Eval --> Metrics[runs/.../metrics.json]

  RunDir --> Probes[scripts/probe_domain.py]
  RunDir --> CKA[scripts/run_cka.py]
  RunDir --> Patch[scripts/run_patching.py]

  Manifest --> TrillExtract[scripts/extract_trillsson.py]
  TrillExtract --> TrillFeats[data/features/trillsson/{split}.npy]
```

## Concrete implementation steps (files to add/modify)

### 1) Data pipeline + manifests

- **Add** [`scripts/prepare_asvspoof5.py`](scripts/prepare_asvspoof5.py)
  - Parse whitespace-separated protocols with columns:

`SPEAKER_ID FLAC_FILE_NAME SPEAKER_GENDER CODEC CODEC_Q CODEC_SEED ATTACK_TAG ATTACK_LABEL KEY TMP`

  - Normalize `CODEC/CODEC_Q`: `"-" -> "NONE"`.
  - Map **KEY**: `bonafide -> 0`, `spoof -> 1`.
  - Build **absolute `audio_path`** from `ASVSPOOF5_ROOT` and `FLAC_FILE_NAME` prefix:
    - `T_` → `ASVSPOOF5_ROOT/flac_T/{FLAC_FILE_NAME}`
    - `D_` → `ASVSPOOF5_ROOT/flac_D/{FLAC_FILE_NAME}`
    - `E_` → `ASVSPOOF5_ROOT/flac_E_eval/{FLAC_FILE_NAME}`
  - Encode domains to ids and persist vocabs:
    - `codec_vocab.json`, `codec_q_vocab.json` saved next to manifests (and later copied into `runs/...`).
  - Save parquet manifests with required columns: `audio_path, y_task, y_codec, y_codec_q, speaker_id, codec_seed, attack_label, attack_tag, codec, codec_q, flac_file`.

- **Keep** `scripts/make_manifest.py` as a thin wrapper (optional) or deprecate it in README; core logic moves into `prepare_asvspoof5.py`.

- **Add** [`src/asvspoof5_domain_invariant_cm/data/audio.py`](src/asvspoof5_domain_invariant_cm/data/audio.py)
  - `load_waveform_torchaudio(path, target_sr=16000) -> (waveform[1,T], sr)`
  - `crop_or_pad_to_n_samples(waveform, n_samples, mode) -> waveform[1,n_samples]`
  - `collate_audio_batch(samples) -> waveform[B,T], attention_mask[B,T], lengths[B], y_task/y_codec/y_codec_q[B], metadata`

- **Replace/extend** [`src/asvspoof5_domain_invariant_cm/data/asvspoof5.py`](src/asvspoof5_domain_invariant_cm/data/asvspoof5.py)
  - Switch to torchaudio-based loading and fixed-length cropping via `data/audio.py`.
  - Fix label mapping to bonafide=0 spoof=1.
  - Ensure it returns per-sample: `waveform[1,T]`, `y_task`, `y_codec`, `y_codec_q`, and metadata.
  - Keep `get_codec_seed_groups` and ensure `codec_seed` is returned in metadata.

- **Update** [`src/asvspoof5_domain_invariant_cm/utils/paths.py`](src/asvspoof5_domain_invariant_cm/utils/paths.py)
  - Make **`ASVSPOOF5_ROOT`** the single source of truth (no `data/raw/asvspoof5` default).
  - Provide a clear error if unset.

### 2) Model alignment (exact ordering + layer selection)

- **Modify** [`src/asvspoof5_domain_invariant_cm/models/backbones.py`](src/asvspoof5_domain_invariant_cm/models/backbones.py)
  - Support `layer_selection.method in {weighted, first_k, last_k, specific}`.
  - For `specific`, allow an explicit `layers: [indices]`.
  - Ensure `LayerWeightedPooling` weights only the **selected layers**.

- **Modify** [`src/asvspoof5_domain_invariant_cm/models/heads.py`](src/asvspoof5_domain_invariant_cm/models/heads.py)
  - Make **stats pooling** the default and explicit: `StatsPooling(mean+std)` returning `[B, 2D]`.
  - Keep mean/attention pooling if desired, but configs will use stats.

- **Refactor** [`src/asvspoof5_domain_invariant_cm/models/dann.py`](src/asvspoof5_domain_invariant_cm/models/dann.py)
  - Make forward exactly:
    - `hidden_states -> selected_layer_mix -> pooled_stats -> projection -> repr(256)`
    - `task_logits = task_head(repr)`
    - `codec_logits, codec_q_logits = domain_disc(GRL(repr))`
  - Return `repr` and `all_hidden_states` for analysis.

- **Add** [`src/asvspoof5_domain_invariant_cm/models/erm.py`](src/asvspoof5_domain_invariant_cm/models/erm.py)
  - Same pipeline as DANN but without GRL/domain heads.

### 3) Training package + runnable training CLI

- **Add** [`src/asvspoof5_domain_invariant_cm/training/loop.py`](src/asvspoof5_domain_invariant_cm/training/loop.py)
  - Train/val loops, AMP optional, grad clipping, early stopping.
  - Save `runs/{exp_name}/best.pt`, `last.pt`, `train_log.jsonl`, `metrics_val.json`.

- **Add** [`src/asvspoof5_domain_invariant_cm/training/sched.py`](src/asvspoof5_domain_invariant_cm/training/sched.py)
  - LR sched (cosine+warmup) and DANN lambda schedules (constant/linear warmup).

- **Add** [`src/asvspoof5_domain_invariant_cm/training/losses.py`](src/asvspoof5_domain_invariant_cm/training/losses.py)
  - Build loss objects from config; support optional class weights for task loss.

- **Implement** [`scripts/train.py`](scripts/train.py)
  - Load YAML configs (data+train+model), resolve and save `config_resolved.yaml`.
  - Build dataloaders from manifests.
  - Build ERM or DANN model.
  - Log task and domain accuracies.
  - Copy vocabs into run dir.

### 4) Evaluation + reports

- **Implement** [`scripts/evaluate.py`](scripts/evaluate.py)
  - Load checkpoint+config, run inference on dev/eval.
  - Save `predictions.tsv` and `metrics.json`.
  - Use consistent score convention: **higher = more bonafide**.

- **Modify** [`src/asvspoof5_domain_invariant_cm/evaluation/metrics.py`](src/asvspoof5_domain_invariant_cm/evaluation/metrics.py)
  - Add `pos_label` (bonafide label = 0) and handle labels accordingly.

- **Add** [`src/asvspoof5_domain_invariant_cm/evaluation/reports.py`](src/asvspoof5_domain_invariant_cm/evaluation/reports.py)
  - Per-domain breakdown tables (CODEC, CODEC_Q, optional ATTACK_LABEL).
  - Bootstrap CIs using existing `bootstrap_metric`.
  - Optional “official scoring compatibility” writer (scorefile emission).

### 5) Probes, CKA, patching (mechanistic)

- **Add** [`scripts/probe_domain.py`](scripts/probe_domain.py)
  - Extract per-layer embeddings on a subset of dev.
  - Train sklearn probes for CODEC/CODEC_Q.
  - Save CSV + plots comparing ERM vs DANN.

- **Add** [`scripts/run_cka.py`](scripts/run_cka.py)
  - Use existing CKA implementation in `src/asvspoof5_domain_invariant_cm/analysis/repr_similarity.py` (or alias via a new `analysis/cka.py`).
  - Save heatmap + CSV.

- **Implement** [`scripts/run_patching.py`](scripts/run_patching.py)
  - Patch **projection repr units** (primary) from DANN → ERM for selected “domain-heavy units”.
  - Measure change in domain discriminator confidence and task score on a small subset.
  - Optionally patch one selected transformer layer output.

### 6) Non-semantic baseline + fallback classical

- **Add** [`scripts/extract_trillsson.py`](scripts/extract_trillsson.py)
  - Read manifest, write `data/features/trillsson/{split}.npy` + metadata CSV.
  - Use TFHub TRILL/TRILLsson; document exact install/usage.

- **Add** [`src/asvspoof5_domain_invariant_cm/models/nonsemantic.py`](src/asvspoof5_domain_invariant_cm/models/nonsemantic.py)
  - Train/eval on cached embeddings (logreg and small MLP).

- **Add (fallback)** LFCC+GMM baseline scripts (minimal):
  - [`scripts/extract_lfcc.py`](scripts/extract_lfcc.py)
  - [`scripts/train_lfcc_gmm.py`](scripts/train_lfcc_gmm.py)

### 7) Thesis-friendly top-level configs

- **Add** single-file runnable configs requested:
  - [`configs/wavlm_erm.yaml`](configs/wavlm_erm.yaml)
  - [`configs/wavlm_dann.yaml`](configs/wavlm_dann.yaml)
  - [`configs/w2v2_erm.yaml`](configs/w2v2_erm.yaml)
  - [`configs/w2v2_dann.yaml`](configs/w2v2_dann.yaml)
  - [`configs/trillsson_baseline.yaml`](configs/trillsson_baseline.yaml)

These will either inline or reference existing modular configs to avoid duplication.

### 8) Minimal unit tests

- **Add** [`tests/test_protocol_parse.py`](tests/test_protocol_parse.py)
  - Whitespace parse, `- -> NONE`, label mapping bonafide=0 spoof=1, prefix-to-audio_dir mapping.

- **Add** [`tests/test_dataset_shapes.py`](tests/test_dataset_shapes.py)
  - Batch shapes, attention masks, dtype checks, fixed-length crop/pad.

### 9) README exact commands

- **Update** [`README.md`](README.md)
  - Use **`ASVSPOOF5_ROOT=/path/to/data/asvspoof5`**.
  - Provide exact commands for prepare/train/evaluate/probes/CKA/patching/TRILLsson.

## Acceptance criteria mapping

- `prepare_asvspoof5.py` → produces manifests + vocabs
- `train.py` → trains wavlm/w2v2 ERM and DANN and saves into `runs/`
- `evaluate.py` → produces overall + per-CODEC + per-CODEC_Q CSV tables
- `probe_domain.py` → plots+CSV ERM vs DANN probe accuracy vs layer
- `run_cka.py` → heatmap+CSV
- `run_patching.py` → JSON report with patching effects
- `extract_trillsson.py` + `trillsson_baseline.yaml` → non-semantic baseline runnable