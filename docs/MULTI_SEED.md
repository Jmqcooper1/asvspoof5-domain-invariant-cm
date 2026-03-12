# Multi-Seed Experiments

Run each model configuration with 3 seeds to measure variance and establish statistical significance.

**Seed 42 is already complete** for all 4 configs. This job runs seeds 123 and 456 only (8 new GPU jobs total).

## Configurations

| Config | Seed 42 | Seeds 123 + 456 |
|--------|---------|-----------------|
| `wavlm_erm` | ✅ Done | 2 new runs |
| `wavlm_dann` | ✅ Done | 2 new runs |
| `w2v2_erm` | ✅ Done | 2 new runs |
| `w2v2_dann` | ✅ Done | 2 new runs |

## Running Experiments

Submit all 4 configs (each launches 2 array jobs = 8 total new runs):

```bash
sbatch scripts/jobs/train_multi_seed.job wavlm_erm
sbatch scripts/jobs/train_multi_seed.job wavlm_dann
sbatch scripts/jobs/train_multi_seed.job w2v2_erm
sbatch scripts/jobs/train_multi_seed.job w2v2_dann
```

Monitor progress:

```bash
squeue -u $USER
```

Each array job trains with one seed, then evaluates on both `dev` and `eval` splits with per-domain breakdown, bootstrap CIs, and score files.

## Output Structure

```
results/predictions/
├── wavlm_erm_seed123_eval/
│   ├── predictions.tsv
│   ├── metrics.json
│   └── scores.txt
├── wavlm_erm_seed123_dev/
│   └── ...
├── wavlm_erm_seed456_eval/
│   └── ...
└── ...
```

Predictions TSV has columns: `flac_file, score, prediction, y_task, codec`.

**Note:** Seed 42 results are already in `/projects/prjs1904/runs/{config}/` — copy or symlink those prediction files into `results/predictions/{config}_seed42_{split}/` to have everything in one place.

## Generating Figures

### DET Curves

```bash
# With real predictions (after experiments complete)
python scripts/plot_det_curves.py --predictions-dir results/predictions/

# Demo mode (synthetic curves for layout testing)
python scripts/plot_det_curves.py --demo
```

Output: `master-thesis-uva/figures/det_curves.{png,pdf}`

### PCA of Representations

First extract representations (see `scripts/extract_representations.py`), then:

```bash
# With real data
python scripts/plot_pca.py \
    --erm-repr results/representations/erm_proj.npy \
    --dann-repr results/representations/dann_proj.npy \
    --labels results/representations/codec_labels.npy

# Demo mode
python scripts/plot_pca.py --demo
```

Output: `master-thesis-uva/figures/pca_representations.{png,pdf}`

## Aggregating Results

To compute mean ± std EER across seeds for a config:

```bash
# Example: collect EER from all wavlm_erm seeds
for f in results/predictions/wavlm_erm_seed*_eval.csv; do
    echo "$f"
done
```

Use the evaluation metrics logged to W&B for systematic comparison, or parse the prediction CSVs with the DET curve script.
