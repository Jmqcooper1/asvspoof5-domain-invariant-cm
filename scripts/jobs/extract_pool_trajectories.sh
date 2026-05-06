#!/usr/bin/env bash
# Extract pool-weight trajectories (every saved checkpoint + best + last)
# for the 6 multi-seed DANN runs across both backbones. Login-node, no GPU.
#
# Usage (from repo root): bash scripts/jobs/extract_pool_trajectories.sh

set -u

cd "${SLURM_SUBMIT_DIR:-.}"
if [[ ! -f "pyproject.toml" ]]; then
  echo "ERROR: run from repo root" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --locked

RUNS_BASE="${RUNS_DIR:-/gpfs/work5/0/prjs1904/runs}"
OUT_DIR="results/pool_weights_trajectory"
mkdir -p "$OUT_DIR"

# (cluster_run_dir, clean_output_name)
RUNS=(
  "wavlm_dann_seed42_v2_1e2d5c7:wavlm_dann_seed42_v2"
  "wavlm_dann_seed123_5223811:wavlm_dann_seed123"
  "wavlm_dann_seed456_5223811:wavlm_dann_seed456"
  "w2v2_dann_seed42_v2_1e2d5c7:w2v2_dann_seed42_v2"
  "w2v2_dann_seed123_5223811:w2v2_dann_seed123"
  "w2v2_dann_seed456_5223811:w2v2_dann_seed456"
)

for entry in "${RUNS[@]}"; do
  src="${entry%%:*}"
  name="${entry##*:}"
  run_dir="${RUNS_BASE}/${src}"
  out="${OUT_DIR}/${name}.json"
  if [[ ! -d "${run_dir}/checkpoints" ]]; then
    echo "WARN  no checkpoints dir for ${src}" >&2
    continue
  fi
  echo ""
  echo "=== ${name} ==="
  uv run python scripts/extract_pool_weights_trajectory.py \
    --run-dir "${run_dir}" \
    --output  "${out}"
done

echo ""
echo "=== Done ==="
ls -la "$OUT_DIR/"
