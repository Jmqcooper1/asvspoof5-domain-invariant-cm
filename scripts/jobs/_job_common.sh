#!/usr/bin/env bash
set -euo pipefail

# Shared helpers for SLURM job scripts.
#
# Responsibilities:
# - Load .env if present
# - Require ASVSPOOF5_ROOT (explicitly set; no defaults)
# - Provide reusable preflight helpers

load_dotenv_if_present() {
  # Expect .env in repo root (jobs run from repo root)
  if [[ -f ".env" ]]; then
    # shellcheck disable=SC1091
    set -a
    . ./.env
    set +a
  fi
}

require_env_var() {
  local var_name="$1"
  # Indirect expansion: ${!var_name}
  if [[ -z "${!var_name:-}" ]]; then
    echo "ERROR: Required environment variable '$var_name' is not set." >&2
    echo "       Set it in .env or export it before submitting jobs." >&2
    return 1
  fi
}

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: Required command '$cmd' not found in PATH." >&2
    return 1
  fi
}

require_dir() {
  local dir_path="$1"
  if [[ ! -d "$dir_path" ]]; then
    echo "ERROR: Required directory not found: $dir_path" >&2
    return 1
  fi
}

require_file() {
  local file_path="$1"
  if [[ ! -f "$file_path" ]]; then
    echo "ERROR: Required file not found: $file_path" >&2
    return 1
  fi
}

require_nonempty_glob() {
  local pattern="$1"
  # bash globbing check
  shopt -s nullglob
  local matches=( $pattern )
  shopt -u nullglob
  if (( ${#matches[@]} == 0 )); then
    echo "ERROR: No files match required pattern: $pattern" >&2
    return 1
  fi
}

print_env_diagnostics() {
  echo "=== Environment diagnostics ==="
  echo "PWD: $(pwd)"
  echo "ASVSPOOF5_ROOT: ${ASVSPOOF5_ROOT}"
  echo "HF_HOME: ${HF_HOME:-}"
  echo "WANDB_PROJECT: ${WANDB_PROJECT:-}"
  echo "WANDB_MODE: ${WANDB_MODE:-}"
  echo "ffmpeg: $(command -v ffmpeg 2>/dev/null || echo 'not-found')"
  echo "==============================="
}

load_ffmpeg_module() {
  # Load FFmpeg module on Snellius/SURF clusters.
  # FFmpeg is not globally installed; must use environment modules.
  # Note: 'module load ffmpeg' fails; must use exact version.
  local ffmpeg_module="${FFMPEG_MODULE:-FFmpeg/7.1.1-GCCcore-14.2.0}"
  
  if command -v module >/dev/null 2>&1; then
    echo "Loading FFmpeg module: $ffmpeg_module"
    module load "$ffmpeg_module" 2>/dev/null || {
      echo "WARNING: Failed to load module $ffmpeg_module" >&2
      echo "         Trying to find available FFmpeg versions..." >&2
      module spider FFmpeg 2>&1 | head -20 || true
      echo "         Set FFMPEG_MODULE env var to override." >&2
    }
  else
    echo "Note: 'module' command not available (not on HPC?). Assuming ffmpeg is in PATH."
  fi
  
  # Verify ffmpeg is now available
  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ERROR: ffmpeg not found after module load. DANN augmentation will fail." >&2
    return 1
  fi
  echo "ffmpeg location: $(command -v ffmpeg)"
}

check_ffmpeg_encoders_or_fail() {
  # DANN augmentation needs at least 2 of these for MP3/AAC/OPUS.
  require_cmd ffmpeg
  local encoders
  encoders="$(ffmpeg -encoders 2>/dev/null || true)"
  if ! echo "$encoders" | grep -E -q 'libmp3lame|aac|libopus'; then
    echo "ERROR: ffmpeg is present but required encoders are missing." >&2
    echo "       Need at least two of: libmp3lame, aac, libopus" >&2
    echo "       Check: ffmpeg -encoders | grep -E 'libmp3lame|aac|libopus'" >&2
    return 1
  fi
}

check_wandb_api_key() {
  # Warn if WANDB_API_KEY is not set (wandb will be disabled)
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "Note: WANDB_API_KEY not set. Wandb logging will be disabled."
    echo "      To enable: add WANDB_API_KEY=your_key to .env"
  else
    echo "WANDB_API_KEY: set (wandb logging enabled)"
  fi
}

require_wandb_api_key() {
  # Fail if WANDB_API_KEY is not set (for jobs that require wandb logging)
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY is required but not set." >&2
    echo "       Set it in .env or export before submitting jobs." >&2
    echo "       Example: export WANDB_API_KEY=your_api_key_here" >&2
    return 1
  fi
  echo "WANDB_API_KEY: set (wandb logging enabled)"
}

load_and_require_asvspoof5_root() {
  load_dotenv_if_present
  require_env_var "ASVSPOOF5_ROOT"
  check_wandb_api_key
  print_env_diagnostics
}

