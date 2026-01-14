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

load_and_require_asvspoof5_root() {
  load_dotenv_if_present
  require_env_var "ASVSPOOF5_ROOT"
  print_env_diagnostics
}

