#!/usr/bin/env bash
# Download ASVspoof 5 dataset from Zenodo
#
# Usage:
#   bash scripts/download_asvspoof5.sh [options]
#
# Options:
#   --full                 Download full dataset (train+dev+eval shards)
#   --parallel N           Connections/splits per file for aria2 (default: 16)
#   --no-split             Disable aria2 splitting (use 1 connection)
#   --ipv4                 Force IPv4 (recommended if IPv6 is slow)
#   --record ID            Override Zenodo record id (default: 14498691)
#   --data-dir PATH        Override destination directory
#   --user-agent UA        Override User-Agent (default: Wget/1.21.4)
#   --no-aria2             Do not use aria2c even if installed (use wget/curl)
#   -h, --help             Show help

set -euo pipefail

ZENODO_RECORD="14498691"
DATA_DIR="${ASVSPOOF5_DATA_ROOT:-data/raw/asvspoof5}"
PARALLEL="${ASVSPOOF5_PARALLEL:-16}"
FORCE_IPV4="${ASVSPOOF5_FORCE_IPV4:-0}"
USER_AGENT="${ASVSPOOF5_USER_AGENT:-Wget/1.21.4}"

FULL_DOWNLOAD=false
USE_ARIA2=true
ARIA2_SPLIT=true

print_help() {
  cat <<'EOF'
Download ASVspoof 5 dataset from Zenodo

Usage:
  bash scripts/download_asvspoof5.sh [options]

Options:
  --full                 Download full dataset (train+dev+eval shards)
  --parallel N           Connections/splits per file for aria2 (default: 16)
  --no-split             Disable aria2 splitting (use 1 connection)
  --ipv4                 Force IPv4
  --record ID            Override Zenodo record id (default: 14498691)
  --data-dir PATH        Override destination directory
  --user-agent UA        Override User-Agent (default: Wget/1.21.4)
  --no-aria2             Do not use aria2c even if installed (use wget/curl)
  -h, --help             Show help

Recommended:
  bash scripts/download_asvspoof5.sh --ipv4 --parallel 16
EOF
}

is_pos_int() {
  [[ "$1" =~ ^[0-9]+$ ]] && [[ "$1" -ge 1 ]]
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)
      FULL_DOWNLOAD=true
      shift
      ;;
    --parallel)
      [[ $# -ge 2 ]] || {
        echo "Error: --parallel requires a value" >&2
        exit 1
      }
      PARALLEL="$2"
      shift 2
      ;;
    --parallel=*)
      PARALLEL="${1#*=}"
      shift
      ;;
    --no-split)
      ARIA2_SPLIT=false
      shift
      ;;
    --ipv4)
      FORCE_IPV4=1
      shift
      ;;
    --record)
      [[ $# -ge 2 ]] || {
        echo "Error: --record requires a value" >&2
        exit 1
      }
      ZENODO_RECORD="$2"
      shift 2
      ;;
    --record=*)
      ZENODO_RECORD="${1#*=}"
      shift
      ;;
    --data-dir)
      [[ $# -ge 2 ]] || {
        echo "Error: --data-dir requires a value" >&2
        exit 1
      }
      DATA_DIR="$2"
      shift 2
      ;;
    --data-dir=*)
      DATA_DIR="${1#*=}"
      shift
      ;;
    --user-agent)
      [[ $# -ge 2 ]] || {
        echo "Error: --user-agent requires a value" >&2
        exit 1
      }
      USER_AGENT="$2"
      shift 2
      ;;
    --user-agent=*)
      USER_AGENT="${1#*=}"
      shift
      ;;
    --no-aria2)
      USE_ARIA2=false
      shift
      ;;
    -h | --help)
      print_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "" >&2
      print_help >&2
      exit 1
      ;;
  esac
done

if ! is_pos_int "$PARALLEL"; then
  echo "Error: --parallel must be a positive integer (got: $PARALLEL)" >&2
  exit 1
fi

BASE_URL="https://zenodo.org/records/${ZENODO_RECORD}/files"

mkdir -p "$DATA_DIR"
cd -- "$DATA_DIR"

echo "Downloading ASVspoof 5 to: $DATA_DIR"
echo "Record: $ZENODO_RECORD"
echo "Parallel (requested): $PARALLEL"
echo "Aria2 split: $ARIA2_SPLIT"
echo "Force IPv4: $FORCE_IPV4"
echo "Use aria2: $USE_ARIA2"
echo "User-Agent: $USER_AGENT"
echo ""

download() {
  local url="$1"

  # Normalize and append ?download=1
  local url_no_q="${url%%\?*}"
  local final_url="${url_no_q}?download=1"
  local filename="${url_no_q##*/}"

  if [[ "$USE_ARIA2" == "true" ]] && command -v aria2c >/dev/null 2>&1; then
    local x=1
    local s=1

    if [[ "$ARIA2_SPLIT" == "true" ]]; then
      x="$PARALLEL"
      s="$PARALLEL"
    fi

    # aria2 constraint: 1..16
    if [[ "$x" -gt 16 ]]; then
      echo "Note: aria2c max connections per server is 16; clamping $x -> 16" >&2
      x=16
      s=16
    fi

    echo "Using aria2c for: $filename" >&2

    if [[ "$FORCE_IPV4" == "1" ]]; then
      aria2c \
        -c \
        -x "$x" \
        -s "$s" \
        -k 1M \
        --file-allocation=none \
        --console-log-level=warn \
        --user-agent="$USER_AGENT" \
        --disable-ipv6=true \
        -o "$filename" \
        "$final_url" && return 0
    else
      aria2c \
        -c \
        -x "$x" \
        -s "$s" \
        -k 1M \
        --file-allocation=none \
        --console-log-level=warn \
        --user-agent="$USER_AGENT" \
        -o "$filename" \
        "$final_url" && return 0
    fi

    echo "aria2c failed; falling back to wget/curl for: $filename" >&2
  fi

  if command -v wget >/dev/null 2>&1; then
    echo "Using wget for: $filename" >&2
    if [[ "$FORCE_IPV4" == "1" ]]; then
      wget -c -4 --user-agent="$USER_AGENT" -O "$filename" "$final_url"
    else
      wget -c --user-agent="$USER_AGENT" -O "$filename" "$final_url"
    fi
    return 0
  fi

  if command -v curl >/dev/null 2>&1; then
    echo "Using curl for: $filename" >&2
    if [[ "$FORCE_IPV4" == "1" ]]; then
      curl -L -C - -A "$USER_AGENT" --ipv4 -o "$filename" "$final_url"
    else
      curl -L -C - -A "$USER_AGENT" -o "$filename" "$final_url"
    fi
    return 0
  fi

  echo "Error: need aria2c, wget, or curl installed." >&2
  exit 1
}

echo "Downloading protocols..."
download "${BASE_URL}/ASVspoof5_protocols.tar.gz"

if [[ "$FULL_DOWNLOAD" == "true" ]]; then
  echo ""
  echo "Downloading full dataset..."

  for shard in aa ab ac ad ae; do
    echo "Downloading flac_T_${shard}.tar..."
    download "${BASE_URL}/flac_T_${shard}.tar"
  done

  for shard in aa ab ac; do
    echo "Downloading flac_D_${shard}.tar..."
    download "${BASE_URL}/flac_D_${shard}.tar"
  done

  echo ""
  echo "Downloading eval audio (this is large)..."
  for shard in aa ab ac ad ae af ag ah ai aj; do
    echo "Downloading flac_E_${shard}.tar..."
    download "${BASE_URL}/flac_E_${shard}.tar"
  done
else
  echo "Downloading minimal subset for development..."
  download "${BASE_URL}/flac_T_aa.tar"
  download "${BASE_URL}/flac_D_aa.tar"

  echo ""
  echo "Downloaded minimal subset. Run with --full for complete dataset."
fi

echo ""
echo "Download complete!"
echo "Next step: bash scripts/unpack_asvspoof5.sh"