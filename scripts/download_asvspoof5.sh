#!/usr/bin/env bash
# Download ASVspoof 5 dataset from Zenodo
# Usage: bash scripts/download_asvspoof5.sh [--full]

set -euo pipefail

# Configuration
ZENODO_RECORD="14498691"
BASE_URL="https://zenodo.org/records/${ZENODO_RECORD}/files"
DATA_DIR="${ASVSPOOF5_DATA_ROOT:-data/raw/asvspoof5}"

# Parse arguments
FULL_DOWNLOAD=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --full)
            FULL_DOWNLOAD=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create data directory
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "Downloading ASVspoof 5 to: $DATA_DIR"

# Always download protocols
echo "Downloading protocols..."
wget -c "${BASE_URL}/ASVspoof5_protocols.tar.gz"

if [ "$FULL_DOWNLOAD" = true ]; then
    echo "Downloading full dataset..."
    
    # Training audio (5 shards)
    for shard in aa ab ac ad ae; do
        echo "Downloading flac_T_${shard}.tar..."
        wget -c "${BASE_URL}/flac_T_${shard}.tar"
    done
    
    # Dev audio (3 shards)
    for shard in aa ab ac; do
        echo "Downloading flac_D_${shard}.tar..."
        wget -c "${BASE_URL}/flac_D_${shard}.tar"
    done
    
    # Eval audio (10 shards) - optional, download last
    echo "Downloading eval audio (this is large)..."
    for shard in aa ab ac ad ae af ag ah ai aj; do
        echo "Downloading flac_E_${shard}.tar..."
        wget -c "${BASE_URL}/flac_E_${shard}.tar"
    done
else
    echo "Downloading minimal subset for development..."
    
    # Just first shard of train and dev
    wget -c "${BASE_URL}/flac_T_aa.tar"
    wget -c "${BASE_URL}/flac_D_aa.tar"
    
    echo ""
    echo "Downloaded minimal subset. Run with --full for complete dataset."
fi

echo ""
echo "Download complete!"
echo "Next step: bash scripts/unpack_asvspoof5.sh"
