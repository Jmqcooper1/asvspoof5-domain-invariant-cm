#!/usr/bin/env bash
# Unpack ASVspoof 5 tar archives
# Usage: bash scripts/unpack_asvspoof5.sh

set -euo pipefail

# Prefer ASVSPOOF5_ROOT (used by training scripts), fall back to legacy ASVSPOOF5_DATA_ROOT
DATA_DIR="${ASVSPOOF5_ROOT:-${ASVSPOOF5_DATA_ROOT:-data/raw/asvspoof5}}"

cd "$DATA_DIR"

echo "Unpacking ASVspoof 5 archives in: $DATA_DIR"

# Unpack protocols into ASVspoof5_protocols/ subfolder
if [ -f "ASVspoof5_protocols.tar.gz" ]; then
    echo "Unpacking protocols..."
    mkdir -p ASVspoof5_protocols
    tar -xzf ASVspoof5_protocols.tar.gz -C ASVspoof5_protocols --strip-components=0
    # If files extracted to root, move them to subfolder
    if [ -f "ASVspoof5.train.tsv" ]; then
        mv ASVspoof5.*.tsv ASVspoof5_protocols/ 2>/dev/null || true
        mv ASVspoof5.*.csv ASVspoof5_protocols/ 2>/dev/null || true
    fi
fi

# Unpack training audio
echo "Unpacking training audio..."
for f in flac_T_*.tar; do
    if [ -f "$f" ]; then
        echo "  Unpacking $f..."
        tar -xf "$f"
    fi
done

# Unpack dev audio
echo "Unpacking dev audio..."
for f in flac_D_*.tar; do
    if [ -f "$f" ]; then
        echo "  Unpacking $f..."
        tar -xf "$f"
    fi
done

# Unpack eval audio
echo "Unpacking eval audio..."
for f in flac_E_*.tar; do
    if [ -f "$f" ]; then
        echo "  Unpacking $f..."
        tar -xf "$f"
    fi
done

echo ""
echo "Unpacking complete!"
echo ""
echo "Directory structure:"
find . -maxdepth 2 -type d | head -20

echo ""
echo "Next step: python scripts/make_manifest.py"
