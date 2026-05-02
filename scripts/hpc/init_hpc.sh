#!/usr/bin/env bash
# SpectralStore AutoDL/HPC first-time initialization.
#
# Recommended AutoDL environment:
# - Ubuntu 22.04
# - Conda/Miniconda image
# - Python 3.11 environment named "spectralstore"
# - High-memory CPU instance, or a low-cost GPU instance used as a CPU box
#
# Run once after uploading code + data via FileZilla/SFTP.
# Usage: bash scripts/hpc/init_hpc.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PIP_INDEX_ARGS=()
if [ "${USE_TUNA_PIP:-0}" = "1" ]; then
    PIP_INDEX_ARGS=(-i https://pypi.tuna.tsinghua.edu.cn/simple)
fi

echo "=== SpectralStore HPC init ==="
echo "Root:     $ROOT"
echo "Python:   $(python --version)"
echo "Conda:    ${CONDA_DEFAULT_ENV:-none}"

PY_MINOR="$(
python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
if [ "$PY_MINOR" != "3.11" ]; then
    echo "WARNING: recommended Python is 3.11; current is ${PY_MINOR}."
    echo "Create it on AutoDL with: conda create -n spectralstore python=3.11 -y"
fi

echo ""
echo "--- upgrade packaging tools ---"
python -m pip install --upgrade pip setuptools wheel "${PIP_INDEX_ARGS[@]}"

echo ""
echo "--- install project dependencies ---"
python -m pip install -e ".[dev,experiments]" "${PIP_INDEX_ARGS[@]}"

echo ""
echo "--- verify import ---"
python -c "
import spectralstore
from spectralstore.compression import create_compressor
from spectralstore.evaluation import load_experiment_config
print('spectralstore import ok')
print('create_compressor ok')
print('load_experiment_config ok')
"

echo ""
echo "--- verify data ---"
MISSING=0
check_data() {
    local f="$1"
    if [ -f "$f" ]; then
        echo "  OK: $f ($(du -h "$f" | cut -f1))"
    else
        echo "  MISSING: $f"
        MISSING=1
    fi
}
check_data data/raw/soc-sign-bitcoinotc.csv.gz
check_data data/raw/soc-sign-bitcoinalpha.csv.gz

if [ "$MISSING" -eq 1 ]; then
    echo ""
    echo "ERROR: data files missing. Upload data/raw/ via FileZilla first."
    exit 1
fi

echo ""
echo "--- output dirs ---"
OUT_BASE="${SPECTRAL_OUTPUT_DIR:-/root/autodl-tmp/spectral_outputs}"
mkdir -p "${OUT_BASE}/logs"
echo "Output: ${OUT_BASE}"

echo ""
echo "=== init complete ==="
echo "Optional validation:"
echo "  python -m ruff check src scripts"
echo "  python -m pytest"
echo "Next: bash scripts/hpc/run_all_mainline.sh"
