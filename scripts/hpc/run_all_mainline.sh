#!/usr/bin/env bash
# SpectralStore AutoDL/HPC mainline rerun (exp1 + exp2 + exp4 + exp5).
#
# Data must be uploaded via FileZilla/SFTP before running.
# Recommended: conda activate spectralstore  # Python 3.11
#
# Usage:
#   screen -S spectralstore
#   conda activate spectralstore
#   export SPECTRAL_OUTPUT_DIR=/root/autodl-tmp/spectral_outputs
#   bash scripts/hpc/run_all_mainline.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

OUT_BASE="${SPECTRAL_OUTPUT_DIR:-/root/autodl-tmp/spectral_outputs}"
LOG_DIR="${OUT_BASE}/logs"
mkdir -p "$LOG_DIR"

echo "=== SpectralStore HPC mainline rerun ==="
echo "Root:     $ROOT"
echo "Output:   $OUT_BASE"
echo "Logs:     $LOG_DIR"
echo "Started:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo ""
echo "--- preflight ---"
python -c "import spectralstore; print('spectralstore ok')"
echo "git commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "python:     $(python --version)"
echo "conda env:  ${CONDA_DEFAULT_ENV:-none}"

echo ""
echo "--- step 1: data check ---"
DATA_OK=true
for f in data/raw/soc-sign-bitcoinotc.csv.gz data/raw/soc-sign-bitcoinalpha.csv.gz; do
    if [ -f "$f" ]; then
        echo "  OK  $f  ($(du -h "$f" | cut -f1))"
    else
        echo "  MISSING  $f"
        DATA_OK=false
    fi
done
if ! $DATA_OK; then
    echo ""
    echo "FATAL: data files missing."
    echo "Upload data/raw/ via FileZilla first, then re-run."
    exit 1
fi

echo ""
echo "--- step 2/9: exp1 theory_validation ---"
python -u scripts/exp1/run_exp1_theory_validation.py \
    --config experiments/configs/exp1/theory_validation.yaml \
    2>&1 | tee "${LOG_DIR}/exp1.log"
echo "exp1 done ($(date -u +%H:%M:%SZ))"

echo ""
echo "--- step 3/9: exp2 bitcoin_sweep ---"
python -u scripts/exp2/run_bitcoin_compression_ratio_sweep.py \
    --config experiments/configs/exp2/bitcoin_sweep.yaml \
    2>&1 | tee "${LOG_DIR}/exp2_sweep.log"
echo "exp2 bitcoin_sweep done ($(date -u +%H:%M:%SZ))"

echo ""
echo "--- step 4/9: exp2 bitcoin_sweep_rmse ---"
python -u scripts/exp2/run_bitcoin_compression_ratio_sweep_rmse.py \
    --config experiments/configs/exp2/bitcoin_sweep_rmse.yaml \
    2>&1 | tee "${LOG_DIR}/exp2_sweep_rmse.log"
echo "exp2 bitcoin_sweep_rmse done ($(date -u +%H:%M:%SZ))"

echo ""
echo "--- step 5/9: exp2 residual_boundary ---"
python -u scripts/exp2/run_bitcoin_residual_boundary_sweep.py \
    --config experiments/configs/exp2/bitcoin_residual_boundary.yaml \
    2>&1 | tee "${LOG_DIR}/exp2_residual_boundary.log"
echo "exp2 residual_boundary done ($(date -u +%H:%M:%SZ))"

echo ""
echo "--- step 6/9: exp4 random_attack ---"
python -u scripts/exp4/run_synthetic_attack_random.py \
    --config experiments/configs/exp4/random_attack.yaml \
    2>&1 | tee "${LOG_DIR}/exp4_random_attack.log"
echo "exp4 random_attack done ($(date -u +%H:%M:%SZ))"

echo ""
echo "--- step 7/9: exp4 targeted_attack ---"
python -u scripts/exp4/run_synthetic_attack_targeted.py \
    --config experiments/configs/exp4/targeted_attack.yaml \
    2>&1 | tee "${LOG_DIR}/exp4_targeted_attack.log"
echo "exp4 targeted_attack done ($(date -u +%H:%M:%SZ))"

echo ""
echo "--- step 8/9: exp4 injection_attack ---"
python -u scripts/exp4/run_synthetic_attack_injection.py \
    --config experiments/configs/exp4/injection_attack.yaml \
    2>&1 | tee "${LOG_DIR}/exp4_injection_attack.log"
echo "exp4 injection_attack done ($(date -u +%H:%M:%SZ))"

echo ""
echo "--- step 9/9: exp5 ard_rank ---"
python -u scripts/exp5/run_ard_rank_selection.py \
    --config experiments/configs/exp5/ard_rank_selection.yaml \
    2>&1 | tee "${LOG_DIR}/exp5_ard_rank.log"
echo "exp5 ard_rank done ($(date -u +%H:%M:%SZ))"

echo ""
echo "=== all done ==="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Logs:     ${LOG_DIR}"
echo ""
echo "Output directories:"
find experiments/results -maxdepth 3 -type f \( -name "*.csv" -o -name "*.json" -o -name "*.md" \) -newer "${LOG_DIR}/exp1.log" 2>/dev/null | sort || true
