#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

OUT_BASE="${SPECTRAL_OUTPUT_DIR:-/root/autodl-tmp/spectral_outputs}"
LOG_DIR="${OUT_BASE}/logs"
mkdir -p "$LOG_DIR"

echo "=== SpectralStore system-direction run (A-D) ==="
echo "Root:     $ROOT"
echo "Output:   $OUT_BASE"
echo "Logs:     $LOG_DIR"
echo "Started:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"

python -c "import spectralstore; print('spectralstore ok')"
echo "python:    $(python --version)"
echo "git commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

echo ""
echo "--- stage A/B: Exp3 query benchmark + index comparison ---"
python -u scripts/exp3/run_query_benchmark.py \
    --config experiments/configs/exp3/query_benchmark.yaml \
    2>&1 | tee "${LOG_DIR}/exp3_query_benchmark.log"
echo "Exp3 done ($(date -u +%H:%M:%SZ))"

echo ""
echo "--- stage C: Exp4_v2 residual-query robustness ---"
python -u scripts/exp4_v2/run_residual_query_robustness.py \
    --config experiments/configs/exp4_v2/residual_query_robustness.yaml \
    2>&1 | tee "${LOG_DIR}/exp4_v2_residual_query_robustness.log"
echo "Exp4_v2 done ($(date -u +%H:%M:%SZ))"

echo ""
echo "--- stage D: ARD diagnostic ---"
python -u scripts/exp5/run_ard_diagnostic.py \
    --config experiments/configs/exp5/ard_diagnostic.yaml \
    2>&1 | tee "${LOG_DIR}/exp5_ard_diagnostic.log"
echo "ARD diagnostic done ($(date -u +%H:%M:%SZ))"

echo ""
echo "=== system-direction run done ==="
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Results:"
echo "  experiments/results/exp3/query_benchmark"
echo "  experiments/results/exp4_v2/residual_query_robustness"
echo "  experiments/results/exp5/ard_diagnostic"
echo "Logs:"
echo "  ${LOG_DIR}/exp3_query_benchmark.log"
echo "  ${LOG_DIR}/exp4_v2_residual_query_robustness.log"
echo "  ${LOG_DIR}/exp5_ard_diagnostic.log"
