# Preliminary Synthetic-Attack Experiment

Goal: compare the robust SpectralStore prototype against the non-robust
asymmetric compressor and simple SVD baselines on attacked temporal SBM data.

The first scenario uses sparse outlier edges because they directly test whether
large residuals can be separated and stored as CSR residuals.

Outputs:

- `results/metrics.json`: per-run and aggregate metrics
- `results/summary.md`: compact comparison table
