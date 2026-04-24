# Preliminary Synthetic-Attack Experiment

Goal: compare the robust SpectralStore prototype against the non-robust
asymmetric compressor and simple SVD baselines on attacked temporal SBM data.

The first scenario uses sparse outlier edges because they directly test whether
large residuals can be separated and stored as CSR residuals.

The sweep script also runs `random_flip` and `targeted_cross_community`. These
attacks are less outlier-like, so the current residual-separation prototype is
expected to help less there. Treat those rows as a diagnostic for threshold
policy and future attack-specific robustification, not as the final robust graph
learning result.

Outputs:

- `results/metrics.json`: per-run and aggregate metrics
- `results/summary.md`: compact comparison table
- `results/sweep_metrics.json`: robustness sweep metrics
- `results/sweep_summary.md`: robustness sweep tables
