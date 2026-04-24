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

The sweep compares two robust residual policies:

- `spectralstore_full_mad`: adaptive threshold based on the upper tail of
  absolute residuals. This is the preferred default because it can avoid forced
  residual extraction when no attack is present.
- `spectralstore_full_quantile`: fixed residual sparsity. This is useful for
  controlled comparisons, but it can over-clean normal noise when the graph is
  clean and under-clean when the attack fraction exceeds the chosen quantile.

Current interpretation:

- Sparse outlier attacks are where residual separation is strongest.
- Random flips and targeted cross-community edges are structural perturbations;
  they need future graph-aware robustification rather than only magnitude-based
  residual thresholding.

Outputs:

- `results/metrics.json`: per-run and aggregate metrics
- `results/summary.md`: compact comparison table
- `results/sweep_metrics.json`: robustness sweep metrics
- `results/sweep_summary.md`: robustness sweep tables
