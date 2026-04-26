# Preliminary Synthetic-SBM Experiment

Goal: compare SpectralStore's asymmetric spectral compressor against tensor
unfolding SVD, CP-ALS, Tucker-HOSVD, SymSVD, and DirectSVD on a controlled
temporal stochastic block model.

This experiment evaluates reconstruction against the latent expected adjacency
matrices, not just against noisy sampled edges.

Outputs:

- `results/metrics.json`: per-run and aggregate metrics
- `results/summary.md`: compact result table
- `results/bound_scaling_metrics.json`: Synthetic-SBM empirical bound scaling metrics
- `results/bound_scaling_summary.md`: compact bound scaling tables
