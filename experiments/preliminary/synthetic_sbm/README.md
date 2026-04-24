# Preliminary Synthetic-SBM Experiment

Goal: compare SpectralStore's asymmetric spectral compressor against SymSVD and
DirectSVD on a controlled temporal stochastic block model.

This experiment evaluates reconstruction against the latent expected adjacency
matrices, not just against noisy sampled edges.

Outputs:

- `results/metrics.json`: per-run and aggregate metrics
- `results/summary.md`: compact result table
