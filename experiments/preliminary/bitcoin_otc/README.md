# Preliminary Bitcoin-OTC Experiment

Goal: run the first real-data comparison between SpectralStore's asymmetric
spectral compressor, robust residual path, and SVD baselines.

Dataset: SNAP Bitcoin-OTC signed trust network.

Outputs:

- `results/metrics.json`: reconstruction and held-out edge metrics
- `results/summary.md`: human-readable result table
- `results/residual_sweep_metrics.json`: robust residual threshold sweep metrics
- `results/residual_sweep_summary.md`: storage-gated robust residual sweep table

The residual sweep treats `compressed_vs_raw_sparse_ratio < 1` as a hard
stage-three gate for Bitcoin-OTC. Candidates must also keep held-out
observed-edge RMSE/MAE within the configured regression tolerance.
