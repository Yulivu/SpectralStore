# Implementation Notes

This document records implementation choices that are intentionally lightweight
or prototype-level. These components are useful for keeping the system runnable
while the core SpectralStore architecture is still being built, but they should
not be treated as final paper-grade implementations.

## Strategy

Current priority is to build SpectralStore as a coherent system first:

1. Factored storage format.
2. Core asymmetric compressor.
3. Robust residual storage.
4. Error-aware query processor.
5. Query optimizer and core queries.
6. Storage/compression accounting.
7. System experiments.

Baselines should remain pluggable and minimal until the system path is stable.
Paper-grade baselines can be added after the interfaces, metrics, and experiment
contracts stop moving.

## Lightweight Or Prototype Components

| Component | Current implementation | Why lightweight | Paper-grade direction |
| --- | --- | --- | --- |
| `AsymmetricSpectralCompressor` | Dense mean/basis SVD with split stitching | Keeps the core path small and testable | Sparse/truncated SVD, tensor-native factorization |
| `RobustAsymmetricSpectralCompressor` | Alternating residual thresholding with MAD/quantile/hybrid rules | Good for sparse outliers, not full robust graph learning | Theory-aware thresholds, robust tensor decomposition |
| `TensorUnfoldingSVDCompressor` | NumPy SVD on mode-1 and mode-2 unfoldings | Simple tensor-entry baseline | Keep as simple baseline; compare with TensorLy methods |
| `CPALSCompressor` | TensorLy `parafac` CP-ALS with component scaling and optional energy pruning | Dense TensorLy baseline, not sparse/observed-edge optimized | Sparse CP or masked observed-edge objective |
| `TuckerHOSVDCompressor` | TensorLy Tucker-ALS followed by projection into the current store format | Uses formal TensorLy Tucker fitting, but the store is still diagonal-factor query format | Native Tucker store/query path or sparse Tucker |
| `QueryEngine.community` | K-means on time-weighted spectral embeddings | No precomputed labels or index yet | Community cache, NMI/ARI experiments, inverted index |
| `link_prob_optimized` / `temporal_trend_optimized` | Simple tolerance rule for reading residuals | Only covers Q1/Q4 and one correction path | Cost-aware optimizer across residual/index/raw paths |
| `ExactMIPSIndex` / `ExactTopNeighborIndex` | Exact dense factor-space scan over right embeddings, with CSR residual candidate reranking | Validates the TOP_NEIGHBOR index path without ANN/PQ complexity | Add PQ/ANN variants and update-aware rebuilds |
| Query latency microbenchmark | Small synthetic fixed-seed script with `time.perf_counter` | Useful for regression checks, not a paper-grade benchmark | Larger datasets, warmup policy, confidence intervals, hardware metadata |
| Experiment management | YAML configs via Hydra/OmegaConf plus `run_metadata.json` outputs | Reproducibility skeleton, not yet full Hydra multirun or MLflow/W&B tracking | Hydra config groups, sweep launchers, optional MLflow/W&B backend |
| Unified evaluation reports | Shared helpers for reconstruction, observed-edge, storage, MRR, and Hits metrics | API is stable enough for scripts, but not yet a full experiment registry/evaluator framework | Dataset-aware evaluators and OGB official evaluator integration |
| `load_ogbl_collab` | Official OGB `LinkPropPredDataset` with yearly snapshots and capped-node preliminary runs | Dense prototype compressors require node caps for smoke-scale runs; 5000 nodes already exceeds the default dense memory budget | Full sparse pipeline and OGB evaluator-based link prediction |
| ARD-like rank pruning | Iterative deterministic component-strength pruning using `|lambda_j| * rms(W_j)` with temporal refit | Engineering approximation to rank selection, not variational Bayesian ARD | Full ARD/VI posterior updates with uncertainty parameters |
| Entrywise bounds | Empirical degree-aware calibration | Useful coverage diagnostics, not formal theorem | Noise-scale and degree-theory bound with calibration |
| Storage estimates | In-memory factor/CSR byte estimates, dense and sparse raw denominators, scalar diagnostic metadata estimate | Fast accounting for early experiments | Serialized bytes, allocator-aware memory reports, residual metadata audit |
| `FactorizedTemporalStore.save_npz/load_npz` | NPZ bundle with dense factor arrays, CSR residual components, and JSON diagnostics | Portable prototype format, not yet a versioned on-disk contract | Versioned manifest, schema validation, compatibility tests |

## Dependency Policy

Core package dependencies should stay small:

- `numpy`
- `scipy`
- `scikit-learn`
- `hydra-core`

Hydra/OmegaConf is now a core dependency because experiment scripts and
reproducibility metadata use YAML configs directly. Optional experiment
dependencies such as TensorLy should not be required for the main test suite.
When TensorLy baselines are added, they should live behind optional imports and
clear skip/error messages.

OGB is an optional experiment dependency. `ogbl-collab` uses the official OGB
loader; a narrow PyTorch 2.6+ compatibility shim is scoped to that loader so OGB
1.3.x processed cache files can be read with `weights_only=False`.

## Current Baseline Policy

Baselines are useful only when they do not slow down core system construction.
For now:

- Treat SymSVD and DirectSVD as complete for their documented definitions:
  mean adjacency, optional symmetrization, then truncated SVD. They still need
  unified evaluation wiring, but not algorithm rewrites.
- Keep TensorUnfoldingSVD as a stable simple tensor-entry baseline.
- Treat CP-ALS and Tucker-ALS as TensorLy-backed dense baselines; they are no
  longer hand-written lightweight implementations, but still need sparse-scale
  variants for large OGB runs.
- Do not spend major implementation time on NMF, BPTF, RPCA, graph summaries, or
  dynamic GNN baselines until the SpectralStore system path is more complete.

## Next System-First Milestones

1. Add versioned serialization manifests and compatibility tests.
2. Add sparse/observed-edge tensor objectives for CP/Tucker.
3. Upgrade ARD-like pruning toward full Bayesian ARD/VI.
4. Add Hydra config groups and sweep launchers.
5. Add broader latency benchmarks after the system path stabilizes.

## Experiment 1 Scope Notes

The current Experiment 1 theory-validation script is aligned with the advisor's
three requested axes: Synthetic-SBM node scaling, Synthetic-SBM time scaling,
and Synthetic-Spiked SNR scaling. Current practical deviations are:

- CP-ALS is excluded from Experiment 1 because TensorLy CP-ALS becomes
  computationally impractical at the larger dense tensor settings on this
  machine.
- The node sweep stops at `n=5000` rather than `n=10000`; the missing point is a
  machine-limit gap, while the remaining points are enough to inspect the trend.
- Repeats are set to `10` rather than `50`, so uncertainty bands are wider but
  the initial trend should still be visible.
