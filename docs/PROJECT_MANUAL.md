# SpectralStore Project Manual

Last updated: 2026-04-30

This is the single engineering + planning manual for SpectralStore. It merges
the previous status, roadmap, implementation notes, and project completion
content. The goal is to keep one up-to-date source of truth.

## 1. Project Definition

SpectralStore is a temporal-graph approximate query processing (AQP) storage and
query engine. The system-level target is:

- compress temporal snapshots into a factorized physical store,
- answer core queries directly in compressed space,
- support calibrated error-aware query paths,
- support robust sparse residual correction and anomaly detection,
- report storage/accuracy/latency tradeoffs under reproducible experiments.

Supported query API:

- `LINK_PROB(u, v, t)`
- `TOP_NEIGHBOR(u, t, k)`
- `COMMUNITY(t)`
- `TEMPORAL_TREND(u, v, t1, t2)`
- `ANOMALY_DETECT(t, threshold)`

## 2. Current Architecture

Core package layout:

- `src/spectralstore/compression`: factorized store + compressors
- `src/spectralstore/query_engine`: Q1-Q5 and optimized query paths
- `src/spectralstore/index`: exact MIPS and ANN prototype
- `src/spectralstore/data_loader`: synthetic + real dataset loaders
- `src/spectralstore/evaluation`: metrics, reports, experiment IO helpers

Main implemented methods:

- `spectralstore_asym`
- `spectralstore_split_asym_unfolding`
- `spectralstore_robust`
- `spectralstore_asym_alternating_robust` (diagnostic compressor)
- `sym_svd`
- `direct_svd`
- `rpca_svd`
- tensor baselines (`tensor_unfolding_svd`, `cp_als`, `tucker_hosvd`)

## 3. Implemented System Capabilities

### 3.1 Storage and Serialization

- `FactorizedTemporalStore` with factors + residuals + diagnostics metadata.
- residual physical formats:
  - snapshot CSR tuple,
  - `TemporalCOOResidualStore` with global `(t, u, v, value)` coordinates.
- unified residual access:
  - `residual_value()`
  - `residual_snapshot()`
  - `residual_row()`
- storage accounting:
  - factor bytes with configurable dtype accounting
  - compressed bytes and ratio vs raw dense/sparse
- NPZ round-trip with schema/manifest:
  - `store_schema_version`
  - `store_manifest_json`
  - compatibility checks and future-version rejection

### 3.2 Compression / Robustness

- asymmetric split construction and multi-split stitching.
- robust residual thresholding with:
  - `mad`, `quantile`, `hybrid` modes
  - threshold scaling
- rank selection modes:
  - `fixed` (default fixed-rank truncation)
  - `ard` (variational ARD shrinkage with ELBO convergence diagnostics)
- storage-aware robust controls:
  - `residual_storage_format` (`csr`, `temporal_coo`, `auto`)
  - `max_sparse_ratio`
  - `storage_gate_action` (`diagnostic`, `drop_residual`, `raise`)
  - `factor_storage_dtype_bytes`

### 3.3 Query / Index

- Q1/Q4 bounded result schema:
  - `estimate`, `bound`, `used_residual`, `method`, `satisfied_error_tolerance`
- Q2 exact index path and ANN prototype path.
- query optimizer paths for Q2:
  - factor-only, factor+residual, indexed, raw fallback.
- Q3/Q4 cache helpers and cache diagnostics.
- Q2/Q3/Q5 batch helper APIs.

### 3.4 Sparse-Native Path Progress

- sparse-native mode now supported for:
  - `spectralstore_asym`
  - `spectralstore_robust`
  - `sym_svd`
  - `direct_svd`
- sparse snapshots can bypass `_as_dense_stack` in those paths.
- smoke outputs are now regenerated ad hoc and should be treated as run artifacts
  instead of committed canonical directories.
- scaling scripts/configs from the deleted `scripts/preliminary` tree are no
  longer part of the active repository state.
- scaling runner now emits run-status diagnostics:
  - `dense_fallback_used`, `dense_fallback_reason`
  - `failure_reason_category`, `failure_stage`
  - dense stack estimate + memory guard fields (`dense_memory_guard_*`)
  - rank-selection fields (`rank_selection_mode`, `effective_rank`, `ard_converged`, `ard_iterations`)

## 4. Data Coverage

Implemented loaders/generators:

- Synthetic-SBM
- Synthetic-Spiked
- Synthetic-Attack (including sparse corruption masks)
- temporal-correlated SBM
- Bitcoin-OTC
- Bitcoin-Alpha
- ogbl-collab preliminary loader

## 5. Experiment Framework Contract

All main scripts use YAML/OmegaConf config loading and support dotlist overrides.
Standard outputs:

- `metrics.json`
- `summary.md`
- `resolved_config.yaml`
- `run_metadata.json`

Common features:

- strict override key validation
- CSV append/resume in long-running checks
- reproducibility metadata and config hash

## 6. Practical Defaults and Theory Priorities

Current calibrated defaults (empirical, not theory constants):

- `query.bound_C = 3.0`
- `robust.threshold_scale = 3.0`

Current implementation priorities:

- align the compression engine to the Thinking theoretical form first.
- use engineering constraints as verification tools, not as mechanism blockers.
- keep empirical defaults (`bound_C`, `threshold_scale`) calibrated per dataset.
- publish diagnostics and negative results together with positive results.

## 7. Canonical Working Plan

The project now progresses with Thinking-theory-first priorities:

1. implement a unified offline loop for asym + tensor + ARD + robust,
2. tighten theoretical consistency before broad engineering expansion,
3. run reproducible fixed-vs-ARD and NoAsym/NoRobust ablations as default evidence,
4. iterate mechanisms directly when metrics fail instead of pausing by gate policy.

Current phase summary:

- Phase 0 (repo hygiene and evidence baseline): completed.
- Phase 1 (store/query contracts): completed.
- Phase 2 (query latency/index tradeoff): completed with benchmark outputs.
- Phase 3 (real-data residual calibration + storage gate): completed.
- Asym mechanism diagnostics: executed and retained as baseline evidence for new theory-aligned iterations.

## 8. Minimal Operations Runbook

Development checks:

```powershell
python -c "import spectralstore; print('spectralstore import ok')"
```

Representative command patterns:

```powershell
python scripts/exp1/run_exp1_theory_validation.py --config <config.yaml>
python scripts/exp2/run_bitcoin_compression_ratio_sweep.py --config <config.yaml>
python scripts/exp2/run_bitcoin_compression_ratio_sweep_rmse.py --config <config.yaml>
python scripts/exp4/run_synthetic_attack_random.py --out-dir experiments/results/exp4/random_attack
```

## 9. Baseline Policy

Current baseline strategy:

- keep core baselines stable: `sym_svd`, `direct_svd`, tensor unfold baseline,
  CP/Tucker, `rpca_svd`.
- optional/heavy baselines may run in staged batches, but do not constrain theory-aligned compressor redesign.
- dynamic GNN and other heavy baselines are secondary to finishing the core Thinking-aligned compression engine.

## 10. Out-of-Scope / Not Immediate

- none at policy level; scope is controlled by available compute and reproducibility.
- near-term focus remains offline compressor quality rather than online incremental updates.
