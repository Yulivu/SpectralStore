# SpectralStore Experiments and Decisions

Last updated: 2026-04-30

This document is the single source for experiment registry, key metrics,
decision gates, and action items. It merges former experiment logs, progress
reports, checklist docs, and estimator decision memos.

## 1. Canonical Result Directories

| Topic | Directory | Primary artifacts |
| --- | --- | --- |
| Exp1-v2 theory regime (iid) | `experiments/results/exp1_v2/standard/` | `exp1_v2_theory_regime.csv`, `summary.md` |
| Exp1-v2 theory regime (hetero) | `experiments/results/exp1_v2/hetero/` | `exp1_v2_theory_regime.csv`, `summary.md` |
| Exp2 Bitcoin storage sweep | `experiments/results/exp2/` | `sweep_results.csv`, `sweep_results_rmse.csv` |
| Exp3 query latency/index | `experiments/results/query_latency/` | `metrics.json`, `summary.md` |
| Phase3 real-data residual calibration | `experiments/results/phase3_realdata/` | `phase3_metrics.json`, `phase3_summary.md`, `phase3_candidates.csv` |
| Asym temporal dependence | `experiments/results/exp_asym_temporal_dependence/` | `exp_asym_temporal_dependence.csv`, `summary.md` |
| Asym mechanism audit | `experiments/results/asym_mechanism_audit_full/` | `asym_mechanism_audit.csv`, `summary.md` |
| Asym decision gate baseline | `experiments/results/asym_decision_gate_main/` | gate CSVs + `summary.md` |
| Asym decision gate H1 | `experiments/results/asym_decision_gate_h1_numsplits8/` | gate CSVs + `summary.md` |
| Sparse-native smoke | `experiments/results/sparse_path_smoke/` | `metrics.json`, `summary.md` |
| Sparse-native robust smoke | `experiments/results/sparse_path_smoke_robust/` | `metrics.json`, `summary.md` |
| Sparse-path local scaling | `experiments/results/sparse_scaling/` | scaling CSV + summaries |

Legacy Exp1 folders are diagnostic only and should not be cited as current
theory evidence.

## 2. Current Evidence Summary

### 2.1 Theory and Mechanism (Exp1-v2 + audits)

- asym construction is real (non-symmetric matrix, U/V gap, output asymmetry).
- split-asym and current asym path are often nearly equivalent unless mechanism
  settings are changed.
- in completed Exp1-v2 runs, asym is not stably better than `sym_svd` on mean
  max-entrywise error and variance.
- temporal-correlation tests show weak mean-ratio improvement trend at high
  `alpha`, but variance ratio remains above 1.

### 2.2 System Contribution (Exp2/Exp3/Phase3)

- Exp3 demonstrates query-path tradeoff with raw/factor/residual/index/cache.
- robust residual path provides practical correction capability, but storage must
  be controlled with explicit sparse-ratio gate.
- Exp2 field naming is normalized to `compressed_vs_raw_sparse_ratio`.
- invalid compression region is explicitly marked where
  `storage_gate_accepted=false`.

### 2.3 Robust Residual + Storage Control

- `TemporalCOOResidualStore` and gate controls are implemented.
- robust real-data calibration completed for `bitcoin_otc` and `bitcoin_alpha`.
- accepted candidate in current run: `quantile_0_9998_auto_drop`.

## 3. Important Calibration and Ratio Findings

### 3.1 Ratio > 1 root cause and fix

Root causes identified:

1. ratio denominator confusion (dense vs raw sparse).
2. per-snapshot CSR residual overhead (`indptr`) at large `T`.
3. high-rank factor bytes + residual + metadata crossing sparse budget.
4. missing hard gate in early implementation.

Implemented fixes:

- `TemporalCOOResidualStore` for global sparse residual events.
- robust storage gate with actions (`diagnostic`, `drop_residual`, `raise`).
- factor dtype accounting (`factor_storage_dtype_bytes`).
- storage regime sweep script and explicit invalid-region reporting.

### 3.2 Latest local robust calibration (sparse-path scaling)

Baseline robust (before drop gate), `density=0.0008`:

- `n=5000`: error `0.610551`, ratio `1.271078`
- `n=10000`: error `0.611028`, ratio `1.160988`

After `storage_gate_action=drop_residual`, `max_sparse_ratio=1.0`:

- `q=0.9998`, `density=0.0008`
  - `n=5000`: error `0.864411`, ratio `0.693715`
  - `n=10000`: error `0.864721`, ratio `0.560596`
- `q=0.9999`, `density=0.0008`
  - `n=5000`: error `0.933836`, ratio `0.501263`
  - `n=10000`: error `0.934115`, ratio `0.360465`
- `q=0.9998`, `density=0.0012`
  - `n=5000`: error `0.911040`, ratio `0.474697`
  - `n=10000`: error `0.911236`, ratio `0.378803`

Interpretation:

- robust can be moved into valid compression region (`ratio < 1`),
- but error increases due to residual dropping,
- this is a controlled storage/accuracy tradeoff, not a bug.

## 4. Asym Evidence Baseline

Historical baseline observations:

- baseline runs (`asym_decision_gate_main`) reported mean error ratio around `1.018` and variance ratio around `2.697`.
- H1 mechanism (`num_splits=8`) improved both metrics directionally to mean ratio around `1.002` and variance ratio around `2.016`.

Current policy:

- treat these numbers as starting evidence, not as stop conditions.
- continue mechanism redesign and reruns under the Thinking-theory-aligned roadmap.
- evaluate each new iteration with the same metric set for direct comparability.

## 5. Sparse-Native Execution Status

Completed:

- sparse-native support for `spectralstore_asym`, `spectralstore_robust`,
  `sym_svd`, `direct_svd`.
- regression tests for bypassing dense stack in sparse mode.
- local and robust smoke outputs generated.
- scaling runner and local/HPC configs added.
- scaling runner memory guard and run-status fields added:
  - `dense_fallback_used`, `dense_fallback_reason`
  - `failure_reason_category`, `failure_stage`
  - dense stack estimate + `dense_memory_guard_*` diagnostics
- ARD rank-selection mode integrated in compressor paths:
  - `rank_selection_mode=fixed|ard`
  - ARD diagnostics in store metadata (`effective_rank`, `ard_elbo_history`, convergence fields)
  - scaling runner exports ARD status fields (`rank_selection_mode`, `effective_rank`, `ard_converged`, `ard_iterations`)

Pending:

- reproducible `>=50K` sparse-scale run evidence.

## 6. Citation and Claim Rules

Allowed claims:

- system framework and reproducibility pipeline are implemented.
- robust sparse-corruption recovery is strong in compliant checks.
- asym construction/mechanism exists and is audited.
- query bound and threshold defaults are empirically calibrated.

Disallowed claims:

- asym is stably better than `sym_svd` without matching evidence in the current run matrix.
- calibrated defaults are universal constants.
- Exp4 random/targeted attack replaces strict sparse-corruption evidence.

## 7. Repro Command Index

Main checks:

```powershell
python -c "import spectralstore; print('spectralstore import ok')"
python scripts/exp1/run_exp1_theory_validation.py --config <config.yaml>
python scripts/exp1/run_lowsnr_diagnostic.py --config <config.yaml>
python scripts/exp2/run_bitcoin_compression_ratio_sweep.py --config <config.yaml>
python scripts/exp2/run_bitcoin_compression_ratio_sweep_rmse.py --config <config.yaml>
python scripts/exp4/run_synthetic_attack_random.py --out-dir experiments/results/exp4/random_attack
```

## 8. Next Actions

1. continue sparse-native path to larger scales (`50K+`) with explicit memory/run
   diagnostics.
2. keep robust calibration in accepted region and report invalid region clearly.
3. continue direct iterations on asym + ARD + robust unified loop and rerun the matrix after each substantive mechanism change.
