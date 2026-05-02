# SpectralStore Experiments and Decisions

Last updated: 2026-05-02

This document is the single source for experiment registry, key metrics,
decision gates, and action items. It merges former experiment logs, progress
reports, checklist docs, and estimator decision memos.

## 1. Canonical Result Directories

| Topic | Directory | Primary artifacts |
| --- | --- | --- |
| Active rerun config set | `experiments/configs/` | `exp1/`, `exp2/`, `exp3/`, `exp4/`, `exp4_v2/`, `exp5/` YAMLs |
| Exp1 theory validation | `experiments/results/exp1/mainline/` | `results.csv`, plots, `metrics.json`, `summary.md` |
| Exp2 Bitcoin storage sweep | `experiments/results/exp2/mainline/` | `sweep_results.csv`, `sweep_results_rmse.csv`, residual-boundary CSV |
| Exp3 query/index benchmark | `experiments/results/exp3/query_benchmark/` | `query_records.csv`, `summary.csv`, `metrics.json`, `summary.md` |
| Exp4_v2 residual-query robustness | `experiments/results/exp4_v2/residual_query_robustness/` | `raw_records.csv`, `summary.csv`, `metrics.json`, `summary.md` |
| Exp5 ARD diagnostic | `experiments/results/exp5/ard_diagnostic/` | ARD diagnostic CSVs + summaries |
| Phase3 real-data residual calibration | `experiments/results/phase3_realdata/` | `phase3_metrics.json`, `phase3_summary.md`, `phase3_candidates.csv` |
| Asym temporal dependence | `experiments/results/exp_asym_temporal_dependence/` | `exp_asym_temporal_dependence.csv`, `summary.md` |
| Asym mechanism audit | `experiments/results/asym_mechanism_audit_full/` | `asym_mechanism_audit.csv`, `summary.md` |
| Asym decision gate baseline | `experiments/results/asym_decision_gate_main/` | gate CSVs + `summary.md` |
| Asym decision gate H1 | `experiments/results/asym_decision_gate_h1_numsplits8/` | gate CSVs + `summary.md` |
| Sparse-native smoke | `experiments/results/sparse_path_smoke/` | `metrics.json`, `summary.md` |
| Sparse-native robust smoke | `experiments/results/sparse_path_smoke_robust/` | `metrics.json`, `summary.md` |
| Sparse-path local scaling | `experiments/results/sparse_scaling/` | scaling CSV + summaries |

Legacy directories are diagnostic only and should not be cited as current
theory evidence after mechanism-level updates.

## 2. Current Evidence Summary

### 2.1 Theory and Mechanism

- current paper-facing method is `spectralstore_thinking`.
- historical `spectralstore_asym`, `spectralstore_robust`, split/unfolding, and
  alternating diagnostic variants have been removed from the public registry.
- existing historical results are diagnostic only. Claim-bearing evidence must be
  regenerated under `experiments/results/exp1/mainline`,
  `experiments/results/exp2/mainline`, `experiments/results/exp3/query_benchmark`,
  `experiments/results/exp4_v2/residual_query_robustness`, and
  `experiments/results/exp5/ard_diagnostic`.
- previous negative evidence remains useful as a warning: asymmetry alone did not
  reliably beat `sym_svd`; the current hypothesis is specifically the unified
  mode-3 + asym + ARD + robust loop.

### 2.2 System Contribution (Exp2/Exp3/Phase3)

- Exp3 demonstrates query-path tradeoff with factor/residual/index paths.
- Exp4_v2 is the current claim-bearing robustness experiment for residual anomaly
  capture and Q1/Q4/Q5 correction.
- robust residual path provides practical correction capability, but storage must
  be controlled with explicit sparse-ratio gate.
- Exp2 field naming is normalized to `compressed_vs_raw_sparse_ratio`.
- invalid compression region is explicitly marked where `storage_gate_accepted=false`.
- `storage_gate_accepted` denotes the final post-action storage state; before-action
  diagnostics are preserved separately.

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

### 3.2 Historical local robust calibration (diagnostic only)

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

## 4. Removed Historical SpectralStore Variants

Removed variants:

- `spectralstore_asym`
- `spectralstore_robust`
- `spectralstore_unfolding_asym`
- `spectralstore_split_asym_unfolding`
- `spectralstore_asym_alternating_robust` / alias

Current policy:

- do not rerun or cite these variants as current SpectralStore methods.
- use `spectralstore_thinking` for the paper method.
- use stable external baselines (`sym_svd`, `direct_svd`, `tensor_unfolding_svd`,
  `nmf`, `rpca_svd`, CP/Tucker when dependencies are available) for comparison.

## 5. Sparse-Native Execution Status

Completed:

- sparse-native baseline support for `sym_svd` and `direct_svd`.
- mainline `spectralstore_thinking` remains dense-stack based pending a
  correctness-preserving sparse mode.
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
- `spectralstore_thinking` implements the audited mode-3 + asym + ARD + robust offline loop.
- query bound and threshold defaults are empirically calibrated.

Disallowed claims:

- `spectralstore_thinking` is stably better than `sym_svd` without matching evidence in the current run matrix.
- calibrated defaults are universal constants.
- legacy Exp4 random/targeted/injection attack replaces Exp4_v2 residual-query evidence.

## 7. Repro Command Index

Main checks:

```powershell
python -c "import spectralstore; print('spectralstore import ok')"
python scripts/data/download_dataset.py bitcoin_otc
python scripts/exp1/run_exp1_theory_validation.py --config experiments/configs/exp1/theory_validation.yaml
python scripts/exp1/run_lowsnr_diagnostic.py
python scripts/exp2/run_bitcoin_compression_ratio_sweep.py --config experiments/configs/exp2/bitcoin_sweep.yaml
python scripts/exp2/run_bitcoin_compression_ratio_sweep_rmse.py --config experiments/configs/exp2/bitcoin_sweep_rmse.yaml
python scripts/exp2/run_bitcoin_residual_boundary_sweep.py --config experiments/configs/exp2/bitcoin_residual_boundary.yaml
python scripts/exp3/run_query_benchmark.py --config experiments/configs/exp3/query_benchmark.yaml
python scripts/exp4_v2/run_residual_query_robustness.py --config experiments/configs/exp4_v2/residual_query_robustness.yaml
python scripts/exp4/run_synthetic_attack_random.py --config experiments/configs/exp4/random_attack.yaml
python scripts/exp4/run_synthetic_attack_targeted.py --config experiments/configs/exp4/targeted_attack.yaml
python scripts/exp4/run_synthetic_attack_injection.py --config experiments/configs/exp4/injection_attack.yaml
python scripts/exp5/run_ard_diagnostic.py --config experiments/configs/exp5/ard_diagnostic.yaml
python scripts/exp5/run_ard_rank_selection.py --config experiments/configs/exp5/ard_rank_selection.yaml
```

AutoDL/HPC system-direction rerun:

```bash
bash scripts/hpc/run_system_direction.sh
```

AutoDL/HPC mainline rerun:

```bash
bash scripts/hpc/init_hpc.sh
bash scripts/hpc/run_all_mainline.sh
```

## 8. Next Actions

1. continue sparse-native path to larger scales (`50K+`) with explicit memory/run
   diagnostics.
2. keep robust calibration in accepted region and report invalid region clearly.
3. continue direct iterations on `spectralstore_thinking` and rerun the matrix after each substantive mechanism change.
4. for new HPC sessions, follow [docs/AUTODL_RUNBOOK.md](AUTODL_RUNBOOK.md) and
   run `scripts/hpc/run_all_mainline.sh` before reusing historical result
   folders in summaries.
