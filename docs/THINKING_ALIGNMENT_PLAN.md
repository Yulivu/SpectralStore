# Thinking Alignment Plan

This document tracks implementation alignment against `Thinking.docx` and
`SpectralStore.md`. It does not change the research target based on current
experimental outcomes.

## 1. Thinking.docx Requirement Checklist

| Requirement | Status | Notes |
|---|---|---|
| T1/T2 snapshot split for asymmetric construction | Implemented and checked | `spectralstore_split_asym_unfolding` uses a random snapshot split. The alignment check replays the split from the YAML seed. |
| Upper triangle from T1 mean | Checked | `upper_triangle_source_error` is reported in CSV. |
| Lower triangle from T2 mean | Checked | `lower_triangle_source_error` is reported in CSV. |
| Diagonal from average of T1/T2 means | Checked | `diag_consistency_error` is reported in CSV. |
| Independent T1/T2 noise under synthetic data | Diagnostic | `noise_correlation_T1_T2` is reported; low absolute correlation supports the assumption. |
| Q1 entrywise bound | Implemented and checked | `FactorizedTemporalStore.precompute_bound_params` caches `sigma_max` and `mu`; Q1 wrapper exposes estimate and bound. |
| Sparse corruption model `A_t = M* + S_t + H_t` | Implemented for smoke checks | Synthetic attack loader now supports sparse spike modes and keeps corruption masks. |
| Sparse anomaly recovery through residual storage | Checked | Robust corruption CSV reports precision, recall, F1, residual storage, and reconstruction error to `M*`. |

## 2. Current Implementation vs Requirements

The implementation has a unified compressor registry, synthetic data loaders,
evaluation metrics, query engine, and experiment output helpers. The new checks
reuse those pieces rather than adding isolated one-off scripts.

| Component | Current mapping |
|---|---|
| Compressor registry | Methods are called through `create_compressor`, including `spectralstore_asym`, `spectralstore_split_asym_unfolding`, `spectralstore_robust`, `sym_svd`, `direct_svd`, and `rpca_svd`. |
| Synthetic data | Uses existing SBM/Spiked/Attack loaders. Sparse corruption modes extend `make_synthetic_attack`. |
| Evaluation | New reusable metrics live in `spectralstore.evaluation.metrics`. |
| Query layer | Legacy `link_prob` still returns a float; `link_prob_with_error` and `link_prob_result` expose bounds without breaking callers. |
| Experiment management | Uses `load_experiment_config` and `write_experiment_outputs`, so runs write metrics, summary, resolved config, and metadata. |
| Calibration wiring | `query.bound_C` and `robust.threshold_scale` are YAML parameters. Defaults are empirical calibration values, not hard-coded theory constants. |

## 3. Experiment Configuration and Running

Config:

```text
experiments/preliminary/thinking_alignment/configs/default.yaml
```

Run all three smoke checks:

```bash
python scripts/preliminary/run_thinking_alignment_checks.py
```

Run with OmegaConf overrides:

```bash
python scripts/preliminary/run_thinking_alignment_checks.py --set split_asym.num_repeats=1
```

The config has three sections:

- `split_asym`: SBM dimensions, rank, split seed, and construction tolerances.
- `entrywise_bound`: dataset, methods, bound constant, and method list for bound precomputation.
- `robust_sparse_corruption`: corruption modes, corruption rate/magnitude, robust threshold settings, and method list.

Calibration and system parameters are also config-driven:

- `calibration.bound_C_list`: sweep values for entrywise-bound calibration.
- `calibration.threshold_scale_list`: sweep values for robust residual threshold calibration.
- `query.bound_C`: calibrated multiplier used by Q1/query-layer bounds.
- `query.loose_tolerance` and `query.tight_tolerance`: smoke tolerances for residual gating checks.
- `robust.threshold_scale`: calibrated multiplier applied to the robust compressor's base residual threshold.

Run calibration:

```bash
python scripts/preliminary/run_thinking_alignment_checks.py --set "checks=[entrywise_bound_calibration,residual_threshold_calibration] calibration.bound_C_list=[1,2,3,5] calibration.threshold_scale_list=[1,2,3]"
```

Run the system wiring sanity check:

```bash
python scripts/preliminary/run_thinking_alignment_checks.py --set "checks=[system_calibration_sanity] query.bound_C=2 robust.threshold_scale=1.5"
```

The current default `query.bound_C=3.0` came from the current
`n=500,T=20,repeats=3` calibration where `spectralstore_asym` needed C=3 for
coverage >= 0.995. The current default `robust.threshold_scale=3.0` came from
the current sparse-corruption threshold calibration. Both are configurable
defaults for this regime, not global theory constants or universal optima.

Run Exp1-v2 theory-regime scaling validation:

```bash
python scripts/preliminary/run_thinking_alignment_checks.py --set "checks=[exp1_v2_theory_regime]"
```

Smoke-scale Exp1-v2:

```bash
python scripts/preliminary/run_thinking_alignment_checks.py --set "checks=[exp1_v2_theory_regime] data.n_list=[200] data.T_list=[5] experiment.repeats=2"
```

Exp1-v2 uses low-SNR SBM regimes from YAML (`medium_snr`, `low_snr`,
`very_low_snr`) and supports `iid`, `heteroskedastic_entry`, and
`heteroskedastic_node` observation noise. It evaluates errors against
`M_star`, writes rows incrementally, and refreshes the CSV with repeat-level
variance and asym-vs-sym ratios after the run. Legacy Exp1 under
`experiments/results/exp1/` remains diagnostic and is not overwritten.

Run the asymmetric temporal-dependence check:

```bash
python scripts/preliminary/run_thinking_alignment_checks.py --set "checks=[exp_asym_temporal_dependence]"
```

Smoke-scale temporal-dependence check:

```bash
python scripts/preliminary/run_thinking_alignment_checks.py --set "checks=[exp_asym_temporal_dependence] data.alpha_list=[0.0] data.n_list=[200] data.T_list=[10] experiment.repeats=2"
```

This experiment uses a temporal-correlated SBM with `A_t = M* + H_t` and
`H_t = alpha H_{t-1} + epsilon_t`. It reports whether asym error or variance
falls below `sym_svd` as `alpha` increases.

## 4. Output Locations

Following the existing preliminary experiment convention, outputs go under:

```text
experiments/preliminary/thinking_alignment/results/
```

Expected outputs:

- `metrics.json`
- `summary.md`
- `resolved_config.yaml`
- `run_metadata.json`
- `split_asym_sanity.csv`
- `entrywise_bound_coverage.csv`
- `robust_sparse_corruption.csv`

Calibration outputs go under:

```text
experiments/preliminary/thinking_alignment/results/calibration/
```

Expected calibration outputs:

- `entrywise_bound_calibration.csv`
- `residual_threshold_calibration.csv`
- `system_calibration_sanity.csv`
- `summary.md`
- `resolved_config.yaml`
- `run_metadata.json`

Current Exp1-v2 outputs go under:

```text
experiments/results/exp1_v2/standard/
experiments/results/exp1_v2/hetero/
```

Expected Exp1-v2 outputs:

- `exp1_v2_theory_regime.csv`
- `summary.md`
- `metrics.json`
- `resolved_config.yaml`
- `run_metadata.json`

Temporal-dependence outputs go under:

```text
experiments/results/exp_asym_temporal_dependence/
```

Expected temporal-dependence outputs:

- `exp_asym_temporal_dependence.csv`
- `summary.md`
- `metrics.json`
- `resolved_config.yaml`
- `run_metadata.json`

## 5. Existing Experiments That Are Compliant

| Experiment | Compliance status |
|---|---|
| Preliminary Synthetic-SBM | Compliant as a reconstruction/bound-scaling diagnostic. It uses YAML configs and existing loaders. |
| Preliminary Synthetic-Attack | Partially compliant. It evaluates Q5 and attack edges, but older attack modes are not the strict `M* + S_t + H_t` sparse corruption model. |
| Bitcoin-OTC preliminary and residual sweep | Compliant for real-data storage/error reporting, but not a direct Thinking synthetic-theory validation. |
| Exp2 Bitcoin compression ratio sweep | Useful system experiment, but it is script-specific and not the primary Thinking alignment check. |
| Asym temporal dependence | Compliant as a targeted diagnostic for whether temporal dependence creates an asym advantage. |

## 6. Existing Experiments That Are Diagnostic Only

| Experiment/result | Why diagnostic |
|---|---|
| Exp1 legacy results under `experiments/results/exp1` | Generated before the new entrywise bound coverage check; should not be treated as final bound validation. |
| Exp1-v2 results under `experiments/results/exp1_v2/standard` and `experiments/results/exp1_v2/hetero` | Compliant with the theory-regime design and should replace legacy Exp1 for current theory-regime discussion. |
| Exp4 random attack | Random flip attack did not stress NMI and does not implement explicit `S_t` sparse corruption masks. |
| Exp4 targeted attack | Useful stress test, but targeted edge addition is not the strict sparse-spike corruption model and showed non-monotone NMI behavior. |
| Low-SNR Exp1 diagnostic | Useful for risk discovery; not a complete specification experiment. |

## 7. Next Steps

1. Re-run calibration when changing dataset scale, rank, corruption model, or tolerance targets.
2. Decide whether Exp1 should be rerun as a bound-coverage experiment rather than only a max-entrywise scaling experiment.
3. Run the Thinking alignment checks at larger `n`, `T`, and repeats only after smoke metrics remain stable.
4. Keep `query.bound_C` and `robust.threshold_scale` configurable; do not promote current empirical values into algorithm constants.
5. Keep negative or non-supportive results as diagnostics; do not rewrite the research goal around them.
