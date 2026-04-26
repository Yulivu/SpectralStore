# SpectralStore Current Status

Last updated: 2026-04-26.

This document summarizes the current implementation and experiment state. It
does not change the research target in `Thinking.docx` or `SpectralStore.md`.

## System Implementation

Implemented core storage and compression:

- `FactorizedTemporalStore` with factorized temporal snapshots, residual CSR
  storage, NPZ round trip, storage accounting, and entrywise bound metadata.
- Compression registry with config-driven compressor construction.
- Main methods:
  - `spectralstore_asym`
  - `spectralstore_split_asym_unfolding`
  - `spectralstore_unfolding_asym`
  - `spectralstore_robust`
  - `sym_svd`
  - `direct_svd`
  - `rpca_svd`
  - dense TensorLy CP/Tucker baselines for small feasible cases.
- Query engine:
  - `LINK_PROB`
  - bounded Q1 result shape: `estimate`, `bound`, `used_residual`, `method`
  - tolerance-aware residual correction
  - top-neighbor queries with exact MIPS helper
  - community query from time-weighted embeddings
  - anomaly query over residuals.
- Calibration wiring:
  - `query.bound_C` is YAML/config driven.
  - `robust.threshold_scale` is YAML/config driven.
  - Current defaults are empirical calibration defaults, not theory constants.
- Data loaders:
  - Bitcoin-OTC
  - Synthetic-SBM
  - Synthetic-Spiked
  - Synthetic-Attack with sparse corruption masks
  - ogbl-collab preliminary loader
- Exp1-v2 theory-regime SBM with iid and heteroskedastic observation noise.
- Temporal-correlated SBM for checking whether asymmetric methods benefit from
  AR(1) temporal dependence in the observation noise.
- Evaluation and experiment management:
  - YAML/OmegaConf config loading and strict overrides
  - CSV/summary/metadata output helpers
  - resume support for long-running alignment, calibration, and Exp1-v2 checks.

## Experiments

### Thinking Alignment

Location:

```text
experiments/preliminary/thinking_alignment/results/
```

Purpose:

- Verify split asymmetric construction against the T1/T2 triangular design.
- Check entrywise bound coverage.
- Check robust sparse corruption behavior under `A_t = M* + S_t + H_t`.

Observed:

- Split asymmetric construction is compliant.
- `spectralstore_split_asym_unfolding` and `spectralstore_asym` are effectively
  equivalent in the current construction path.
- C=1 entrywise bound is not stable enough for query use without calibration.
- Robust residual storage is useful under sparse corruption, but threshold
  calibration is needed to avoid residual over-triggering without corruption.

### Calibration

Location:

```text
experiments/preliminary/thinking_alignment/results_n500_t20_r3_calibration/
```

Observed:

- In the current `n=500,T=20,repeats=3` sweep, `spectralstore_asym` needs
  approximately `query.bound_C=3` to reach coverage >= 0.995.
- `robust.threshold_scale=3` is the best current sparse-corruption tradeoff in
  the tested regime.
- These values are configurable defaults, not global constants.

### Exp1-v2 Theory Regime

Current result locations:

```text
experiments/results/exp1_v2/standard/
experiments/results/exp1_v2/hetero/
```

Standard iid run:

- Rows: 640
- Regimes: `medium_snr`, `low_snr`
- Noise: `iid`
- Sweeps: `n_sweep`, `T_sweep`
- Methods: `spectralstore_asym`, `spectralstore_split_asym_unfolding`,
  `sym_svd`, `direct_svd`

Key standard observations:

- Closest slope to -0.5: `-0.378889` for
  `n_sweep / spectralstore_asym / low_snr / iid`, R^2 `0.943054`.
- Mean asym max-error ratio vs `sym_svd`: `1.20647`.
- No variance reduction was observed on average; variance ratio `6.71843`.

Heteroskedastic-entry run:

- Rows: 240
- Regimes: `medium_snr`, `low_snr`
- Noise: `heteroskedastic_entry`
- Sweeps: `n_sweep`, `T_sweep`
- Methods: `spectralstore_asym`, `spectralstore_split_asym_unfolding`,
  `sym_svd`, `direct_svd`

Key heteroskedastic observations:

- Closest slope to -0.5: `-0.51133` for
  `T_sweep / spectralstore_asym / medium_snr / heteroskedastic_entry`,
  R^2 `0.811836`.
- Mean asym max-error ratio vs `sym_svd`: `1.25189`.
- No variance reduction was observed on average; variance ratio `7.57832`.

Interpretation:

- Exp1-v2 is now the theory-regime experiment to cite for entrywise scaling.
- It records both supportive and non-supportive outcomes without changing the
  method definitions.
- Current asym methods show more theory-like slopes in some regimes, but they do
  not yet beat `sym_svd` on mean max-entrywise error in these completed runs.

### Exp2 Bitcoin-OTC

Location:

```text
experiments/results/exp2/
```

Purpose:

- Compression ratio sweep on Bitcoin-OTC.
- Metrics include reconstruction errors, storage ratio, and held-out RMSE.

Status:

- Completed as a system/storage experiment.
- Useful for practical behavior, not a direct synthetic theory validation.

### Asym Temporal Dependence

Location:

```text
experiments/results/exp_asym_temporal_dependence/
```

Purpose:

- Test whether `spectralstore_asym` starts to outperform `sym_svd` when
  observations follow `A_t = M* + H_t` with `H_t = alpha H_{t-1} + epsilon_t`.
- Sweep `alpha` to separate iid noise (`alpha=0`) from temporally correlated
  noise.

Smoke status:

- `alpha=0`, `n=200`, `T=10`, `repeats=2` ran successfully.
- CSV append and resume behavior were verified.
- The smoke result reproduced the expected iid no-advantage behavior:
  `spectralstore_asym` mean error ratio vs `sym_svd` was about `1.00896`.

Interpretation:

- The experiment is now implemented and ready for larger alpha sweeps.
- A negative result is valid evidence; the summary explicitly reports when no
  asym advantage is observed.

### Exp4 Robustness

Location:

```text
experiments/results/exp4/
```

Purpose:

- Synthetic attack robustness with random and targeted attacks.
- Includes `rpca_svd` baseline using matrix PCP/ADMM on the temporal mean.

Status:

- Useful robustness diagnostics.
- Targeted/random attack variants are not the strict Thinking sparse-corruption
  model; the alignment sparse-corruption check is the more compliant test.

## Legacy / Removable Results

The following old Exp1 directories are legacy diagnostic outputs and can be
removed after confirming no external reference depends on them:

```text
experiments/results/exp1/
experiments/results/exp1_smoke/
experiments/results/exp1_pilot/
experiments/results/exp1_low_snr_smoke/
```

They were produced before Exp1-v2 and should not be treated as the current
theory-regime result.

## Current Validation

Most recent test command:

```powershell
python -m pytest -p no:cacheprovider
```

Result:

```text
60 passed, 1 warning
```

The warning is the local joblib CPU-core detection warning and is not a test
failure.
