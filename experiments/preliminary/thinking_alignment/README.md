# Thinking Alignment Checks

This preliminary experiment contains smoke-scale checks that align the current
implementation with the research specification in `Thinking.docx` and
`SpectralStore.md`.

Run:

```bash
python scripts/preliminary/run_thinking_alignment_checks.py
```

The runner appends each completed setting to its CSV immediately and skips
completed setting keys on restart, so interrupted runs can be resumed by
rerunning the same command.

Shared smoke-scale overrides are available through `data.n`, `data.T`, and
`experiment.repeats`:

```bash
python scripts/preliminary/run_thinking_alignment_checks.py --set "data.n=500 data.T=20 experiment.repeats=3"
```

Calibration checks reuse the same runner and write to `results/calibration/`
by default so they do not overwrite the base alignment outputs:

```bash
python scripts/preliminary/run_thinking_alignment_checks.py --set "checks=[entrywise_bound_calibration,residual_threshold_calibration] calibration.bound_C_list=[1,2,3,5] calibration.threshold_scale_list=[1,2,3]"
```

Calibration rows are appended as each setting finishes and are skipped on
restart when the same setting key is already present.

The current calibrated system defaults live in YAML, not in compressor code:

- `query.bound_C: 3.0`
- `robust.threshold_scale: 3.0`

These values came from the current calibration sweep
(`n=500,T=20,repeats=3`) and should be treated as configurable empirical
defaults, not global theory constants. Recalibrate when the data regime,
rank, corruption model, or service tolerance changes.

System-level calibration wiring can be smoke-tested through the same runner:

```bash
python scripts/preliminary/run_thinking_alignment_checks.py --set "checks=[system_calibration_sanity] query.bound_C=2 robust.threshold_scale=1.5"
```

This writes under `results/calibration/` by default and checks that Q1 returns
`estimate`, `bound`, `used_residual`, and `method`; that `query.bound_C` scales
the bound; and that `robust.threshold_scale` changes residual storage.

Exp1-v2 theory-regime validation also uses the same runner. It does not
overwrite legacy Exp1 outputs; by default it writes to
`experiments/results/exp1_v2/theory_regime/`:

```bash
python scripts/preliminary/run_thinking_alignment_checks.py --set "checks=[exp1_v2_theory_regime]"
```

Small smoke:

```bash
python scripts/preliminary/run_thinking_alignment_checks.py --set "checks=[exp1_v2_theory_regime] data.n_list=[200] data.T_list=[5] experiment.repeats=2"
```

Regime/noise override:

```bash
python scripts/preliminary/run_thinking_alignment_checks.py --set "checks=[exp1_v2_theory_regime] data.regime_list=[medium_snr,low_snr] data.noise_type=heteroskedastic_entry"
```

Exp1-v2 appends each setting to CSV immediately and skips completed
`method+n+T+regime+noise+seed+repeat` keys on restart. After a run completes,
the CSV is refreshed with repeat-level variance and asym-vs-sym ratio columns
without recomputing completed settings.

Outputs:

- `results/metrics.json`: structured metrics for all checks
- `results/summary.md`: compact alignment summary
- `results/split_asym_sanity.csv`: T1/T2 construction checks
- `results/entrywise_bound_coverage.csv`: bound coverage and violation metrics
- `results/robust_sparse_corruption.csv`: sparse corruption reconstruction,
  anomaly, storage, and timing metrics
- `results/calibration/entrywise_bound_calibration.csv`: bound-constant sweep
- `results/calibration/residual_threshold_calibration.csv`: robust threshold sweep
- `results/calibration/system_calibration_sanity.csv`: config wiring smoke check
- `experiments/results/exp1_v2/theory_regime/exp1_v2_theory_regime.csv`: Exp1-v2 theory-regime sweep
- `results/resolved_config.yaml`
- `results/run_metadata.json`
