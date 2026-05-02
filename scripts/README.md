# Scripts

Command-line entry points for dataset download, preprocessing, and experiments.

Scripts are grouped by purpose where possible:

```text
scripts/
  data/
    download_dataset.py
  exp1/
    run_exp1_theory_validation.py
    run_lowsnr_diagnostic.py
    plot_theory_from_results.py
    time_exp1_methods.py
  exp2/
    run_bitcoin_compression_ratio_sweep.py
    run_bitcoin_compression_ratio_sweep_rmse.py
    run_bitcoin_residual_boundary_sweep.py
  exp3/
    run_query_benchmark.py
  exp4/
    run_synthetic_attack_random.py
    run_synthetic_attack_targeted.py
    run_synthetic_attack_injection.py
  exp4_v2/
    run_residual_query_robustness.py
  exp5/
    run_ard_diagnostic.py
    run_ard_rank_selection.py
  hpc/
    init_hpc.sh
    run_all_mainline.sh
    run_system_direction.sh
```

Scripts should be thin wrappers around code in `src/spectralstore/` so the core
logic remains importable and testable.

Experiment scripts use YAML configs through Hydra/OmegaConf. Each run writes:

- `metrics.json`
- `summary.md`
- `run_metadata.json`

`run_metadata.json` records the command, resolved config, Python/platform
details, selected package versions, and git branch/commit/dirty status.

Most experiment scripts support OmegaConf dotlist overrides:

```bash
python scripts/exp1/run_exp1_theory_validation.py --config <config.yaml> --set num_repeats=1
```

Experiment 1 is now under `scripts/exp1/`:

```bash
python scripts/exp1/run_exp1_theory_validation.py
python scripts/exp1/run_exp1_theory_validation.py --refresh-from-results
python scripts/exp1/run_lowsnr_diagnostic.py
python scripts/exp1/plot_theory_from_results.py
```

Current active entry points are under `scripts/exp1`, `scripts/exp2`,
`scripts/exp3`, `scripts/exp4`, `scripts/exp4_v2`, and `scripts/exp5`.
Use explicit `--config` arguments for reproducibility. AutoDL/HPC setup is documented in
[`docs/AUTODL_RUNBOOK.md`](../docs/AUTODL_RUNBOOK.md).

The resolved configuration is also written to `resolved_config.yaml` next to the
metrics and summary.

Deprecated preliminary/scaling/smoke script references have been removed from
this repository state.

System-direction rerun for the current A-D route:

```bash
bash scripts/hpc/run_system_direction.sh
```

It runs Exp3 query/index, Exp4_v2 residual-query robustness, and the ARD
diagnostic. Large runs should use HPC; local runs should override configs to
small `num_nodes`, `num_steps`, and repeat counts.
