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
  preliminary/
    run_preliminary_*.py
    run_bitcoin_residual_sweep.py
    run_query_latency_microbenchmark.py
  scaling/
    run_synthetic_*_scaling.py
  smoke/
    run_smoke_quickstart.py
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
python scripts/preliminary/run_preliminary_synthetic_sbm.py --set num_repeats=1 --set rank=3
```

Experiment 1 is now under `scripts/exp1/`:

```bash
python scripts/exp1/run_exp1_theory_validation.py
python scripts/exp1/run_exp1_theory_validation.py --refresh-from-results
python scripts/exp1/run_lowsnr_diagnostic.py
python scripts/exp1/plot_theory_from_results.py
```

The resolved configuration is also written to `resolved_config.yaml` next to the
metrics and summary.

`run_bitcoin_residual_sweep.py` is the current stage-three residual validation
entry point for Bitcoin-OTC. It writes `residual_sweep_metrics.json` and
`residual_sweep_summary.md`, and marks a robust setting accepted only if sparse
storage ratio and held-out error regression stay within the YAML thresholds.

`run_preliminary_synthetic_attack.py` is also the Q5 validation entry point. It
reports injected anomaly edges, Q5 detected edges, Q5 precision, and Q5 recall.

`run_preliminary_ogbl_collab.py` loads `ogbl-collab` through OGB's official
`LinkPropPredDataset` and runs a capped-node preliminary real-data comparison.
It uses the unified evaluation report helpers for reconstruction,
observed-edge, storage, MRR, and Hits metrics.
The script checks the dense tensor memory budget before compression; large
`max_nodes` settings require a sparse compressor path.
