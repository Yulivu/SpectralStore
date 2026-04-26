# Experiments

Each experiment lives in its own subdirectory:

```text
experiments/
  preliminary/
    bitcoin_otc/
      configs/
      results/
      README.md
  smoke/
    quickstart/
      configs/
      results/
      README.md
```

Configs are YAML files loaded through Hydra/OmegaConf. Scripts accept dotlist
overrides with repeated `--set` flags, for example:

```bash
python scripts/preliminary/run_query_latency_microbenchmark.py --set num_queries=20 --set rank=3
```

Each run writes:

- `metrics.json` or an experiment-specific metrics filename
- `summary.md` or an experiment-specific summary filename
- `resolved_config.yaml`
- `run_metadata.json`

`run_metadata.json` records the command, resolved config hash, environment,
package versions, git commit, and dirty status. `results/` folders are ignored by
Git except for `.gitkeep` placeholders. Commit configs and README files so each
experiment remains reproducible.

Bitcoin-OTC residual sweeps use `configs/residual_sweep.yaml`. A candidate
passes the stage-three storage gate only when its compressed-vs-raw-sparse ratio
is below the configured limit, currently `1.0`, and held-out observed-edge error
does not exceed the configured regression tolerance.

The `preliminary/ogbl_collab` experiment uses the official OGB loader. Keep the
default node cap for smoke runs; full-scale OGB experiments should be added only
after the sparse compressor path is ready.
