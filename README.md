# SpectralStore

SpectralStore is a research-oriented storage and query engine for temporal
graphs. It focuses on approximate query processing (AQP) over compressed
factorized graph snapshots, with robust sparse residual correction and
reproducible system experiments.

## What The System Does

- compress temporal graph snapshots into factorized storage,
- answer Q1-Q5 directly in compressed space,
- provide calibrated error-aware query outputs for Q1/Q4,
- support residual-based anomaly detection and query correction,
- expose storage/accuracy/latency tradeoffs for system evaluation.

Supported query API:

| Query | Purpose |
| --- | --- |
| `LINK_PROB(u, v, t)` | estimate connection score/probability at time `t` |
| `TOP_NEIGHBOR(u, t, k)` | top-`k` likely neighbors at time `t` |
| `COMMUNITY(t)` | community inference for snapshot `t` |
| `TEMPORAL_TREND(u, v, t1, t2)` | pair trend over a time window |
| `ANOMALY_DETECT(t, threshold)` | residual-based anomaly edge detection |

## Minimal Docs

Core project documentation:

- [docs/PROJECT_MANUAL.md](docs/PROJECT_MANUAL.md)
- [docs/EXPERIMENTS_AND_DECISIONS.md](docs/EXPERIMENTS_AND_DECISIONS.md)
- [docs/AUTODL_RUNBOOK.md](docs/AUTODL_RUNBOOK.md)
- [docs/SYSTEM_DIRECTION_A_TO_D_ZH.md](docs/SYSTEM_DIRECTION_A_TO_D_ZH.md)

## Repository Layout

```text
data/
  raw/        Downloaded datasets (ignored by Git)
  interim/    Temporary conversions (ignored by Git)
  processed/  Derived artifacts (ignored by Git)
src/spectralstore/
  compression/      Compression algorithms and factorized stores
  query_engine/     Query execution over compressed representations
  index/            Exact/ANN index prototypes
  data_loader/      Dataset loaders and synthetic generators
  evaluation/       Metrics and experiment helpers
scripts/            Active experiment and data entry points
docs/               Project manual and decisions
```

## Quick Start

```powershell
python -m pip install -e ".[dev]"
python -c "import spectralstore; print('spectralstore import ok')"
python scripts/data/download_dataset.py bitcoin_otc
```

Run representative experiments:

```powershell
python scripts/exp1/run_exp1_theory_validation.py --config experiments/configs/exp1/theory_validation.yaml
python scripts/exp2/run_bitcoin_compression_ratio_sweep.py --config experiments/configs/exp2/bitcoin_sweep.yaml
python scripts/exp2/run_bitcoin_compression_ratio_sweep_rmse.py --config experiments/configs/exp2/bitcoin_sweep_rmse.yaml
python scripts/exp3/run_query_benchmark.py --config experiments/configs/exp3/query_benchmark.yaml
python scripts/exp4_v2/run_residual_query_robustness.py --config experiments/configs/exp4_v2/residual_query_robustness.yaml
python scripts/exp5/run_ard_diagnostic.py --config experiments/configs/exp5/ard_diagnostic.yaml
```

Historical robustness/rank-selection diagnostics:

```powershell
python scripts/exp4/run_synthetic_attack_random.py --config experiments/configs/exp4/random_attack.yaml
python scripts/exp4/run_synthetic_attack_targeted.py --config experiments/configs/exp4/targeted_attack.yaml
python scripts/exp4/run_synthetic_attack_injection.py --config experiments/configs/exp4/injection_attack.yaml
python scripts/exp5/run_ard_rank_selection.py --config experiments/configs/exp5/ard_rank_selection.yaml
```

For AutoDL/HPC reruns, follow [docs/AUTODL_RUNBOOK.md](docs/AUTODL_RUNBOOK.md):

```bash
bash scripts/hpc/init_hpc.sh
bash scripts/hpc/run_system_direction.sh
bash scripts/hpc/run_all_mainline.sh
```

Primary outputs by script family:

- `exp1`: `results.csv`, plots, `metrics.json`, `summary.md`, `resolved_config.yaml`, `run_metadata.json`
- `exp2`: sweep CSV files + plots + markdown summaries
- `exp3`: query/index CSV summaries and compressed-domain Q1-Q5 metrics
- `exp4_v2`: residual anomaly/query-correction CSV summaries
- `exp5`: ARD diagnostic and rank-selection CSV summaries

## Current Snapshot

- Phase 0/1/2/3 core objectives are completed.
- Current system-direction rerun is `exp3 + exp4_v2 + exp5/ard_diagnostic`.
- Current historical mainline rerun remains `exp1 + exp2 + exp4 + exp5`.
- Historical results remain useful as diagnostics, but publication-facing
  claims should be regenerated with the current compressor implementation.

## License

This project is released under the MIT License.
