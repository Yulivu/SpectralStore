# SpectralStore

SpectralStore is a research-oriented storage and query engine for temporal graphs.
It explores approximate query processing over compressed graph representations built
from robust asymmetric spectral and tensor factorization.

The project starts from the research plan in [SpectralStore.md](SpectralStore.md),
which describes the motivation, algorithms, supported queries, datasets, baselines,
and evaluation plan.

## Goals

- Compress temporal graph snapshots into compact factorized representations.
- Answer graph queries directly in the compressed domain.
- Provide error-aware approximate query results.
- Support anomaly-aware residual storage for robust query correction.
- Offer reproducible experiments for spectral, tensor, and graph-compression baselines.

## Planned Query API

| Query | Purpose |
| --- | --- |
| `LINK_PROB(u, v, t)` | Estimate the connection probability for a node pair at time `t`. |
| `TOP_NEIGHBOR(u, t, k)` | Return the top `k` likely neighbors of node `u` at time `t`. |
| `COMMUNITY(t)` | Return or compute communities for a snapshot. |
| `TEMPORAL_TREND(u, v, t1, t2)` | Track a node-pair score over a time window. |
| `ANOMALY_DETECT(t, threshold)` | Return edges whose residual exceeds a threshold. |

## Repository Layout

```text
data/
  raw/        Original downloaded datasets, ignored by Git
  interim/    Temporary conversions, ignored by Git
  processed/  Stable derived artifacts, ignored by Git
scripts/      Dataset download and experiment entry points
experiments/  Experiment configs, notes, and result folders
tests/smoke/  End-to-end smoke tests on tiny inputs
src/spectralstore/
  compression/      Compression algorithms and factorized stores
  query_engine/     Query execution over compressed representations
  index/            Approximate search and temporal indexes
  data_loader/      Dataset loading utilities
  evaluation/       Metrics and experiment helpers
  baselines/        Baseline method wrappers
tests/              Unit tests
```

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
pytest
```

Run the tiny smoke experiment:

```bash
python scripts/run_smoke_quickstart.py
```

Download the first real dataset:

```bash
python scripts/download_dataset.py bitcoin_otc
```

Run the preliminary Bitcoin-OTC comparison:

```bash
python scripts/run_preliminary_bitcoin.py
```

Results are written under `experiments/preliminary/bitcoin_otc/results/`.

Run the preliminary Synthetic-SBM comparison:

```bash
python scripts/run_preliminary_synthetic_sbm.py
```

Results are written under `experiments/preliminary/synthetic_sbm/results/`.

Run the preliminary Synthetic-Attack robust residual comparison:

```bash
python scripts/run_preliminary_synthetic_attack.py
```

Results are written under `experiments/preliminary/synthetic_attack/results/`.

Run the Synthetic-Attack robustness sweep:

```bash
python scripts/run_preliminary_synthetic_attack_sweep.py
```

Sweep results are written to `sweep_metrics.json` and `sweep_summary.md` in the same results folder.

Run the Synthetic-SBM entrywise-bound scaling experiment:

```bash
python scripts/run_synthetic_sbm_bound_scaling.py
```

Bound scaling results are written to `bound_scaling_metrics.json` and
`bound_scaling_summary.md` in the Synthetic-SBM results folder.

Run the Synthetic-Spiked scaling experiment:

```bash
python scripts/run_synthetic_spiked_scaling.py
```

Spiked scaling results are written to `spiked_scaling_metrics.json` and
`spiked_scaling_summary.md` in the Synthetic-Spiked results folder.

## Status

This repository is in the initial implementation phase. It currently includes:

- a factorized temporal store,
- an asymmetric spectral compressor,
- multi-split asymmetric spectral ensembling,
- a tensor-unfolding SVD compressor baseline,
- a first robust residual compressor,
- adaptive MAD, fixed-quantile, and hybrid residual threshold policies,
- configurable degree-aware empirical entrywise error bounds,
- a first query optimizer for error-tolerant link and trend queries,
- SymSVD and DirectSVD baselines,
- Bitcoin-OTC loading,
- Synthetic-SBM, Synthetic-Spiked, and Synthetic-Attack generation,
- smoke tests, preliminary real-data experiments, and preliminary synthetic experiments.

The implementation roadmap is tracked in [docs/TECHNICAL_ROADMAP.md](docs/TECHNICAL_ROADMAP.md).

## License

This project is released under the MIT License.
