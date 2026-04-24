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

## Status

This repository is in the initial implementation phase. It currently includes:

- a factorized temporal store,
- an asymmetric spectral compressor,
- SymSVD and DirectSVD baselines,
- Bitcoin-OTC loading,
- smoke tests and a preliminary real-data experiment.

## License

This project is released under the MIT License.
