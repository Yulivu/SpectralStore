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

Experiment and tensor-baseline extras can be installed with:

```bash
python -m pip install -e ".[dev,experiments]"
```

Run the tiny smoke experiment:

```bash
python scripts/run_smoke_quickstart.py
```

Experiment scripts default to YAML configs under `experiments/**/configs/` and
write `metrics.json`, `summary.md`, and `run_metadata.json` under the matching
results directory.

Most experiment scripts support OmegaConf overrides:

```bash
python scripts/run_preliminary_synthetic_sbm.py --set num_repeats=1 --set rank=3
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
The Bitcoin script reports storage ratios, held-out observed-edge metrics, Q4
temporal trend checks, and residual statistics.

Run the Bitcoin-OTC residual threshold sweep:

```bash
python scripts/run_bitcoin_residual_sweep.py
```

The sweep accepts robust residual settings only when compressed storage is
smaller than raw sparse CSR storage and held-out observed-edge error does not
regress beyond the configured tolerance.

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

Run the Q5 anomaly detection validation:

```bash
python scripts/run_preliminary_synthetic_attack.py
```

The Synthetic-Attack summary reports injected anomaly edges, Q5 detected edges,
Q5 precision, and Q5 recall.

Run the preliminary ogbl-collab integration:

```bash
python scripts/run_preliminary_ogbl_collab.py --set max_nodes=200
```

This uses OGB's official `LinkPropPredDataset` loader and writes results under
`experiments/preliminary/ogbl_collab/results/`.
The summary includes unified reconstruction, observed-edge, storage, MRR, and
Hits metrics.

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
- dense CP-ALS and Tucker-HOSVD tensor baselines,
- YAML/OmegaConf experiment configs with reproducibility metadata,
- a first robust residual compressor,
- adaptive MAD, fixed-quantile, and hybrid residual threshold policies,
- configurable degree-aware empirical entrywise error bounds,
- a first query optimizer for error-tolerant link and trend queries,
- a first community query using time-weighted spectral embeddings,
- storage cost and compression-ratio estimates,
- SymSVD and DirectSVD baselines,
- Bitcoin-OTC loading,
- Synthetic-SBM, Synthetic-Spiked, and Synthetic-Attack generation,
- smoke tests, preliminary real-data experiments, and preliminary synthetic experiments.

The implementation roadmap is tracked in [docs/TECHNICAL_ROADMAP.md](docs/TECHNICAL_ROADMAP.md).
Prototype and lightweight implementation choices are tracked in
[docs/IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md).
The current implementation and experiment status is summarized in
[docs/CURRENT_STATUS.md](docs/CURRENT_STATUS.md).

## License

This project is released under the MIT License.
