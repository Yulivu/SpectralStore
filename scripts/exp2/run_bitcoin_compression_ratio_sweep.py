"""Run Exp2 compression-ratio sweep on Bitcoin-OTC."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spectralstore.compression import (  # noqa: E402
    available_compressors,
    create_compressor,
    spectral_config_from_mapping,
)
from spectralstore.data_loader import load_bitcoin_otc  # noqa: E402
from spectralstore.evaluation import (  # noqa: E402
    load_experiment_config,
    max_entrywise_error,
    mean_entrywise_error,
    relative_frobenius_error_against_dense,
    set_reproducibility_seed,
    storage_report,
)


FIELDNAMES = [
    "rank",
    "method",
    "max_entrywise",
    "mean_entrywise",
    "frobenius",
    "storage_ratio",
    "compressed_bytes",
    "raw_sparse_bytes",
    "raw_dense_bytes",
    "run_seconds",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments/preliminary/bitcoin_otc/configs/default.yaml",
    )
    parser.add_argument("--out-dir", default="experiments/results/exp2")
    parser.add_argument("--min-rank", type=int, default=2)
    parser.add_argument("--max-rank", type=int, default=20)
    parser.add_argument("--rank-step", type=int, default=2)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()

    config = load_experiment_config(args.config, args.overrides)
    set_reproducibility_seed(config.get("random_seed"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_bitcoin_otc(config["raw_path"], max_nodes=config["max_nodes"])
    expected_snapshots = [snapshot.toarray() for snapshot in dataset.snapshots]
    ranks = list(range(args.min_rank, args.max_rank + 1, args.rank_step))
    methods = list(available_compressors())

    rows = []
    for rank in ranks:
        for method in methods:
            started_at = time.perf_counter()
            compressor_config = spectral_config_from_mapping(config, rank=rank)
            store = create_compressor(method, compressor_config).fit_transform(dataset.snapshots)
            include_residual = bool(store.residuals)
            storage = storage_report(store, dataset.snapshots)
            elapsed = time.perf_counter() - started_at
            row = {
                "rank": rank,
                "method": method,
                "max_entrywise": max_entrywise_error(
                    expected_snapshots,
                    store,
                    include_residual=include_residual,
                ),
                "mean_entrywise": mean_entrywise_error(
                    expected_snapshots,
                    store,
                    include_residual=include_residual,
                ),
                "frobenius": relative_frobenius_error_against_dense(
                    expected_snapshots,
                    store,
                    include_residual=include_residual,
                ),
                "storage_ratio": storage["compressed_vs_raw_sparse_ratio"],
                "compressed_bytes": storage["compressed_bytes"],
                "raw_sparse_bytes": storage["raw_sparse_bytes"],
                "raw_dense_bytes": storage["raw_dense_bytes"],
                "run_seconds": elapsed,
            }
            rows.append(row)
            print(f"[rank={rank}, method={method}] done in {elapsed:.1f}s", flush=True)

    results_csv = out_dir / "sweep_results.csv"
    write_rows(results_csv, rows)
    plot_sweep(rows, out_dir / "exp2_sweep.png")
    print(f"wrote {results_csv}")
    print(f"wrote {out_dir / 'exp2_sweep.png'}")


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_sweep(rows: list[dict[str, object]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    methods = sorted({str(row["method"]) for row in rows})
    for method in methods:
        method_rows = sorted(
            [row for row in rows if row["method"] == method],
            key=lambda row: float(row["storage_ratio"]),
        )
        x = [float(row["storage_ratio"]) for row in method_rows]
        y = [float(row["max_entrywise"]) for row in method_rows]
        ax.plot(x, y, marker="o", linewidth=1.6, label=method)

    ax.set_xlabel("storage ratio vs raw sparse CSR")
    ax.set_ylabel("max entrywise error")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
