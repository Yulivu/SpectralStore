"""Run Exp2 factor-only vs residual-corrected evaluation on Bitcoin-OTC."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spectralstore.compression import (  # noqa: E402
    create_compressor,
    spectral_config_from_mapping,
)
from spectralstore.data_loader import load_bitcoin_otc  # noqa: E402
from spectralstore.evaluation import (  # noqa: E402
    load_experiment_config,
    observed_edge_rmse,
    relative_frobenius_error,
    residual_nnz,
    residual_sparsity,
    set_reproducibility_seed,
    split_observed_edges,
)


FIELDNAMES = [
    "rank",
    "method",
    "held_out_rmse_factor_only",
    "held_out_rmse_with_residual",
    "train_frobenius_factor_only",
    "train_frobenius_with_residual",
    "compressed_vs_raw_sparse_ratio_factor_only",
    "compressed_vs_raw_sparse_ratio_with_residual",
    "factor_bytes",
    "residual_bytes",
    "metadata_bytes",
    "compressed_bytes_with_residual",
    "raw_sparse_bytes",
    "residual_nnz",
    "residual_sparsity",
    "held_out_edges",
    "run_seconds",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML config.",
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--min-rank", type=int, default=2)
    parser.add_argument("--max-rank", type=int, default=20)
    parser.add_argument("--rank-step", type=int, default=2)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Optional method list. Defaults to all registry methods.",
    )
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()

    config = load_experiment_config(args.config, args.overrides)
    set_reproducibility_seed(config.get("random_seed"))
    out_dir = Path(args.out_dir or config.get("output_dir", "experiments/results/exp2"))
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_bitcoin_otc(config["raw_path"], max_nodes=config["max_nodes"])
    train_snapshots, held_out = split_observed_edges(
        dataset.snapshots,
        test_fraction=config["test_fraction"],
        random_seed=config["random_seed"],
    )
    raw_sparse_bytes = _raw_sparse_csr_bytes(train_snapshots)
    ranks = list(range(args.min_rank, args.max_rank + 1, args.rank_step))
    methods = args.methods if args.methods is not None else list(config.get("methods", ["spectralstore_thinking"]))

    rows = []
    for rank in ranks:
        for method in methods:
            started_at = time.perf_counter()
            compressor_config = spectral_config_from_mapping(config, rank=rank)
            store = create_compressor(method, compressor_config).fit_transform(train_snapshots)
            elapsed = time.perf_counter() - started_at
            factor_bytes = store.factor_bytes()
            residual_bytes = store.residual_bytes()
            metadata_bytes = store.metadata_bytes()
            compressed_bytes = store.compressed_bytes()
            row = {
                "rank": rank,
                "method": method,
                "held_out_rmse_factor_only": observed_edge_rmse(
                    held_out,
                    store,
                    include_residual=False,
                ),
                "held_out_rmse_with_residual": observed_edge_rmse(
                    held_out,
                    store,
                    include_residual=True,
                ),
                "train_frobenius_factor_only": relative_frobenius_error(
                    train_snapshots,
                    store,
                    include_residual=False,
                ),
                "train_frobenius_with_residual": relative_frobenius_error(
                    train_snapshots,
                    store,
                    include_residual=True,
                ),
                "compressed_vs_raw_sparse_ratio_factor_only": factor_bytes
                / max(raw_sparse_bytes, 1),
                "compressed_vs_raw_sparse_ratio_with_residual": compressed_bytes
                / max(raw_sparse_bytes, 1),
                "factor_bytes": factor_bytes,
                "residual_bytes": residual_bytes,
                "metadata_bytes": metadata_bytes,
                "compressed_bytes_with_residual": compressed_bytes,
                "raw_sparse_bytes": raw_sparse_bytes,
                "residual_nnz": residual_nnz(store),
                "residual_sparsity": residual_sparsity(store),
                "held_out_edges": len(held_out),
                "run_seconds": elapsed,
            }
            rows.append(row)
            print(
                f"[rank={rank}, method={method}] "
                f"factor_rmse={row['held_out_rmse_factor_only']:.4f} "
                f"residual_rmse={row['held_out_rmse_with_residual']:.4f} "
                f"done in {elapsed:.1f}s",
                flush=True,
            )

    output_path = out_dir / "sweep_results_residual_boundary.csv"
    write_rows(output_path, rows)
    print(f"wrote {output_path}")


def _raw_sparse_csr_bytes(snapshots) -> int:
    total = 0
    for snapshot in snapshots:
        csr = snapshot.tocsr()
        total += csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes
    return int(total)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
