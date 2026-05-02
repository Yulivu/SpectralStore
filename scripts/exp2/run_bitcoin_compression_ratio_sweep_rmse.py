"""Run Exp2 held-out RMSE compression-ratio sweep on Bitcoin-OTC."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    set_reproducibility_seed,
    split_observed_edges,
    storage_report,
)


FIELDNAMES = [
    "rank",
    "effective_rank",
    "method",
    "held_out_rmse",
    "compressed_vs_raw_sparse_ratio",
    "storage_ratio_gt_one",
    "storage_gate_accepted",
    "storage_gate_action_taken",
    "invalid_compression_region",
    "factor_bytes",
    "residual_bytes",
    "metadata_bytes",
    "compressed_bytes",
    "raw_sparse_bytes",
    "raw_dense_bytes",
    "source_unfolding_shape",
    "target_unfolding_shape",
    "source_unfolding_nnz",
    "target_unfolding_nnz",
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
    ranks = list(range(args.min_rank, args.max_rank + 1, args.rank_step))
    methods = list(config.get("methods", ["spectralstore_thinking"]))

    rows = []
    for rank in ranks:
        for method in methods:
            started_at = time.perf_counter()
            compressor_config = spectral_config_from_mapping(config, rank=rank)
            store = create_compressor(method, compressor_config).fit_transform(train_snapshots)
            include_residual = bool(store.residuals)
            storage = storage_report(store, train_snapshots)
            unfolding = unfolding_diagnostics(store)
            elapsed = time.perf_counter() - started_at
            diagnostics = store.threshold_diagnostics or {}
            sparse_ratio = storage["compressed_vs_raw_sparse_ratio"]
            storage_gate_accepted = bool(
                diagnostics.get("storage_gate_accepted", sparse_ratio <= 1.0)
            )
            row = {
                "rank": rank,
                "effective_rank": store.rank,
                "method": method,
                "held_out_rmse": observed_edge_rmse(
                    held_out,
                    store,
                    include_residual=include_residual,
                ),
                "compressed_vs_raw_sparse_ratio": sparse_ratio,
                "storage_ratio_gt_one": int(sparse_ratio > 1.0),
                "storage_gate_accepted": storage_gate_accepted,
                "storage_gate_action_taken": diagnostics.get("storage_gate_action_taken", "none"),
                "invalid_compression_region": int(not storage_gate_accepted),
                "factor_bytes": storage["factor_bytes"],
                "residual_bytes": storage["residual_bytes"],
                "metadata_bytes": storage["metadata_bytes"],
                "compressed_bytes": storage["compressed_bytes"],
                "raw_sparse_bytes": storage["raw_sparse_bytes"],
                "raw_dense_bytes": storage["raw_dense_bytes"],
                "source_unfolding_shape": unfolding["source_unfolding_shape"],
                "target_unfolding_shape": unfolding["target_unfolding_shape"],
                "source_unfolding_nnz": unfolding["source_unfolding_nnz"],
                "target_unfolding_nnz": unfolding["target_unfolding_nnz"],
                "held_out_edges": len(held_out),
                "run_seconds": elapsed,
            }
            rows.append(row)
            print(f"[rank={rank}, method={method}] done in {elapsed:.1f}s", flush=True)

    results_csv = out_dir / "sweep_results_rmse.csv"
    write_rows(results_csv, rows)
    plot_sweep(rows, out_dir / "exp2_sweep_rmse.png")
    write_summary(rows, out_dir / "exp2_sweep_rmse_summary.md")
    print(f"wrote {results_csv}")
    print(f"wrote {out_dir / 'exp2_sweep_rmse.png'}")
    print(f"wrote {out_dir / 'exp2_sweep_rmse_summary.md'}")


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def unfolding_diagnostics(store) -> dict[str, object]:
    diagnostics = store.threshold_diagnostics or {}
    source_shape = diagnostics.get("source_unfolding_shape")
    target_shape = diagnostics.get("target_unfolding_shape")
    return {
        "source_unfolding_shape": json.dumps(source_shape) if source_shape is not None else "NA",
        "target_unfolding_shape": json.dumps(target_shape) if target_shape is not None else "NA",
        "source_unfolding_nnz": diagnostics.get("source_unfolding_nnz", "NA"),
        "target_unfolding_nnz": diagnostics.get("target_unfolding_nnz", "NA"),
    }


def plot_sweep(rows: list[dict[str, object]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    methods = sorted({str(row["method"]) for row in rows})
    for method in methods:
        method_rows = sorted(
            [row for row in rows if row["method"] == method],
            key=lambda row: float(row["compressed_vs_raw_sparse_ratio"]),
        )
        x = [float(row["compressed_vs_raw_sparse_ratio"]) for row in method_rows]
        y = [float(row["held_out_rmse"]) for row in method_rows]
        ax.plot(x, y, marker="o", linewidth=1.6, label=method)
        invalid_rows = [row for row in method_rows if not bool(row["storage_gate_accepted"])]
        if invalid_rows:
            ax.scatter(
                [float(row["compressed_vs_raw_sparse_ratio"]) for row in invalid_rows],
                [float(row["held_out_rmse"]) for row in invalid_rows],
                marker="x",
                color="red",
                s=40,
            )

    ax.axvline(1.0, color="red", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("compressed_vs_raw_sparse_ratio")
    ax.set_ylabel("held-out RMSE")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(rows: list[dict[str, object]], output_path: Path) -> None:
    total = len(rows)
    invalid = sum(1 for row in rows if not bool(row["storage_gate_accepted"]))
    methods = sorted({str(row["method"]) for row in rows})
    lines = [
        "# Exp2 RMSE Sweep Summary",
        "",
        "- invalid compression region 定义: `storage_gate_accepted=false`",
        f"- total points: {total}",
        f"- invalid points: {invalid}",
        "",
        "## Per-Method Invalid Counts",
        "",
    ]
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        method_invalid = sum(1 for row in method_rows if not bool(row["storage_gate_accepted"]))
        lines.append(f"- {method}: {method_invalid}/{len(method_rows)} invalid")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
