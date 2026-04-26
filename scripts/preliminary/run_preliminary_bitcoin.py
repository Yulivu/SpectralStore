"""Run a first Bitcoin-OTC comparison between SpectralStore and SVD baselines."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spectralstore.compression import (
    create_compressor,
    spectral_config_from_mapping,
)
from spectralstore.data_loader import load_bitcoin_otc
from spectralstore.evaluation import (
    load_experiment_config,
    observed_edges_from_snapshots,
    observed_edge_report,
    residual_nnz,
    residual_sparsity,
    reconstruction_report,
    set_reproducibility_seed,
    split_observed_edges,
    storage_report,
    write_experiment_outputs,
)
from spectralstore.query_engine import QueryEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments/preliminary/bitcoin_otc/configs/default.yaml",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/preliminary/bitcoin_otc/results",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        help="OmegaConf dotlist override, e.g. --set max_nodes=100",
    )
    args = parser.parse_args()
    started_at = time.perf_counter()

    config = load_experiment_config(args.config, args.overrides)
    set_reproducibility_seed(config.get("random_seed"))
    out_dir = Path(args.out_dir)

    dataset = load_bitcoin_otc(config["raw_path"], max_nodes=config["max_nodes"])
    train_snapshots, held_out = split_observed_edges(
        dataset.snapshots,
        test_fraction=config["test_fraction"],
        random_seed=config["random_seed"],
    )
    train_dense_snapshots = [snapshot.toarray() for snapshot in train_snapshots]

    compressor_config = spectral_config_from_mapping(config)
    method_names = list(config["methods"])
    methods = {
        name: create_compressor(name, compressor_config)
        for name in method_names
    }
    train_observed_edges = observed_edges_from_snapshots(train_snapshots)
    q4_queries = sample_temporal_trend_queries(
        train_snapshots,
        num_queries=config["q4_num_queries"],
        window_size=config["q4_window_size"],
        random_seed=config["random_seed"],
    )

    metrics = {
        "dataset": {
            "name": dataset.name,
            "num_nodes": dataset.num_nodes,
            "num_steps": dataset.num_steps,
            "held_out_edges": len(held_out),
            "train_observed_edges": len(train_observed_edges),
            "q4_queries": len(q4_queries),
        },
        "methods": {},
    }

    for name, compressor in methods.items():
        store = compressor.fit_transform(train_snapshots)
        include_residual = bool(store.residuals)
        reconstruction = reconstruction_report(
            train_snapshots,
            store,
            include_residual=include_residual,
            expected_snapshots=train_dense_snapshots,
        )
        metrics["methods"][name] = {
            "rank": store.rank,
            "reconstruction": reconstruction,
            "train_observed_edges": observed_edge_report(
                train_observed_edges,
                store,
                include_residual=include_residual,
                prefix="train_observed_edge",
            ),
            "held_out_observed_edges": observed_edge_report(
                held_out,
                store,
                include_residual=include_residual,
                prefix="held_out_observed_edge",
            ),
            "storage": storage_report(store, train_snapshots),
            "q4_temporal_trend": temporal_trend_metrics(
                train_snapshots,
                QueryEngine(store),
                q4_queries,
                include_residual=include_residual,
                max_examples=config["q4_summary_examples"],
            ),
            "residual": residual_metrics(
                store,
                top_k=config["residual_top_k"],
            ),
        }

    summary = render_summary(metrics)
    write_experiment_outputs(
        out_dir=out_dir,
        metrics=metrics,
        summary=summary,
        config_path=args.config,
        config=config,
        started_at=started_at,
    )
    print(summary)

def sample_temporal_trend_queries(
    snapshots,
    *,
    num_queries: int,
    window_size: int,
    random_seed: int,
) -> list[tuple[int, int, int, int]]:
    if not snapshots or num_queries <= 0:
        return []
    rng = np.random.default_rng(random_seed + 20_000)
    pairs = sorted(
        {(int(u), int(v)) for _t, u, v, _w in observed_edges_from_snapshots(snapshots)}
    )
    if not pairs:
        return []

    window = max(1, min(window_size, len(snapshots)))
    max_start = len(snapshots) - window
    queries: list[tuple[int, int, int, int]] = []
    for _ in range(num_queries):
        u, v = pairs[int(rng.integers(len(pairs)))]
        start = int(rng.integers(max_start + 1)) if max_start > 0 else 0
        queries.append((u, v, start, start + window - 1))
    return queries


def temporal_trend_metrics(
    snapshots: list[sparse.csr_matrix],
    engine: QueryEngine,
    queries: list[tuple[int, int, int, int]],
    *,
    include_residual: bool,
    max_examples: int,
) -> dict[str, object]:
    absolute_errors: list[float] = []
    examples = []
    for query_index, (u, v, t1, t2) in enumerate(queries):
        raw = [float(snapshots[t][u, v]) for t in range(t1, t2 + 1)]
        estimated = engine.temporal_trend(
            u,
            v,
            t1,
            t2,
            include_residual=include_residual,
        )
        absolute_errors.extend(abs(a - b) for a, b in zip(raw, estimated))
        if query_index < max_examples:
            examples.append(
                {
                    "u": u,
                    "v": v,
                    "t1": t1,
                    "t2": t2,
                    "raw": raw,
                    "estimated": [float(value) for value in estimated],
                }
            )

    return {
        "num_queries": len(queries),
        "mean_absolute_error": float(np.mean(absolute_errors)) if absolute_errors else float("nan"),
        "max_absolute_error": float(np.max(absolute_errors)) if absolute_errors else float("nan"),
        "examples": examples,
    }


def residual_metrics(store, *, top_k: int) -> dict[str, object]:
    top_entries: list[dict[str, float | int]] = []
    for t, residual in enumerate(store.residuals):
        coo = residual.tocoo()
        for row, col, value in zip(coo.row, coo.col, coo.data):
            top_entries.append(
                {
                    "t": int(t),
                    "u": int(row),
                    "v": int(col),
                    "value": float(value),
                    "abs_value": float(abs(value)),
                }
            )
    top_entries.sort(key=lambda entry: float(entry["abs_value"]), reverse=True)
    diagnostics = store.threshold_diagnostics or {}
    return {
        "nnz": residual_nnz(store),
        "sparsity": residual_sparsity(store),
        "estimated_threshold": diagnostics.get("estimated_threshold"),
        "top_entries": top_entries[:top_k],
    }


def render_summary(metrics: dict) -> str:
    dataset = metrics["dataset"]
    lines = [
        "# Bitcoin-OTC Preliminary Results",
        "",
        f"- nodes: {dataset['num_nodes']}",
        f"- temporal snapshots: {dataset['num_steps']}",
        f"- held-out observed edges: {dataset['held_out_edges']}",
        f"- Q4 trend queries: {dataset['q4_queries']}",
        "",
        "| method | train rel. Frobenius | max entrywise | mean entrywise | held-out RMSE | storage ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in metrics["methods"].items():
        reconstruction = values["reconstruction"]
        held_out_edges = values["held_out_observed_edges"]
        storage = values["storage"]
        lines.append(
            "| "
            f"{name} | "
            f"{reconstruction['relative_frobenius_error']:.4f} | "
            f"{reconstruction['max_entrywise_error']:.4f} | "
            f"{reconstruction['mean_entrywise_error']:.4f} | "
            f"{held_out_edges['held_out_observed_edge_rmse']:.4f} | "
            f"{storage['compressed_vs_raw_sparse_ratio']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Q4 Examples",
            "",
        ]
    )
    for name, values in metrics["methods"].items():
        examples = values["q4_temporal_trend"]["examples"]
        if not examples:
            continue
        lines.append(f"### {name}")
        for example in examples:
            lines.append(
                "- "
                f"({example['u']}, {example['v']}), "
                f"t={example['t1']}..{example['t2']}: "
                f"raw={example['raw']}, estimated={example['estimated']}"
            )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

