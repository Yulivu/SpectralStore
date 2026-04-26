"""Run a preliminary ogbl-collab comparison."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from spectralstore.compression import create_compressor, spectral_config_from_mapping  # noqa: E402
from spectralstore.data_loader import load_ogbl_collab  # noqa: E402
from spectralstore.evaluation import (  # noqa: E402
    load_experiment_config,
    observed_edges_from_snapshots,
    observed_edge_report,
    ranking_report,
    reconstruction_report,
    sample_temporal_negative_edges,
    set_reproducibility_seed,
    split_observed_edges,
    storage_report,
    validate_dense_stack_memory_budget,
    write_experiment_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments/preliminary/ogbl_collab/configs/default.yaml",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/preliminary/ogbl_collab/results",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        help="OmegaConf dotlist override, e.g. --set max_nodes=200",
    )
    args = parser.parse_args()
    started_at = time.perf_counter()

    config = load_experiment_config(args.config, args.overrides)
    set_reproducibility_seed(config.get("random_seed"))
    min_year = config.get("year_start")
    if min_year is None:
        min_year = config.get("min_year")
    dataset = load_ogbl_collab(
        root=config["root"],
        max_nodes=config["max_nodes"],
        min_year=min_year,
        max_year=config.get("max_year"),
    )
    validate_dense_memory_budget(
        dataset.num_nodes,
        dataset.num_steps,
        limit_gb=config["dense_memory_limit_gb"],
    )
    train_snapshots, held_out = split_observed_edges(
        dataset.snapshots,
        test_fraction=config["test_fraction"],
        random_seed=config["random_seed"],
    )
    train_dense_snapshots = [snapshot.toarray() for snapshot in train_snapshots]
    train_edges = observed_edges_from_snapshots(train_snapshots)
    ranking_negatives = sample_temporal_negative_edges(
        held_out,
        train_snapshots,
        negatives_per_positive=config["negative_edges_per_positive"],
        random_seed=config["random_seed"],
    )
    compressor_config = spectral_config_from_mapping(config)

    metrics = {
        "dataset": {
            "name": dataset.name,
            "num_nodes": dataset.num_nodes,
            "num_steps": dataset.num_steps,
            "time_bins": dataset.time_bins,
            "year_start": min_year,
            "max_year": config.get("max_year"),
            "held_out_edges": len(held_out),
            "train_observed_edges": len(train_edges),
        },
        "methods": {},
    }
    for name in config["methods"]:
        store = create_compressor(name, compressor_config).fit_transform(train_snapshots)
        metrics["methods"][name] = {
            "rank": store.rank,
            "reconstruction": reconstruction_report(
                train_snapshots,
                store,
                expected_snapshots=train_dense_snapshots,
            ),
            "train_observed_edges": observed_edge_report(
                train_edges,
                store,
                prefix="train_observed_edge",
            ),
            "held_out_observed_edges": observed_edge_report(
                held_out,
                store,
                prefix="held_out_observed_edge",
            ),
            "ranking": ranking_report(
                held_out,
                ranking_negatives,
                store,
                hits_at=(10, 50),
            ),
            "storage": storage_report(store, train_snapshots),
        }

    summary = render_summary(metrics)
    write_experiment_outputs(
        out_dir=args.out_dir,
        metrics=metrics,
        summary=summary,
        config_path=args.config,
        config=config,
        started_at=started_at,
    )
    print(summary)


def render_summary(metrics: dict) -> str:
    dataset = metrics["dataset"]
    lines = [
        "# ogbl-collab Preliminary Results",
        "",
        f"- nodes: {dataset['num_nodes']}",
        f"- temporal snapshots: {dataset['num_steps']}",
        f"- year_start: {dataset['year_start']}",
        f"- years: {', '.join(dataset['time_bins'])}",
        f"- held-out observed edges: {dataset['held_out_edges']}",
        "",
        "| method | train rel. Frob | max entrywise | mean entrywise | train edge RMSE | held-out RMSE | "
        "held-out MAE | MRR | Hits@10 | Hits@50 | compressed bytes | sparse ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in metrics["methods"].items():
        reconstruction = values["reconstruction"]
        train_edges = values["train_observed_edges"]
        held_out_edges = values["held_out_observed_edges"]
        ranking = values["ranking"]
        storage = values["storage"]
        lines.append(
            "| "
            f"{name} | "
            f"{reconstruction['relative_frobenius_error']:.4f} | "
            f"{reconstruction['max_entrywise_error']:.4f} | "
            f"{reconstruction['mean_entrywise_error']:.4f} | "
            f"{train_edges['train_observed_edge_rmse']:.4f} | "
            f"{held_out_edges['held_out_observed_edge_rmse']:.4f} | "
            f"{held_out_edges['held_out_observed_edge_mae']:.4f} | "
            f"{ranking['mrr']:.4f} | "
            f"{ranking['hits_at_10']:.4f} | "
            f"{ranking['hits_at_50']:.4f} | "
            f"{storage['compressed_bytes']} | "
            f"{storage['compressed_vs_raw_sparse_ratio']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def validate_dense_memory_budget(num_nodes: int, num_steps: int, *, limit_gb: float) -> None:
    validate_dense_stack_memory_budget(
        num_nodes,
        num_steps,
        limit_gb=limit_gb,
        label="ogbl-collab run",
    )


if __name__ == "__main__":
    main()

