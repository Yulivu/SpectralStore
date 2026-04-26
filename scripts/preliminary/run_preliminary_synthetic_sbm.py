"""Run a controlled Synthetic-SBM comparison."""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spectralstore.compression import (
    AsymmetricSpectralCompressor,
    CPALSCompressor,
    DirectSVDCompressor,
    SymmetricSVDCompressor,
    TensorUnfoldingSVDCompressor,
    TuckerHOSVDCompressor,
    spectral_config_from_mapping,
)
from spectralstore.data_loader import make_temporal_sbm
from spectralstore.evaluation import (
    community_clustering_scores,
    load_experiment_config,
    max_entrywise_error,
    mean_entrywise_error,
    percentile_entrywise_error,
    relative_frobenius_error_against_dense,
    set_reproducibility_seed,
    write_experiment_outputs,
)
from spectralstore.query_engine import QueryEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments/preliminary/synthetic_sbm/configs/default.yaml",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/preliminary/synthetic_sbm/results",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        help="OmegaConf dotlist override, e.g. --set num_repeats=1",
    )
    args = parser.parse_args()
    started_at = time.perf_counter()

    config = load_experiment_config(args.config, args.overrides)
    set_reproducibility_seed(config.get("random_seed"))
    out_dir = Path(args.out_dir)

    per_run = []
    aggregates: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for repeat in range(config["num_repeats"]):
        seed = config["random_seed"] + repeat
        dataset = make_temporal_sbm(
            num_nodes=config["num_nodes"],
            num_steps=config["num_steps"],
            num_communities=config["num_communities"],
            p_in=config["p_in"],
            p_out=config["p_out"],
            temporal_jitter=config["temporal_jitter"],
            directed=config["directed"],
            random_seed=seed,
        )

        compressor_config = spectral_config_from_mapping(config, random_seed=seed)
        methods = {
            "spectralstore_asym": AsymmetricSpectralCompressor(compressor_config),
            "tensor_unfolding_svd": TensorUnfoldingSVDCompressor(compressor_config),
            "cp_als": CPALSCompressor(compressor_config),
            "tucker_hosvd": TuckerHOSVDCompressor(compressor_config),
            "baseline_sym_svd": SymmetricSVDCompressor(compressor_config),
            "baseline_direct_svd": DirectSVDCompressor(compressor_config),
        }

        run_result = {"repeat": repeat, "seed": seed, "methods": {}}
        for name, compressor in methods.items():
            store = compressor.fit_transform(dataset.snapshots)
            community_labels = QueryEngine(store).community(
                0,
                num_communities=config["num_communities"],
                random_seed=seed,
            )
            community_scores = community_clustering_scores(
                dataset.communities,
                community_labels,
            )
            values = {
                "max_entrywise_error": max_entrywise_error(dataset.expected_snapshots, store),
                "p95_entrywise_error": percentile_entrywise_error(
                    dataset.expected_snapshots,
                    store,
                    95.0,
                ),
                "p99_entrywise_error": percentile_entrywise_error(
                    dataset.expected_snapshots,
                    store,
                    99.0,
                ),
                "mean_entrywise_error": mean_entrywise_error(dataset.expected_snapshots, store),
                "relative_frobenius_error": relative_frobenius_error_against_dense(
                    dataset.expected_snapshots,
                    store,
                ),
                "compression_ratio": store.compression_ratio(),
                "community_nmi": community_scores["community_nmi"],
                "community_ari": community_scores["community_ari"],
                "compressed_bytes": float(store.compressed_bytes()),
                "raw_dense_bytes": float(store.raw_dense_bytes()),
                "raw_sparse_bytes": float(store.raw_sparse_csr_bytes(dataset.snapshots)),
                "compressed_vs_raw_dense_ratio": store.compressed_vs_raw_dense_ratio(),
                "compressed_vs_raw_sparse_ratio": store.compressed_vs_raw_sparse_ratio(
                    dataset.snapshots,
                ),
            }
            run_result["methods"][name] = values
            for metric, value in values.items():
                aggregates[name][metric].append(value)
        per_run.append(run_result)

    metrics = {
        "dataset": {
            "name": "synthetic_sbm",
            "num_nodes": config["num_nodes"],
            "num_steps": config["num_steps"],
            "num_communities": config["num_communities"],
            "num_repeats": config["num_repeats"],
            "rank": config["rank"],
            "num_splits": config.get("num_splits", 1),
        },
        "aggregates": summarize_aggregates(aggregates),
        "per_run": per_run,
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


def summarize_aggregates(
    aggregates: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, dict[str, float]]]:
    summarized = {}
    for method, metric_values in aggregates.items():
        summarized[method] = {}
        for metric, values in metric_values.items():
            array = np.asarray(values, dtype=float)
            summarized[method][metric] = {
                "mean": float(np.mean(array)),
                "std": float(np.std(array, ddof=0)),
            }
    return summarized


def render_summary(metrics: dict) -> str:
    dataset = metrics["dataset"]
    lines = [
        "# Synthetic-SBM Preliminary Results",
        "",
        f"- nodes: {dataset['num_nodes']}",
        f"- temporal snapshots: {dataset['num_steps']}",
        f"- communities: {dataset['num_communities']}",
        f"- repeats: {dataset['num_repeats']}",
        f"- rank: {dataset['rank']}",
        f"- asymmetric split ensemble: {dataset['num_splits']}",
        "",
        "| method | max entrywise | p95 entrywise | p99 entrywise | mean entrywise | "
        "relative Frobenius | community NMI | community ARI |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, metric_values in metrics["aggregates"].items():
        lines.append(
            "| "
            f"{method} | "
            f"{format_mean_std(metric_values['max_entrywise_error'])} | "
            f"{format_mean_std(metric_values['p95_entrywise_error'])} | "
            f"{format_mean_std(metric_values['p99_entrywise_error'])} | "
            f"{format_mean_std(metric_values['mean_entrywise_error'])} | "
            f"{format_mean_std(metric_values['relative_frobenius_error'])} | "
            f"{format_mean_std(metric_values['community_nmi'])} | "
            f"{format_mean_std(metric_values['community_ari'])} |"
        )
    lines.extend(
        [
            "",
            "| method | compressed bytes | raw dense bytes | raw sparse bytes | "
            "ratio vs dense | ratio vs sparse |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method, metric_values in metrics["aggregates"].items():
        lines.append(
            "| "
            f"{method} | "
            f"{format_mean_std(metric_values['compressed_bytes'])} | "
            f"{format_mean_std(metric_values['raw_dense_bytes'])} | "
            f"{format_mean_std(metric_values['raw_sparse_bytes'])} | "
            f"{format_mean_std(metric_values['compressed_vs_raw_dense_ratio'])} | "
            f"{format_mean_std(metric_values['compressed_vs_raw_sparse_ratio'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def format_mean_std(values: dict[str, float]) -> str:
    return f"{values['mean']:.4f} +/- {values['std']:.4f}"


if __name__ == "__main__":
    main()

