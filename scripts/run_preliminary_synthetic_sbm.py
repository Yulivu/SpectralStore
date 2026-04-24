"""Run a controlled Synthetic-SBM comparison."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spectralstore.compression import (
    AsymmetricSpectralCompressor,
    DirectSVDCompressor,
    SpectralCompressionConfig,
    SymmetricSVDCompressor,
    TensorUnfoldingSVDCompressor,
)
from spectralstore.data_loader import make_temporal_sbm
from spectralstore.evaluation import (
    max_entrywise_error,
    mean_entrywise_error,
    relative_frobenius_error_against_dense,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments/preliminary/synthetic_sbm/configs/default.json",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/preliminary/synthetic_sbm/results",
    )
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

        compressor_config = SpectralCompressionConfig(
            rank=config["rank"],
            random_seed=seed,
            num_splits=config.get("num_splits", 1),
        )
        methods = {
            "spectralstore_asym": AsymmetricSpectralCompressor(compressor_config),
            "tensor_unfolding_svd": TensorUnfoldingSVDCompressor(compressor_config),
            "baseline_sym_svd": SymmetricSVDCompressor(compressor_config),
            "baseline_direct_svd": DirectSVDCompressor(compressor_config),
        }

        run_result = {"repeat": repeat, "seed": seed, "methods": {}}
        for name, compressor in methods.items():
            store = compressor.fit_transform(dataset.snapshots)
            values = {
                "max_entrywise_error": max_entrywise_error(dataset.expected_snapshots, store),
                "mean_entrywise_error": mean_entrywise_error(dataset.expected_snapshots, store),
                "relative_frobenius_error": relative_frobenius_error_against_dense(
                    dataset.expected_snapshots,
                    store,
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

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "summary.md").write_text(render_summary(metrics), encoding="utf-8")
    print(render_summary(metrics))


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
        "| method | max entrywise | mean entrywise | relative Frobenius |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method, metric_values in metrics["aggregates"].items():
        lines.append(
            "| "
            f"{method} | "
            f"{format_mean_std(metric_values['max_entrywise_error'])} | "
            f"{format_mean_std(metric_values['mean_entrywise_error'])} | "
            f"{format_mean_std(metric_values['relative_frobenius_error'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def format_mean_std(values: dict[str, float]) -> str:
    return f"{values['mean']:.4f} +/- {values['std']:.4f}"


if __name__ == "__main__":
    main()
