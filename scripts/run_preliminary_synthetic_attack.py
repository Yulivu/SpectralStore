"""Run a preliminary Synthetic-Attack robust residual experiment."""

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
    RobustAsymmetricSpectralCompressor,
    SpectralCompressionConfig,
    SymmetricSVDCompressor,
)
from spectralstore.data_loader import make_synthetic_attack
from spectralstore.evaluation import (
    anomaly_precision_recall,
    max_entrywise_error,
    mean_entrywise_error,
    residual_nnz,
    residual_sparsity,
    relative_frobenius_error_against_dense,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments/preliminary/synthetic_attack/configs/default.json",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/preliminary/synthetic_attack/results",
    )
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_run = []
    aggregates: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for repeat in range(config["num_repeats"]):
        seed = config["random_seed"] + repeat
        dataset = make_synthetic_attack(
            num_nodes=config["num_nodes"],
            num_steps=config["num_steps"],
            num_communities=config["num_communities"],
            p_in=config["p_in"],
            p_out=config["p_out"],
            temporal_jitter=config["temporal_jitter"],
            attack_kind=config["attack_kind"],
            attack_fraction=config["attack_fraction"],
            outlier_weight=config["outlier_weight"],
            directed=config["directed"],
            random_seed=seed,
        )

        base_config = SpectralCompressionConfig(
            rank=config["rank"],
            random_seed=seed,
            num_splits=config["num_splits"],
        )
        robust_mad_config = SpectralCompressionConfig(
            rank=config["rank"],
            random_seed=seed,
            num_splits=config["num_splits"],
            robust_iterations=config["robust_iterations"],
            residual_threshold_mode=config.get("residual_threshold_mode", "mad"),
            residual_mad_multiplier=config.get("residual_mad_multiplier", 45.0),
        )
        robust_quantile_config = SpectralCompressionConfig(
            rank=config["rank"],
            random_seed=seed,
            num_splits=config["num_splits"],
            robust_iterations=config["robust_iterations"],
            residual_threshold_mode="quantile",
            residual_quantile=config["residual_quantile"],
        )
        methods = {
            "spectralstore_full_mad": RobustAsymmetricSpectralCompressor(robust_mad_config),
            "spectralstore_full_quantile": RobustAsymmetricSpectralCompressor(robust_quantile_config),
            "spectralstore_no_robust": AsymmetricSpectralCompressor(base_config),
            "baseline_sym_svd": SymmetricSVDCompressor(base_config),
            "baseline_direct_svd": DirectSVDCompressor(base_config),
        }

        run_result = {"repeat": repeat, "seed": seed, "attack_edges": len(dataset.attack_edges), "methods": {}}
        for name, compressor in methods.items():
            store = compressor.fit_transform(dataset.snapshots)
            precision, recall = anomaly_precision_recall(dataset.attack_edges, store)
            values = {
                "max_entrywise_error": max_entrywise_error(dataset.expected_snapshots, store),
                "mean_entrywise_error": mean_entrywise_error(dataset.expected_snapshots, store),
                "relative_frobenius_error": relative_frobenius_error_against_dense(
                    dataset.expected_snapshots,
                    store,
                ),
                "anomaly_precision": precision,
                "anomaly_recall": recall,
                "residual_nnz": float(residual_nnz(store)),
                "residual_sparsity": residual_sparsity(store),
            }
            run_result["methods"][name] = values
            for metric, value in values.items():
                aggregates[name][metric].append(value)
        per_run.append(run_result)

    metrics = {
        "dataset": {
            "name": "synthetic_attack",
            "attack_kind": config["attack_kind"],
            "attack_fraction": config["attack_fraction"],
            "num_nodes": config["num_nodes"],
            "num_steps": config["num_steps"],
            "num_communities": config["num_communities"],
            "num_repeats": config["num_repeats"],
            "rank": config["rank"],
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
        "# Synthetic-Attack Preliminary Results",
        "",
        f"- attack kind: {dataset['attack_kind']}",
        f"- attack fraction: {dataset['attack_fraction']}",
        f"- nodes: {dataset['num_nodes']}",
        f"- temporal snapshots: {dataset['num_steps']}",
        f"- communities: {dataset['num_communities']}",
        f"- repeats: {dataset['num_repeats']}",
        f"- rank: {dataset['rank']}",
        "",
        "| method | max entrywise | mean entrywise | rel. Frobenius | anomaly P | anomaly R | residual nnz | residual sparsity |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, metric_values in metrics["aggregates"].items():
        lines.append(
            "| "
            f"{method} | "
            f"{format_mean_std(metric_values['max_entrywise_error'])} | "
            f"{format_mean_std(metric_values['mean_entrywise_error'])} | "
            f"{format_mean_std(metric_values['relative_frobenius_error'])} | "
            f"{format_mean_std(metric_values['anomaly_precision'])} | "
            f"{format_mean_std(metric_values['anomaly_recall'])} | "
            f"{format_mean_std(metric_values['residual_nnz'])} | "
            f"{format_mean_std(metric_values['residual_sparsity'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def format_mean_std(values: dict[str, float]) -> str:
    return f"{values['mean']:.4f} +/- {values['std']:.4f}"


if __name__ == "__main__":
    main()
