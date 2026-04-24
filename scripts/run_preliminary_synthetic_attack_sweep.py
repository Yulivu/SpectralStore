"""Run Synthetic-Attack robustness sweep experiments."""

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
        default="experiments/preliminary/synthetic_attack/configs/sweep.json",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/preliminary/synthetic_attack/results",
    )
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_results = []
    for attack_kind in config["attack_kinds"]:
        for attack_fraction in config["attack_fractions"]:
            sweep_results.append(run_one_setting(config, attack_kind, attack_fraction))

    metrics = {
        "dataset": {
            "name": "synthetic_attack_sweep",
            "attack_kinds": config["attack_kinds"],
            "attack_fractions": config["attack_fractions"],
            "num_nodes": config["num_nodes"],
            "num_steps": config["num_steps"],
            "num_communities": config["num_communities"],
            "num_repeats": config["num_repeats"],
            "rank": config["rank"],
        },
        "sweep": sweep_results,
    }

    (out_dir / "sweep_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "sweep_summary.md").write_text(render_summary(metrics), encoding="utf-8")
    print(render_summary(metrics))


def run_one_setting(config: dict, attack_kind: str, attack_fraction: float) -> dict:
    per_run = []
    aggregates: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    attack_edge_counts = []

    for repeat in range(config["num_repeats"]):
        seed = config["random_seed"] + repeat
        dataset = make_synthetic_attack(
            num_nodes=config["num_nodes"],
            num_steps=config["num_steps"],
            num_communities=config["num_communities"],
            p_in=config["p_in"],
            p_out=config["p_out"],
            temporal_jitter=config["temporal_jitter"],
            attack_kind=attack_kind,
            attack_fraction=attack_fraction,
            outlier_weight=config["outlier_weight"],
            directed=config["directed"],
            random_seed=seed,
        )
        attack_edge_counts.append(len(dataset.attack_edges))

        base_config = SpectralCompressionConfig(
            rank=config["rank"],
            random_seed=seed,
            num_splits=config["num_splits"],
        )
        robust_config = SpectralCompressionConfig(
            rank=config["rank"],
            random_seed=seed,
            num_splits=config["num_splits"],
            robust_iterations=config["robust_iterations"],
            residual_quantile=config["residual_quantile"],
        )
        methods = {
            "spectralstore_full": RobustAsymmetricSpectralCompressor(robust_config),
            "spectralstore_no_robust": AsymmetricSpectralCompressor(base_config),
            "baseline_sym_svd": SymmetricSVDCompressor(base_config),
            "baseline_direct_svd": DirectSVDCompressor(base_config),
        }

        run_result = {
            "repeat": repeat,
            "seed": seed,
            "attack_edges": len(dataset.attack_edges),
            "methods": {},
        }
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

    summarized = summarize_aggregates(aggregates)
    summarized["spectralstore_full"]["robustness_gain_vs_best_competitor"] = {
        "mean": robustness_gain_vs_best_competitor(summarized),
        "std": 0.0,
    }

    return {
        "attack_kind": attack_kind,
        "attack_fraction": attack_fraction,
        "attack_edges_mean": float(np.mean(np.asarray(attack_edge_counts, dtype=float))),
        "aggregates": summarized,
        "per_run": per_run,
    }


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


def robustness_gain_vs_best_competitor(aggregates: dict[str, dict[str, dict[str, float]]]) -> float:
    full_error = aggregates["spectralstore_full"]["relative_frobenius_error"]["mean"]
    competitor_errors = [
        aggregates["spectralstore_no_robust"]["relative_frobenius_error"]["mean"],
        aggregates["baseline_sym_svd"]["relative_frobenius_error"]["mean"],
        aggregates["baseline_direct_svd"]["relative_frobenius_error"]["mean"],
    ]
    return float(min(competitor_errors) / max(full_error, 1e-12))


def render_summary(metrics: dict) -> str:
    dataset = metrics["dataset"]
    lines = [
        "# Synthetic-Attack Sweep Results",
        "",
        f"- attack kinds: {', '.join(dataset['attack_kinds'])}",
        f"- attack fractions: {', '.join(str(value) for value in dataset['attack_fractions'])}",
        f"- nodes: {dataset['num_nodes']}",
        f"- temporal snapshots: {dataset['num_steps']}",
        f"- repeats: {dataset['num_repeats']}",
        f"- rank: {dataset['rank']}",
        "",
        "| attack | fraction | full rel. Frob | no-robust rel. Frob | best competitor gain | anomaly P | anomaly R | residual sparsity |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in metrics["sweep"]:
        aggregates = result["aggregates"]
        full = aggregates["spectralstore_full"]
        no_robust = aggregates["spectralstore_no_robust"]
        lines.append(
            "| "
            f"{result['attack_kind']} | "
            f"{result['attack_fraction']:.3f} | "
            f"{format_mean_std(full['relative_frobenius_error'])} | "
            f"{format_mean_std(no_robust['relative_frobenius_error'])} | "
            f"{full['robustness_gain_vs_best_competitor']['mean']:.3f}x | "
            f"{format_mean_std(full['anomaly_precision'])} | "
            f"{format_mean_std(full['anomaly_recall'])} | "
            f"{format_mean_std(full['residual_sparsity'])} |"
        )
    lines.append("")
    lines.extend(render_detailed_method_tables(metrics))
    return "\n".join(lines)


def render_detailed_method_tables(metrics: dict) -> list[str]:
    lines = ["## Detailed Relative Frobenius Error", ""]
    for attack_kind in metrics["dataset"]["attack_kinds"]:
        lines.extend(
            [
                f"### {attack_kind}",
                "",
                "| fraction | spectralstore_full | no_robust | sym_svd | direct_svd |",
                "| ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for result in metrics["sweep"]:
            if result["attack_kind"] != attack_kind:
                continue
            aggregates = result["aggregates"]
            lines.append(
                "| "
                f"{result['attack_fraction']:.3f} | "
                f"{format_mean_std(aggregates['spectralstore_full']['relative_frobenius_error'])} | "
                f"{format_mean_std(aggregates['spectralstore_no_robust']['relative_frobenius_error'])} | "
                f"{format_mean_std(aggregates['baseline_sym_svd']['relative_frobenius_error'])} | "
                f"{format_mean_std(aggregates['baseline_direct_svd']['relative_frobenius_error'])} |"
            )
        lines.append("")
    return lines


def format_mean_std(values: dict[str, float]) -> str:
    return f"{values['mean']:.4f} +/- {values['std']:.4f}"


if __name__ == "__main__":
    main()
