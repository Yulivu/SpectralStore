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
        robust_hybrid_config = SpectralCompressionConfig(
            rank=config["rank"],
            random_seed=seed,
            num_splits=config["num_splits"],
            robust_iterations=config["robust_iterations"],
            residual_threshold_mode="hybrid",
            residual_mad_multiplier=config.get("residual_mad_multiplier", 45.0),
            residual_quantile=config["residual_quantile"],
            residual_hybrid_tail_quantile=config.get("residual_hybrid_tail_quantile", 0.95),
            residual_hybrid_tail_ratio=config.get("residual_hybrid_tail_ratio", 2.0),
        )
        methods = {
            "full_mad": RobustAsymmetricSpectralCompressor(robust_mad_config),
            "full_quantile": RobustAsymmetricSpectralCompressor(robust_quantile_config),
            "full_hybrid": RobustAsymmetricSpectralCompressor(robust_hybrid_config),
            "no_robust": AsymmetricSpectralCompressor(base_config),
            "sym_svd": SymmetricSVDCompressor(base_config),
            "direct_svd": DirectSVDCompressor(base_config),
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
            if store.threshold_diagnostics is not None:
                values.update(
                    {
                        "estimated_threshold": float(
                            store.threshold_diagnostics["estimated_threshold"]
                        ),
                        "noise_scale": float(store.threshold_diagnostics["noise_scale"]),
                        "diagnostic_residual_nnz": float(
                            store.threshold_diagnostics["residual_nnz"]
                        ),
                        "diagnostic_residual_sparsity": float(
                            store.threshold_diagnostics["residual_sparsity"]
                        ),
                    }
                )
            run_result["methods"][name] = values
            for metric, value in values.items():
                aggregates[name][metric].append(value)
        per_run.append(run_result)

    summarized = summarize_aggregates(aggregates)
    summarized["full_mad"]["robustness_gain_vs_best_competitor"] = {
        "mean": robustness_gain_vs_best_competitor(summarized, "full_mad"),
        "std": 0.0,
    }
    summarized["full_quantile"]["robustness_gain_vs_best_competitor"] = {
        "mean": robustness_gain_vs_best_competitor(summarized, "full_quantile"),
        "std": 0.0,
    }
    summarized["full_hybrid"]["robustness_gain_vs_best_competitor"] = {
        "mean": robustness_gain_vs_best_competitor(summarized, "full_hybrid"),
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


def robustness_gain_vs_best_competitor(
    aggregates: dict[str, dict[str, dict[str, float]]],
    method: str,
) -> float:
    full_error = aggregates[method]["relative_frobenius_error"]["mean"]
    competitor_errors = [
        aggregates["no_robust"]["relative_frobenius_error"]["mean"],
        aggregates["sym_svd"]["relative_frobenius_error"]["mean"],
        aggregates["direct_svd"]["relative_frobenius_error"]["mean"],
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
        "| attack | fraction | full MAD rel. Frob | full quantile rel. Frob | "
        "full hybrid rel. Frob | no-robust rel. Frob | MAD gain | quantile gain | "
        "hybrid gain | hybrid threshold | hybrid noise | hybrid residual sparsity |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in metrics["sweep"]:
        aggregates = result["aggregates"]
        full_mad = aggregates["full_mad"]
        full_quantile = aggregates["full_quantile"]
        full_hybrid = aggregates["full_hybrid"]
        no_robust = aggregates["no_robust"]
        lines.append(
            "| "
            f"{result['attack_kind']} | "
            f"{result['attack_fraction']:.3f} | "
            f"{format_mean_std(full_mad['relative_frobenius_error'])} | "
            f"{format_mean_std(full_quantile['relative_frobenius_error'])} | "
            f"{format_mean_std(full_hybrid['relative_frobenius_error'])} | "
            f"{format_mean_std(no_robust['relative_frobenius_error'])} | "
            f"{full_mad['robustness_gain_vs_best_competitor']['mean']:.3f}x | "
            f"{full_quantile['robustness_gain_vs_best_competitor']['mean']:.3f}x | "
            f"{full_hybrid['robustness_gain_vs_best_competitor']['mean']:.3f}x | "
            f"{format_mean_std(full_hybrid['estimated_threshold'])} | "
            f"{format_mean_std(full_hybrid['noise_scale'])} | "
            f"{format_mean_std(full_hybrid['residual_sparsity'])} |"
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
                "| fraction | full_mad | full_quantile | full_hybrid | no_robust | "
                "sym_svd | direct_svd |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for result in metrics["sweep"]:
            if result["attack_kind"] != attack_kind:
                continue
            aggregates = result["aggregates"]
            lines.append(
                "| "
                f"{result['attack_fraction']:.3f} | "
                f"{format_mean_std(aggregates['full_mad']['relative_frobenius_error'])} | "
                f"{format_mean_std(aggregates['full_quantile']['relative_frobenius_error'])} | "
                f"{format_mean_std(aggregates['full_hybrid']['relative_frobenius_error'])} | "
                f"{format_mean_std(aggregates['no_robust']['relative_frobenius_error'])} | "
                f"{format_mean_std(aggregates['sym_svd']['relative_frobenius_error'])} | "
                f"{format_mean_std(aggregates['direct_svd']['relative_frobenius_error'])} |"
            )
        lines.append("")
    return lines


def format_mean_std(values: dict[str, float]) -> str:
    return f"{values['mean']:.4f} +/- {values['std']:.4f}"


if __name__ == "__main__":
    main()
