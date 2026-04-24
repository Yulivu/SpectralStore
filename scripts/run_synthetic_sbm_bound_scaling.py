"""Run Synthetic-SBM empirical entrywise-bound scaling experiments."""

from __future__ import annotations

import argparse
import json
import math
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
    TensorUnfoldingSVDCompressor,
)
from spectralstore.data_loader import make_temporal_sbm
from spectralstore.evaluation import (
    entrywise_bound_coverage,
    max_entrywise_error,
    max_entrywise_error_bound,
    mean_entrywise_error,
    mean_entrywise_error_bound,
    relative_frobenius_error_against_dense,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments/preliminary/synthetic_sbm/configs/bound_scaling.json",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/preliminary/synthetic_sbm/results",
    )
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scaling_results = []
    for mode in ("sweep_n", "sweep_t"):
        if mode == "sweep_n":
            settings = [(n, config["num_steps_values"][1]) for n in config["num_nodes_values"]]
        else:
            settings = [(config["num_nodes_values"][1], t) for t in config["num_steps_values"]]
        for num_nodes, num_steps in settings:
            scaling_results.append(run_one_setting(config, mode, num_nodes, num_steps))

    metrics = {
        "dataset": {
            "name": "synthetic_sbm_bound_scaling",
            "num_nodes_values": config["num_nodes_values"],
            "num_steps_values": config["num_steps_values"],
            "num_communities": config["num_communities"],
            "num_repeats": config["num_repeats"],
            "rank": config["rank"],
            "num_splits": config["num_splits"],
        },
        "scaling": scaling_results,
    }

    (out_dir / "bound_scaling_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "bound_scaling_summary.md").write_text(
        render_summary(metrics),
        encoding="utf-8",
    )
    print(render_summary(metrics))


def run_one_setting(config: dict, mode: str, num_nodes: int, num_steps: int) -> dict:
    per_run = []
    aggregates: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for repeat in range(config["num_repeats"]):
        seed = config["random_seed"] + repeat
        dataset = make_temporal_sbm(
            num_nodes=num_nodes,
            num_steps=num_steps,
            num_communities=config["num_communities"],
            p_in=config["p_in"],
            p_out=config["p_out"],
            temporal_jitter=config["temporal_jitter"],
            directed=config["directed"],
            random_seed=seed,
        )

        base_config = SpectralCompressionConfig(
            rank=config["rank"],
            random_seed=seed,
            num_splits=config["num_splits"],
        )
        methods = {
            "spectralstore_asym": AsymmetricSpectralCompressor(base_config),
            "tensor_unfolding_svd": TensorUnfoldingSVDCompressor(base_config),
            "sym_svd": SymmetricSVDCompressor(base_config),
            "direct_svd": DirectSVDCompressor(base_config),
        }
        for coverage in config.get("entrywise_bound_coverages", [1.0]):
            robust_config = SpectralCompressionConfig(
                rank=config["rank"],
                random_seed=seed,
                num_splits=config["num_splits"],
                robust_iterations=config["robust_iterations"],
                residual_threshold_mode=config["residual_threshold_mode"],
                residual_mad_multiplier=config["residual_mad_multiplier"],
                residual_quantile=config["residual_quantile"],
                residual_hybrid_tail_quantile=config["residual_hybrid_tail_quantile"],
                residual_hybrid_tail_ratio=config["residual_hybrid_tail_ratio"],
                entrywise_bound_coverage=coverage,
            )
            methods[f"full_hybrid_cov{int(round(coverage * 100))}"] = (
                RobustAsymmetricSpectralCompressor(robust_config)
            )

        observed_snapshots = [snapshot.toarray() for snapshot in dataset.snapshots]
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
            if store.threshold_diagnostics is not None:
                mean_bound = mean_entrywise_error_bound(store)
                max_bound = max_entrywise_error_bound(store)
                values.update(
                    {
                        "observed_bound_coverage": entrywise_bound_coverage(
                            observed_snapshots,
                            store,
                            include_residual=True,
                        ),
                        "expected_bound_coverage": entrywise_bound_coverage(
                            dataset.expected_snapshots,
                            store,
                            include_residual=True,
                        ),
                        "mean_entrywise_bound": mean_bound,
                        "max_entrywise_bound": max_bound,
                        "mean_bound_tightness": mean_bound
                        / max(values["mean_entrywise_error"], 1e-12),
                        "max_bound_tightness": max_bound
                        / max(values["max_entrywise_error"], 1e-12),
                    }
                )
            run_result["methods"][name] = values
            for metric, value in values.items():
                aggregates[name][metric].append(value)
        per_run.append(run_result)

    summarized = summarize_aggregates(aggregates)
    return {
        "mode": mode,
        "num_nodes": num_nodes,
        "num_steps": num_steps,
        "theory_scale": theory_scale(num_nodes, num_steps, config["rank"]),
        "aggregates": summarized,
        "per_run": per_run,
    }


def theory_scale(num_nodes: int, num_steps: int, rank: int) -> float:
    return float(math.sqrt(max(rank, 1) * math.log(max(num_nodes, 2)) / (num_nodes * num_steps)))


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
        "# Synthetic-SBM Bound Scaling Results",
        "",
        f"- node sweep: {', '.join(str(value) for value in dataset['num_nodes_values'])}",
        f"- time sweep: {', '.join(str(value) for value in dataset['num_steps_values'])}",
        f"- communities: {dataset['num_communities']}",
        f"- repeats: {dataset['num_repeats']}",
        f"- rank: {dataset['rank']}",
        f"- asymmetric split ensemble: {dataset['num_splits']}",
        "",
    ]
    lines.extend(render_mode_table(metrics, "sweep_n"))
    lines.extend(render_mode_table(metrics, "sweep_t"))
    return "\n".join(lines)


def render_mode_table(metrics: dict, mode: str) -> list[str]:
    title = "Node Scaling" if mode == "sweep_n" else "Temporal Scaling"
    lines = [
        f"## {title}",
        "",
        "| n | T | theory scale | asym max err | tensor max err | sym max err | "
        "cov100 mean bound | cov99 mean bound | cov95 mean bound | cov95 coverage | "
        "cov95 tightness |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in metrics["scaling"]:
        if result["mode"] != mode:
            continue
        aggregates = result["aggregates"]
        cov100 = aggregates["full_hybrid_cov100"]
        cov99 = aggregates["full_hybrid_cov99"]
        cov95 = aggregates["full_hybrid_cov95"]
        lines.append(
            "| "
            f"{result['num_nodes']} | "
            f"{result['num_steps']} | "
            f"{result['theory_scale']:.4f} | "
            f"{format_mean_std(aggregates['spectralstore_asym']['max_entrywise_error'])} | "
            f"{format_mean_std(aggregates['tensor_unfolding_svd']['max_entrywise_error'])} | "
            f"{format_mean_std(aggregates['sym_svd']['max_entrywise_error'])} | "
            f"{format_mean_std(cov100['mean_entrywise_bound'])} | "
            f"{format_mean_std(cov99['mean_entrywise_bound'])} | "
            f"{format_mean_std(cov95['mean_entrywise_bound'])} | "
            f"{format_mean_std(cov95['observed_bound_coverage'])} | "
            f"{format_mean_std(cov95['mean_bound_tightness'])} |"
        )
    lines.append("")
    return lines


def format_mean_std(values: dict[str, float]) -> str:
    return f"{values['mean']:.4f} +/- {values['std']:.4f}"


if __name__ == "__main__":
    main()
