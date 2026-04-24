"""Run Synthetic-Spiked scaling experiments."""

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
    SpectralCompressionConfig,
    SymmetricSVDCompressor,
    TensorUnfoldingSVDCompressor,
)
from spectralstore.data_loader import make_synthetic_spiked
from spectralstore.evaluation import (
    max_entrywise_error,
    mean_entrywise_error,
    relative_frobenius_error_against_dense,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments/preliminary/synthetic_spiked/configs/scaling.json",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/preliminary/synthetic_spiked/results",
    )
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scaling_results = []
    for mode in ("sweep_n", "sweep_t", "sweep_snr"):
        for num_nodes, num_steps, snr in settings_for_mode(config, mode):
            scaling_results.append(run_one_setting(config, mode, num_nodes, num_steps, snr))

    metrics = {
        "dataset": {
            "name": "synthetic_spiked_scaling",
            "num_nodes_values": config["num_nodes_values"],
            "num_steps_values": config["num_steps_values"],
            "snr_values": config["snr_values"],
            "rank": config["rank"],
            "num_splits": config["num_splits"],
            "num_repeats": config["num_repeats"],
        },
        "scaling": scaling_results,
    }

    (out_dir / "spiked_scaling_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "spiked_scaling_summary.md").write_text(
        render_summary(metrics),
        encoding="utf-8",
    )
    print(render_summary(metrics))


def settings_for_mode(config: dict, mode: str) -> list[tuple[int, int, float]]:
    center_n = config["num_nodes_values"][1]
    center_t = config["num_steps_values"][1]
    center_snr = config["snr_values"][2]
    if mode == "sweep_n":
        return [(n, center_t, center_snr) for n in config["num_nodes_values"]]
    if mode == "sweep_t":
        return [(center_n, t, center_snr) for t in config["num_steps_values"]]
    if mode == "sweep_snr":
        return [(center_n, center_t, snr) for snr in config["snr_values"]]
    raise ValueError(f"unsupported scaling mode: {mode}")


def run_one_setting(config: dict, mode: str, num_nodes: int, num_steps: int, snr: float) -> dict:
    per_run = []
    aggregates: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for repeat in range(config["num_repeats"]):
        seed = config["random_seed"] + repeat
        dataset = make_synthetic_spiked(
            num_nodes=num_nodes,
            num_steps=num_steps,
            rank=config["rank"],
            snr=snr,
            random_seed=seed,
        )
        compressor_config = SpectralCompressionConfig(
            rank=config["rank"],
            random_seed=seed,
            num_splits=config["num_splits"],
        )
        methods = {
            "spectralstore_asym": AsymmetricSpectralCompressor(compressor_config),
            "tensor_unfolding_svd": TensorUnfoldingSVDCompressor(compressor_config),
            "sym_svd": SymmetricSVDCompressor(compressor_config),
            "direct_svd": DirectSVDCompressor(compressor_config),
        }

        run_result = {"repeat": repeat, "seed": seed, "methods": {}}
        expected_max = max(
            float(np.max(np.abs(snapshot)))
            for snapshot in dataset.expected_snapshots
        )
        for name, compressor in methods.items():
            store = compressor.fit_transform(dataset.snapshots)
            max_error = max_entrywise_error(dataset.expected_snapshots, store)
            values = {
                "max_entrywise_error": max_error,
                "normalized_max_entrywise_error": max_error / max(expected_max, 1e-12),
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

    return {
        "mode": mode,
        "num_nodes": num_nodes,
        "num_steps": num_steps,
        "snr": snr,
        "theory_scale": theory_scale(num_nodes, num_steps, config["rank"], snr),
        "aggregates": summarize_aggregates(aggregates),
        "per_run": per_run,
    }


def theory_scale(num_nodes: int, num_steps: int, rank: int, snr: float) -> float:
    numerator = max(rank, 1) * math.log(max(num_nodes, 2))
    return float(math.sqrt(numerator / (num_nodes * num_steps)) / snr)


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
        "# Synthetic-Spiked Scaling Results",
        "",
        f"- node sweep: {', '.join(str(value) for value in dataset['num_nodes_values'])}",
        f"- time sweep: {', '.join(str(value) for value in dataset['num_steps_values'])}",
        f"- SNR sweep: {', '.join(str(value) for value in dataset['snr_values'])}",
        f"- repeats: {dataset['num_repeats']}",
        f"- rank: {dataset['rank']}",
        f"- asymmetric split ensemble: {dataset['num_splits']}",
        "",
    ]
    for mode, title in (
        ("sweep_n", "Node Scaling"),
        ("sweep_t", "Temporal Scaling"),
        ("sweep_snr", "SNR Scaling"),
    ):
        lines.extend(render_mode_table(metrics, mode, title))
    return "\n".join(lines)


def render_mode_table(metrics: dict, mode: str, title: str) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| n | T | SNR | theory scale | asym norm max | tensor norm max | "
        "sym norm max | direct norm max | asym rel. Frob | tensor rel. Frob |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in metrics["scaling"]:
        if result["mode"] != mode:
            continue
        aggregates = result["aggregates"]
        asym_norm = aggregates["spectralstore_asym"]["normalized_max_entrywise_error"]
        tensor_norm = aggregates["tensor_unfolding_svd"]["normalized_max_entrywise_error"]
        sym_norm = aggregates["sym_svd"]["normalized_max_entrywise_error"]
        direct_norm = aggregates["direct_svd"]["normalized_max_entrywise_error"]
        lines.append(
            "| "
            f"{result['num_nodes']} | "
            f"{result['num_steps']} | "
            f"{result['snr']:.2f} | "
            f"{result['theory_scale']:.4f} | "
            f"{format_mean_std(asym_norm)} | "
            f"{format_mean_std(tensor_norm)} | "
            f"{format_mean_std(sym_norm)} | "
            f"{format_mean_std(direct_norm)} | "
            f"{format_mean_std(aggregates['spectralstore_asym']['relative_frobenius_error'])} | "
            f"{format_mean_std(aggregates['tensor_unfolding_svd']['relative_frobenius_error'])} |"
        )
    lines.append("")
    return lines


def format_mean_std(values: dict[str, float]) -> str:
    return f"{values['mean']:.4f} +/- {values['std']:.4f}"


if __name__ == "__main__":
    main()
