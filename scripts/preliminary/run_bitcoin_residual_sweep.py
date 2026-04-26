"""Sweep Bitcoin-OTC residual thresholds under a sparse-storage constraint."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from spectralstore.compression import (  # noqa: E402
    AsymmetricSpectralCompressor,
    RobustAsymmetricSpectralCompressor,
    spectral_config_from_mapping,
)
from spectralstore.data_loader import load_bitcoin_otc  # noqa: E402
from spectralstore.evaluation import (  # noqa: E402
    load_experiment_config,
    observed_edges_from_snapshots,
    observed_edge_mae,
    observed_edge_rmse,
    residual_nnz,
    residual_sparsity,
    relative_frobenius_error,
    set_reproducibility_seed,
    split_observed_edges,
    storage_report,
    write_experiment_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments/preliminary/bitcoin_otc/configs/residual_sweep.yaml",
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
        help="OmegaConf dotlist override, e.g. --set candidates.0.residual_quantile=0.9999",
    )
    args = parser.parse_args()
    started_at = time.perf_counter()

    config = load_experiment_config(args.config, args.overrides)
    set_reproducibility_seed(config.get("random_seed"))

    dataset = load_bitcoin_otc(config["raw_path"], max_nodes=config["max_nodes"])
    train_snapshots, held_out = split_observed_edges(
        dataset.snapshots,
        test_fraction=config["test_fraction"],
        random_seed=config["random_seed"],
    )
    train_edges = observed_edges_from_snapshots(train_snapshots)

    baseline = evaluate_baseline(config, train_snapshots, train_edges, held_out)
    candidates = [
        evaluate_candidate(config, candidate, train_snapshots, train_edges, held_out, baseline)
        for candidate in config["candidates"]
    ]
    passing = [candidate for candidate in candidates if candidate["accepted"]]
    best = min(
        passing,
        key=lambda item: (
            item["held_out_observed_edge_rmse"],
            item["storage"]["compressed_vs_raw_sparse_ratio"],
        ),
        default=None,
    )

    metrics = {
        "dataset": {
            "name": dataset.name,
            "num_nodes": dataset.num_nodes,
            "num_steps": dataset.num_steps,
            "held_out_edges": len(held_out),
            "train_observed_edges": len(train_edges),
        },
        "acceptance": config["acceptance"],
        "baseline": baseline,
        "candidates": candidates,
        "best_accepted_candidate": best["name"] if best is not None else None,
    }
    summary = render_summary(metrics)
    write_experiment_outputs(
        out_dir=args.out_dir,
        metrics=metrics,
        summary=summary,
        config_path=args.config,
        config=config,
        started_at=started_at,
        metrics_filename="residual_sweep_metrics.json",
        summary_filename="residual_sweep_summary.md",
    )
    print(summary)


def evaluate_baseline(
    config: dict,
    train_snapshots: list[sparse.csr_matrix],
    train_edges: list[tuple[int, int, int, float]],
    held_out: list[tuple[int, int, int, float]],
) -> dict:
    compressor_config = spectral_config_from_mapping(config)
    store = AsymmetricSpectralCompressor(compressor_config).fit_transform(train_snapshots)
    return evaluate_store(
        "spectralstore_asym",
        store,
        train_snapshots,
        train_edges,
        held_out,
        include_residual=False,
    )


def evaluate_candidate(
    config: dict,
    candidate: dict,
    train_snapshots: list[sparse.csr_matrix],
    train_edges: list[tuple[int, int, int, float]],
    held_out: list[tuple[int, int, int, float]],
    baseline: dict,
) -> dict:
    compressor_config = spectral_config_from_mapping(
        config,
        **{
            key: value
            for key, value in candidate.items()
            if key != "name"
        },
    )
    store = RobustAsymmetricSpectralCompressor(compressor_config).fit_transform(train_snapshots)
    values = evaluate_store(
        candidate["name"],
        store,
        train_snapshots,
        train_edges,
        held_out,
        include_residual=True,
    )
    values["candidate_config"] = candidate
    values["accepted"] = is_accepted(values, baseline, config["acceptance"])
    values["acceptance_failures"] = acceptance_failures(values, baseline, config["acceptance"])
    return values


def evaluate_store(
    name: str,
    store,
    train_snapshots: list[sparse.csr_matrix],
    train_edges: list[tuple[int, int, int, float]],
    held_out: list[tuple[int, int, int, float]],
    *,
    include_residual: bool,
) -> dict:
    diagnostics = store.threshold_diagnostics or {}
    return {
        "name": name,
        "rank": store.rank,
        "relative_frobenius_error_train": relative_frobenius_error(
            train_snapshots,
            store,
            include_residual=include_residual,
        ),
        "train_observed_edge_rmse": observed_edge_rmse(
            train_edges,
            store,
            include_residual=include_residual,
        ),
        "train_observed_edge_mae": observed_edge_mae(
            train_edges,
            store,
            include_residual=include_residual,
        ),
        "held_out_observed_edge_rmse": observed_edge_rmse(
            held_out,
            store,
            include_residual=include_residual,
        ),
        "held_out_observed_edge_mae": observed_edge_mae(
            held_out,
            store,
            include_residual=include_residual,
        ),
        "storage": storage_report(store, train_snapshots),
        "residual_nnz": residual_nnz(store),
        "residual_sparsity": residual_sparsity(store),
        "estimated_threshold": diagnostics.get("estimated_threshold"),
        "noise_scale": diagnostics.get("noise_scale"),
    }


def is_accepted(candidate: dict, baseline: dict, acceptance: dict) -> bool:
    return not acceptance_failures(candidate, baseline, acceptance)


def acceptance_failures(candidate: dict, baseline: dict, acceptance: dict) -> list[str]:
    failures = []
    sparse_ratio = candidate["storage"]["compressed_vs_raw_sparse_ratio"]
    if sparse_ratio >= acceptance["max_sparse_ratio"]:
        failures.append("sparse_ratio")

    max_rmse = baseline["held_out_observed_edge_rmse"] * (
        1.0 + acceptance["max_heldout_rmse_regression"]
    )
    if candidate["held_out_observed_edge_rmse"] > max_rmse:
        failures.append("heldout_rmse")

    max_mae = baseline["held_out_observed_edge_mae"] * (
        1.0 + acceptance["max_heldout_mae_regression"]
    )
    if candidate["held_out_observed_edge_mae"] > max_mae:
        failures.append("heldout_mae")
    return failures


def render_summary(metrics: dict) -> str:
    dataset = metrics["dataset"]
    baseline = metrics["baseline"]
    lines = [
        "# Bitcoin-OTC Residual Sweep",
        "",
        f"- nodes: {dataset['num_nodes']}",
        f"- temporal snapshots: {dataset['num_steps']}",
        f"- held-out observed edges: {dataset['held_out_edges']}",
        f"- acceptance sparse ratio: < {metrics['acceptance']['max_sparse_ratio']}",
        f"- best accepted candidate: {metrics['best_accepted_candidate']}",
        "",
        "## Baseline",
        "",
        "| method | held-out RMSE | held-out MAE | sparse ratio | compressed bytes |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            "| "
            f"{baseline['name']} | "
            f"{baseline['held_out_observed_edge_rmse']:.4f} | "
            f"{baseline['held_out_observed_edge_mae']:.4f} | "
            f"{baseline['storage']['compressed_vs_raw_sparse_ratio']:.4f} | "
            f"{baseline['storage']['compressed_bytes']} |"
        ),
        "",
        "## Robust Candidates",
        "",
        "| candidate | accepted | failures | threshold | residual nnz | sparse ratio | "
        "train rel. Frob | held-out RMSE | held-out MAE |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate in metrics["candidates"]:
        failures = ",".join(candidate["acceptance_failures"]) or "-"
        threshold = candidate["estimated_threshold"]
        threshold_text = f"{threshold:.6f}" if threshold is not None else "nan"
        lines.append(
            "| "
            f"{candidate['name']} | "
            f"{candidate['accepted']} | "
            f"{failures} | "
            f"{threshold_text} | "
            f"{candidate['residual_nnz']} | "
            f"{candidate['storage']['compressed_vs_raw_sparse_ratio']:.4f} | "
            f"{candidate['relative_frobenius_error_train']:.4f} | "
            f"{candidate['held_out_observed_edge_rmse']:.4f} | "
            f"{candidate['held_out_observed_edge_mae']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

