from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spectralstore.compression import SpectralCompressionConfig, create_compressor  # noqa: E402
from spectralstore.data_loader import make_low_rank_temporal_graph, make_temporal_sbm  # noqa: E402
from spectralstore.evaluation import (  # noqa: E402
    load_experiment_config,
    relative_frobenius_error_against_dense,
    set_reproducibility_seed,
)

FIELDNAMES = [
    "dataset",
    "true_rank",
    "num_nodes",
    "num_steps",
    "noise_scale",
    "repeat",
    "rank_mode",
    "requested_rank",
    "effective_rank",
    "ard_converged",
    "ard_iterations",
    "ard_refit_after_selection",
    "frobenius_error",
    "runtime",
]

SUMMARY_FIELDNAMES = [
    "dataset",
    "true_rank",
    "num_nodes",
    "num_steps",
    "noise_scale",
    "rank_mode",
    "effective_rank_mean",
    "rank_accuracy_mean",
    "frobenius_error_mean",
    "runtime_mean",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()

    config = load_experiment_config(args.config, args.overrides)
    set_reproducibility_seed(config.get("random_seed"))
    out_dir = Path(args.out_dir or config.get("output_dir", "experiments/results/exp5/ard_diagnostic"))
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = run_sweep(config)
    summary_rows = summarize(rows)
    write_csv(out_dir / "ard_diagnostic_raw.csv", rows, FIELDNAMES)
    write_csv(out_dir / "ard_diagnostic_summary.csv", summary_rows, SUMMARY_FIELDNAMES)
    write_metrics(out_dir / "metrics.json", rows, summary_rows, config)
    write_summary(out_dir / "summary.md", summary_rows)
    print(f"wrote {out_dir / 'ard_diagnostic_raw.csv'}")
    print(f"wrote {out_dir / 'ard_diagnostic_summary.csv'}")


def run_sweep(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    base_seed = int(config.get("random_seed", 0))
    repeats = int(config.get("num_repeats", 1))
    for dataset_name in config.get("datasets", ["exact_low_rank"]):
        for true_rank in config.get("true_rank_values", [3]):
            for num_nodes in config.get("num_nodes_values", [80]):
                for num_steps in config.get("num_steps_values", [6]):
                    for noise_scale in config.get("noise_scale_values", [0.0]):
                        for repeat in range(repeats):
                            seed = base_seed + repeat
                            snapshots, expected = make_dataset(
                                str(dataset_name),
                                true_rank=int(true_rank),
                                num_nodes=int(num_nodes),
                                num_steps=int(num_steps),
                                noise_scale=float(noise_scale),
                                random_seed=seed,
                                config=config,
                            )
                            for rank_mode, requested_rank in rank_candidates(
                                int(true_rank),
                                int(config.get("ard_max_rank", 12)),
                            ):
                                started = time.perf_counter()
                                store = create_compressor(
                                    "spectralstore_thinking",
                                    build_config(config, rank_mode, requested_rank, seed),
                                ).fit_transform(snapshots)
                                runtime = time.perf_counter() - started
                                diagnostics = store.threshold_diagnostics or {}
                                row = {
                                    "dataset": dataset_name,
                                    "true_rank": true_rank,
                                    "num_nodes": num_nodes,
                                    "num_steps": num_steps,
                                    "noise_scale": noise_scale,
                                    "repeat": repeat,
                                    "rank_mode": rank_mode,
                                    "requested_rank": requested_rank,
                                    "effective_rank": int(diagnostics.get("effective_rank", store.rank)),
                                    "ard_converged": bool(diagnostics.get("ard_converged", False)),
                                    "ard_iterations": int(diagnostics.get("ard_iterations", 0)),
                                    "ard_refit_after_selection": bool(
                                        diagnostics.get("ard_refit_after_selection", False)
                                    ),
                                    "frobenius_error": relative_frobenius_error_against_dense(
                                        expected,
                                        store,
                                        include_residual=False,
                                    ),
                                    "runtime": runtime,
                                }
                                rows.append(row)
                                print(
                                    f"[ARD diagnostic, {dataset_name}, r*={true_rank}, "
                                    f"mode={rank_mode}, eff={row['effective_rank']}] "
                                    f"done in {runtime:.2f}s",
                                    flush=True,
                                )
    return rows


def make_dataset(
    dataset_name: str,
    *,
    true_rank: int,
    num_nodes: int,
    num_steps: int,
    noise_scale: float,
    random_seed: int,
    config: dict[str, Any],
):
    if dataset_name == "exact_low_rank":
        snapshots = make_low_rank_temporal_graph(
            num_nodes=num_nodes,
            num_steps=num_steps,
            rank=true_rank,
            noise_scale=noise_scale,
            random_seed=random_seed,
        )
        expected = [snapshot.toarray() for snapshot in snapshots]
        return snapshots, expected
    if dataset_name == "sbm_observed":
        sbm_cfg = config.get("sbm", {})
        dataset = make_temporal_sbm(
            num_nodes=num_nodes,
            num_steps=num_steps,
            num_communities=int(sbm_cfg.get("num_communities", true_rank)),
            p_in=float(sbm_cfg.get("p_in", 0.25)),
            p_out=float(sbm_cfg.get("p_out", 0.04)),
            temporal_jitter=float(sbm_cfg.get("temporal_jitter", 0.08)),
            directed=bool(sbm_cfg.get("directed", True)),
            random_seed=random_seed,
        )
        expected = [snapshot.toarray() for snapshot in dataset.snapshots]
        return dataset.snapshots, expected
    raise ValueError(f"unsupported ARD diagnostic dataset: {dataset_name}")


def rank_candidates(true_rank: int, max_rank: int) -> list[tuple[str, int]]:
    return [
        ("fixed_exact", true_rank),
        ("fixed_over", max_rank),
        ("ard", max_rank),
    ]


def build_config(
    run_cfg: dict[str, Any],
    rank_mode: str,
    rank: int,
    seed: int,
) -> SpectralCompressionConfig:
    common = {
        "rank": rank,
        "random_seed": seed,
        "robust_iterations": int(run_cfg.get("robust_iterations", 1)),
        "residual_threshold_mode": str(run_cfg.get("residual_threshold_mode", "mad")),
        "residual_quantile": float(run_cfg.get("residual_quantile", 0.98)),
        "residual_threshold_scale": float(run_cfg.get("residual_threshold_scale", 1.0)),
    }
    if rank_mode != "ard":
        return SpectralCompressionConfig(**common, rank_selection_mode="fixed")
    ard = run_cfg.get("ard", {})
    return SpectralCompressionConfig(
        **common,
        rank_selection_mode="ard",
        ard_max_rank=rank,
        ard_prior_alpha=float(ard.get("prior_alpha", 1e-2)),
        ard_prior_beta=float(ard.get("prior_beta", 1.0)),
        ard_max_iterations=int(ard.get("max_iterations", 100)),
        ard_tolerance=float(ard.get("tolerance", 1e-6)),
        ard_min_effective_ratio=float(ard.get("min_effective_ratio", 0.05)),
        ard_min_rank=int(ard.get("min_rank", 1)),
    )


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["dataset"],
                int(row["true_rank"]),
                int(row["num_nodes"]),
                int(row["num_steps"]),
                float(row["noise_scale"]),
                row["rank_mode"],
            )
        ].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        dataset, true_rank, num_nodes, num_steps, noise_scale, rank_mode = key
        effective = np.asarray([float(row["effective_rank"]) for row in values], dtype=float)
        frob = np.asarray([float(row["frobenius_error"]) for row in values], dtype=float)
        runtime = np.asarray([float(row["runtime"]) for row in values], dtype=float)
        output.append(
            {
                "dataset": dataset,
                "true_rank": true_rank,
                "num_nodes": num_nodes,
                "num_steps": num_steps,
                "noise_scale": noise_scale,
                "rank_mode": rank_mode,
                "effective_rank_mean": float(np.mean(effective)),
                "rank_accuracy_mean": float(np.mean(effective == float(true_rank))),
                "frobenius_error_mean": float(np.mean(frob)),
                "runtime_mean": float(np.mean(runtime)),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_metrics(
    path: Path,
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    payload = {
        "experiment": "ARD diagnostic",
        "purpose": "small closed-loop rank selection and reconstruction sanity check",
        "config": config,
        "num_raw_rows": len(rows),
        "num_summary_rows": len(summary_rows),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# ARD 小闭环诊断摘要",
        "",
        "- 目的：先在精确低秩和小型 SBM 上判断 ARD 是选择问题、尺度问题，还是数据真秩定义问题。",
        "- 当前修复策略：ARD 负责选择分量；选择后用保留的 SVD 分量重新投影，避免 shrinkage 尺度污染最终重构。",
        "",
        "## 结果概览",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row['dataset']}` r*={row['true_rank']} n={row['num_nodes']} "
            f"T={row['num_steps']} noise={float(row['noise_scale']):.4f} "
            f"`{row['rank_mode']}`: eff={float(row['effective_rank_mean']):.2f}, "
            f"rank_acc={float(row['rank_accuracy_mean']):.2f}, "
            f"frob={float(row['frobenius_error_mean']):.6f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
