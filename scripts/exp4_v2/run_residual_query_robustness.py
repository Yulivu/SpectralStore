from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spectralstore.compression import create_compressor, spectral_config_from_mapping  # noqa: E402
from spectralstore.data_loader import make_synthetic_attack  # noqa: E402
from spectralstore.evaluation import (  # noqa: E402
    anomaly_precision_recall_f1,
    load_experiment_config,
    q5_anomaly_detection_scores,
    relative_frobenius_error_against_dense,
    set_reproducibility_seed,
    storage_report,
)
from spectralstore.query_engine import QueryEngine  # noqa: E402

RAW_FIELDNAMES = [
    "attack_kind",
    "attack_fraction",
    "method",
    "repeat",
    "effective_rank",
    "runtime",
    "residual_precision",
    "residual_recall",
    "residual_f1",
    "residual_predicted_count",
    "residual_truth_count",
    "q5_precision",
    "q5_recall",
    "q5_f1",
    "q5_detected_edges",
    "q1_attack_rmse_factor_only",
    "q1_attack_rmse_factor_residual",
    "q1_attack_mae_factor_only",
    "q1_attack_mae_factor_residual",
    "q1_attack_rmse_improvement",
    "q4_attack_rmse_factor_only",
    "q4_attack_rmse_factor_residual",
    "q4_attack_rmse_improvement",
    "frobenius_factor_only",
    "frobenius_factor_residual",
    "factor_bytes",
    "residual_bytes",
    "metadata_bytes",
    "compressed_bytes",
    "raw_sparse_bytes",
    "compressed_vs_raw_sparse_ratio",
    "storage_gate_action_taken",
    "storage_gate_accepted",
]

SUMMARY_FIELDNAMES = [
    "attack_kind",
    "attack_fraction",
    "method",
    "residual_precision_mean",
    "residual_precision_std",
    "residual_recall_mean",
    "residual_recall_std",
    "residual_f1_mean",
    "residual_f1_std",
    "q5_precision_mean",
    "q5_recall_mean",
    "q5_f1_mean",
    "q1_attack_rmse_factor_only_mean",
    "q1_attack_rmse_factor_residual_mean",
    "q1_attack_rmse_improvement_mean",
    "q4_attack_rmse_factor_only_mean",
    "q4_attack_rmse_factor_residual_mean",
    "q4_attack_rmse_improvement_mean",
    "compressed_vs_raw_sparse_ratio_mean",
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
    out_dir = Path(
        args.out_dir
        or config.get("output_dir", "experiments/results/exp4_v2/residual_query_robustness")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "raw_records.csv"
    rows = run_sweep(config, raw_path)
    summary_rows = summarize(rows)
    write_csv(out_dir / "summary.csv", summary_rows, SUMMARY_FIELDNAMES)
    write_metrics(out_dir / "metrics.json", rows, summary_rows, config)
    write_summary(out_dir / "summary.md", summary_rows)
    print(f"wrote {raw_path}")
    print(f"wrote {out_dir / 'summary.csv'}")


def run_sweep(config: dict[str, Any], raw_path: Path) -> list[dict[str, Any]]:
    rows = read_csv(raw_path)
    completed = {
        (
            str(row["attack_kind"]),
            round(float(row["attack_fraction"]), 12),
            str(row["method"]),
            int(row["repeat"]),
        )
        for row in rows
    }
    base_seed = int(config.get("random_seed", 0))
    total_repeats = int(config.get("num_repeats", 1))
    methods = list(config.get("methods", ["spectralstore_thinking"]))
    attack_kinds = list(config.get("attack_kinds", ["sparse_spike"]))
    attack_fractions = list(config.get("attack_fractions", [0.002]))
    dataset_cfg = config.get("dataset", {})

    for attack_kind in attack_kinds:
        for attack_fraction in attack_fractions:
            for repeat in range(total_repeats):
                seed = base_seed + repeat
                dataset = make_synthetic_attack(
                    num_nodes=int(dataset_cfg["num_nodes"]),
                    num_steps=int(dataset_cfg["num_steps"]),
                    num_communities=int(dataset_cfg["num_communities"]),
                    p_in=float(dataset_cfg["p_in"]),
                    p_out=float(dataset_cfg["p_out"]),
                    temporal_jitter=float(dataset_cfg.get("temporal_jitter", 0.08)),
                    attack_kind=str(attack_kind),
                    attack_fraction=float(attack_fraction),
                    outlier_weight=float(dataset_cfg.get("outlier_weight", 4.0)),
                    corruption_magnitude=float(dataset_cfg.get("corruption_magnitude", 4.0)),
                    directed=bool(dataset_cfg.get("directed", True)),
                    random_seed=seed,
                )
                expected = [
                    np.asarray(snapshot, dtype=float)
                    for snapshot in dataset.expected_snapshots or []
                ]
                attack_queries = list(dataset.attack_edges)
                for method in methods:
                    key = (
                        str(attack_kind),
                        round(float(attack_fraction), 12),
                        str(method),
                        int(repeat),
                    )
                    if key in completed:
                        continue
                    started = time.perf_counter()
                    compressor_config = spectral_config_from_mapping(
                        {
                            **config.get("compressor", {}),
                            "random_seed": seed,
                        }
                    )
                    store = create_compressor(method, compressor_config).fit_transform(
                        dataset.snapshots
                    )
                    runtime = time.perf_counter() - started
                    engine = QueryEngine(store, method=method)
                    storage = storage_report(
                        store,
                        dataset.snapshots,
                        factor_dtype_bytes=config.get("compressor", {}).get("factor_storage_dtype_bytes"),
                    )
                    diagnostics = store.threshold_diagnostics or {}
                    residual_scores = anomaly_precision_recall_f1(dataset.attack_edges, store)
                    q5_scores = q5_anomaly_detection_scores(
                        dataset.attack_edges,
                        engine,
                        threshold=float(config.get("q5_threshold", 0.0)),
                    )
                    q5_precision = float(q5_scores["q5_precision"])
                    q5_recall = float(q5_scores["q5_recall"])
                    q5_f1 = harmonic_f1(q5_precision, q5_recall)
                    q1_factor, q1_residual = attack_q1_errors(
                        engine,
                        expected,
                        attack_queries,
                    )
                    q4_factor, q4_residual = attack_q4_errors(
                        engine,
                        expected,
                        attack_queries,
                        window=int(config.get("q4_window", 3)),
                    )
                    row = {
                        "attack_kind": attack_kind,
                        "attack_fraction": attack_fraction,
                        "method": method,
                        "repeat": repeat,
                        "effective_rank": int(diagnostics.get("effective_rank", store.rank)),
                        "runtime": runtime,
                        "residual_precision": residual_scores["precision"],
                        "residual_recall": residual_scores["recall"],
                        "residual_f1": residual_scores["f1"],
                        "residual_predicted_count": residual_scores["anomaly_predicted_count"],
                        "residual_truth_count": residual_scores["anomaly_truth_count"],
                        "q5_precision": q5_precision,
                        "q5_recall": q5_recall,
                        "q5_f1": q5_f1,
                        "q5_detected_edges": q5_scores["q5_detected_edges"],
                        "q1_attack_rmse_factor_only": q1_factor["rmse"],
                        "q1_attack_rmse_factor_residual": q1_residual["rmse"],
                        "q1_attack_mae_factor_only": q1_factor["mae"],
                        "q1_attack_mae_factor_residual": q1_residual["mae"],
                        "q1_attack_rmse_improvement": improvement(
                            q1_factor["rmse"],
                            q1_residual["rmse"],
                        ),
                        "q4_attack_rmse_factor_only": q4_factor["rmse"],
                        "q4_attack_rmse_factor_residual": q4_residual["rmse"],
                        "q4_attack_rmse_improvement": improvement(
                            q4_factor["rmse"],
                            q4_residual["rmse"],
                        ),
                        "frobenius_factor_only": relative_frobenius_error_against_dense(
                            expected,
                            store,
                            include_residual=False,
                        ) if expected else float("nan"),
                        "frobenius_factor_residual": relative_frobenius_error_against_dense(
                            expected,
                            store,
                            include_residual=True,
                        ) if expected else float("nan"),
                        "factor_bytes": storage["factor_bytes"],
                        "residual_bytes": storage["residual_bytes"],
                        "metadata_bytes": storage["metadata_bytes"],
                        "compressed_bytes": storage["compressed_bytes"],
                        "raw_sparse_bytes": storage["raw_sparse_bytes"],
                        "compressed_vs_raw_sparse_ratio": storage["compressed_vs_raw_sparse_ratio"],
                        "storage_gate_action_taken": diagnostics.get("storage_gate_action_taken", "none"),
                        "storage_gate_accepted": diagnostics.get("storage_gate_accepted", True),
                    }
                    rows.append(row)
                    completed.add(key)
                    append_csv(raw_path, row, RAW_FIELDNAMES)
                    print(
                        f"[Exp4_v2, {attack_kind}, eps={float(attack_fraction):.4f}, "
                        f"method={method}, repeat={repeat + 1}/{total_repeats}] "
                        f"done in {runtime:.1f}s",
                        flush=True,
                    )
    return rows


def attack_q1_errors(
    engine: QueryEngine,
    expected: list[np.ndarray],
    attack_edges: list[tuple[int, int, int]],
) -> tuple[dict[str, float], dict[str, float]]:
    if not expected or not attack_edges:
        nan = {"rmse": float("nan"), "mae": float("nan")}
        return nan, nan
    truth = np.asarray([expected[t][u, v] for t, u, v in attack_edges], dtype=float)
    factor = np.asarray(
        [
            engine.link_prob(u, v, t, include_residual=False)
            for t, u, v in attack_edges
        ],
        dtype=float,
    )
    residual = np.asarray(
        [
            engine.link_prob(u, v, t, include_residual=True)
            for t, u, v in attack_edges
        ],
        dtype=float,
    )
    return error_report(factor, truth), error_report(residual, truth)


def attack_q4_errors(
    engine: QueryEngine,
    expected: list[np.ndarray],
    attack_edges: list[tuple[int, int, int]],
    *,
    window: int,
) -> tuple[dict[str, float], dict[str, float]]:
    if not expected or not attack_edges:
        nan = {"rmse": float("nan"), "mae": float("nan")}
        return nan, nan
    half = max(0, int(window) // 2)
    factor_values = []
    residual_values = []
    truth_values = []
    for t, u, v in attack_edges:
        start = max(0, t - half)
        end = min(engine.store.num_steps - 1, t + half)
        factor_values.extend(engine.temporal_trend(u, v, start, end, include_residual=False))
        residual_values.extend(engine.temporal_trend(u, v, start, end, include_residual=True))
        truth_values.extend(float(expected[tt][u, v]) for tt in range(start, end + 1))
    truth = np.asarray(truth_values, dtype=float)
    return (
        error_report(np.asarray(factor_values, dtype=float), truth),
        error_report(np.asarray(residual_values, dtype=float), truth),
    )


def error_report(values: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"rmse": float("nan"), "mae": float("nan")}
    diff = values - truth
    return {
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mae": float(np.mean(np.abs(diff))),
    }


def improvement(before: float, after: float) -> float:
    if not np.isfinite(before) or before <= 0.0:
        return float("nan")
    return float((before - after) / before)


def harmonic_f1(precision: float, recall: float) -> float:
    return float(2.0 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["attack_kind"]),
                float(row["attack_fraction"]),
                str(row["method"]),
            )
        ].append(row)
    output = []
    for (attack_kind, attack_fraction, method), values in sorted(grouped.items()):
        summary = {
            "attack_kind": attack_kind,
            "attack_fraction": attack_fraction,
            "method": method,
        }
        for metric in [
            "residual_precision",
            "residual_recall",
            "residual_f1",
        ]:
            summary[f"{metric}_mean"] = mean(values, metric)
            summary[f"{metric}_std"] = std(values, metric)
        for metric in [
            "q5_precision",
            "q5_recall",
            "q5_f1",
            "q1_attack_rmse_factor_only",
            "q1_attack_rmse_factor_residual",
            "q1_attack_rmse_improvement",
            "q4_attack_rmse_factor_only",
            "q4_attack_rmse_factor_residual",
            "q4_attack_rmse_improvement",
            "compressed_vs_raw_sparse_ratio",
            "runtime",
        ]:
            summary[f"{metric}_mean"] = mean(values, metric)
        output.append(summary)
    return output


def mean(rows: list[dict[str, Any]], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    return float(np.nanmean(values)) if values.size else float("nan")


def std(rows: list[dict[str, Any]], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    return float(np.nanstd(values, ddof=1)) if values.size > 1 else 0.0


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_csv(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    import pandas as pd

    pd.DataFrame([row], columns=fieldnames).to_csv(
        path,
        mode="a",
        header=not os.path.exists(path),
        index=False,
    )


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
        "experiment": "Exp4_v2",
        "purpose": "residual anomaly capture and query correction",
        "config": config,
        "num_raw_rows": len(rows),
        "num_summary_rows": len(summary_rows),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Exp4_v2 残差-查询鲁棒性摘要",
        "",
        "- 目的：验证鲁棒残差是否捕获注入异常边，并修正 Q1/Q4/Q5 查询。",
        "- 与旧 Exp4 的区别：旧 Exp4 主要看因子表示在攻击下是否崩溃；Exp4_v2 直接评价残差命中率和查询修正收益。",
        "",
        "## 结果概览",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row['attack_kind']}` eps={float(row['attack_fraction']):.4f}, "
            f"`{row['method']}`: residual F1={float(row['residual_f1_mean']):.4f}, "
            f"Q5 F1={float(row['q5_f1_mean']):.4f}, "
            f"Q1 RMSE improvement={float(row['q1_attack_rmse_improvement_mean']):.4f}, "
            f"sparse ratio={float(row['compressed_vs_raw_sparse_ratio_mean']):.4f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
