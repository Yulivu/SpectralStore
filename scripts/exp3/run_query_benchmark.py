from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spectralstore.compression import create_compressor, spectral_config_from_mapping  # noqa: E402
from spectralstore.data_loader import (  # noqa: E402
    SyntheticTemporalGraph,
    load_bitcoin_alpha,
    load_bitcoin_otc,
    make_synthetic_attack,
    make_temporal_sbm,
)
from spectralstore.evaluation import (  # noqa: E402
    load_experiment_config,
    q5_anomaly_detection_scores,
    set_reproducibility_seed,
    storage_report,
)
from spectralstore.query_engine import QueryEngine  # noqa: E402

FIELDNAMES = [
    "method",
    "query",
    "path",
    "num_queries",
    "rank",
    "effective_rank",
    "build_seconds",
    "query_seconds",
    "mean_latency_ms",
    "throughput_qps",
    "q1_rmse",
    "q1_mae",
    "q2_recall_at_k",
    "q2_mean_overlap",
    "q3_nmi",
    "q4_rmse",
    "q4_mae",
    "q5_precision",
    "q5_recall",
    "q5_f1",
    "factor_bytes",
    "residual_bytes",
    "metadata_bytes",
    "compressed_bytes",
    "raw_sparse_bytes",
    "compressed_vs_raw_sparse_ratio",
    "storage_gate_action_taken",
    "storage_gate_accepted",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()

    config = load_experiment_config(args.config, args.overrides)
    set_reproducibility_seed(config.get("random_seed"))
    out_dir = Path(args.out_dir or config.get("output_dir", "experiments/results/exp3"))
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(config)
    dense = [snapshot.toarray() for snapshot in dataset.snapshots]
    rng = np.random.default_rng(int(config.get("random_seed", 0)))
    workloads = build_workloads(dataset, dense, config, rng)

    rows: list[dict[str, Any]] = []
    methods = list(config.get("methods", ["spectralstore_thinking"]))
    for method in methods:
        started = time.perf_counter()
        compressor_config = spectral_config_from_mapping(config.get("compressor", {}))
        store = create_compressor(method, compressor_config).fit_transform(dataset.snapshots)
        build_seconds = time.perf_counter() - started
        engine = QueryEngine(store, raw_snapshots=dataset.snapshots, method=method)
        storage = storage_report(
            store,
            dataset.snapshots,
            factor_dtype_bytes=config.get("compressor", {}).get("factor_storage_dtype_bytes"),
        )
        diagnostics = store.threshold_diagnostics or {}
        common = {
            "method": method,
            "rank": int(config.get("compressor", {}).get("rank", store.rank)),
            "effective_rank": int(diagnostics.get("effective_rank", store.rank)),
            "build_seconds": build_seconds,
            "factor_bytes": storage["factor_bytes"],
            "residual_bytes": storage["residual_bytes"],
            "metadata_bytes": storage["metadata_bytes"],
            "compressed_bytes": storage["compressed_bytes"],
            "raw_sparse_bytes": storage["raw_sparse_bytes"],
            "compressed_vs_raw_sparse_ratio": storage["compressed_vs_raw_sparse_ratio"],
            "storage_gate_action_taken": diagnostics.get("storage_gate_action_taken", "none"),
            "storage_gate_accepted": diagnostics.get("storage_gate_accepted", True),
        }
        rows.extend(run_q1(engine, dense, workloads["q1"], common, config))
        rows.extend(run_q2(engine, dense, workloads["q2"], common, config))
        rows.extend(run_q3(engine, dataset, workloads["q3"], common, config))
        rows.extend(run_q4(engine, dense, workloads["q4"], common, config))
        rows.extend(run_q5(engine, dataset, common, config))
        print(f"[exp3, method={method}] done in {build_seconds:.2f}s", flush=True)

    write_csv(out_dir / "query_records.csv", rows, FIELDNAMES)
    write_csv(out_dir / "summary.csv", rows, FIELDNAMES)
    write_metrics(out_dir / "metrics.json", rows, config)
    write_summary(out_dir / "summary.md", rows, dataset)
    print(f"wrote {out_dir / 'query_records.csv'}")
    print(f"wrote {out_dir / 'summary.csv'}")


def load_dataset(config: dict[str, Any]) -> SyntheticTemporalGraph:
    dataset_cfg = config.get("dataset", {})
    kind = dataset_cfg.get("kind", "synthetic_attack")
    if kind == "synthetic_attack":
        return make_synthetic_attack(
            num_nodes=int(dataset_cfg["num_nodes"]),
            num_steps=int(dataset_cfg["num_steps"]),
            num_communities=int(dataset_cfg["num_communities"]),
            p_in=float(dataset_cfg["p_in"]),
            p_out=float(dataset_cfg["p_out"]),
            temporal_jitter=float(dataset_cfg.get("temporal_jitter", 0.08)),
            attack_kind=str(dataset_cfg.get("attack_kind", "sparse_spike")),
            attack_fraction=float(dataset_cfg.get("attack_fraction", 0.002)),
            outlier_weight=float(dataset_cfg.get("outlier_weight", 3.0)),
            corruption_magnitude=float(dataset_cfg.get("corruption_magnitude", 4.0)),
            directed=bool(dataset_cfg.get("directed", True)),
            random_seed=int(config.get("random_seed", 0)),
        )
    if kind == "synthetic_sbm":
        return make_temporal_sbm(
            num_nodes=int(dataset_cfg["num_nodes"]),
            num_steps=int(dataset_cfg["num_steps"]),
            num_communities=int(dataset_cfg["num_communities"]),
            p_in=float(dataset_cfg["p_in"]),
            p_out=float(dataset_cfg["p_out"]),
            temporal_jitter=float(dataset_cfg.get("temporal_jitter", 0.08)),
            directed=bool(dataset_cfg.get("directed", True)),
            random_seed=int(config.get("random_seed", 0)),
        )
    if kind == "bitcoin_otc":
        return load_bitcoin_otc(
            dataset_cfg["raw_path"],
            max_nodes=int(dataset_cfg["max_nodes"]),
            normalize_ratings=bool(dataset_cfg.get("normalize_ratings", True)),
        )
    if kind == "bitcoin_alpha":
        return load_bitcoin_alpha(
            dataset_cfg["raw_path"],
            max_nodes=int(dataset_cfg["max_nodes"]),
            normalize_ratings=bool(dataset_cfg.get("normalize_ratings", True)),
        )
    raise ValueError(f"unsupported dataset kind: {kind}")


def build_workloads(
    dataset: SyntheticTemporalGraph,
    dense: list[np.ndarray],
    config: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    workload = config.get("workload", {})
    num_nodes = int(dataset.num_nodes)
    num_steps = int(dataset.num_steps)
    return {
        "q1": sample_link_queries(
            dense,
            int(workload.get("num_q1", 1000)),
            rng,
        ),
        "q2": [
            (
                int(rng.integers(num_nodes)),
                int(rng.integers(num_steps)),
                int(workload.get("q2_k", 10)),
            )
            for _ in range(int(workload.get("num_q2", 300)))
        ],
        "q3": [
            int(rng.integers(num_steps))
            for _ in range(int(workload.get("num_q3", min(num_steps, 20))))
        ],
        "q4": sample_trend_queries(
            num_nodes,
            num_steps,
            int(workload.get("num_q4", 300)),
            int(workload.get("q4_window", 5)),
            rng,
        ),
    }


def sample_link_queries(
    dense: list[np.ndarray],
    count: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, int, float]]:
    num_steps = len(dense)
    num_nodes = dense[0].shape[0]
    queries = []
    for _ in range(max(0, count)):
        t = int(rng.integers(num_steps))
        u = int(rng.integers(num_nodes))
        v = int(rng.integers(num_nodes))
        queries.append((u, v, t, float(dense[t][u, v])))
    return queries


def sample_trend_queries(
    num_nodes: int,
    num_steps: int,
    count: int,
    window: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, int, int]]:
    queries = []
    max_window = max(1, min(window, num_steps))
    for _ in range(max(0, count)):
        t1 = int(rng.integers(max(1, num_steps - max_window + 1)))
        t2 = min(num_steps - 1, t1 + max_window - 1)
        queries.append((int(rng.integers(num_nodes)), int(rng.integers(num_nodes)), t1, t2))
    return queries


def run_q1(
    engine: QueryEngine,
    dense: list[np.ndarray],
    queries: list[tuple[int, int, int, float]],
    common: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    paths = config.get("query_paths", {}).get("q1", ["factor_only", "factor_residual"])
    for path in paths:
        include_residual = path != "factor_only"
        started = time.perf_counter()
        estimates = np.asarray(
            [
                engine.link_prob(u, v, t, include_residual=include_residual)
                for u, v, t, _truth in queries
            ],
            dtype=float,
        )
        elapsed = time.perf_counter() - started
        truth = np.asarray([truth for *_entry, truth in queries], dtype=float)
        rows.append(
            make_row(
                common,
                query="Q1",
                path=path,
                num_queries=len(queries),
                query_seconds=elapsed,
                q1_rmse=rmse(estimates, truth),
                q1_mae=mae(estimates, truth),
            )
        )
    return rows


def run_q2(
    engine: QueryEngine,
    dense: list[np.ndarray],
    queries: list[tuple[int, int, int]],
    common: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    paths = config.get("query_paths", {}).get(
        "q2",
        ["scan_factor_only", "exact_index", "ann_index"],
    )
    index_build_seconds = {"scan_factor_only": 0.0, "scan_residual": 0.0}
    if any("exact_index" in path for path in paths):
        started = time.perf_counter()
        engine.build_exact_top_neighbor_index()
        index_build_seconds["exact_index"] = time.perf_counter() - started
        index_build_seconds["exact_index_residual"] = index_build_seconds["exact_index"]
    if any("ann_index" in path for path in paths):
        ann_cfg = config.get("ann_index", {})
        started = time.perf_counter()
        engine.build_ann_top_neighbor_index(
            projection_dim=ann_cfg.get("projection_dim"),
            candidate_multiplier=int(ann_cfg.get("candidate_multiplier", 4)),
            random_seed=int(config.get("random_seed", 0)),
        )
        index_build_seconds["ann_index"] = time.perf_counter() - started
        index_build_seconds["ann_index_residual"] = index_build_seconds["ann_index"]

    truth_sets = [
        set(idx for idx, _score in raw_top_neighbor(dense[t], u, k))
        for u, t, k in queries
    ]
    for path in paths:
        started = time.perf_counter()
        results = [
            q2_query(engine, path, u, t, k)
            for u, t, k in queries
        ]
        elapsed = time.perf_counter() - started
        recalls = []
        overlaps = []
        for truth, result in zip(truth_sets, results):
            predicted = {idx for idx, _score in result}
            overlap = len(truth & predicted)
            recalls.append(overlap / max(len(truth), 1))
            overlaps.append(overlap)
        rows.append(
            make_row(
                common,
                query="Q2",
                path=path,
                num_queries=len(queries),
                build_seconds=float(common["build_seconds"])
                + float(index_build_seconds.get(path, 0.0)),
                query_seconds=elapsed,
                q2_recall_at_k=float(np.mean(recalls)) if recalls else float("nan"),
                q2_mean_overlap=float(np.mean(overlaps)) if overlaps else float("nan"),
            )
        )
    return rows


def q2_query(
    engine: QueryEngine,
    path: str,
    u: int,
    t: int,
    k: int,
) -> list[tuple[int, float]]:
    if path == "scan_factor_only":
        return engine.top_neighbor(u, t, k, include_residual=False, use_index=False)
    if path == "scan_residual":
        return engine.top_neighbor(u, t, k, include_residual=True, use_index=False)
    if path == "exact_index":
        return engine.top_neighbor(u, t, k, include_residual=False, use_index=True)
    if path == "exact_index_residual":
        return engine.top_neighbor(u, t, k, include_residual=True, use_index=True)
    if path == "ann_index":
        return engine.top_neighbor(
            u,
            t,
            k,
            include_residual=False,
            use_index=True,
            index_mode="ann",
        )
    if path == "ann_index_residual":
        return engine.top_neighbor(
            u,
            t,
            k,
            include_residual=True,
            use_index=True,
            index_mode="ann",
        )
    raise ValueError(f"unsupported Q2 path: {path}")


def raw_top_neighbor(snapshot: np.ndarray, u: int, k: int) -> list[tuple[int, float]]:
    row = snapshot[u].copy()
    row[u] = -np.inf
    limit = min(k, row.shape[0])
    indices = np.argpartition(-row, limit - 1)[:limit]
    ordered = indices[np.argsort(-row[indices])]
    return [(int(idx), float(row[idx])) for idx in ordered]


def run_q3(
    engine: QueryEngine,
    dataset: SyntheticTemporalGraph,
    times: list[int],
    common: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    if dataset.communities is None:
        return [
            make_row(
                common,
                query="Q3",
                path="community_cached",
                num_queries=0,
                query_seconds=0.0,
            )
        ]
    started = time.perf_counter()
    scores = [
        normalized_mutual_info_score(
            dataset.communities,
            engine.community_cached(
                t,
                num_communities=int(config.get("dataset", {}).get("num_communities", engine.store.rank)),
                random_seed=int(config.get("random_seed", 0)),
            ),
        )
        for t in times
    ]
    elapsed = time.perf_counter() - started
    return [
        make_row(
            common,
            query="Q3",
            path="community_cached",
            num_queries=len(times),
            query_seconds=elapsed,
            q3_nmi=float(np.mean(scores)) if scores else float("nan"),
        )
    ]


def run_q4(
    engine: QueryEngine,
    dense: list[np.ndarray],
    queries: list[tuple[int, int, int, int]],
    common: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    paths = config.get("query_paths", {}).get("q4", ["factor_only", "factor_residual"])
    for path in paths:
        include_residual = path != "factor_only"
        started = time.perf_counter()
        estimates = []
        truth = []
        for u, v, t1, t2 in queries:
            estimates.extend(
                engine.temporal_trend(
                    u,
                    v,
                    t1,
                    t2,
                    include_residual=include_residual,
                )
            )
            truth.extend(float(dense[t][u, v]) for t in range(t1, t2 + 1))
        elapsed = time.perf_counter() - started
        estimates_arr = np.asarray(estimates, dtype=float)
        truth_arr = np.asarray(truth, dtype=float)
        rows.append(
            make_row(
                common,
                query="Q4",
                path=path,
                num_queries=len(queries),
                query_seconds=elapsed,
                q4_rmse=rmse(estimates_arr, truth_arr),
                q4_mae=mae(estimates_arr, truth_arr),
            )
        )
    return rows


def run_q5(
    engine: QueryEngine,
    dataset: SyntheticTemporalGraph,
    common: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    threshold = float(config.get("workload", {}).get("q5_threshold", 0.0))
    started = time.perf_counter()
    scores = q5_anomaly_detection_scores(
        dataset.attack_edges,
        engine,
        threshold=threshold,
    )
    elapsed = time.perf_counter() - started
    precision = float(scores["q5_precision"])
    recall = float(scores["q5_recall"])
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return [
        make_row(
            common,
            query="Q5",
            path=f"residual_threshold_{threshold:g}",
            num_queries=engine.store.num_steps,
            query_seconds=elapsed,
            q5_precision=precision,
            q5_recall=recall,
            q5_f1=f1,
        )
    ]


def make_row(
    common: dict[str, Any],
    *,
    query: str,
    path: str,
    num_queries: int,
    query_seconds: float,
    build_seconds: float | None = None,
    **metrics: Any,
) -> dict[str, Any]:
    row = {name: float("nan") for name in FIELDNAMES}
    row.update(common)
    row.update(
        {
            "query": query,
            "path": path,
            "num_queries": int(num_queries),
            "build_seconds": float(common["build_seconds"] if build_seconds is None else build_seconds),
            "query_seconds": float(query_seconds),
            "mean_latency_ms": latency_ms(query_seconds, num_queries),
            "throughput_qps": throughput(query_seconds, num_queries),
        }
    )
    row.update(metrics)
    return row


def rmse(estimates: np.ndarray, truth: np.ndarray) -> float:
    if estimates.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((estimates - truth) ** 2)))


def mae(estimates: np.ndarray, truth: np.ndarray) -> float:
    if estimates.size == 0:
        return float("nan")
    return float(np.mean(np.abs(estimates - truth)))


def latency_ms(seconds: float, count: int) -> float:
    if count <= 0:
        return float("nan")
    return float(1000.0 * seconds / count)


def throughput(seconds: float, count: int) -> float:
    if seconds <= 0.0:
        return float("inf")
    return float(count / seconds)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_metrics(path: Path, rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    metrics = {
        "config": config,
        "num_rows": len(rows),
        "queries": sorted({str(row["query"]) for row in rows}),
        "methods": sorted({str(row["method"]) for row in rows}),
        "best_latency_by_query": best_latency_by_query(rows),
    }
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def best_latency_by_query(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output = {}
    for query in sorted({str(row["query"]) for row in rows}):
        candidates = [
            row for row in rows
            if row["query"] == query and not math.isnan(float(row["mean_latency_ms"]))
        ]
        if candidates:
            best = min(candidates, key=lambda row: float(row["mean_latency_ms"]))
            output[query] = {
                "method": best["method"],
                "path": best["path"],
                "mean_latency_ms": best["mean_latency_ms"],
            }
    return output


def write_summary(
    path: Path,
    rows: list[dict[str, Any]],
    dataset: SyntheticTemporalGraph,
) -> None:
    lines = [
        "# Exp3 查询与索引基准摘要",
        "",
        f"- 数据集：`{dataset.name}`，节点数 {dataset.num_nodes}，时间片 {dataset.num_steps}",
        f"- 输出行数：{len(rows)}",
        "- 本实验覆盖 Q1-Q5，并把 Q2 的 `scan`、`exact_index`、`ann_index` 路径放在同一张表中。",
        "",
        "## 最快路径",
        "",
    ]
    for query, item in best_latency_by_query(rows).items():
        lines.append(
            f"- {query}: `{item['method']}` / `{item['path']}`，"
            f"{float(item['mean_latency_ms']):.6f} ms/query"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
