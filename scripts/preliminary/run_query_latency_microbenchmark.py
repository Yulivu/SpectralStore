"""Run a small reproducible query-latency microbenchmark."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spectralstore.compression import (  # noqa: E402
    AsymmetricSpectralCompressor,
    RobustAsymmetricSpectralCompressor,
    spectral_config_from_mapping,
)
from spectralstore.data_loader import make_synthetic_attack  # noqa: E402
from spectralstore.evaluation import (  # noqa: E402
    load_experiment_config,
    set_reproducibility_seed,
    write_experiment_outputs,
)
from spectralstore.query_engine import QueryEngine  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments/preliminary/query_latency/configs/default.yaml",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/preliminary/query_latency/results",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        help="OmegaConf dotlist override, e.g. --set num_queries=10",
    )
    args = parser.parse_args()
    started_at = time.perf_counter()

    config = load_experiment_config(args.config, args.overrides)
    set_reproducibility_seed(config.get("random_seed"))
    out_dir = Path(args.out_dir)

    dataset = make_synthetic_attack(
        num_nodes=config["num_nodes"],
        num_steps=config["num_steps"],
        num_communities=config["num_communities"],
        p_in=config["p_in"],
        p_out=config["p_out"],
        temporal_jitter=config["temporal_jitter"],
        directed=config["directed"],
        random_seed=config["random_seed"],
    )
    dense_snapshots = [snapshot.toarray() for snapshot in dataset.snapshots]

    compressor_config = spectral_config_from_mapping(
        config,
        residual_threshold_mode="hybrid",
    )
    factor_store = AsymmetricSpectralCompressor(compressor_config).fit_transform(
        dataset.snapshots
    )
    residual_store = RobustAsymmetricSpectralCompressor(compressor_config).fit_transform(
        dataset.snapshots
    )
    factor_engine = QueryEngine(factor_store)
    residual_engine = QueryEngine(residual_store)
    indexed_engine = QueryEngine(factor_store)
    indexed_engine.build_exact_top_neighbor_index()

    queries = make_queries(config)
    measurements = {
        "raw_dense": benchmark_raw_dense(dense_snapshots, queries, config),
        "factorized": benchmark_engine(
            factor_engine,
            queries,
            config,
            include_residual=False,
            use_index=False,
        ),
        "factorized_residual": benchmark_engine(
            residual_engine,
            queries,
            config,
            include_residual=True,
            use_index=False,
        ),
        "indexed_top_neighbor": benchmark_engine(
            indexed_engine,
            queries,
            config,
            include_residual=False,
            use_index=True,
            top_neighbor_only=True,
        ),
    }

    metrics = {
        "config": config,
        "queries": {
            "q1_link_prob": len(queries["pairs"]),
            "q2_top_neighbor": len(queries["top_neighbor"]),
            "q4_temporal_trend": len(queries["trends"]),
            "q5_anomaly_detect": len(queries["anomaly_times"]),
        },
        "measurements": measurements,
    }
    summary = render_summary(metrics)
    write_experiment_outputs(
        out_dir=out_dir,
        metrics=metrics,
        summary=summary,
        config_path=args.config,
        config=config,
        started_at=started_at,
    )
    print(summary)


def make_queries(config: dict) -> dict[str, list[tuple[int, ...]]]:
    rng = np.random.default_rng(config["random_seed"])
    n = config["num_nodes"]
    steps = config["num_steps"]
    count = config["num_queries"]
    trend_length = min(config["trend_length"], steps)
    return {
        "pairs": [
            (int(rng.integers(n)), int(rng.integers(n)), int(rng.integers(steps)))
            for _ in range(count)
        ],
        "top_neighbor": [
            (int(rng.integers(n)), int(rng.integers(steps))) for _ in range(count)
        ],
        "trends": [
            (
                int(rng.integers(n)),
                int(rng.integers(n)),
                int(start := rng.integers(0, steps - trend_length + 1)),
                int(start + trend_length - 1),
            )
            for _ in range(count)
        ],
        "anomaly_times": [(int(rng.integers(steps)),) for _ in range(count)],
    }


def benchmark_raw_dense(
    dense_snapshots: list[np.ndarray],
    queries: dict[str, list[tuple[int, ...]]],
    config: dict,
) -> dict[str, dict[str, float]]:
    threshold = config["anomaly_threshold"]
    top_k = config["top_k"]
    return {
        "q1_link_prob": time_calls(
            lambda: [dense_snapshots[t][u, v] for u, v, t in queries["pairs"]],
            len(queries["pairs"]),
        ),
        "q2_top_neighbor": time_calls(
            lambda: [
                raw_top_neighbor(dense_snapshots[t][u], u, top_k)
                for u, t in queries["top_neighbor"]
            ],
            len(queries["top_neighbor"]),
        ),
        "q4_temporal_trend": time_calls(
            lambda: [
                [dense_snapshots[t][u, v] for t in range(t1, t2 + 1)]
                for u, v, t1, t2 in queries["trends"]
            ],
            len(queries["trends"]),
        ),
        "q5_anomaly_detect": time_calls(
            lambda: [
                np.argwhere(np.abs(dense_snapshots[t]) > threshold)
                for (t,) in queries["anomaly_times"]
            ],
            len(queries["anomaly_times"]),
        ),
    }


def benchmark_engine(
    engine: QueryEngine,
    queries: dict[str, list[tuple[int, ...]]],
    config: dict,
    *,
    include_residual: bool,
    use_index: bool,
    top_neighbor_only: bool = False,
) -> dict[str, dict[str, float] | None]:
    top_k = config["top_k"]
    threshold = config["anomaly_threshold"]
    result: dict[str, dict[str, float] | None] = {
        "q1_link_prob": None,
        "q2_top_neighbor": time_calls(
            lambda: [
                engine.top_neighbor(
                    u,
                    t,
                    top_k,
                    include_residual=include_residual,
                    use_index=use_index,
                )
                for u, t in queries["top_neighbor"]
            ],
            len(queries["top_neighbor"]),
        ),
        "q4_temporal_trend": None,
        "q5_anomaly_detect": None,
    }
    if top_neighbor_only:
        return result

    result["q1_link_prob"] = time_calls(
        lambda: [
            engine.link_prob(u, v, t, include_residual=include_residual)
            for u, v, t in queries["pairs"]
        ],
        len(queries["pairs"]),
    )
    result["q4_temporal_trend"] = time_calls(
        lambda: [
            engine.temporal_trend(u, v, t1, t2, include_residual=include_residual)
            for u, v, t1, t2 in queries["trends"]
        ],
        len(queries["trends"]),
    )
    result["q5_anomaly_detect"] = time_calls(
        lambda: [engine.anomaly_detect(t, threshold) for (t,) in queries["anomaly_times"]],
        len(queries["anomaly_times"]),
    )
    return result


def time_calls(call: Callable[[], object], query_count: int) -> dict[str, float]:
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        call()
        samples.append(time.perf_counter() - start)
    return {
        "query_count": float(query_count),
        "mean_latency_seconds": float(statistics.mean(samples) / max(query_count, 1)),
        "total_seconds_mean": float(statistics.mean(samples)),
    }


def raw_top_neighbor(row: np.ndarray, u: int, k: int) -> list[tuple[int, float]]:
    scores = row.copy()
    scores[u] = -np.inf
    limit = min(k, scores.shape[0])
    candidate_indices = np.argpartition(-scores, limit - 1)[:limit]
    ordered = candidate_indices[np.argsort(-scores[candidate_indices])]
    return [(int(idx), float(scores[idx])) for idx in ordered]


def render_summary(metrics: dict) -> str:
    lines = [
        "# Query Latency Microbenchmark",
        "",
        f"- seed: {metrics['config']['random_seed']}",
        f"- nodes: {metrics['config']['num_nodes']}",
        f"- temporal snapshots: {metrics['config']['num_steps']}",
        f"- queries per query type: {metrics['config']['num_queries']}",
        "",
        "| path | query | query count | mean latency seconds |",
        "| --- | --- | ---: | ---: |",
    ]
    for path, queries in metrics["measurements"].items():
        for query_name, values in queries.items():
            if values is None:
                continue
            lines.append(
                "| "
                f"{path} | {query_name} | "
                f"{int(values['query_count'])} | "
                f"{values['mean_latency_seconds']:.8f} |"
            )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

