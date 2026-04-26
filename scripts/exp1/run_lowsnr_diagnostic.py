"""Run a focused low-SNR Synthetic-SBM diagnostic."""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spectralstore.compression import SpectralCompressionConfig, create_compressor  # noqa: E402
from spectralstore.data_loader import make_temporal_sbm  # noqa: E402
from spectralstore.evaluation import (  # noqa: E402
    max_entrywise_error,
    mean_entrywise_error,
    relative_frobenius_error_against_dense,
)


NUM_NODES = 2000
NUM_STEPS = 20
RANK = 5
NUM_COMMUNITIES = 5
P_IN = 0.15
P_OUT = 0.10
TEMPORAL_JITTER = 0.08
DIRECTED = True
NUM_REPEATS = 10
RANDOM_SEED = 401
NUM_SPLITS = 1

METHODS = (
    "spectralstore_asym",
    "sym_svd",
    "direct_svd",
)


def main() -> None:
    started = time.perf_counter()
    metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for repeat in range(NUM_REPEATS):
        seed = RANDOM_SEED + repeat
        print(f"repeat {repeat + 1}/{NUM_REPEATS} seed={seed}", flush=True)
        dataset = make_temporal_sbm(
            num_nodes=NUM_NODES,
            num_steps=NUM_STEPS,
            num_communities=NUM_COMMUNITIES,
            p_in=P_IN,
            p_out=P_OUT,
            temporal_jitter=TEMPORAL_JITTER,
            directed=DIRECTED,
            random_seed=seed,
        )
        config = SpectralCompressionConfig(
            rank=RANK,
            random_seed=seed,
            num_splits=NUM_SPLITS,
        )

        for method in METHODS:
            method_started = time.perf_counter()
            store = create_compressor(method, config).fit_transform(dataset.snapshots)
            seconds = time.perf_counter() - method_started
            metrics[method]["seconds"].append(seconds)
            metrics[method]["max_entrywise"].append(
                max_entrywise_error(dataset.expected_snapshots, store)
            )
            metrics[method]["mean_entrywise"].append(
                mean_entrywise_error(dataset.expected_snapshots, store)
            )
            metrics[method]["relative_frobenius"].append(
                relative_frobenius_error_against_dense(dataset.expected_snapshots, store)
            )
            print(f"  {method}: {seconds:.3f}s", flush=True)

    print()
    print(
        f"Low-SNR Synthetic-SBM diagnostic: n={NUM_NODES}, T={NUM_STEPS}, "
        f"r={RANK}, p={P_IN}, q={P_OUT}, repeats={NUM_REPEATS}"
    )
    print(render_table(metrics))
    print(f"\ntotal seconds: {time.perf_counter() - started:.3f}")


def render_table(metrics: dict[str, dict[str, list[float]]]) -> str:
    lines = [
        "| method | seconds | max entrywise | mean entrywise | rel. Frobenius |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        values = metrics[method]
        lines.append(
            "| "
            f"{method} | "
            f"{mean_std(values['seconds'])} | "
            f"{mean_std(values['max_entrywise'])} | "
            f"{mean_std(values['mean_entrywise'])} | "
            f"{mean_std(values['relative_frobenius'])} |"
        )
    return "\n".join(lines)


def mean_std(values: list[float]) -> str:
    array = np.asarray(values, dtype=float)
    return f"{np.mean(array):.6g} +/- {np.std(array, ddof=0):.3g}"


if __name__ == "__main__":
    main()
