"""Time one Exp1 setting method-by-method."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spectralstore.compression import SpectralCompressionConfig, create_compressor  # noqa: E402
from spectralstore.data_loader import make_temporal_sbm  # noqa: E402
from spectralstore.evaluation import (  # noqa: E402
    max_entrywise_error,
    mean_entrywise_error,
    relative_frobenius_error_against_dense,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-nodes", type=int, default=500)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=401)
    parser.add_argument("--tensor-iterations", type=int, default=5)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["spectralstore_asym", "sym_svd", "direct_svd", "cp_als"],
    )
    args = parser.parse_args()

    print(
        "Timing Exp1 single setting: "
        f"n={args.num_nodes}, T={args.num_steps}, r={args.rank}, repeat=1"
    )
    dataset_started = time.perf_counter()
    dataset = make_temporal_sbm(
        num_nodes=args.num_nodes,
        num_steps=args.num_steps,
        num_communities=args.rank,
        p_in=0.3,
        p_out=0.05,
        temporal_jitter=0.08,
        directed=True,
        random_seed=args.random_seed,
    )
    dataset_seconds = time.perf_counter() - dataset_started
    print(f"dataset_generation_seconds: {dataset_seconds:.3f}")
    print(
        "| method | seconds | max entrywise | mean entrywise | rel. Frobenius |"
    )
    print("| --- | ---: | ---: | ---: | ---: |")

    for method in args.methods:
        config = SpectralCompressionConfig(
            rank=args.rank,
            random_seed=args.random_seed,
            tensor_iterations=args.tensor_iterations,
        )
        started = time.perf_counter()
        store = create_compressor(method, config).fit_transform(dataset.snapshots)
        seconds = time.perf_counter() - started
        max_error = max_entrywise_error(dataset.expected_snapshots, store)
        mean_error = mean_entrywise_error(dataset.expected_snapshots, store)
        frob = relative_frobenius_error_against_dense(dataset.expected_snapshots, store)
        print(
            "| "
            f"{method} | {seconds:.3f} | {max_error:.6g} | "
            f"{mean_error:.6g} | {frob:.6g} |"
        )


if __name__ == "__main__":
    main()
