"""Run a tiny end-to-end smoke experiment."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spectralstore.compression import AsymmetricSpectralCompressor, spectral_config_from_mapping
from spectralstore.data_loader import make_low_rank_temporal_graph
from spectralstore.evaluation import (
    load_experiment_config,
    relative_frobenius_error,
    set_reproducibility_seed,
    write_experiment_outputs,
)
from spectralstore.query_engine import QueryEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments/smoke/quickstart/configs/default.yaml",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/smoke/quickstart/results",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        help="OmegaConf dotlist override, e.g. --set num_nodes=12",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    out_dir = Path(args.out_dir)
    started_at = time.perf_counter()
    config = load_experiment_config(config_path, args.overrides)
    set_reproducibility_seed(config.get("random_seed"))

    snapshots = make_low_rank_temporal_graph(**config)
    store = AsymmetricSpectralCompressor(
        spectral_config_from_mapping(config)
    ).fit_transform(snapshots)
    engine = QueryEngine(store)

    result = {
        "relative_frobenius_error": relative_frobenius_error(snapshots, store),
        "example_link_prob": engine.link_prob(0, 1, 0),
        "example_top_neighbors": engine.top_neighbor(0, 0, 3),
    }
    summary = render_summary(result)
    write_experiment_outputs(
        out_dir=out_dir,
        metrics=result,
        summary=summary,
        config_path=config_path,
        config=config,
        started_at=started_at,
    )
    print(summary)


def render_summary(metrics: dict) -> str:
    return "\n".join(
        [
            "# Quickstart Smoke Results",
            "",
            f"- relative Frobenius error: {metrics['relative_frobenius_error']:.4f}",
            f"- example link probability: {metrics['example_link_prob']:.4f}",
            f"- example top neighbors: {metrics['example_top_neighbors']}",
            "",
        ]
    )


if __name__ == "__main__":
    main()

