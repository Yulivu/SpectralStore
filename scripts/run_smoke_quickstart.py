"""Run a tiny end-to-end smoke experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spectralstore.compression import AsymmetricSpectralCompressor, SpectralCompressionConfig
from spectralstore.data_loader import make_low_rank_temporal_graph
from spectralstore.evaluation import relative_frobenius_error
from spectralstore.query_engine import QueryEngine


def main() -> None:
    config_path = Path("experiments/smoke/quickstart/configs/default.json")
    out_dir = Path("experiments/smoke/quickstart/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    snapshots = make_low_rank_temporal_graph(**config)
    store = AsymmetricSpectralCompressor(
        SpectralCompressionConfig(rank=config["rank"], random_seed=config["random_seed"])
    ).fit_transform(snapshots)
    engine = QueryEngine(store)

    result = {
        "relative_frobenius_error": relative_frobenius_error(snapshots, store),
        "example_link_prob": engine.link_prob(0, 1, 0),
        "example_top_neighbors": engine.top_neighbor(0, 0, 3),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
