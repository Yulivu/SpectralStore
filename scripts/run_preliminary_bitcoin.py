"""Run a first Bitcoin-OTC comparison between SpectralStore and SVD baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spectralstore.compression import (
    AsymmetricSpectralCompressor,
    DirectSVDCompressor,
    SpectralCompressionConfig,
    SymmetricSVDCompressor,
)
from spectralstore.data_loader import load_bitcoin_otc
from spectralstore.evaluation import observed_edge_mae, observed_edge_rmse, relative_frobenius_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments/preliminary/bitcoin_otc/configs/default.json",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/preliminary/bitcoin_otc/results",
    )
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_bitcoin_otc(config["raw_path"], max_nodes=config["max_nodes"])
    train_snapshots, held_out = split_observed_edges(
        dataset.snapshots,
        test_fraction=config["test_fraction"],
        random_seed=config["random_seed"],
    )

    compressor_config = SpectralCompressionConfig(
        rank=config["rank"],
        random_seed=config["random_seed"],
    )
    methods = {
        "spectralstore_asym": AsymmetricSpectralCompressor(compressor_config),
        "baseline_sym_svd": SymmetricSVDCompressor(compressor_config),
        "baseline_direct_svd": DirectSVDCompressor(compressor_config),
    }

    metrics = {
        "dataset": {
            "name": dataset.name,
            "num_nodes": len(dataset.node_ids),
            "num_steps": len(dataset.snapshots),
            "held_out_edges": len(held_out),
        },
        "methods": {},
    }

    for name, compressor in methods.items():
        store = compressor.fit_transform(train_snapshots)
        metrics["methods"][name] = {
            "rank": store.rank,
            "relative_frobenius_error_train": relative_frobenius_error(train_snapshots, store),
            "held_out_observed_edge_rmse": observed_edge_rmse(held_out, store),
            "held_out_observed_edge_mae": observed_edge_mae(held_out, store),
        }

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "summary.md").write_text(render_summary(metrics), encoding="utf-8")
    print(render_summary(metrics))


def split_observed_edges(
    snapshots: list[sparse.csr_matrix],
    *,
    test_fraction: float,
    random_seed: int,
) -> tuple[list[sparse.csr_matrix], list[tuple[int, int, int, float]]]:
    rng = np.random.default_rng(random_seed)
    train_snapshots = []
    held_out: list[tuple[int, int, int, float]] = []

    for t, snapshot in enumerate(snapshots):
        coo = snapshot.tocoo()
        if coo.nnz == 0:
            train_snapshots.append(snapshot.copy())
            continue

        is_test = rng.random(coo.nnz) < test_fraction
        held_out.extend(
            (t, int(u), int(v), float(weight))
            for u, v, weight in zip(coo.row[is_test], coo.col[is_test], coo.data[is_test])
        )

        train = sparse.coo_matrix(
            (coo.data[~is_test], (coo.row[~is_test], coo.col[~is_test])),
            shape=snapshot.shape,
        ).tocsr()
        train_snapshots.append(train)

    return train_snapshots, held_out


def render_summary(metrics: dict) -> str:
    dataset = metrics["dataset"]
    lines = [
        "# Bitcoin-OTC Preliminary Results",
        "",
        f"- nodes: {dataset['num_nodes']}",
        f"- temporal snapshots: {dataset['num_steps']}",
        f"- held-out observed edges: {dataset['held_out_edges']}",
        "",
        "| method | train relative Frobenius | held-out RMSE | held-out MAE |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, values in metrics["methods"].items():
        lines.append(
            "| "
            f"{name} | "
            f"{values['relative_frobenius_error_train']:.4f} | "
            f"{values['held_out_observed_edge_rmse']:.4f} | "
            f"{values['held_out_observed_edge_mae']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
