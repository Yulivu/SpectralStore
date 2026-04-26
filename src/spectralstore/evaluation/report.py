"""Unified evaluation reports for compressed temporal graph stores."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import sparse

from spectralstore.compression import FactorizedTemporalStore
from spectralstore.evaluation.metrics import (
    max_entrywise_error,
    mean_entrywise_error,
    observed_edge_mae,
    observed_edge_rmse,
    percentile_entrywise_error,
    relative_frobenius_error,
    relative_frobenius_error_against_dense,
)

TemporalEdge = tuple[int, int, int, float]
TemporalPair = tuple[int, int, int]


def observed_edges_from_snapshots(snapshots: list[sparse.spmatrix]) -> list[TemporalEdge]:
    edges: list[TemporalEdge] = []
    for t, snapshot in enumerate(snapshots):
        coo = snapshot.tocoo()
        edges.extend(
            (t, int(u), int(v), float(weight))
            for u, v, weight in zip(coo.row, coo.col, coo.data)
        )
    return edges


def split_observed_edges(
    snapshots: list[sparse.spmatrix],
    *,
    test_fraction: float,
    random_seed: int,
) -> tuple[list[sparse.csr_matrix], list[TemporalEdge]]:
    if not 0.0 <= test_fraction <= 1.0:
        raise ValueError("test_fraction must be between 0 and 1")

    rng = np.random.default_rng(random_seed)
    train_snapshots = []
    held_out: list[TemporalEdge] = []

    for t, snapshot in enumerate(snapshots):
        coo = snapshot.tocoo()
        if coo.nnz == 0:
            train_snapshots.append(snapshot.tocsr().copy())
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


def storage_report(
    store: FactorizedTemporalStore,
    snapshots: list[sparse.spmatrix],
) -> dict[str, float | int]:
    return {
        "factor_bytes": store.factor_bytes(),
        "residual_bytes": store.residual_bytes(),
        "metadata_bytes": store.metadata_bytes(),
        "compressed_bytes": store.compressed_bytes(),
        "raw_dense_bytes": store.raw_dense_bytes(),
        "raw_sparse_bytes": store.raw_sparse_csr_bytes(snapshots),
        "compressed_vs_raw_dense_ratio": store.compressed_vs_raw_dense_ratio(),
        "compressed_vs_raw_sparse_ratio": store.compressed_vs_raw_sparse_ratio(snapshots),
    }


def reconstruction_report(
    snapshots: list[sparse.spmatrix],
    store: FactorizedTemporalStore,
    *,
    include_residual: bool = False,
    expected_snapshots: list[np.ndarray] | None = None,
    entrywise_percentiles: Sequence[float] = (95.0, 99.0),
) -> dict[str, float]:
    if expected_snapshots is None:
        return {
            "relative_frobenius_error": relative_frobenius_error(
                snapshots,
                store,
                include_residual=include_residual,
            )
        }

    report = {
        "relative_frobenius_error": relative_frobenius_error_against_dense(
            expected_snapshots,
            store,
            include_residual=include_residual,
        ),
        "max_entrywise_error": max_entrywise_error(
            expected_snapshots,
            store,
            include_residual=include_residual,
        ),
        "mean_entrywise_error": mean_entrywise_error(
            expected_snapshots,
            store,
            include_residual=include_residual,
        ),
    }
    for percentile in entrywise_percentiles:
        report[f"p{int(percentile)}_entrywise_error"] = percentile_entrywise_error(
            expected_snapshots,
            store,
            percentile,
            include_residual=include_residual,
        )
    return report


def observed_edge_report(
    edges: list[TemporalEdge],
    store: FactorizedTemporalStore,
    *,
    include_residual: bool = False,
    prefix: str = "observed_edge",
) -> dict[str, float | int]:
    return {
        f"{prefix}_count": len(edges),
        f"{prefix}_rmse": observed_edge_rmse(
            edges,
            store,
            include_residual=include_residual,
        ),
        f"{prefix}_mae": observed_edge_mae(
            edges,
            store,
            include_residual=include_residual,
        ),
    }


def ranking_report(
    positive_edges: list[TemporalEdge],
    negative_edges: list[list[TemporalPair]],
    store: FactorizedTemporalStore,
    *,
    include_residual: bool = False,
    hits_at: Sequence[int] = (10, 50, 100),
) -> dict[str, float | int]:
    if len(positive_edges) != len(negative_edges):
        raise ValueError("positive_edges and negative_edges must have the same length")
    if not positive_edges:
        return {
            "ranking_queries": 0,
            "mrr": float("nan"),
            **{f"hits_at_{k}": float("nan") for k in hits_at},
        }

    ranks = []
    for positive, negatives in zip(positive_edges, negative_edges):
        t, u, v, _weight = positive
        positive_score = store.link_score(u, v, t, include_residual=include_residual)
        negative_scores = [
            store.link_score(neg_u, neg_v, neg_t, include_residual=include_residual)
            for neg_t, neg_u, neg_v in negatives
        ]
        rank = 1 + sum(score >= positive_score for score in negative_scores)
        ranks.append(rank)

    rank_array = np.asarray(ranks, dtype=float)
    report: dict[str, float | int] = {
        "ranking_queries": len(positive_edges),
        "mrr": float(np.mean(1.0 / rank_array)),
        "mean_rank": float(np.mean(rank_array)),
    }
    for k in hits_at:
        report[f"hits_at_{k}"] = float(np.mean(rank_array <= k))
    return report


def sample_temporal_negative_edges(
    positive_edges: list[TemporalEdge],
    snapshots: list[sparse.spmatrix],
    *,
    negatives_per_positive: int,
    random_seed: int,
) -> list[list[TemporalPair]]:
    rng = np.random.default_rng(random_seed)
    observed = [
        set(zip(snapshot.tocoo().row.astype(int), snapshot.tocoo().col.astype(int)))
        for snapshot in snapshots
    ]
    num_sources, num_targets = snapshots[0].shape if snapshots else (0, 0)
    negatives: list[list[TemporalPair]] = []
    for t, _u, _v, _weight in positive_edges:
        chosen: set[tuple[int, int]] = set()
        max_unique = max(num_sources * num_targets - len(observed[t]), 0)
        target_count = min(negatives_per_positive, max_unique)
        attempts = 0
        while len(chosen) < target_count and attempts < max(target_count * 50, 100):
            attempts += 1
            u = int(rng.integers(num_sources))
            v = int(rng.integers(num_targets))
            if (u, v) in observed[t] or (u, v) in chosen:
                continue
            chosen.add((u, v))
        negatives.append([(t, u, v) for u, v in sorted(chosen)])
    return negatives
