"""Evaluation metrics for compressed temporal graph stores."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from spectralstore.compression import FactorizedTemporalStore


def relative_frobenius_error(
    snapshots: list[sparse.spmatrix],
    store: FactorizedTemporalStore,
    *,
    include_residual: bool = False,
) -> float:
    numerator = 0.0
    denominator = 0.0
    for t, snapshot in enumerate(snapshots):
        dense = snapshot.toarray()
        reconstruction = store.dense_snapshot(t, include_residual=include_residual)
        numerator += float(np.sum((dense - reconstruction) ** 2))
        denominator += float(np.sum(dense**2))
    return float(np.sqrt(numerator / max(denominator, 1e-12)))


def observed_edge_rmse(
    held_out_edges: list[tuple[int, int, int, float]],
    store: FactorizedTemporalStore,
    *,
    include_residual: bool = False,
) -> float:
    if not held_out_edges:
        return float("nan")
    errors = [
        (store.link_score(u, v, t, include_residual=include_residual) - weight) ** 2
        for t, u, v, weight in held_out_edges
    ]
    return float(np.sqrt(np.mean(errors)))


def observed_edge_mae(
    held_out_edges: list[tuple[int, int, int, float]],
    store: FactorizedTemporalStore,
    *,
    include_residual: bool = False,
) -> float:
    if not held_out_edges:
        return float("nan")
    errors = [
        abs(store.link_score(u, v, t, include_residual=include_residual) - weight)
        for t, u, v, weight in held_out_edges
    ]
    return float(np.mean(errors))


def max_entrywise_error(
    expected_snapshots: list[np.ndarray],
    store: FactorizedTemporalStore,
    *,
    include_residual: bool = False,
) -> float:
    errors = []
    for t, expected in enumerate(expected_snapshots):
        reconstruction = store.dense_snapshot(t, include_residual=include_residual)
        errors.append(float(np.max(np.abs(expected - reconstruction))))
    return float(np.max(errors))


def mean_entrywise_error(
    expected_snapshots: list[np.ndarray],
    store: FactorizedTemporalStore,
    *,
    include_residual: bool = False,
) -> float:
    total = 0.0
    count = 0
    for t, expected in enumerate(expected_snapshots):
        reconstruction = store.dense_snapshot(t, include_residual=include_residual)
        total += float(np.sum(np.abs(expected - reconstruction)))
        count += expected.size
    return float(total / max(count, 1))


def relative_frobenius_error_against_dense(
    expected_snapshots: list[np.ndarray],
    store: FactorizedTemporalStore,
    *,
    include_residual: bool = False,
) -> float:
    numerator = 0.0
    denominator = 0.0
    for t, expected in enumerate(expected_snapshots):
        reconstruction = store.dense_snapshot(t, include_residual=include_residual)
        numerator += float(np.sum((expected - reconstruction) ** 2))
        denominator += float(np.sum(expected**2))
    return float(np.sqrt(numerator / max(denominator, 1e-12)))
