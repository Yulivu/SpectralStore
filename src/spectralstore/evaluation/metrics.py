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
