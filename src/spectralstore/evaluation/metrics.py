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


def entrywise_bound_coverage(
    expected_snapshots: list[np.ndarray],
    store: FactorizedTemporalStore,
    *,
    include_residual: bool = True,
) -> float:
    """Return the fraction of entries covered by the store's empirical bound."""
    if store.threshold_diagnostics is None:
        return float("nan")
    covered = 0
    total = 0
    for t, expected in enumerate(expected_snapshots):
        reconstruction = store.dense_snapshot(t, include_residual=include_residual)
        bound = store.entrywise_error_bound_matrix(t, include_residual=include_residual)
        if bound is None:
            return float("nan")
        covered += int(np.sum(np.abs(expected - reconstruction) <= bound + 1e-12))
        total += expected.size
    return float(covered / max(total, 1))


def mean_entrywise_error_bound(
    store: FactorizedTemporalStore,
    *,
    include_residual: bool = True,
) -> float:
    """Return the mean empirical entrywise bound across all stored snapshots."""
    values = []
    for t in range(store.num_steps):
        bound = store.entrywise_error_bound_matrix(t, include_residual=include_residual)
        if bound is None:
            return float("nan")
        values.append(float(np.mean(bound)))
    return float(np.mean(values))


def max_entrywise_error_bound(
    store: FactorizedTemporalStore,
    *,
    include_residual: bool = True,
) -> float:
    """Return the max empirical entrywise bound across all stored snapshots."""
    values = []
    for t in range(store.num_steps):
        bound = store.entrywise_error_bound_matrix(t, include_residual=include_residual)
        if bound is None:
            return float("nan")
        values.append(float(np.max(bound)))
    return float(np.max(values))


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


def residual_nnz(store: FactorizedTemporalStore) -> int:
    return int(sum(residual.nnz for residual in store.residuals))


def residual_sparsity(store: FactorizedTemporalStore) -> float:
    if not store.residuals:
        return 0.0
    total_entries = sum(residual.shape[0] * residual.shape[1] for residual in store.residuals)
    return float(residual_nnz(store) / max(total_entries, 1))


def anomaly_precision_recall(
    attack_edges: tuple[tuple[int, int, int], ...],
    store: FactorizedTemporalStore,
) -> tuple[float, float]:
    truth = set(attack_edges)
    predicted: set[tuple[int, int, int]] = set()
    for t, residual in enumerate(store.residuals):
        coo = residual.tocoo()
        predicted.update((t, int(row), int(col)) for row, col in zip(coo.row, coo.col))

    if not predicted:
        precision = 0.0
    else:
        precision = len(predicted & truth) / len(predicted)

    if not truth:
        recall = 0.0
    else:
        recall = len(predicted & truth) / len(truth)
    return float(precision), float(recall)
