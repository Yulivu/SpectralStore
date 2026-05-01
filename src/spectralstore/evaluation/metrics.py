"""Evaluation metrics for compressed temporal graph stores."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from spectralstore.compression import FactorizedTemporalStore
from spectralstore.query_engine import QueryEngine


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


def community_clustering_scores(
    true_labels: np.ndarray,
    predicted_labels: list[int] | np.ndarray,
) -> dict[str, float]:
    return {
        "community_nmi": float(
            normalized_mutual_info_score(true_labels, predicted_labels)
        ),
        "community_ari": float(adjusted_rand_score(true_labels, predicted_labels)),
    }


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


def percentile_entrywise_error(
    expected_snapshots: list[np.ndarray],
    store: FactorizedTemporalStore,
    percentile: float,
    *,
    include_residual: bool = False,
) -> float:
    errors = []
    for t, expected in enumerate(expected_snapshots):
        reconstruction = store.dense_snapshot(t, include_residual=include_residual)
        errors.append(np.abs(expected - reconstruction).ravel())
    return float(np.percentile(np.concatenate(errors), percentile))


def entrywise_bound_coverage(
    expected_snapshots: list[np.ndarray],
    store: FactorizedTemporalStore,
    *,
    include_residual: bool = True,
) -> float:
    """Return the fraction of entries covered by the store's empirical bound."""
    if (
        store.threshold_diagnostics is None
        and (store.bound_sigma_max is None or store.bound_mu is None)
    ):
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


def entrywise_bound_report(
    expected_snapshots: list[np.ndarray],
    store: FactorizedTemporalStore,
    *,
    include_residual: bool = False,
) -> dict[str, float]:
    """Report entrywise errors, bound coverage, and violation margins."""
    errors = []
    bounds = []
    for t, expected in enumerate(expected_snapshots):
        reconstruction = store.dense_snapshot(t, include_residual=include_residual)
        bound = store.entrywise_error_bound_matrix(t, include_residual=include_residual)
        errors.append(np.abs(expected - reconstruction).ravel())
        if bound is None:
            bounds.append(None)
        else:
            bounds.append(np.asarray(bound, dtype=float).ravel())

    error_values = np.concatenate(errors)
    if any(bound is None for bound in bounds):
        return {
            "max_entrywise_error": float(np.max(error_values)),
            "mean_entrywise_error": float(np.mean(error_values)),
            "mean_bound_value": float("nan"),
            "max_bound_value": float("nan"),
            "coverage": float("nan"),
            "violation_rate": float("nan"),
            "mean_violation_margin": float("nan"),
            "max_violation_margin": float("nan"),
            "mean_bound_over_error": float("nan"),
            "median_bound_over_error": float("nan"),
        }

    bound_values = np.concatenate([bound for bound in bounds if bound is not None])
    margins = error_values - bound_values
    violations = margins > 1e-12
    positive_margins = margins[violations]
    return {
        "max_entrywise_error": float(np.max(error_values)),
        "mean_entrywise_error": float(np.mean(error_values)),
        "mean_bound_value": float(np.mean(bound_values)),
        "max_bound_value": float(np.max(bound_values)),
        "coverage": float(np.mean(~violations)),
        "violation_rate": float(np.mean(violations)),
        "mean_violation_margin": (
            float(np.mean(positive_margins)) if positive_margins.size else 0.0
        ),
        "max_violation_margin": (
            float(np.max(positive_margins)) if positive_margins.size else 0.0
        ),
        "mean_bound_over_error": float(
            np.mean(bound_values / np.maximum(error_values, 1e-12))
        ),
        "median_bound_over_error": float(
            np.median(bound_values / np.maximum(error_values, 1e-12))
        ),
    }


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


def entrywise_bound_tightness(
    expected_snapshots: list[np.ndarray],
    store: FactorizedTemporalStore,
    *,
    include_residual: bool = True,
) -> dict[str, float]:
    mean_error = mean_entrywise_error(
        expected_snapshots,
        store,
        include_residual=include_residual,
    )
    max_error = max_entrywise_error(
        expected_snapshots,
        store,
        include_residual=include_residual,
    )
    mean_bound = mean_entrywise_error_bound(store, include_residual=include_residual)
    max_bound = max_entrywise_error_bound(store, include_residual=include_residual)
    return {
        "mean_bound_tightness": float(mean_bound / max(mean_error, 1e-12)),
        "max_bound_tightness": float(max_bound / max(max_error, 1e-12)),
    }


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
    if hasattr(store.residuals, "nnz"):
        return int(store.residuals.nnz)
    return int(sum(residual.nnz for residual in store.residuals))


def residual_sparsity(store: FactorizedTemporalStore) -> float:
    if not store.residuals:
        return 0.0
    total_entries = store.num_steps * store.num_nodes * store.right.shape[0]
    return float(residual_nnz(store) / max(total_entries, 1))


def anomaly_precision_recall(
    attack_edges: tuple[tuple[int, int, int], ...],
    store: FactorizedTemporalStore,
) -> tuple[float, float]:
    truth = set(attack_edges)
    predicted: set[tuple[int, int, int]] = set()
    for t in range(store.num_steps):
        coo = store.residual_snapshot(t).tocoo()
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


def anomaly_precision_recall_f1(
    attack_edges: tuple[tuple[int, int, int], ...],
    store: FactorizedTemporalStore,
) -> dict[str, float | int]:
    """Evaluate residual-stored anomaly detections against sparse corruption truth."""
    truth = set(attack_edges)
    predicted: set[tuple[int, int, int]] = set()
    for t in range(store.num_steps):
        coo = store.residual_snapshot(t).tocoo()
        predicted.update((t, int(row), int(col)) for row, col in zip(coo.row, coo.col))

    true_positives = len(predicted & truth)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(truth) if truth else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    return {
        "anomaly_truth_count": len(truth),
        "anomaly_predicted_count": len(predicted),
        "anomaly_true_positives": true_positives,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def subspace_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Projection-matrix distance between two column subspaces."""
    rank = min(first.shape[1], second.shape[1])
    if rank <= 0:
        return float("nan")
    first_q, _ = np.linalg.qr(first[:, :rank])
    second_q, _ = np.linalg.qr(second[:, :rank])
    first_projection = first_q @ first_q.T
    second_projection = second_q @ second_q.T
    return float(np.linalg.norm(first_projection - second_projection, ord="fro") / np.sqrt(2 * rank))


def reconstruction_difference(
    first: FactorizedTemporalStore,
    second: FactorizedTemporalStore,
    *,
    include_residual: bool = False,
) -> float:
    """Relative Frobenius difference between two reconstructed stores."""
    numerator = 0.0
    denominator = 0.0
    for t in range(first.num_steps):
        first_snapshot = first.dense_snapshot(t, include_residual=include_residual)
        second_snapshot = second.dense_snapshot(t, include_residual=include_residual)
        numerator += float(np.sum((first_snapshot - second_snapshot) ** 2))
        denominator += float(np.sum(first_snapshot**2))
    return float(np.sqrt(numerator / max(denominator, 1e-12)))


def split_asym_construction_report(
    snapshots: list[sparse.spmatrix],
    expected_snapshots: list[np.ndarray],
    *,
    random_seed: int,
) -> dict[str, float | int | bool]:
    """Replay the T1/T2 split construction and report specification checks."""
    if len(snapshots) < 2:
        raise ValueError("at least two snapshots are required")
    dense = [
        snapshot.toarray() if sparse.issparse(snapshot) else np.asarray(snapshot)
        for snapshot in snapshots
    ]
    rng = np.random.default_rng(random_seed)
    order = rng.permutation(len(dense))
    split = max(1, len(dense) // 2)
    first_indices = order[:split]
    second_indices = order[split:]
    if second_indices.size == 0:
        second_indices = first_indices

    first_mean = np.mean([dense[int(index)] for index in first_indices], axis=0)
    second_mean = np.mean([dense[int(index)] for index in second_indices], axis=0)
    stitched = (
        np.triu(first_mean, k=1)
        + np.tril(second_mean, k=-1)
        + np.diag(0.5 * (np.diag(first_mean) + np.diag(second_mean)))
    )

    upper = np.triu(np.ones_like(stitched, dtype=bool), k=1)
    lower = np.tril(np.ones_like(stitched, dtype=bool), k=-1)
    diag = np.eye(stitched.shape[0], dtype=bool)
    upper_error = np.max(np.abs(stitched[upper] - first_mean[upper]))
    lower_error = np.max(np.abs(stitched[lower] - second_mean[lower]))
    diag_target = 0.5 * (np.diag(first_mean) + np.diag(second_mean))
    diag_error = np.max(np.abs(np.diag(stitched) - diag_target))

    first_expected = np.mean([expected_snapshots[int(index)] for index in first_indices], axis=0)
    second_expected = np.mean([expected_snapshots[int(index)] for index in second_indices], axis=0)
    first_noise = (first_mean - first_expected)[~diag].ravel()
    second_noise = (second_mean - second_expected)[~diag].ravel()
    if np.std(first_noise) <= 1e-12 or np.std(second_noise) <= 1e-12:
        noise_correlation = float("nan")
    else:
        noise_correlation = float(np.corrcoef(first_noise, second_noise)[0, 1])

    overlap = set(int(index) for index in first_indices) & set(int(index) for index in second_indices)
    return {
        "split_first_size": int(first_indices.size),
        "split_second_size": int(second_indices.size),
        "split_overlap_count": int(len(overlap)),
        "split_is_disjoint": len(overlap) == 0,
        "upper_triangle_source_error": float(upper_error),
        "lower_triangle_source_error": float(lower_error),
        "diag_consistency_error": float(diag_error),
        "noise_correlation_T1_T2": noise_correlation,
    }


def q5_anomaly_detection_scores(
    attack_edges: tuple[tuple[int, int, int], ...],
    engine: QueryEngine,
    *,
    threshold: float,
) -> dict[str, float | int]:
    """Evaluate Q5 anomaly detection against injected temporal attack edges."""
    truth = set(attack_edges)
    predicted: set[tuple[int, int, int]] = set()
    for t in range(engine.store.num_steps):
        predicted.update(
            (t, int(u), int(v))
            for u, v, _value in engine.anomaly_detect(t, threshold)
        )

    true_positives = len(predicted & truth)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(truth) if truth else 0.0
    return {
        "injected_anomaly_edges": len(truth),
        "q5_detected_edges": len(predicted),
        "q5_true_positives": true_positives,
        "q5_precision": float(precision),
        "q5_recall": float(recall),
    }
