import numpy as np
from scipy import sparse

from spectralstore.compression import AsymmetricSpectralCompressor, SpectralCompressionConfig
from spectralstore.evaluation import (
    observed_edges_from_snapshots,
    observed_edge_report,
    ranking_report,
    reconstruction_report,
    sample_temporal_negative_edges,
    split_observed_edges,
    storage_report,
    validate_dense_stack_memory_budget,
)


def test_unified_reports_include_reconstruction_storage_and_observed_edges() -> None:
    snapshots = [
        sparse.csr_matrix([[0.0, 2.0], [0.0, 0.0]]),
        sparse.csr_matrix([[0.0, 1.5], [0.5, 0.0]]),
    ]
    store = AsymmetricSpectralCompressor(
        SpectralCompressionConfig(rank=1, random_seed=4)
    ).fit_transform(snapshots)
    observed_edges = [(0, 0, 1, 2.0), (1, 1, 0, 0.5)]

    reconstruction = reconstruction_report(snapshots, store)
    observed = observed_edge_report(observed_edges, store, prefix="held_out")
    storage = storage_report(store, snapshots)

    assert "relative_frobenius_error" in reconstruction
    assert observed["held_out_count"] == 2
    assert "held_out_rmse" in observed
    assert storage["compressed_bytes"] > 0
    assert storage["raw_sparse_bytes"] > 0
    assert storage["compressed_vs_raw_sparse_ratio"] > 0.0


def test_reconstruction_report_includes_entrywise_metrics_against_expected_dense() -> None:
    snapshots = [sparse.csr_matrix(np.eye(3)), sparse.csr_matrix(np.eye(3))]
    store = AsymmetricSpectralCompressor(
        SpectralCompressionConfig(rank=1, random_seed=5)
    ).fit_transform(snapshots)

    report = reconstruction_report(
        snapshots,
        store,
        expected_snapshots=[np.eye(3), np.eye(3)],
        entrywise_percentiles=(95,),
    )

    assert set(report) == {
        "relative_frobenius_error",
        "max_entrywise_error",
        "mean_entrywise_error",
        "p95_entrywise_error",
    }


def test_ranking_report_computes_mrr_and_hits() -> None:
    snapshot = sparse.csr_matrix(
        [[0.0, 3.0, 1.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]]
    )
    snapshots = [snapshot, snapshot]
    store = AsymmetricSpectralCompressor(
        SpectralCompressionConfig(rank=2, random_seed=6)
    ).fit_transform(snapshots)
    positives = [(0, 0, 1, 3.0)]
    negatives = [[(0, 0, 2), (0, 1, 2)]]

    report = ranking_report(positives, negatives, store, hits_at=(1, 2))

    assert report["ranking_queries"] == 1
    assert 0.0 < report["mrr"] <= 1.0
    assert 0.0 <= report["hits_at_1"] <= 1.0
    assert 0.0 <= report["hits_at_2"] <= 1.0


def test_sample_temporal_negative_edges_avoids_observed_edges() -> None:
    snapshots = [sparse.csr_matrix([[0.0, 1.0], [0.0, 0.0]])]
    positives = [(0, 0, 1, 1.0)]

    negatives = sample_temporal_negative_edges(
        positives,
        snapshots,
        negatives_per_positive=2,
        random_seed=7,
    )

    assert all((u, v) != (0, 1) for _t, u, v in negatives[0])
    assert len(negatives[0]) == 2


def test_observed_edge_helpers_split_and_collect_sparse_snapshots() -> None:
    snapshots = [
        sparse.csr_matrix([[0.0, 1.0], [2.0, 0.0]]),
        sparse.csr_matrix([[0.0, 0.0], [3.0, 0.0]]),
    ]

    all_edges = observed_edges_from_snapshots(snapshots)
    train_snapshots, held_out = split_observed_edges(
        snapshots,
        test_fraction=1.0,
        random_seed=8,
    )

    assert all_edges == [(0, 0, 1, 1.0), (0, 1, 0, 2.0), (1, 1, 0, 3.0)]
    assert held_out == all_edges
    assert all(snapshot.nnz == 0 for snapshot in train_snapshots)


def test_ogbl_collab_dense_memory_budget_rejects_oversized_runs() -> None:
    validate_dense_stack_memory_budget(100, 10, limit_gb=1.0, label="ogbl-collab run")

    try:
        validate_dense_stack_memory_budget(
            5000,
            34,
            limit_gb=4.0,
            label="ogbl-collab run",
        )
    except MemoryError as exc:
        assert "dense input stack" in str(exc)
    else:
        raise AssertionError("expected oversized ogbl-collab run to fail")
