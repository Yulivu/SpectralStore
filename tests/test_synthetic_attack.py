from spectralstore.compression import RobustAsymmetricSpectralCompressor, SpectralCompressionConfig
from spectralstore.data_loader import make_synthetic_attack
from spectralstore.evaluation import (
    entrywise_bound_coverage,
    relative_frobenius_error_against_dense,
    residual_nnz,
)


def test_synthetic_attack_returns_metadata_and_expected_shapes() -> None:
    dataset = make_synthetic_attack(
        num_nodes=30,
        num_steps=3,
        num_communities=3,
        attack_kind="sparse_outlier_edges",
        attack_fraction=0.02,
        random_seed=12,
    )

    assert len(dataset.snapshots) == 3
    assert len(dataset.expected_snapshots) == 3
    assert dataset.snapshots[0].shape == (30, 30)
    assert dataset.expected_snapshots[0].shape == (30, 30)
    assert dataset.attack_kind == "sparse_outlier_edges"
    assert len(dataset.attack_edges) > 0


def test_synthetic_attack_severity_increases_attack_edges() -> None:
    small = make_synthetic_attack(
        num_nodes=30,
        num_steps=2,
        attack_fraction=0.01,
        random_seed=4,
    )
    large = make_synthetic_attack(
        num_nodes=30,
        num_steps=2,
        attack_fraction=0.05,
        random_seed=4,
    )

    assert len(large.attack_edges) > len(small.attack_edges)


def test_synthetic_attack_allows_zero_attack_fraction() -> None:
    dataset = make_synthetic_attack(
        num_nodes=30,
        num_steps=2,
        attack_fraction=0.0,
        random_seed=4,
    )

    assert len(dataset.attack_edges) == 0


def test_robust_compressor_produces_residuals_per_snapshot() -> None:
    dataset = make_synthetic_attack(
        num_nodes=32,
        num_steps=4,
        num_communities=4,
        attack_fraction=0.04,
        random_seed=21,
    )
    compressor = RobustAsymmetricSpectralCompressor(
        SpectralCompressionConfig(
            rank=4,
            num_splits=3,
            residual_threshold_mode="quantile",
            residual_quantile=0.95,
            robust_iterations=2,
            random_seed=21,
        )
    )

    store = compressor.fit_transform(dataset.snapshots)

    assert len(store.residuals) == len(dataset.snapshots)
    assert sum(residual.nnz for residual in store.residuals) > 0


def test_mad_threshold_is_less_aggressive_without_attack_than_quantile() -> None:
    dataset = make_synthetic_attack(
        num_nodes=32,
        num_steps=4,
        num_communities=4,
        attack_fraction=0.0,
        random_seed=22,
    )
    mad_store = RobustAsymmetricSpectralCompressor(
        SpectralCompressionConfig(
            rank=4,
            num_splits=3,
            residual_threshold_mode="mad",
            residual_mad_multiplier=45.0,
            robust_iterations=1,
            random_seed=22,
        )
    ).fit_transform(dataset.snapshots)
    quantile_store = RobustAsymmetricSpectralCompressor(
        SpectralCompressionConfig(
            rank=4,
            num_splits=3,
            residual_threshold_mode="quantile",
            residual_quantile=0.95,
            robust_iterations=1,
            random_seed=22,
        )
    ).fit_transform(dataset.snapshots)

    assert residual_nnz(mad_store) < residual_nnz(quantile_store)


def test_threshold_diagnostics_are_recorded() -> None:
    dataset = make_synthetic_attack(
        num_nodes=32,
        num_steps=4,
        num_communities=4,
        attack_fraction=0.02,
        random_seed=24,
    )
    store = RobustAsymmetricSpectralCompressor(
        SpectralCompressionConfig(
            rank=4,
            num_splits=3,
            residual_threshold_mode="hybrid",
            residual_quantile=0.985,
            robust_iterations=2,
            random_seed=24,
        )
    ).fit_transform(dataset.snapshots)

    assert store.threshold_diagnostics is not None
    for field in (
        "estimated_threshold",
        "noise_scale",
        "residual_sparsity",
        "residual_nnz",
    ):
        assert field in store.threshold_diagnostics
    assert store.threshold_diagnostics["residual_nnz"] == residual_nnz(store)


def test_entrywise_bound_covers_fitted_snapshots() -> None:
    dataset = make_synthetic_attack(
        num_nodes=32,
        num_steps=4,
        num_communities=4,
        attack_kind="sparse_outlier_edges",
        attack_fraction=0.02,
        random_seed=26,
    )
    store = RobustAsymmetricSpectralCompressor(
        SpectralCompressionConfig(
            rank=4,
            num_splits=3,
            residual_threshold_mode="hybrid",
            residual_quantile=0.985,
            robust_iterations=2,
            random_seed=26,
        )
    ).fit_transform(dataset.snapshots)
    observed = [snapshot.toarray() for snapshot in dataset.snapshots]

    assert entrywise_bound_coverage(observed, store, include_residual=True) == 1.0
    assert entrywise_bound_coverage(observed, store, include_residual=False) == 1.0


def test_hybrid_threshold_does_not_over_extract_without_attack() -> None:
    dataset = make_synthetic_attack(
        num_nodes=40,
        num_steps=4,
        num_communities=4,
        attack_fraction=0.0,
        random_seed=25,
    )
    hybrid_store = RobustAsymmetricSpectralCompressor(
        SpectralCompressionConfig(
            rank=4,
            num_splits=3,
            residual_threshold_mode="hybrid",
            residual_quantile=0.985,
            residual_mad_multiplier=45.0,
            robust_iterations=1,
            random_seed=25,
        )
    ).fit_transform(dataset.snapshots)
    quantile_store = RobustAsymmetricSpectralCompressor(
        SpectralCompressionConfig(
            rank=4,
            num_splits=3,
            residual_threshold_mode="quantile",
            residual_quantile=0.985,
            robust_iterations=1,
            random_seed=25,
        )
    ).fit_transform(dataset.snapshots)

    assert residual_nnz(hybrid_store) < residual_nnz(quantile_store)
    assert hybrid_store.threshold_diagnostics is not None
    assert hybrid_store.threshold_diagnostics["hybrid_cap_active"] is False


def test_hybrid_threshold_beats_mad_under_high_sparse_outlier_attack() -> None:
    dataset = make_synthetic_attack(
        num_nodes=60,
        num_steps=4,
        num_communities=4,
        attack_kind="sparse_outlier_edges",
        attack_fraction=0.05,
        outlier_weight=3.0,
        directed=True,
        random_seed=33,
    )
    mad_store = RobustAsymmetricSpectralCompressor(
        SpectralCompressionConfig(
            rank=4,
            num_splits=4,
            residual_threshold_mode="mad",
            residual_quantile=0.985,
            residual_mad_multiplier=45.0,
            robust_iterations=2,
            random_seed=33,
        )
    ).fit_transform(dataset.snapshots)
    hybrid_store = RobustAsymmetricSpectralCompressor(
        SpectralCompressionConfig(
            rank=4,
            num_splits=4,
            residual_threshold_mode="hybrid",
            residual_quantile=0.985,
            residual_mad_multiplier=45.0,
            robust_iterations=2,
            random_seed=33,
        )
    ).fit_transform(dataset.snapshots)

    mad_error = relative_frobenius_error_against_dense(dataset.expected_snapshots, mad_store)
    hybrid_error = relative_frobenius_error_against_dense(dataset.expected_snapshots, hybrid_store)

    assert hybrid_error < mad_error
    assert hybrid_store.threshold_diagnostics is not None
    assert hybrid_store.threshold_diagnostics["hybrid_cap_active"] is True
