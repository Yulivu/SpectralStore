from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from spectralstore.compression import (
    AsymmetricSpectralCompressor,
    CPALSCompressor,
    DirectSVDCompressor,
    FactorizedTemporalStore,
    PROTOTYPE_COMPRESSORS,
    RPCASVDCompressor,
    RobustAsymmetricSpectralCompressor,
    SplitAsymmetricUnfoldingCompressor,
    SparseUnfoldingAsymmetricCompressor,
    SpectralCompressionConfig,
    SymmetricSVDCompressor,
    TensorUnfoldingSVDCompressor,
    TuckerHOSVDCompressor,
    available_compressors,
    create_compressor,
    spectral_config_from_mapping,
)
from spectralstore.data_loader import make_low_rank_temporal_graph
from spectralstore.evaluation import relative_frobenius_error


def test_spectral_compressors_fit_temporal_snapshots() -> None:
    snapshots = make_low_rank_temporal_graph(
        num_nodes=18,
        num_steps=4,
        rank=3,
        random_seed=5,
    )
    config = SpectralCompressionConfig(rank=3, random_seed=5, num_splits=3)

    for compressor_cls in [
        AsymmetricSpectralCompressor,
        DirectSVDCompressor,
        SymmetricSVDCompressor,
        SparseUnfoldingAsymmetricCompressor,
        TensorUnfoldingSVDCompressor,
        CPALSCompressor,
        TuckerHOSVDCompressor,
    ]:
        store = compressor_cls(config).fit_transform(snapshots)
        assert store.num_nodes == 18
        assert store.num_steps == 4
        assert store.rank == 3
        assert relative_frobenius_error(snapshots, store) < 1.0


def test_tensorly_cp_and_tucker_report_formal_backends() -> None:
    snapshots = make_low_rank_temporal_graph(
        num_nodes=8,
        num_steps=3,
        rank=2,
        random_seed=8,
    )
    config = SpectralCompressionConfig(rank=2, random_seed=8, tensor_iterations=3)

    cp_store = CPALSCompressor(config).fit_transform(snapshots)
    tucker_store = TuckerHOSVDCompressor(config).fit_transform(snapshots)

    assert cp_store.threshold_diagnostics["tensorly_backend"] == "parafac"
    assert cp_store.threshold_diagnostics["tensor_method"] == "cp_als"
    assert tucker_store.threshold_diagnostics["tensorly_backend"] == "tucker"
    assert tucker_store.threshold_diagnostics["tensor_method"] == "tucker_als"
    assert "tucker_core_shape" in tucker_store.threshold_diagnostics


def test_compressor_registry_builds_expected_methods() -> None:
    config = SpectralCompressionConfig(rank=2, random_seed=3)
    snapshots = make_low_rank_temporal_graph(
        num_nodes=10,
        num_steps=3,
        rank=2,
        random_seed=3,
    )

    for name in [
        "spectralstore_asym",
        "spectralstore_unfolding_asym",
        "spectralstore_split_asym_unfolding",
        "spectralstore_robust",
        "tensor_unfolding_svd",
        "sym_svd",
        "direct_svd",
        "rpca_svd",
    ]:
        store = create_compressor(name, config).fit_transform(snapshots)
        assert store.num_nodes == 10
        assert store.num_steps == 3

    sym_from_registry = create_compressor("sym_svd", config).fit_transform(snapshots)
    sym_direct = SymmetricSVDCompressor(config).fit_transform(snapshots)
    direct_from_registry = create_compressor("direct_svd", config).fit_transform(snapshots)
    direct_direct = DirectSVDCompressor(config).fit_transform(snapshots)

    assert np.allclose(sym_from_registry.dense_snapshot(0), sym_direct.dense_snapshot(0))
    assert np.allclose(direct_from_registry.dense_snapshot(0), direct_direct.dense_snapshot(0))
    assert {
        "cp_als",
        "spectralstore_split_asym_unfolding",
        "spectralstore_unfolding_asym",
        "tucker_hosvd",
        "rpca_svd",
    } <= set(available_compressors())
    assert PROTOTYPE_COMPRESSORS == frozenset()


def test_rpca_svd_uses_matrix_pcp_backend() -> None:
    snapshots = make_low_rank_temporal_graph(
        num_nodes=8,
        num_steps=3,
        rank=2,
        random_seed=22,
    )
    store = RPCASVDCompressor(
        SpectralCompressionConfig(rank=2, random_seed=22, rpca_iterations=3)
    ).fit_transform(snapshots)

    assert store.num_nodes == 8
    assert store.num_steps == 3
    assert store.rank == 2
    assert store.threshold_diagnostics["tensor_method"] == "rpca_svd"
    assert store.threshold_diagnostics["rpca_backend"] == "matrix_pcp_admm"
    assert store.threshold_diagnostics["rpca_sparse_nnz"] >= 0


def test_split_asym_unfolding_uses_independent_left_and_right_factors() -> None:
    snapshots = make_low_rank_temporal_graph(
        num_nodes=10,
        num_steps=4,
        rank=2,
        random_seed=21,
    )
    store = SplitAsymmetricUnfoldingCompressor(
        SpectralCompressionConfig(rank=2, random_seed=21)
    ).fit_transform(snapshots)

    assert store.num_nodes == 10
    assert store.num_steps == 4
    assert store.rank == 2
    assert store.threshold_diagnostics["tensor_method"] == "split_asym_unfolding"
    assert not np.allclose(store.left, store.right)
    weights = store.lambdas * store.temporal[0]
    assert np.allclose(
        store.dense_snapshot(0, include_residual=False),
        (store.left * weights) @ store.right.T,
    )


def test_spectral_config_from_mapping_uses_known_fields_only() -> None:
    config = spectral_config_from_mapping(
        {
            "rank": 3,
            "random_seed": 11,
            "dataset_only_key": "ignored",
        },
        residual_threshold_mode="quantile",
    )

    assert config.rank == 3
    assert config.random_seed == 11
    assert config.residual_threshold_mode == "quantile"

    with pytest.raises(KeyError, match="unknown_field"):
        spectral_config_from_mapping({"rank": 3}, unknown_field=True)


def test_store_reports_storage_costs() -> None:
    snapshots = make_low_rank_temporal_graph(
        num_nodes=12,
        num_steps=3,
        rank=2,
        random_seed=6,
    )
    store = AsymmetricSpectralCompressor(SpectralCompressionConfig(rank=2)).fit_transform(snapshots)

    assert store.factor_bytes() > 0
    assert store.raw_dense_bytes() == 3 * 12 * 12 * 8
    assert store.compressed_bytes() == store.factor_bytes()
    assert 0.0 < store.compression_ratio() < 1.0


def test_factorized_store_npz_round_trip_preserves_queries() -> None:
    snapshots = make_low_rank_temporal_graph(
        num_nodes=10,
        num_steps=3,
        rank=2,
        random_seed=7,
    )
    store = RobustAsymmetricSpectralCompressor(
        SpectralCompressionConfig(
            rank=2,
            residual_threshold_mode="quantile",
            residual_quantile=0.9,
            random_seed=7,
        )
    ).fit_transform(snapshots)
    path = Path("tests") / "_factorized_store_roundtrip_test.npz"

    try:
        store.save_npz(path)
        loaded = FactorizedTemporalStore.load_npz(path)
    finally:
        path.unlink(missing_ok=True)

    assert loaded.rank == store.rank
    assert loaded.num_nodes == store.num_nodes
    assert loaded.num_steps == store.num_steps
    assert loaded.threshold_diagnostics == store.threshold_diagnostics
    assert loaded.entrywise_bound_scale == store.entrywise_bound_scale
    assert len(loaded.residuals) == len(store.residuals)
    assert loaded.compressed_bytes() == store.compressed_bytes()
    for t in range(store.num_steps):
        assert np.allclose(
            loaded.dense_snapshot(t, include_residual=True),
            store.dense_snapshot(t, include_residual=True),
        )
        assert loaded.link_score(0, 1, t) == store.link_score(0, 1, t)


def test_tensor_rank_energy_prunes_low_energy_components() -> None:
    pattern = np.array([1.0, 0.5, -0.25, 0.75, -0.5])
    snapshots = [
        sparse.csr_matrix(scale * np.outer(pattern, pattern))
        for scale in [1.0, 1.5, 0.75, 2.0]
    ]

    store = TensorUnfoldingSVDCompressor(
        SpectralCompressionConfig(
            rank=4,
            tensor_rank_energy=0.99,
            tensor_min_rank=1,
        )
    ).fit_transform(snapshots)

    assert store.rank == 1
    assert store.threshold_diagnostics["effective_rank"] == 1
    assert store.threshold_diagnostics["requested_rank"] == 4
    assert store.lambdas[0] > 0.0


def test_ard_like_rank_pruning_reduces_effective_svd_rank() -> None:
    pattern = np.array([1.0, -0.5, 0.25, 0.75, -1.25, 0.5])
    snapshots = [
        sparse.csr_matrix(scale * np.outer(pattern, pattern))
        for scale in [1.0, 0.9, 1.1, 1.2]
    ]
    unpruned = AsymmetricSpectralCompressor(
        SpectralCompressionConfig(rank=4, random_seed=10)
    ).fit_transform(snapshots)
    pruned = AsymmetricSpectralCompressor(
        SpectralCompressionConfig(
            rank=4,
            random_seed=10,
            rank_pruning_mode="ard_like",
            rank_pruning_threshold=0.05,
            rank_pruning_min_rank=1,
            rank_pruning_iterations=3,
        )
    ).fit_transform(snapshots)

    assert unpruned.rank == 4
    assert pruned.rank == 1
    assert pruned.factor_bytes() < unpruned.factor_bytes()
    assert pruned.threshold_diagnostics["rank_pruning_mode"] == "ard_like"
    assert pruned.threshold_diagnostics["effective_rank"] == 1
    assert pruned.threshold_diagnostics["rank_pruning_refit"] is True
    assert pruned.threshold_diagnostics["rank_pruning_history"][0]["input_rank"] == 4
    assert np.isfinite(pruned.link_score(0, 1, 0))


def test_robust_compressor_rank_pruning_recomputes_residuals() -> None:
    pattern = np.array([1.0, -0.5, 0.25, 0.75, -1.25, 0.5])
    dense_snapshots = [scale * np.outer(pattern, pattern) for scale in [1.0, 0.9, 1.1, 1.2]]
    dense_snapshots[0][0, 1] += 5.0
    snapshots = [sparse.csr_matrix(snapshot) for snapshot in dense_snapshots]

    store = RobustAsymmetricSpectralCompressor(
        SpectralCompressionConfig(
            rank=4,
            random_seed=11,
            rank_pruning_mode="ard_like",
            rank_pruning_threshold=0.05,
            rank_pruning_min_rank=1,
            rank_pruning_iterations=3,
            residual_threshold_mode="quantile",
            residual_quantile=0.95,
        )
    ).fit_transform(snapshots)

    assert store.rank == 1
    assert store.threshold_diagnostics["effective_rank"] == 1
    assert len(store.residuals) == len(snapshots)
    assert sum(residual.nnz for residual in store.residuals) > 0


def test_robust_residual_threshold_scale_controls_residual_storage() -> None:
    dense_snapshots = [
        np.array(
            [
                [0.0, 1.0, 0.2],
                [1.0, 0.0, 0.3],
                [0.2, 0.3, 0.0],
            ]
        ),
        np.array(
            [
                [0.0, 1.1, 0.25],
                [1.1, 0.0, 0.35],
                [0.25, 0.35, 0.0],
            ]
        ),
    ]
    snapshots = [sparse.csr_matrix(snapshot) for snapshot in dense_snapshots]
    common = {
        "rank": 1,
        "random_seed": 17,
        "residual_threshold_mode": "quantile",
        "residual_quantile": 0.5,
    }

    low_scale = RobustAsymmetricSpectralCompressor(
        SpectralCompressionConfig(**common, residual_threshold_scale=0.5)
    ).fit_transform(snapshots)
    high_scale = RobustAsymmetricSpectralCompressor(
        SpectralCompressionConfig(**common, residual_threshold_scale=3.0)
    ).fit_transform(snapshots)

    assert low_scale.threshold_diagnostics["residual_threshold_scale"] == 0.5
    assert high_scale.threshold_diagnostics["residual_threshold_scale"] == 3.0
    assert sum(residual.nnz for residual in low_scale.residuals) >= sum(
        residual.nnz for residual in high_scale.residuals
    )
