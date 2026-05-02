from spectralstore.compression import SpectralCompressionConfig, create_compressor
from spectralstore.data_loader import make_low_rank_temporal_graph
from spectralstore.evaluation import relative_frobenius_error_against_dense


def test_ard_refit_preserves_low_rank_reconstruction_scale() -> None:
    snapshots = make_low_rank_temporal_graph(
        num_nodes=40,
        num_steps=5,
        rank=3,
        noise_scale=0.0,
        random_seed=1,
    )
    expected = [snapshot.toarray() for snapshot in snapshots]
    fixed = create_compressor(
        "spectralstore_thinking",
        SpectralCompressionConfig(
            rank=3,
            rank_selection_mode="fixed",
            robust_iterations=1,
            random_seed=1,
        ),
    ).fit_transform(snapshots)
    ard = create_compressor(
        "spectralstore_thinking",
        SpectralCompressionConfig(
            rank=6,
            rank_selection_mode="ard",
            ard_max_rank=6,
            ard_prior_alpha=1e-2,
            ard_prior_beta=1.0,
            ard_min_effective_ratio=0.05,
            ard_max_iterations=50,
            robust_iterations=1,
            random_seed=1,
        ),
    ).fit_transform(snapshots)

    fixed_error = relative_frobenius_error_against_dense(
        expected,
        fixed,
        include_residual=False,
    )
    ard_error = relative_frobenius_error_against_dense(
        expected,
        ard,
        include_residual=False,
    )

    assert ard.rank == 3
    assert ard_error <= fixed_error * 1.05
