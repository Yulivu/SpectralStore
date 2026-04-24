from spectralstore.compression import (
    AsymmetricSpectralCompressor,
    DirectSVDCompressor,
    SpectralCompressionConfig,
    SymmetricSVDCompressor,
    TensorUnfoldingSVDCompressor,
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
        TensorUnfoldingSVDCompressor,
    ]:
        store = compressor_cls(config).fit_transform(snapshots)
        assert store.num_nodes == 18
        assert store.num_steps == 4
        assert store.rank == 3
        assert relative_frobenius_error(snapshots, store) < 1.0
