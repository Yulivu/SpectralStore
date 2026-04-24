from spectralstore.compression import AsymmetricSpectralCompressor, SpectralCompressionConfig
from spectralstore.data_loader import make_low_rank_temporal_graph
from spectralstore.evaluation import relative_frobenius_error
from spectralstore.query_engine import QueryEngine


def test_quickstart_smoke_runs_end_to_end() -> None:
    snapshots = make_low_rank_temporal_graph(
        num_nodes=16,
        num_steps=3,
        rank=2,
        random_seed=3,
    )
    store = AsymmetricSpectralCompressor(
        SpectralCompressionConfig(rank=2, random_seed=3)
    ).fit_transform(snapshots)
    engine = QueryEngine(store)

    assert store.rank == 2
    assert relative_frobenius_error(snapshots, store) < 0.15
    assert isinstance(engine.link_prob(0, 1, 0), float)
    assert len(engine.top_neighbor(0, 0, 3)) == 3
