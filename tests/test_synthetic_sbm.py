from spectralstore.compression import AsymmetricSpectralCompressor, SpectralCompressionConfig
from spectralstore.data_loader import make_synthetic_spiked, make_temporal_sbm
from spectralstore.evaluation import community_clustering_scores
from spectralstore.query_engine import QueryEngine


def test_temporal_sbm_returns_snapshots_and_ground_truth() -> None:
    dataset = make_temporal_sbm(
        num_nodes=30,
        num_steps=4,
        num_communities=3,
        random_seed=9,
    )

    assert len(dataset.snapshots) == 4
    assert dataset.name == "synthetic_sbm"
    assert dataset.num_nodes == 30
    assert dataset.num_steps == 4
    assert len(dataset.expected_snapshots) == 4
    assert dataset.snapshots[0].shape == (30, 30)
    assert dataset.expected_snapshots[0].shape == (30, 30)
    assert dataset.communities.shape == (30,)


def test_synthetic_spiked_returns_gaussian_snapshots_and_ground_truth() -> None:
    dataset = make_synthetic_spiked(
        num_nodes=24,
        num_steps=3,
        rank=3,
        snr=2.0,
        random_seed=10,
    )

    assert len(dataset.snapshots) == 3
    assert dataset.name == "synthetic_spiked"
    assert dataset.num_nodes == 24
    assert dataset.num_steps == 3
    assert len(dataset.expected_snapshots) == 3
    assert dataset.snapshots[0].shape == (24, 24)
    assert dataset.expected_snapshots[0].shape == (24, 24)
    assert dataset.communities.shape == (24,)


def test_synthetic_sbm_community_metrics_are_available() -> None:
    dataset = make_temporal_sbm(
        num_nodes=24,
        num_steps=4,
        num_communities=2,
        p_in=0.8,
        p_out=0.02,
        random_seed=12,
    )
    store = AsymmetricSpectralCompressor(
        SpectralCompressionConfig(rank=2, random_seed=12)
    ).fit_transform(dataset.snapshots)
    labels = QueryEngine(store).community(0, num_communities=2, random_seed=12)

    scores = community_clustering_scores(dataset.communities, labels)

    assert set(scores) == {"community_nmi", "community_ari"}
    assert 0.0 <= scores["community_nmi"] <= 1.0
    assert -1.0 <= scores["community_ari"] <= 1.0


def test_store_reports_sparse_storage_ratios() -> None:
    dataset = make_temporal_sbm(
        num_nodes=12,
        num_steps=3,
        num_communities=2,
        p_in=0.5,
        p_out=0.01,
        random_seed=13,
    )
    store = AsymmetricSpectralCompressor(
        SpectralCompressionConfig(rank=2, random_seed=13)
    ).fit_transform(dataset.snapshots)

    assert store.raw_sparse_csr_bytes(dataset.snapshots) > 0
    assert store.compressed_vs_raw_dense_ratio() > 0.0
    assert store.compressed_vs_raw_sparse_ratio(dataset.snapshots) > 0.0
