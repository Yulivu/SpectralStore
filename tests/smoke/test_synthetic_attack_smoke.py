from spectralstore.compression import RobustAsymmetricSpectralCompressor, SpectralCompressionConfig
from spectralstore.data_loader import make_synthetic_attack
from spectralstore.evaluation import anomaly_precision_recall, relative_frobenius_error_against_dense
from spectralstore.query_engine import QueryEngine


def test_synthetic_attack_smoke_runs_end_to_end() -> None:
    dataset = make_synthetic_attack(
        num_nodes=24,
        num_steps=3,
        num_communities=3,
        attack_fraction=0.03,
        random_seed=8,
    )
    store = RobustAsymmetricSpectralCompressor(
        SpectralCompressionConfig(
            rank=3,
            num_splits=2,
            residual_quantile=0.95,
            robust_iterations=1,
            random_seed=8,
        )
    ).fit_transform(dataset.snapshots)
    engine = QueryEngine(store)
    precision, recall = anomaly_precision_recall(dataset.attack_edges, store)

    assert relative_frobenius_error_against_dense(dataset.expected_snapshots, store) < 2.0
    assert isinstance(engine.link_prob(0, 1, 0), float)
    assert len(engine.anomaly_detect(0, threshold=0.0)) > 0
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
