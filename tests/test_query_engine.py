import numpy as np
from scipy import sparse

from spectralstore import ExactMIPSIndex, FactorizedTemporalStore, QueryEngine


def test_link_prob_uses_factorization_and_residual() -> None:
    store = FactorizedTemporalStore(
        left=np.array([[1.0], [2.0]]),
        right=np.array([[3.0], [4.0]]),
        temporal=np.array([[1.0], [0.5]]),
        lambdas=np.array([2.0]),
        residuals=(
            sparse.csr_matrix([[0.0, 0.25], [0.0, 0.0]]),
            sparse.csr_matrix((2, 2)),
        ),
    )

    engine = QueryEngine(store)

    assert engine.link_prob(0, 1, 0) == 8.25
    assert engine.link_prob(0, 1, 0, include_residual=False) == 8.0


def test_link_prob_with_error_uses_threshold_diagnostics() -> None:
    store = FactorizedTemporalStore(
        left=np.array([[1.0], [2.0]]),
        right=np.array([[3.0], [4.0]]),
        temporal=np.array([[1.0], [0.5]]),
        lambdas=np.array([2.0]),
        residuals=(
            sparse.csr_matrix([[0.0, 0.25], [0.0, 0.0]]),
            sparse.csr_matrix((2, 2)),
        ),
        threshold_diagnostics={"estimated_threshold": 0.1},
    )

    engine = QueryEngine(store)
    corrected = engine.link_prob_with_error(0, 1, 0)
    uncorrected = engine.link_prob_with_error(0, 1, 0, include_residual=False)

    assert corrected.value == 8.25
    assert corrected.error_bound == 0.1
    assert uncorrected.value == 8.0
    assert uncorrected.error_bound == 0.35


def test_link_prob_optimizer_reads_residual_only_when_needed() -> None:
    store = FactorizedTemporalStore(
        left=np.array([[1.0], [2.0]]),
        right=np.array([[3.0], [4.0]]),
        temporal=np.array([[1.0]]),
        lambdas=np.array([2.0]),
        residuals=(sparse.csr_matrix([[0.0, 0.25], [0.0, 0.0]]),),
        threshold_diagnostics={"estimated_threshold": 0.1},
    )
    engine = QueryEngine(store)

    loose = engine.link_prob_optimized(0, 1, 0, error_tolerance=0.4)
    tight = engine.link_prob_optimized(0, 1, 0, error_tolerance=0.2)

    assert loose.value == 8.0
    assert loose.error_bound == 0.35
    assert loose.used_residual is False
    assert loose.satisfied_error_tolerance is True
    assert tight.value == 8.25
    assert tight.error_bound == 0.1
    assert tight.used_residual is True
    assert tight.satisfied_error_tolerance is True


def test_link_prob_with_error_returns_none_without_diagnostics() -> None:
    store = FactorizedTemporalStore(
        left=np.array([[1.0]]),
        right=np.array([[2.0]]),
        temporal=np.array([[1.0]]),
        lambdas=np.array([0.5]),
    )

    result = QueryEngine(store).link_prob_with_error(0, 0, 0)

    assert result.value == 1.0
    assert result.error_bound is None


def test_link_prob_with_error_uses_precomputed_theoretical_bound() -> None:
    store = FactorizedTemporalStore(
        left=np.eye(2),
        right=np.eye(2),
        temporal=np.ones((2, 2)),
        lambdas=np.ones(2),
        threshold_diagnostics={"estimated_threshold": 99.0},
    )
    snapshots = [
        sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0]]),
        sparse.csr_matrix([[1.0, 0.5], [0.0, 1.0]]),
    ]

    store.precompute_bound_params(snapshots)
    result = QueryEngine(store).link_prob_with_error(0, 1, 0)

    assert result.error_bound == store.entrywise_bound(0, 1)
    assert result.error_bound < 99.0


def test_link_prob_result_preserves_q1_dict_shape() -> None:
    store = FactorizedTemporalStore(
        left=np.array([[1.0]]),
        right=np.array([[2.0]]),
        temporal=np.array([[1.0]]),
        lambdas=np.array([0.5]),
    )

    result = QueryEngine(store, method="toy_method").link_prob_result(0, 0, 0)

    assert result == {
        "estimate": 1.0,
        "bound": None,
        "used_residual": True,
        "method": "toy_method",
    }


def test_query_engine_applies_bound_calibration_only_to_theoretical_bounds() -> None:
    store = FactorizedTemporalStore(
        left=np.eye(2),
        right=np.eye(2),
        temporal=np.ones((1, 2)),
        lambdas=np.ones(2),
    )
    snapshots = [sparse.csr_matrix([[1.0, 0.25], [0.0, 1.0]])]
    store.precompute_bound_params(snapshots)

    base = QueryEngine(store, bound_C=1.0).link_prob_with_error(0, 1, 0)
    calibrated = QueryEngine(store, bound_C=3.0).link_prob_with_error(0, 1, 0)

    assert base.error_bound is not None
    assert calibrated.error_bound == 3.0 * base.error_bound


def test_query_engine_reads_bound_calibration_from_config() -> None:
    store = FactorizedTemporalStore(
        left=np.eye(2),
        right=np.eye(2),
        temporal=np.ones((1, 2)),
        lambdas=np.ones(2),
    )
    snapshots = [sparse.csr_matrix([[1.0, 0.25], [0.0, 1.0]])]
    store.precompute_bound_params(snapshots)

    base = QueryEngine.from_config(store, {"query": {"bound_C": 1.0}}).link_prob_with_error(
        0,
        1,
        0,
    )
    calibrated = QueryEngine.from_config(
        store,
        {"query": {"bound_C": 2.5}},
    ).link_prob_with_error(0, 1, 0)

    assert base.error_bound is not None
    assert calibrated.error_bound == 2.5 * base.error_bound


def test_top_neighbor_orders_scores() -> None:
    store = FactorizedTemporalStore(
        left=np.array([[1.0], [0.1], [0.5]]),
        right=np.array([[1.0], [3.0], [2.0]]),
        temporal=np.array([[1.0]]),
        lambdas=np.array([1.0]),
    )

    engine = QueryEngine(store)

    assert engine.top_neighbor(0, 0, 2) == [(1, 3.0), (2, 2.0)]


def test_temporal_trend_is_inclusive() -> None:
    store = FactorizedTemporalStore(
        left=np.array([[1.0]]),
        right=np.array([[2.0]]),
        temporal=np.array([[1.0], [2.0], [3.0]]),
        lambdas=np.array([0.5]),
    )

    engine = QueryEngine(store)

    assert engine.temporal_trend(0, 0, 0, 2) == [1.0, 2.0, 3.0]


def test_temporal_trend_with_error_is_inclusive() -> None:
    store = FactorizedTemporalStore(
        left=np.array([[1.0]]),
        right=np.array([[2.0]]),
        temporal=np.array([[1.0], [2.0], [3.0]]),
        lambdas=np.array([0.5]),
        threshold_diagnostics={"estimated_threshold": 0.2},
    )

    trend = QueryEngine(store).temporal_trend_with_error(0, 0, 0, 2)

    assert [result.value for result in trend] == [1.0, 2.0, 3.0]
    assert [result.error_bound for result in trend] == [0.2, 0.2, 0.2]


def test_temporal_trend_optimizer_is_inclusive() -> None:
    store = FactorizedTemporalStore(
        left=np.array([[1.0]]),
        right=np.array([[2.0]]),
        temporal=np.array([[1.0], [2.0], [3.0]]),
        lambdas=np.array([0.5]),
        threshold_diagnostics={"estimated_threshold": 0.2},
    )

    trend = QueryEngine(store).temporal_trend_optimized(0, 0, 0, 2, error_tolerance=0.3)

    assert [result.value for result in trend] == [1.0, 2.0, 3.0]
    assert [result.used_residual for result in trend] == [False, False, False]
    assert [result.satisfied_error_tolerance for result in trend] == [True, True, True]


def test_community_clusters_weighted_embeddings() -> None:
    store = FactorizedTemporalStore(
        left=np.array([[1.0], [1.1], [-1.0], [-1.1]]),
        right=np.array([[1.0], [1.1], [-1.0], [-1.1]]),
        temporal=np.array([[1.0]]),
        lambdas=np.array([1.0]),
    )

    labels = QueryEngine(store).community(0, num_communities=2, random_seed=3)

    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_community_returns_one_label_per_node() -> None:
    store = FactorizedTemporalStore(
        left=np.array([[1.0], [1.1], [-1.0], [-1.1]]),
        right=np.array([[1.0], [1.1], [-1.0], [-1.1]]),
        temporal=np.array([[1.0]]),
        lambdas=np.array([1.0]),
    )

    labels = QueryEngine(store).community(0, num_communities=2, random_seed=3)

    assert len(labels) == store.num_nodes


def test_exact_index_top_neighbor_matches_dense_scan_without_residual() -> None:
    store = FactorizedTemporalStore(
        left=np.array([[1.0, 0.0], [0.2, 0.1], [0.1, 0.4]]),
        right=np.array([[0.5, 0.0], [3.0, 0.1], [1.0, 2.0]]),
        temporal=np.array([[1.0, 0.5]]),
        lambdas=np.array([2.0, 1.0]),
    )
    engine = QueryEngine(store, top_neighbor_index=ExactMIPSIndex.from_store(store))

    assert engine.top_neighbor(0, 0, 2, use_index=True) == engine.top_neighbor(0, 0, 2)


def test_exact_index_top_neighbor_reranks_residual_candidates() -> None:
    store = FactorizedTemporalStore(
        left=np.array([[1.0], [0.1], [0.5]]),
        right=np.array([[1.0], [3.0], [2.0]]),
        temporal=np.array([[1.0]]),
        lambdas=np.array([1.0]),
        residuals=(sparse.csr_matrix([[0.0, -2.5, 3.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),),
    )
    engine = QueryEngine(store)

    assert engine.top_neighbor(0, 0, 2, include_residual=True, use_index=True) == (
        engine.top_neighbor(0, 0, 2, include_residual=True)
    )
