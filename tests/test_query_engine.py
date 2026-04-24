import numpy as np
from scipy import sparse

from spectralstore import FactorizedTemporalStore, QueryEngine


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
