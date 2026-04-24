"""Query engine for factorized temporal graph stores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spectralstore.compression import FactorizedTemporalStore


@dataclass(frozen=True)
class BoundedQueryResult:
    """Approximate query result with an empirical absolute error bound."""

    value: float
    error_bound: float | None
    used_residual: bool | None = None
    satisfied_error_tolerance: bool | None = None


class QueryEngine:
    """Execute first-milestone queries over a factorized store."""

    def __init__(self, store: FactorizedTemporalStore) -> None:
        self.store = store

    def link_prob(self, u: int, v: int, t: int, *, include_residual: bool = True) -> float:
        return self.store.link_score(u, v, t, include_residual=include_residual)

    def link_prob_with_error(
        self,
        u: int,
        v: int,
        t: int,
        *,
        include_residual: bool = True,
    ) -> BoundedQueryResult:
        return BoundedQueryResult(
            value=self.store.link_score(u, v, t, include_residual=include_residual),
            error_bound=self.store.entrywise_error_bound(
                u,
                v,
                t,
                include_residual=include_residual,
            ),
            used_residual=include_residual,
        )

    def link_prob_optimized(
        self,
        u: int,
        v: int,
        t: int,
        *,
        error_tolerance: float,
    ) -> BoundedQueryResult:
        low_rank_bound = self.store.entrywise_error_bound(
            u,
            v,
            t,
            include_residual=False,
        )
        if low_rank_bound is not None and low_rank_bound <= error_tolerance:
            return BoundedQueryResult(
                value=self.store.link_score(u, v, t, include_residual=False),
                error_bound=low_rank_bound,
                used_residual=False,
                satisfied_error_tolerance=True,
            )

        corrected_bound = self.store.entrywise_error_bound(
            u,
            v,
            t,
            include_residual=True,
        )
        return BoundedQueryResult(
            value=self.store.link_score(u, v, t, include_residual=True),
            error_bound=corrected_bound,
            used_residual=True,
            satisfied_error_tolerance=(
                corrected_bound is not None and corrected_bound <= error_tolerance
            ),
        )

    def top_neighbor(
        self,
        u: int,
        t: int,
        k: int,
        *,
        include_residual: bool = False,
        exclude_self: bool = True,
    ) -> list[tuple[int, float]]:
        if k <= 0:
            return []

        scores = self.store.dense_snapshot(t, include_residual=include_residual)[u].copy()
        if exclude_self and 0 <= u < scores.shape[0]:
            scores[u] = -np.inf

        limit = min(k, scores.shape[0])
        candidate_indices = np.argpartition(-scores, limit - 1)[:limit]
        ordered = candidate_indices[np.argsort(-scores[candidate_indices])]
        return [(int(idx), float(scores[idx])) for idx in ordered]

    def temporal_trend(
        self,
        u: int,
        v: int,
        t1: int,
        t2: int,
        *,
        include_residual: bool = True,
    ) -> list[float]:
        if t1 > t2:
            raise ValueError("t1 must be less than or equal to t2")
        return [
            self.store.link_score(u, v, t, include_residual=include_residual)
            for t in range(t1, t2 + 1)
        ]

    def temporal_trend_with_error(
        self,
        u: int,
        v: int,
        t1: int,
        t2: int,
        *,
        include_residual: bool = True,
    ) -> list[BoundedQueryResult]:
        if t1 > t2:
            raise ValueError("t1 must be less than or equal to t2")
        return [
            self.link_prob_with_error(u, v, t, include_residual=include_residual)
            for t in range(t1, t2 + 1)
        ]

    def temporal_trend_optimized(
        self,
        u: int,
        v: int,
        t1: int,
        t2: int,
        *,
        error_tolerance: float,
    ) -> list[BoundedQueryResult]:
        if t1 > t2:
            raise ValueError("t1 must be less than or equal to t2")
        return [
            self.link_prob_optimized(u, v, t, error_tolerance=error_tolerance)
            for t in range(t1, t2 + 1)
        ]

    def anomaly_detect(self, t: int, threshold: float) -> list[tuple[int, int, float]]:
        if not self.store.residuals:
            return []
        matrix = self.store.residuals[t].tocoo()
        keep = np.abs(matrix.data) > threshold
        return [
            (int(row), int(col), float(value))
            for row, col, value in zip(matrix.row[keep], matrix.col[keep], matrix.data[keep])
        ]
