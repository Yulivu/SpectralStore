"""Query engine for factorized temporal graph stores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

from spectralstore.compression import FactorizedTemporalStore
from spectralstore.index import ExactMIPSIndex


@dataclass(frozen=True)
class BoundedQueryResult:
    """Approximate query result with an empirical absolute error bound."""

    value: float
    error_bound: float | None
    used_residual: bool | None = None
    satisfied_error_tolerance: bool | None = None
    method: str = "unknown"

    @property
    def estimate(self) -> float:
        return self.value

    @property
    def bound(self) -> float | None:
        return self.error_bound

    def as_dict(self) -> dict[str, float | bool | str | None]:
        return {
            "estimate": self.value,
            "bound": self.error_bound,
            "used_residual": self.used_residual,
            "method": self.method,
        }


class QueryEngine:
    """Execute first-milestone queries over a factorized store."""

    def __init__(
        self,
        store: FactorizedTemporalStore,
        *,
        top_neighbor_index: ExactMIPSIndex | None = None,
        bound_C: float = 1.0,
        method: str = "unknown",
    ) -> None:
        self.store = store
        self.top_neighbor_index = top_neighbor_index
        self.bound_C = float(bound_C)
        self.method = method

    @classmethod
    def from_config(
        cls,
        store: FactorizedTemporalStore,
        config: dict,
        *,
        top_neighbor_index: ExactMIPSIndex | None = None,
        method: str = "unknown",
    ) -> "QueryEngine":
        query_config = config.get("query", config)
        return cls(
            store,
            top_neighbor_index=top_neighbor_index,
            bound_C=float(query_config.get("bound_C", 1.0)),
            method=method,
        )

    def build_exact_top_neighbor_index(self) -> ExactMIPSIndex:
        self.top_neighbor_index = ExactMIPSIndex.from_store(self.store)
        return self.top_neighbor_index

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
            error_bound=self._calibrated_entrywise_bound(
                u,
                v,
                t,
                include_residual=include_residual,
            ),
            used_residual=include_residual,
            method=self.method,
        )

    def link_prob_result(
        self,
        u: int,
        v: int,
        t: int,
        *,
        include_residual: bool = True,
    ) -> dict[str, float | bool | str | None]:
        """Return Q1 in a dict shape while preserving the legacy float API."""
        return self.link_prob_with_error(
            u,
            v,
            t,
            include_residual=include_residual,
        ).as_dict()

    def link_prob_optimized(
        self,
        u: int,
        v: int,
        t: int,
        *,
        error_tolerance: float,
    ) -> BoundedQueryResult:
        low_rank_bound = self._calibrated_entrywise_bound(
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
                method=self.method,
            )

        use_residual = bool(self.store.residuals)
        corrected_bound = self._calibrated_entrywise_bound(
            u,
            v,
            t,
            include_residual=use_residual,
        )
        return BoundedQueryResult(
            value=self.store.link_score(u, v, t, include_residual=use_residual),
            error_bound=corrected_bound,
            used_residual=use_residual,
            satisfied_error_tolerance=(
                corrected_bound is not None and corrected_bound <= error_tolerance
            ),
            method=self.method,
        )

    def _calibrated_entrywise_bound(
        self,
        u: int,
        v: int,
        t: int,
        *,
        include_residual: bool,
    ) -> float | None:
        bound = self.store.entrywise_error_bound(
            u,
            v,
            t,
            include_residual=include_residual,
        )
        if bound is None:
            return None
        if self.store.bound_sigma_max is None or self.store.bound_mu is None:
            return bound
        return float(self.bound_C * bound)

    def top_neighbor(
        self,
        u: int,
        t: int,
        k: int,
        *,
        include_residual: bool = False,
        exclude_self: bool = True,
        use_index: bool = False,
    ) -> list[tuple[int, float]]:
        if k <= 0:
            return []
        if t < 0 or t >= self.store.num_steps:
            raise IndexError("time index out of range")
        if u < 0 or u >= self.store.num_nodes:
            raise IndexError("source node index out of range")

        if use_index and (not include_residual or not self.store.residuals):
            index = self.top_neighbor_index or self.build_exact_top_neighbor_index()
            query = self.store.left[u] * self.store.lambdas * self.store.temporal[t]
            return index.search(query, k, exclude=u if exclude_self else None)
        if use_index and include_residual and self.store.residuals:
            return self._top_neighbor_index_with_residual_rerank(
                u,
                t,
                k,
                exclude_self=exclude_self,
            )

        scores = self.store.dense_snapshot(t, include_residual=include_residual)[u].copy()
        if exclude_self and 0 <= u < scores.shape[0]:
            scores[u] = -np.inf

        limit = min(k, scores.shape[0])
        candidate_indices = np.argpartition(-scores, limit - 1)[:limit]
        ordered = candidate_indices[np.argsort(-scores[candidate_indices])]
        return [(int(idx), float(scores[idx])) for idx in ordered]

    def _top_neighbor_index_with_residual_rerank(
        self,
        u: int,
        t: int,
        k: int,
        *,
        exclude_self: bool,
    ) -> list[tuple[int, float]]:
        index = self.top_neighbor_index or self.build_exact_top_neighbor_index()
        residual_row = self.store.residuals[t].getrow(u)
        query = self.store.left[u] * self.store.lambdas * self.store.temporal[t]
        candidate_budget = min(
            self.store.right.shape[0],
            k + residual_row.nnz + (1 if exclude_self else 0),
        )
        candidates = {
            idx
            for idx, _score in index.search(
                query,
                candidate_budget,
                exclude=u if exclude_self else None,
            )
        }
        candidates.update(int(idx) for idx in residual_row.indices)
        if exclude_self:
            candidates.discard(u)

        scored = [
            (
                idx,
                self.store.link_score(
                    u,
                    idx,
                    t,
                    include_residual=True,
                ),
            )
            for idx in candidates
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [(int(idx), float(score)) for idx, score in scored[:k]]

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

    def community(
        self,
        t: int,
        *,
        num_communities: int | None = None,
        random_seed: int = 0,
    ) -> list[int]:
        if t < 0 or t >= self.store.num_steps:
            raise IndexError("time index out of range")
        clusters = num_communities or self.store.rank
        if clusters <= 0:
            raise ValueError("num_communities must be positive")
        clusters = min(clusters, self.store.num_nodes)

        weights = np.sqrt(np.abs(self.store.lambdas * self.store.temporal[t]))
        embedding = np.hstack([self.store.left * weights, self.store.right * weights])
        labels = KMeans(
            n_clusters=clusters,
            random_state=random_seed,
            n_init=10,
        ).fit_predict(embedding)
        return [int(label) for label in labels]

    def anomaly_detect(self, t: int, threshold: float) -> list[tuple[int, int, float]]:
        if not self.store.residuals:
            return []
        matrix = self.store.residuals[t].tocoo()
        keep = np.abs(matrix.data) > threshold
        return [
            (int(row), int(col), float(value))
            for row, col, value in zip(matrix.row[keep], matrix.col[keep], matrix.data[keep])
        ]
