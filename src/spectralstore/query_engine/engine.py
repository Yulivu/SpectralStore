"""Query engine for factorized temporal graph stores.

The engine exposes scalar query methods and batch helpers for:
- Q1 `LINK_PROB`
- Q2 `TOP_NEIGHBOR`
- Q3 `COMMUNITY`
- Q4 `TEMPORAL_TREND`
- Q5 `ANOMALY_DETECT`
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans

from spectralstore.compression import FactorizedTemporalStore
from spectralstore.index import ExactMIPSIndex, RandomProjectionANNMIPSIndex

QUERY_RESULT_SCHEMA_VERSION = 1
QUERY_RESULT_FIELDS = (
    "estimate",
    "bound",
    "used_residual",
    "method",
    "satisfied_error_tolerance",
)
LINK_QUERY_RESULT_SCHEMA: dict[str, object] = {
    "version": QUERY_RESULT_SCHEMA_VERSION,
    "fields": QUERY_RESULT_FIELDS,
}


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
            "satisfied_error_tolerance": self.satisfied_error_tolerance,
        }


@dataclass(frozen=True)
class TopNeighborQueryPlan:
    """Plan metadata and output for optimized Q2 execution."""

    neighbors: list[tuple[int, float]]
    path: str
    estimated_error_bound: float | None
    satisfied_error_tolerance: bool | None


class QueryEngine:
    """Execute temporal-graph queries over a `FactorizedTemporalStore`."""

    def __init__(
        self,
        store: FactorizedTemporalStore,
        *,
        top_neighbor_index: ExactMIPSIndex | None = None,
        ann_top_neighbor_index: RandomProjectionANNMIPSIndex | None = None,
        raw_snapshots: list[sparse.spmatrix] | None = None,
        bound_C: float = 1.0,
        method: str = "unknown",
    ) -> None:
        self.store = store
        self.top_neighbor_index = top_neighbor_index
        self.ann_top_neighbor_index = ann_top_neighbor_index
        self.raw_snapshots = raw_snapshots
        self.bound_C = float(bound_C)
        self.method = method
        self._trend_cache: dict[tuple[int, int, int, int, bool], list[float]] = {}
        self._community_cache: dict[tuple[int, int, int], list[int]] = {}
        self._cache_stats = {
            "trend_hits": 0,
            "trend_misses": 0,
            "community_hits": 0,
            "community_misses": 0,
        }

    @classmethod
    def from_config(
        cls,
        store: FactorizedTemporalStore,
        config: dict,
        *,
        top_neighbor_index: ExactMIPSIndex | None = None,
        ann_top_neighbor_index: RandomProjectionANNMIPSIndex | None = None,
        raw_snapshots: list[sparse.spmatrix] | None = None,
        method: str = "unknown",
    ) -> "QueryEngine":
        query_config = config.get("query", config)
        return cls(
            store,
            top_neighbor_index=top_neighbor_index,
            ann_top_neighbor_index=ann_top_neighbor_index,
            raw_snapshots=raw_snapshots,
            bound_C=float(query_config.get("bound_C", 1.0)),
            method=method,
        )

    def build_exact_top_neighbor_index(self) -> ExactMIPSIndex:
        """Build and cache the exact MIPS index used by indexed Q2 paths."""
        self.top_neighbor_index = ExactMIPSIndex.from_store(self.store)
        return self.top_neighbor_index

    def build_ann_top_neighbor_index(
        self,
        *,
        projection_dim: int | None = None,
        candidate_multiplier: int = 4,
        random_seed: int = 0,
    ) -> RandomProjectionANNMIPSIndex:
        """Build and cache a random-projection ANN MIPS index for Q2."""
        self.ann_top_neighbor_index = RandomProjectionANNMIPSIndex.from_store(
            self.store,
            projection_dim=projection_dim,
            candidate_multiplier=candidate_multiplier,
            random_seed=random_seed,
        )
        return self.ann_top_neighbor_index

    def link_prob(self, u: int, v: int, t: int, *, include_residual: bool = True) -> float:
        """Q1 scalar estimate for one `(u, v, t)` entry."""
        return self.store.link_score(u, v, t, include_residual=include_residual)

    def link_prob_batch(
        self,
        queries: list[tuple[int, int, int]],
        *,
        include_residual: bool = True,
    ) -> list[float]:
        """Q1 batch estimate for multiple `(u, v, t)` entries."""
        return [
            self.link_prob(u, v, t, include_residual=include_residual)
            for u, v, t in queries
        ]

    def link_prob_with_error(
        self,
        u: int,
        v: int,
        t: int,
        *,
        include_residual: bool = True,
    ) -> BoundedQueryResult:
        """Q1 scalar estimate with bound metadata."""
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
        """Return Q1 in the fixed dict schema."""
        return self.link_prob_with_error(
            u,
            v,
            t,
            include_residual=include_residual,
        ).as_dict()

    def temporal_trend_result(
        self,
        u: int,
        v: int,
        t1: int,
        t2: int,
        *,
        include_residual: bool = True,
    ) -> list[dict[str, float | bool | str | None]]:
        """Return Q4 trend in the same dict schema as Q1."""
        return [
            result.as_dict()
            for result in self.temporal_trend_with_error(
                u,
                v,
                t1,
                t2,
                include_residual=include_residual,
            )
        ]

    def temporal_trend_optimized_result(
        self,
        u: int,
        v: int,
        t1: int,
        t2: int,
        *,
        error_tolerance: float,
    ) -> list[dict[str, float | bool | str | None]]:
        """Return optimized Q4 trend with fixed dict schema."""
        return [
            result.as_dict()
            for result in self.temporal_trend_optimized(
                u,
                v,
                t1,
                t2,
                error_tolerance=error_tolerance,
            )
        ]

    def link_prob_optimized(
        self,
        u: int,
        v: int,
        t: int,
        *,
        error_tolerance: float,
    ) -> BoundedQueryResult:
        """Q1 optimized path that decides residual usage by error tolerance."""
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
        index_mode: str = "exact",
    ) -> list[tuple[int, float]]:
        """Q2 top-k neighbors for one node and one time step."""
        if k <= 0:
            return []
        if t < 0 or t >= self.store.num_steps:
            raise IndexError("time index out of range")
        if u < 0 or u >= self.store.num_nodes:
            raise IndexError("source node index out of range")

        index = self._resolve_index(index_mode) if use_index else None
        if use_index and index is not None and (not include_residual or not self.store.residuals):
            query = self.store.left[u] * self.store.lambdas * self.store.temporal[t]
            return index.search(query, k, exclude=u if exclude_self else None)
        if use_index and index is not None and include_residual and self.store.residuals:
            return self._top_neighbor_index_with_residual_rerank(
                u,
                t,
                k,
                exclude_self=exclude_self,
                index_mode=index_mode,
            )

        scores = self.store.dense_snapshot(t, include_residual=include_residual)[u].copy()
        if exclude_self and 0 <= u < scores.shape[0]:
            scores[u] = -np.inf

        limit = min(k, scores.shape[0])
        candidate_indices = np.argpartition(-scores, limit - 1)[:limit]
        ordered = candidate_indices[np.argsort(-scores[candidate_indices])]
        return [(int(idx), float(scores[idx])) for idx in ordered]

    def top_neighbor_batch(
        self,
        queries: list[tuple[int, int, int]],
        *,
        include_residual: bool = False,
        exclude_self: bool = True,
        use_index: bool = False,
        index_mode: str = "exact",
    ) -> list[list[tuple[int, float]]]:
        """Q2 batch interface for `(u, t, k)` queries."""
        return [
            self.top_neighbor(
                u,
                t,
                k,
                include_residual=include_residual,
                exclude_self=exclude_self,
                use_index=use_index,
                index_mode=index_mode,
            )
            for u, t, k in queries
        ]

    def _top_neighbor_index_with_residual_rerank(
        self,
        u: int,
        t: int,
        k: int,
        *,
        exclude_self: bool,
        index_mode: str,
    ) -> list[tuple[int, float]]:
        index = self._resolve_index(index_mode)
        if index is None:
            return self.top_neighbor(
                u,
                t,
                k,
                include_residual=True,
                exclude_self=exclude_self,
                use_index=False,
            )
        residual_row = self.store.residual_row(t, u)
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
        """Q4 scalar trend values across `[t1, t2]`."""
        if t1 > t2:
            raise ValueError("t1 must be less than or equal to t2")
        return [
            self.store.link_score(u, v, t, include_residual=include_residual)
            for t in range(t1, t2 + 1)
        ]

    def temporal_trend_cached(
        self,
        u: int,
        v: int,
        t1: int,
        t2: int,
        *,
        include_residual: bool = True,
    ) -> list[float]:
        """Q4 trend path with in-memory cache."""
        key = (u, v, t1, t2, include_residual)
        cached = self._trend_cache.get(key)
        if cached is not None:
            self._cache_stats["trend_hits"] += 1
            return list(cached)
        self._cache_stats["trend_misses"] += 1
        values = self.temporal_trend(u, v, t1, t2, include_residual=include_residual)
        self._trend_cache[key] = list(values)
        return values

    def temporal_trend_with_error(
        self,
        u: int,
        v: int,
        t1: int,
        t2: int,
        *,
        include_residual: bool = True,
    ) -> list[BoundedQueryResult]:
        """Q4 trend values with bound metadata across `[t1, t2]`."""
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
        """Q4 optimized trend path with per-step tolerance decisions."""
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
        """Q3 community assignment labels for one time step."""
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

    def community_cached(
        self,
        t: int,
        *,
        num_communities: int | None = None,
        random_seed: int = 0,
    ) -> list[int]:
        """Q3 community path with in-memory cache."""
        clusters = num_communities or self.store.rank
        key = (t, int(clusters), int(random_seed))
        cached = self._community_cache.get(key)
        if cached is not None:
            self._cache_stats["community_hits"] += 1
            return list(cached)
        self._cache_stats["community_misses"] += 1
        labels = self.community(t, num_communities=num_communities, random_seed=random_seed)
        self._community_cache[key] = list(labels)
        return labels

    def community_batch(
        self,
        times: list[int],
        *,
        num_communities: int | None = None,
        random_seed: int = 0,
    ) -> list[list[int]]:
        """Q3 batch interface for a list of time steps."""
        return [
            self.community_cached(
                t,
                num_communities=num_communities,
                random_seed=random_seed,
            )
            for t in times
        ]

    def anomaly_detect(self, t: int, threshold: float) -> list[tuple[int, int, float]]:
        """Q5 anomalies for one time step and threshold."""
        if not self.store.residuals:
            return []
        matrix = self.store.residual_snapshot(t).tocoo()
        keep = np.abs(matrix.data) > threshold
        return [
            (int(row), int(col), float(value))
            for row, col, value in zip(matrix.row[keep], matrix.col[keep], matrix.data[keep])
        ]

    def anomaly_detect_batch(
        self,
        queries: list[tuple[int, float]],
    ) -> list[list[tuple[int, int, float]]]:
        """Q5 batch interface for `(t, threshold)` requests."""
        return [self.anomaly_detect(t, threshold) for t, threshold in queries]

    def top_neighbor_optimized(
        self,
        u: int,
        t: int,
        k: int,
        *,
        error_tolerance: float,
        exclude_self: bool = True,
        prefer_index: bool = True,
        allow_raw_fallback: bool = True,
    ) -> TopNeighborQueryPlan:
        """Q2 optimized path over factor-only/factor+residual/indexed/raw."""
        residual_row = self.store.residual_row(t, u)
        residual_bound = (
            float(np.max(np.abs(residual_row.data)))
            if residual_row.nnz > 0
            else 0.0
        )
        use_residual = residual_bound > error_tolerance and bool(self.store.residuals)
        satisfied = residual_bound <= error_tolerance or use_residual

        if allow_raw_fallback and self.raw_snapshots is not None and self.store.num_nodes <= 512:
            neighbors = self._raw_top_neighbor(u, t, k, exclude_self=exclude_self)
            return TopNeighborQueryPlan(
                neighbors=neighbors,
                path="raw_fallback",
                estimated_error_bound=0.0,
                satisfied_error_tolerance=True,
            )

        if prefer_index and k <= 64:
            if self.top_neighbor_index is not None:
                neighbors = self.top_neighbor(
                    u,
                    t,
                    k,
                    include_residual=use_residual,
                    exclude_self=exclude_self,
                    use_index=True,
                    index_mode="exact",
                )
                return TopNeighborQueryPlan(
                    neighbors=neighbors,
                    path="indexed_exact_residual" if use_residual else "indexed_exact_factor_only",
                    estimated_error_bound=residual_bound if not use_residual else 0.0,
                    satisfied_error_tolerance=satisfied,
                )
            if self.ann_top_neighbor_index is not None:
                neighbors = self.top_neighbor(
                    u,
                    t,
                    k,
                    include_residual=use_residual,
                    exclude_self=exclude_self,
                    use_index=True,
                    index_mode="ann",
                )
                return TopNeighborQueryPlan(
                    neighbors=neighbors,
                    path="indexed_ann_residual" if use_residual else "indexed_ann_factor_only",
                    estimated_error_bound=residual_bound if not use_residual else 0.0,
                    satisfied_error_tolerance=satisfied,
                )

        neighbors = self.top_neighbor(
            u,
            t,
            k,
            include_residual=use_residual,
            exclude_self=exclude_self,
            use_index=False,
        )
        return TopNeighborQueryPlan(
            neighbors=neighbors,
            path="factor_residual_scan" if use_residual else "factor_only_scan",
            estimated_error_bound=residual_bound if not use_residual else 0.0,
            satisfied_error_tolerance=satisfied,
        )

    def cache_stats(self) -> dict[str, int]:
        """Return trend/community cache hit-miss counters."""
        return {key: int(value) for key, value in self._cache_stats.items()}

    def clear_caches(self) -> None:
        """Clear query caches and reset counters."""
        self._trend_cache.clear()
        self._community_cache.clear()
        for key in self._cache_stats:
            self._cache_stats[key] = 0

    def _resolve_index(self, index_mode: str):
        if index_mode == "exact":
            return self.top_neighbor_index or self.build_exact_top_neighbor_index()
        if index_mode == "ann":
            return self.ann_top_neighbor_index or self.build_ann_top_neighbor_index()
        raise ValueError(f"unsupported index_mode: {index_mode}")

    def _raw_top_neighbor(
        self,
        u: int,
        t: int,
        k: int,
        *,
        exclude_self: bool,
    ) -> list[tuple[int, float]]:
        if self.raw_snapshots is None:
            raise ValueError("raw_snapshots are required for raw fallback")
        row = self.raw_snapshots[t].tocsr().getrow(u).toarray().reshape(-1)
        if exclude_self and 0 <= u < row.shape[0]:
            row[u] = -np.inf
        limit = min(k, row.shape[0])
        candidate_indices = np.argpartition(-row, limit - 1)[:limit]
        ordered = candidate_indices[np.argsort(-row[candidate_indices])]
        return [(int(idx), float(row[idx])) for idx in ordered]
