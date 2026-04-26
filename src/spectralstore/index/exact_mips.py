"""Exact factor-space maximum inner product search indexes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spectralstore.compression import FactorizedTemporalStore


@dataclass(frozen=True)
class ExactMIPSIndex:
    """Exact MIPS over target embeddings.

    The index is intentionally simple: it stores dense right-side embeddings and
    evaluates all inner products in factor space. This keeps Q2 on the index
    path without introducing ANN/PQ behavior yet.
    """

    right_embeddings: np.ndarray

    @classmethod
    def from_right_embeddings(cls, right_embeddings: np.ndarray) -> "ExactMIPSIndex":
        embeddings = np.asarray(right_embeddings, dtype=float)
        if embeddings.ndim != 2:
            raise ValueError("right_embeddings must be a matrix")
        return cls(right_embeddings=embeddings)

    @classmethod
    def from_store(cls, store: FactorizedTemporalStore) -> "ExactMIPSIndex":
        return cls.from_right_embeddings(store.right)

    def search(
        self,
        query_embedding: np.ndarray,
        k: int,
        *,
        exclude: int | None = None,
    ) -> list[tuple[int, float]]:
        if k <= 0:
            return []
        query = np.asarray(query_embedding, dtype=float)
        if query.ndim != 1:
            raise ValueError("query_embedding must be a vector")
        if query.shape[0] != self.right_embeddings.shape[1]:
            raise ValueError("query embedding dimension does not match index")

        scores = self.right_embeddings @ query
        if exclude is not None and 0 <= exclude < scores.shape[0]:
            scores = scores.copy()
            scores[exclude] = -np.inf

        limit = min(k, scores.shape[0])
        candidate_indices = np.argpartition(-scores, limit - 1)[:limit]
        ordered = candidate_indices[np.argsort(-scores[candidate_indices])]
        return [(int(idx), float(scores[idx])) for idx in ordered]


ExactTopNeighborIndex = ExactMIPSIndex
