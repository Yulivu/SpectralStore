"""Approximate MIPS index via random projection candidate pruning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spectralstore.compression import FactorizedTemporalStore


@dataclass(frozen=True)
class RandomProjectionANNMIPSIndex:
    """ANN MIPS prototype with exact rerank on a projected candidate pool.

    This is a lightweight PQ/ANN placeholder for Phase 2 system experiments.
    It projects right embeddings into a lower-dimensional space to prune
    candidates, then reranks those candidates with exact inner products.
    """

    right_embeddings: np.ndarray
    projection_matrix: np.ndarray
    projected_right_embeddings: np.ndarray
    candidate_multiplier: int = 4

    @classmethod
    def from_right_embeddings(
        cls,
        right_embeddings: np.ndarray,
        *,
        projection_dim: int | None = None,
        candidate_multiplier: int = 4,
        random_seed: int = 0,
    ) -> "RandomProjectionANNMIPSIndex":
        embeddings = np.asarray(right_embeddings, dtype=float)
        if embeddings.ndim != 2:
            raise ValueError("right_embeddings must be a matrix")
        dim = embeddings.shape[1]
        proj_dim = int(projection_dim) if projection_dim is not None else max(2, dim // 2)
        proj_dim = max(1, min(dim, proj_dim))
        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be positive")

        rng = np.random.default_rng(random_seed)
        projection = rng.standard_normal(size=(dim, proj_dim))
        projection /= np.sqrt(float(proj_dim))
        projected = embeddings @ projection
        return cls(
            right_embeddings=embeddings,
            projection_matrix=projection,
            projected_right_embeddings=projected,
            candidate_multiplier=int(candidate_multiplier),
        )

    @classmethod
    def from_store(
        cls,
        store: FactorizedTemporalStore,
        *,
        projection_dim: int | None = None,
        candidate_multiplier: int = 4,
        random_seed: int = 0,
    ) -> "RandomProjectionANNMIPSIndex":
        return cls.from_right_embeddings(
            store.right,
            projection_dim=projection_dim,
            candidate_multiplier=candidate_multiplier,
            random_seed=random_seed,
        )

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

        projected_query = query @ self.projection_matrix
        coarse_scores = self.projected_right_embeddings @ projected_query
        if exclude is not None and 0 <= exclude < coarse_scores.shape[0]:
            coarse_scores = coarse_scores.copy()
            coarse_scores[exclude] = -np.inf

        candidate_count = min(
            coarse_scores.shape[0],
            max(k, int(self.candidate_multiplier) * k),
        )
        coarse_idx = np.argpartition(-coarse_scores, candidate_count - 1)[:candidate_count]
        fine_scores = self.right_embeddings[coarse_idx] @ query
        order = np.argsort(-fine_scores)
        top = coarse_idx[order[: min(k, coarse_idx.size)]]
        return [(int(idx), float(self.right_embeddings[idx] @ query)) for idx in top]
