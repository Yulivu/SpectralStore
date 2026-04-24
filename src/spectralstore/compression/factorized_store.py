"""Factorized temporal graph storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class FactorizedTemporalStore:
    """Low-rank temporal graph representation.

    The score for edge ``(u, v)`` at time ``t`` is:

    ``sum_j lambdas[j] * left[u, j] * right[v, j] * temporal[t, j]``
    """

    left: np.ndarray
    right: np.ndarray
    temporal: np.ndarray
    lambdas: np.ndarray
    residuals: tuple[sparse.csr_matrix, ...] = ()
    threshold_diagnostics: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        rank = self.lambdas.shape[0]
        if self.left.ndim != 2 or self.right.ndim != 2 or self.temporal.ndim != 2:
            raise ValueError("left, right, and temporal factors must be matrices")
        if self.left.shape[1] != rank:
            raise ValueError("left factor rank does not match lambdas")
        if self.right.shape[1] != rank:
            raise ValueError("right factor rank does not match lambdas")
        if self.temporal.shape[1] != rank:
            raise ValueError("temporal factor rank does not match lambdas")
        if self.residuals and len(self.residuals) != self.temporal.shape[0]:
            raise ValueError("residual count must match the number of time steps")

    @property
    def rank(self) -> int:
        return int(self.lambdas.shape[0])

    @property
    def num_nodes(self) -> int:
        return int(self.left.shape[0])

    @property
    def num_steps(self) -> int:
        return int(self.temporal.shape[0])

    def link_score(self, u: int, v: int, t: int, *, include_residual: bool = True) -> float:
        self._validate_query_indices(u, v, t)
        base = float(np.dot(self.lambdas * self.left[u] * self.temporal[t], self.right[v]))
        if include_residual and self.residuals:
            base += float(self.residuals[t][u, v])
        return base

    def entrywise_error_bound(
        self,
        u: int,
        v: int,
        t: int,
        *,
        include_residual: bool = True,
    ) -> float | None:
        """Return an empirical omitted-residual bound for one entry.

        Robust residual stores separate entries above the estimated threshold.
        After residual correction, the remaining unmaterialized entrywise error
        is bounded empirically by that threshold on the fitted snapshots.
        """
        self._validate_query_indices(u, v, t)
        if self.threshold_diagnostics is None:
            return None
        threshold = self.threshold_diagnostics.get("estimated_threshold")
        if threshold is None:
            return None

        bound = float(threshold)
        if not include_residual and self.residuals:
            bound += abs(float(self.residuals[t][u, v]))
        return bound

    def dense_snapshot(self, t: int, *, include_residual: bool = True) -> np.ndarray:
        if t < 0 or t >= self.num_steps:
            raise IndexError("time index out of range")
        weights = self.lambdas * self.temporal[t]
        snapshot = (self.left * weights) @ self.right.T
        if include_residual and self.residuals:
            snapshot = snapshot + self.residuals[t].toarray()
        return snapshot

    def _validate_query_indices(self, u: int, v: int, t: int) -> None:
        if u < 0 or u >= self.num_nodes:
            raise IndexError("source node index out of range")
        if v < 0 or v >= self.right.shape[0]:
            raise IndexError("target node index out of range")
        if t < 0 or t >= self.num_steps:
            raise IndexError("time index out of range")
