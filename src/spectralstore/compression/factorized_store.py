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
    source_degree_scale: np.ndarray | None = None
    target_degree_scale: np.ndarray | None = None
    entrywise_bound_scale: float | None = None

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
        if (
            self.source_degree_scale is not None
            and self.source_degree_scale.shape[0] != self.num_nodes
        ):
            raise ValueError("source degree scale must match the number of source nodes")
        if (
            self.target_degree_scale is not None
            and self.target_degree_scale.shape[0] != self.right.shape[0]
        ):
            raise ValueError("target degree scale must match the number of target nodes")

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

        degree_bound = self._degree_aware_bound(u, v)
        bound = degree_bound if degree_bound is not None else float(threshold)
        if not include_residual and self.residuals:
            bound += abs(float(self.residuals[t][u, v]))
        return bound

    def entrywise_error_bound_matrix(
        self,
        t: int,
        *,
        include_residual: bool = True,
    ) -> np.ndarray | None:
        if t < 0 or t >= self.num_steps:
            raise IndexError("time index out of range")
        if self.threshold_diagnostics is None:
            return None
        threshold = self.threshold_diagnostics.get("estimated_threshold")
        if threshold is None:
            return None

        if (
            self.source_degree_scale is not None
            and self.target_degree_scale is not None
            and self.entrywise_bound_scale is not None
        ):
            edge_scale = 0.5 * (
                self.source_degree_scale[:, None] + self.target_degree_scale[None, :]
            )
            bound = float(self.entrywise_bound_scale) * edge_scale
        else:
            bound = np.full((self.num_nodes, self.right.shape[0]), float(threshold))

        if not include_residual and self.residuals:
            bound = bound + np.abs(self.residuals[t].toarray())
        return np.asarray(bound, dtype=float)

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

    def _degree_aware_bound(self, u: int, v: int) -> float | None:
        if (
            self.source_degree_scale is None
            or self.target_degree_scale is None
            or self.entrywise_bound_scale is None
        ):
            return None
        edge_scale = 0.5 * (self.source_degree_scale[u] + self.target_degree_scale[v])
        return float(self.entrywise_bound_scale * edge_scale)
