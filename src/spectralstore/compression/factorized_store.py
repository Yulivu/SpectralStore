"""Factorized temporal graph storage."""

from __future__ import annotations

import json
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
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
    bound_sigma_max: float | None = None
    bound_mu: np.ndarray | None = None
    bound_constant: float = 1.0

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
        if self.bound_mu is not None and self.bound_mu.shape[0] != self.num_nodes:
            raise ValueError("bound mu must match the number of source nodes")

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

    def precompute_bound_params(
        self,
        snapshots: list[sparse.spmatrix | np.ndarray],
        *,
        constant: float = 1.0,
    ) -> None:
        """Precompute parameters for the Section 2.4 entrywise error bound.

        The cached bound uses residual MAD for the noise scale and mean row
        degree for the node-wise ``mu`` terms. The store is frozen for normal
        factor data, but this method intentionally caches derived metadata.
        """
        if len(snapshots) != self.num_steps:
            raise ValueError("snapshot count must match the number of time steps")
        if constant < 0.0:
            raise ValueError("bound constant must be non-negative")

        sigma_values = []
        degree_sum = np.zeros(self.num_nodes, dtype=float)
        for t, snapshot in enumerate(snapshots):
            matrix = snapshot.toarray() if sparse.issparse(snapshot) else np.asarray(snapshot)
            matrix = np.asarray(matrix, dtype=float)
            if matrix.shape != (self.num_nodes, self.right.shape[0]):
                raise ValueError("snapshot shape must match store dimensions")

            reconstruction = self.dense_snapshot(t, include_residual=True)
            residual = matrix - reconstruction
            sigma_values.append(float(np.median(np.abs(residual)) / 0.6745))
            degree_sum += np.asarray(matrix.sum(axis=1), dtype=float).reshape(-1)

        mean_degree = degree_sum / float(self.num_steps)
        mu = np.maximum(mean_degree / float(max(self.num_nodes, 1)), 1.0 / float(max(self.num_nodes, 1)))
        object.__setattr__(self, "bound_sigma_max", float(np.max(sigma_values)))
        object.__setattr__(self, "bound_mu", mu)
        object.__setattr__(self, "bound_constant", float(constant))

    def entrywise_bound(self, u: int, v: int) -> float:
        """Return the Section 2.4 entrywise theoretical error bound."""
        if u < 0 or u >= self.num_nodes:
            raise IndexError("source node index out of range")
        if v < 0 or v >= self.right.shape[0]:
            raise IndexError("target node index out of range")
        if self.bound_sigma_max is None or self.bound_mu is None:
            raise ValueError("bound parameters have not been precomputed")
        if v >= self.bound_mu.shape[0]:
            raise ValueError("entrywise bound currently requires square graph node indexing")

        n = max(self.num_nodes, 1)
        log_n = max(float(np.log(n)), 0.0)
        base = (
            float(self.bound_constant)
            * float(self.bound_sigma_max)
            * np.sqrt(float(self.rank) * log_n)
            / np.sqrt(float(n * self.num_steps))
        )
        degree_term = (1.0 / np.sqrt(self.bound_mu[u])) + (1.0 / np.sqrt(self.bound_mu[v]))
        return float(base * degree_term)

    def entrywise_error_bound(
        self,
        u: int,
        v: int,
        t: int,
        *,
        include_residual: bool = True,
    ) -> float | None:
        """Return the best available absolute error bound for one entry.

        If Section 2.4 bound parameters have been precomputed, that theoretical
        entrywise bound is returned. Otherwise this falls back to the empirical
        omitted-residual bound used by robust residual stores.
        """
        self._validate_query_indices(u, v, t)
        if self.bound_sigma_max is not None and self.bound_mu is not None:
            bound = self.entrywise_bound(u, v)
            if not include_residual and self.residuals:
                bound += abs(float(self.residuals[t][u, v]))
            return bound

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
        if self.bound_sigma_max is not None and self.bound_mu is not None:
            n = max(self.num_nodes, 1)
            log_n = max(float(np.log(n)), 0.0)
            base = (
                float(self.bound_constant)
                * float(self.bound_sigma_max)
                * np.sqrt(float(self.rank) * log_n)
                / np.sqrt(float(n * self.num_steps))
            )
            degree_scale = 1.0 / np.sqrt(self.bound_mu)
            bound = base * (degree_scale[:, None] + degree_scale[None, :])
            if not include_residual and self.residuals:
                bound = bound + np.abs(self.residuals[t].toarray())
            return np.asarray(bound, dtype=float)

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

    def factor_bytes(self) -> int:
        return int(
            self.left.nbytes
            + self.right.nbytes
            + self.temporal.nbytes
            + self.lambdas.nbytes
        )

    def residual_bytes(self) -> int:
        return int(
            sum(
                residual.data.nbytes + residual.indices.nbytes + residual.indptr.nbytes
                for residual in self.residuals
            )
        )

    def metadata_bytes(self) -> int:
        total = 0
        if self.source_degree_scale is not None:
            total += self.source_degree_scale.nbytes
        if self.target_degree_scale is not None:
            total += self.target_degree_scale.nbytes
        if self.threshold_diagnostics is not None:
            total += _diagnostics_bytes(self.threshold_diagnostics)
        if self.entrywise_bound_scale is not None:
            total += np.asarray(self.entrywise_bound_scale, dtype=float).nbytes
        if self.bound_sigma_max is not None:
            total += np.asarray(self.bound_sigma_max, dtype=float).nbytes
        if self.bound_mu is not None:
            total += self.bound_mu.nbytes
        if self.bound_sigma_max is not None or self.bound_mu is not None:
            total += np.asarray(self.bound_constant, dtype=float).nbytes
        return int(total)

    def compressed_bytes(self, *, include_metadata: bool = True) -> int:
        total = self.factor_bytes() + self.residual_bytes()
        if include_metadata:
            total += self.metadata_bytes()
        return int(total)

    def raw_dense_bytes(self, *, dtype_bytes: int = 8) -> int:
        return int(self.num_steps * self.num_nodes * self.right.shape[0] * dtype_bytes)

    @staticmethod
    def raw_sparse_csr_bytes(snapshots: list[sparse.spmatrix]) -> int:
        total = 0
        for snapshot in snapshots:
            csr = snapshot.tocsr()
            total += csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes
        return int(total)

    def compression_ratio(
        self,
        *,
        dtype_bytes: int = 8,
        include_metadata: bool = True,
    ) -> float:
        return float(
            self.compressed_bytes(include_metadata=include_metadata)
            / max(self.raw_dense_bytes(dtype_bytes=dtype_bytes), 1)
        )

    def compressed_vs_raw_dense_ratio(
        self,
        *,
        dtype_bytes: int = 8,
        include_metadata: bool = True,
    ) -> float:
        return self.compression_ratio(
            dtype_bytes=dtype_bytes,
            include_metadata=include_metadata,
        )

    def compressed_vs_raw_sparse_ratio(
        self,
        snapshots: list[sparse.spmatrix],
        *,
        include_metadata: bool = True,
    ) -> float:
        return float(
            self.compressed_bytes(include_metadata=include_metadata)
            / max(self.raw_sparse_csr_bytes(snapshots), 1)
        )

    def save_npz(self, path: str | PathLike[str]) -> None:
        """Serialize the store to a portable NPZ bundle.

        The format stores dense factor arrays directly and sparse residuals as
        CSR component arrays. Diagnostics are JSON-encoded to avoid pickle.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        arrays: dict[str, np.ndarray] = {
            "left": self.left,
            "right": self.right,
            "temporal": self.temporal,
            "lambdas": self.lambdas,
            "residual_count": np.asarray(len(self.residuals), dtype=np.int64),
            "threshold_diagnostics_json": np.asarray(
                json.dumps(_json_ready(self.threshold_diagnostics), sort_keys=True)
                if self.threshold_diagnostics is not None
                else "",
            ),
            "has_source_degree_scale": np.asarray(
                self.source_degree_scale is not None,
                dtype=bool,
            ),
            "has_target_degree_scale": np.asarray(
                self.target_degree_scale is not None,
                dtype=bool,
            ),
            "has_entrywise_bound_scale": np.asarray(
                self.entrywise_bound_scale is not None,
                dtype=bool,
            ),
            "has_bound_sigma_max": np.asarray(
                self.bound_sigma_max is not None,
                dtype=bool,
            ),
            "has_bound_mu": np.asarray(
                self.bound_mu is not None,
                dtype=bool,
            ),
            "bound_constant": np.asarray(self.bound_constant, dtype=float),
        }
        if self.source_degree_scale is not None:
            arrays["source_degree_scale"] = self.source_degree_scale
        if self.target_degree_scale is not None:
            arrays["target_degree_scale"] = self.target_degree_scale
        if self.entrywise_bound_scale is not None:
            arrays["entrywise_bound_scale"] = np.asarray(
                self.entrywise_bound_scale,
                dtype=float,
            )
        if self.bound_sigma_max is not None:
            arrays["bound_sigma_max"] = np.asarray(self.bound_sigma_max, dtype=float)
        if self.bound_mu is not None:
            arrays["bound_mu"] = self.bound_mu
        for index, residual in enumerate(self.residuals):
            csr = residual.tocsr()
            arrays[f"residual_{index}_data"] = csr.data
            arrays[f"residual_{index}_indices"] = csr.indices
            arrays[f"residual_{index}_indptr"] = csr.indptr
            arrays[f"residual_{index}_shape"] = np.asarray(csr.shape, dtype=np.int64)

        np.savez_compressed(output_path, **arrays)

    @classmethod
    def load_npz(cls, path: str | PathLike[str]) -> "FactorizedTemporalStore":
        with np.load(Path(path), allow_pickle=False) as bundle:
            residuals = []
            residual_count = int(bundle["residual_count"])
            for index in range(residual_count):
                shape = tuple(int(value) for value in bundle[f"residual_{index}_shape"])
                residuals.append(
                    sparse.csr_matrix(
                        (
                            bundle[f"residual_{index}_data"],
                            bundle[f"residual_{index}_indices"],
                            bundle[f"residual_{index}_indptr"],
                        ),
                        shape=shape,
                    )
                )

            diagnostics_json = str(bundle["threshold_diagnostics_json"])
            threshold_diagnostics = (
                json.loads(diagnostics_json) if diagnostics_json else None
            )
            source_degree_scale = (
                bundle["source_degree_scale"]
                if bool(bundle["has_source_degree_scale"])
                else None
            )
            target_degree_scale = (
                bundle["target_degree_scale"]
                if bool(bundle["has_target_degree_scale"])
                else None
            )
            entrywise_bound_scale = (
                float(bundle["entrywise_bound_scale"])
                if bool(bundle["has_entrywise_bound_scale"])
                else None
            )
            bound_sigma_max = (
                float(bundle["bound_sigma_max"])
                if "has_bound_sigma_max" in bundle and bool(bundle["has_bound_sigma_max"])
                else None
            )
            bound_mu = (
                bundle["bound_mu"]
                if "has_bound_mu" in bundle and bool(bundle["has_bound_mu"])
                else None
            )
            bound_constant = (
                float(bundle["bound_constant"])
                if "bound_constant" in bundle
                else 1.0
            )
            return cls(
                left=bundle["left"],
                right=bundle["right"],
                temporal=bundle["temporal"],
                lambdas=bundle["lambdas"],
                residuals=tuple(residuals),
                threshold_diagnostics=threshold_diagnostics,
                source_degree_scale=source_degree_scale,
                target_degree_scale=target_degree_scale,
                entrywise_bound_scale=entrywise_bound_scale,
                bound_sigma_max=bound_sigma_max,
                bound_mu=bound_mu,
                bound_constant=bound_constant,
            )

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


def _diagnostics_bytes(diagnostics: dict[str, Any]) -> int:
    """Estimate in-memory payload bytes for scalar threshold diagnostics."""
    total = 0
    for key, value in diagnostics.items():
        total += len(str(key).encode("utf-8"))
        if isinstance(value, str):
            total += len(value.encode("utf-8"))
        elif isinstance(value, bool):
            total += np.asarray(value, dtype=bool).nbytes
        elif isinstance(value, int):
            total += np.asarray(value, dtype=np.int64).nbytes
        elif isinstance(value, float):
            total += np.asarray(value, dtype=float).nbytes
        elif value is None:
            total += 0
        else:
            total += np.asarray(value).nbytes
    return int(total)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value
