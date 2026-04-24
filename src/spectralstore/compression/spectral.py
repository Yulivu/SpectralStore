"""Spectral compressors for temporal graph snapshots."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from spectralstore.compression import FactorizedTemporalStore


ArrayLikeSnapshot = np.ndarray | sparse.spmatrix


@dataclass(frozen=True)
class SpectralCompressionConfig:
    rank: int = 8
    residual_threshold: float | None = None
    random_seed: int = 0


class AsymmetricSpectralCompressor:
    """First SpectralStore compressor using asymmetric split-snapshot SVD."""

    def __init__(self, config: SpectralCompressionConfig | None = None) -> None:
        self.config = config or SpectralCompressionConfig()

    def fit_transform(self, snapshots: list[ArrayLikeSnapshot]) -> FactorizedTemporalStore:
        dense = _as_dense_stack(snapshots)
        if dense.shape[0] < 2:
            raise ValueError("at least two temporal snapshots are required")

        rng = np.random.default_rng(self.config.random_seed)
        order = rng.permutation(dense.shape[0])
        split = max(1, dense.shape[0] // 2)
        first = dense[order[:split]].mean(axis=0)
        second = dense[order[split:]].mean(axis=0)
        if order[split:].size == 0:
            second = first

        stitched = np.triu(first) + np.tril(second, k=-1)
        np.fill_diagonal(stitched, 0.5 * (np.diag(first) + np.diag(second)))
        return _factorize_from_basis(dense, stitched, self.config)


class SymmetricSVDCompressor:
    """Baseline that symmetrizes the mean adjacency matrix before SVD."""

    def __init__(self, config: SpectralCompressionConfig | None = None) -> None:
        self.config = config or SpectralCompressionConfig()

    def fit_transform(self, snapshots: list[ArrayLikeSnapshot]) -> FactorizedTemporalStore:
        dense = _as_dense_stack(snapshots)
        mean = dense.mean(axis=0)
        sym_mean = 0.5 * (mean + mean.T)
        return _factorize_from_basis(dense, sym_mean, self.config)


class DirectSVDCompressor:
    """Baseline that applies SVD directly to the mean adjacency matrix."""

    def __init__(self, config: SpectralCompressionConfig | None = None) -> None:
        self.config = config or SpectralCompressionConfig()

    def fit_transform(self, snapshots: list[ArrayLikeSnapshot]) -> FactorizedTemporalStore:
        dense = _as_dense_stack(snapshots)
        return _factorize_from_basis(dense, dense.mean(axis=0), self.config)


def _as_dense_stack(snapshots: list[ArrayLikeSnapshot]) -> np.ndarray:
    if not snapshots:
        raise ValueError("snapshots cannot be empty")
    dense = [snapshot.toarray() if sparse.issparse(snapshot) else np.asarray(snapshot) for snapshot in snapshots]
    return np.stack(dense).astype(float, copy=False)


def _factorize_from_basis(
    dense_snapshots: np.ndarray,
    basis: np.ndarray,
    config: SpectralCompressionConfig,
) -> FactorizedTemporalStore:
    rank = min(config.rank, min(basis.shape))
    left_full, singular_values, right_t_full = np.linalg.svd(basis, full_matrices=False)
    left = left_full[:, :rank]
    right = right_t_full[:rank, :].T
    lambdas = singular_values[:rank].copy()
    safe_lambdas = np.where(np.abs(lambdas) > 1e-12, lambdas, 1.0)

    temporal = np.empty((dense_snapshots.shape[0], rank), dtype=float)
    for t, snapshot in enumerate(dense_snapshots):
        projected = np.einsum("ij,ij->j", left, snapshot @ right)
        temporal[t] = projected / safe_lambdas

    residuals: tuple[sparse.csr_matrix, ...] = ()
    if config.residual_threshold is not None:
        residual_matrices = []
        for t, snapshot in enumerate(dense_snapshots):
            weights = lambdas * temporal[t]
            reconstruction = (left * weights) @ right.T
            residual = snapshot - reconstruction
            residual[np.abs(residual) <= config.residual_threshold] = 0.0
            residual_matrices.append(sparse.csr_matrix(residual))
        residuals = tuple(residual_matrices)

    return FactorizedTemporalStore(
        left=left,
        right=right,
        temporal=temporal,
        lambdas=lambdas,
        residuals=residuals,
    )
