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
    residual_quantile: float = 0.98
    robust_iterations: int = 1
    random_seed: int = 0
    num_splits: int = 1


class AsymmetricSpectralCompressor:
    """First SpectralStore compressor using asymmetric split-snapshot SVD."""

    def __init__(self, config: SpectralCompressionConfig | None = None) -> None:
        self.config = config or SpectralCompressionConfig()

    def fit_transform(self, snapshots: list[ArrayLikeSnapshot]) -> FactorizedTemporalStore:
        dense = _as_dense_stack(snapshots)
        if dense.shape[0] < 2:
            raise ValueError("at least two temporal snapshots are required")

        stitched = _asymmetric_basis(dense, self.config)
        return _factorize_from_basis(dense, stitched, self.config)


class RobustAsymmetricSpectralCompressor:
    """Asymmetric spectral compressor with sparse residual separation."""

    def __init__(self, config: SpectralCompressionConfig | None = None) -> None:
        self.config = config or SpectralCompressionConfig()

    def fit_transform(self, snapshots: list[ArrayLikeSnapshot]) -> FactorizedTemporalStore:
        dense = _as_dense_stack(snapshots)
        cleaned = dense.copy()
        iterations = max(1, self.config.robust_iterations)

        for _ in range(iterations):
            basis = _asymmetric_basis(cleaned, self.config)
            store = _factorize_from_basis(cleaned, basis, self.config, residuals=())
            residual_stack = _residual_stack(dense, store)
            sparse_residuals = _threshold_residuals(residual_stack, self.config)
            cleaned = dense - np.stack([residual.toarray() for residual in sparse_residuals])

        basis = _asymmetric_basis(cleaned, self.config)
        final_store = _factorize_from_basis(cleaned, basis, self.config, residuals=())
        final_residuals = _threshold_residuals(_residual_stack(dense, final_store), self.config)
        return FactorizedTemporalStore(
            left=final_store.left,
            right=final_store.right,
            temporal=final_store.temporal,
            lambdas=final_store.lambdas,
            residuals=final_residuals,
        )


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


def _asymmetric_basis(dense: np.ndarray, config: SpectralCompressionConfig) -> np.ndarray:
    rng = np.random.default_rng(config.random_seed)
    stitched = np.zeros(dense.shape[1:], dtype=float)
    num_splits = max(1, config.num_splits)
    for _ in range(num_splits):
        order = rng.permutation(dense.shape[0])
        split = max(1, dense.shape[0] // 2)
        first = dense[order[:split]].mean(axis=0)
        second = dense[order[split:]].mean(axis=0)
        if order[split:].size == 0:
            second = first

        split_stitched = np.triu(first) + np.tril(second, k=-1)
        np.fill_diagonal(split_stitched, 0.5 * (np.diag(first) + np.diag(second)))
        stitched += split_stitched
    return stitched / num_splits


def _factorize_from_basis(
    dense_snapshots: np.ndarray,
    basis: np.ndarray,
    config: SpectralCompressionConfig,
    *,
    residuals: tuple[sparse.csr_matrix, ...] | None = None,
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

    store_residuals: tuple[sparse.csr_matrix, ...] = residuals or ()
    if residuals is None and config.residual_threshold is not None:
        residual_matrices = []
        for t, snapshot in enumerate(dense_snapshots):
            weights = lambdas * temporal[t]
            reconstruction = (left * weights) @ right.T
            residual = snapshot - reconstruction
            residual[np.abs(residual) <= config.residual_threshold] = 0.0
            residual_matrices.append(sparse.csr_matrix(residual))
        store_residuals = tuple(residual_matrices)

    return FactorizedTemporalStore(
        left=left,
        right=right,
        temporal=temporal,
        lambdas=lambdas,
        residuals=store_residuals,
    )


def _residual_stack(dense_snapshots: np.ndarray, store: FactorizedTemporalStore) -> np.ndarray:
    residuals = []
    for t, snapshot in enumerate(dense_snapshots):
        residuals.append(snapshot - store.dense_snapshot(t, include_residual=False))
    return np.stack(residuals)


def _threshold_residuals(
    residual_stack: np.ndarray,
    config: SpectralCompressionConfig,
) -> tuple[sparse.csr_matrix, ...]:
    abs_residuals = np.abs(residual_stack)
    if config.residual_threshold is None:
        threshold = float(np.quantile(abs_residuals, config.residual_quantile))
    else:
        threshold = config.residual_threshold

    residual_matrices = []
    for residual in residual_stack:
        separated = residual.copy()
        separated[np.abs(separated) < threshold] = 0.0
        residual_matrices.append(sparse.csr_matrix(separated))
    return tuple(residual_matrices)
