"""Spectral compressors for temporal graph snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

from spectralstore.compression import FactorizedTemporalStore


ArrayLikeSnapshot = np.ndarray | sparse.spmatrix


@dataclass(frozen=True)
class SpectralCompressionConfig:
    rank: int = 8
    residual_threshold: float | None = None
    residual_threshold_mode: str = "mad"
    residual_quantile: float = 0.98
    residual_mad_multiplier: float = 45.0
    residual_hybrid_tail_quantile: float = 0.95
    residual_hybrid_tail_ratio: float = 2.0
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
            sparse_residuals, _ = _threshold_residuals(residual_stack, self.config)
            cleaned = dense - np.stack([residual.toarray() for residual in sparse_residuals])

        basis = _asymmetric_basis(cleaned, self.config)
        final_store = _factorize_from_basis(cleaned, basis, self.config, residuals=())
        final_residual_stack = _residual_stack(dense, final_store)
        final_residuals, diagnostics = _threshold_residuals(final_residual_stack, self.config)
        bound_metadata = _degree_aware_bound_metadata(
            dense,
            final_residual_stack,
            final_residuals,
        )
        return FactorizedTemporalStore(
            left=final_store.left,
            right=final_store.right,
            temporal=final_store.temporal,
            lambdas=final_store.lambdas,
            residuals=final_residuals,
            threshold_diagnostics={**diagnostics, **bound_metadata["diagnostics"]},
            source_degree_scale=bound_metadata["source_degree_scale"],
            target_degree_scale=bound_metadata["target_degree_scale"],
            entrywise_bound_scale=bound_metadata["entrywise_bound_scale"],
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


class TensorUnfoldingSVDCompressor:
    """Tensor-entry baseline using mode-1 and mode-2 unfolding SVD factors."""

    def __init__(self, config: SpectralCompressionConfig | None = None) -> None:
        self.config = config or SpectralCompressionConfig()

    def fit_transform(self, snapshots: list[ArrayLikeSnapshot]) -> FactorizedTemporalStore:
        dense = _as_dense_stack(snapshots)
        rank = min(self.config.rank, dense.shape[1], dense.shape[2])
        source_unfolding = dense.transpose(1, 0, 2).reshape(dense.shape[1], -1)
        target_unfolding = dense.transpose(2, 0, 1).reshape(dense.shape[2], -1)
        left = _truncated_left_singular_vectors(source_unfolding, rank)
        right = _truncated_left_singular_vectors(target_unfolding, rank)
        temporal = _project_temporal_weights(dense, left, right)
        return FactorizedTemporalStore(
            left=left,
            right=right,
            temporal=temporal,
            lambdas=np.ones(rank, dtype=float),
        )


def _as_dense_stack(snapshots: list[ArrayLikeSnapshot]) -> np.ndarray:
    if not snapshots:
        raise ValueError("snapshots cannot be empty")
    dense = [
        snapshot.toarray() if sparse.issparse(snapshot) else np.asarray(snapshot)
        for snapshot in snapshots
    ]
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


def _truncated_left_singular_vectors(matrix: np.ndarray, rank: int) -> np.ndarray:
    left_full, _singular_values, _right_t_full = np.linalg.svd(matrix, full_matrices=False)
    return left_full[:, :rank]


def _project_temporal_weights(
    dense_snapshots: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    temporal = np.empty((dense_snapshots.shape[0], left.shape[1]), dtype=float)
    for t, snapshot in enumerate(dense_snapshots):
        temporal[t] = np.einsum("ij,ij->j", left, snapshot @ right)
    return temporal


def _residual_stack(dense_snapshots: np.ndarray, store: FactorizedTemporalStore) -> np.ndarray:
    residuals = []
    for t, snapshot in enumerate(dense_snapshots):
        residuals.append(snapshot - store.dense_snapshot(t, include_residual=False))
    return np.stack(residuals)


def _threshold_residuals(
    residual_stack: np.ndarray,
    config: SpectralCompressionConfig,
) -> tuple[tuple[sparse.csr_matrix, ...], dict[str, Any]]:
    if config.residual_threshold is None:
        threshold, diagnostics = _adaptive_residual_threshold(residual_stack, config)
    else:
        threshold = float(config.residual_threshold)
        diagnostics = {
            "mode": "fixed",
            "estimated_threshold": threshold,
            "noise_scale": 0.0,
            "mad_sigma": 0.0,
            "residual_center": 0.0,
            "residual_mad": 0.0,
            "quantile_threshold": threshold,
            "hybrid_cap_active": False,
        }

    residual_matrices = []
    for residual in residual_stack:
        separated = residual.copy()
        separated[np.abs(separated) < threshold] = 0.0
        residual_matrices.append(sparse.csr_matrix(separated))
    residuals = tuple(residual_matrices)
    nnz = int(sum(residual.nnz for residual in residuals))
    total_entries = int(sum(residual.shape[0] * residual.shape[1] for residual in residuals))
    diagnostics = {
        **diagnostics,
        "estimated_threshold": threshold,
        "residual_nnz": nnz,
        "residual_sparsity": float(nnz / max(total_entries, 1)),
    }
    return residuals, diagnostics


def _adaptive_residual_threshold(
    residual_stack: np.ndarray,
    config: SpectralCompressionConfig,
) -> tuple[float, dict[str, Any]]:
    abs_residuals = np.abs(residual_stack).ravel()
    center = float(np.median(abs_residuals))
    mad = float(np.median(np.abs(abs_residuals - center)))
    sigma = float(1.4826 * mad)
    quantile_threshold = float(np.quantile(abs_residuals, config.residual_quantile))
    tail_threshold = float(np.quantile(abs_residuals, config.residual_hybrid_tail_quantile))
    if sigma <= 1e-12:
        mad_threshold = float(np.max(abs_residuals) + 1.0)
    else:
        mad_threshold = float(center + config.residual_mad_multiplier * sigma)

    diagnostics: dict[str, Any] = {
        "mode": config.residual_threshold_mode,
        "noise_scale": sigma,
        "mad_sigma": sigma,
        "residual_center": center,
        "residual_mad": mad,
        "mad_threshold": mad_threshold,
        "quantile_threshold": quantile_threshold,
        "tail_quantile": config.residual_hybrid_tail_quantile,
        "tail_threshold": tail_threshold,
        "tail_ratio": float(quantile_threshold / max(tail_threshold, 1e-12)),
        "hybrid_cap_active": False,
    }

    if config.residual_threshold_mode == "quantile":
        return quantile_threshold, diagnostics
    if config.residual_threshold_mode == "mad":
        return mad_threshold, diagnostics
    if config.residual_threshold_mode == "hybrid":
        cap_active = (
            mad_threshold > quantile_threshold
            and quantile_threshold > config.residual_hybrid_tail_ratio * max(tail_threshold, 1e-12)
        )
        diagnostics["hybrid_cap_active"] = cap_active
        if cap_active:
            return quantile_threshold, diagnostics
        return mad_threshold, diagnostics
    raise ValueError(f"unsupported residual_threshold_mode: {config.residual_threshold_mode}")


def _degree_aware_bound_metadata(
    dense_snapshots: np.ndarray,
    residual_stack: np.ndarray,
    residuals: tuple[sparse.csr_matrix, ...],
) -> dict[str, Any]:
    source_degree_scale, target_degree_scale = _degree_scales(dense_snapshots)
    edge_scale = 0.5 * (source_degree_scale[:, None] + target_degree_scale[None, :])
    residual_dense = np.stack([residual.toarray() for residual in residuals])
    omitted_residual = residual_stack - residual_dense
    per_entry_scale = np.abs(omitted_residual) / np.maximum(edge_scale[None, :, :], 1e-12)
    entrywise_bound_scale = float(np.max(per_entry_scale))
    return {
        "source_degree_scale": source_degree_scale,
        "target_degree_scale": target_degree_scale,
        "entrywise_bound_scale": entrywise_bound_scale,
        "diagnostics": {
            "degree_aware_bound_scale": entrywise_bound_scale,
            "source_degree_scale_mean": float(np.mean(source_degree_scale)),
            "target_degree_scale_mean": float(np.mean(target_degree_scale)),
            "source_degree_scale_max": float(np.max(source_degree_scale)),
            "target_degree_scale_max": float(np.max(target_degree_scale)),
        },
    }


def _degree_scales(dense_snapshots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weighted_degree = np.abs(dense_snapshots)
    source_degree = weighted_degree.sum(axis=2).mean(axis=0)
    target_degree = weighted_degree.sum(axis=1).mean(axis=0)
    mean_degree = float(np.mean(0.5 * (source_degree + target_degree)))
    if mean_degree <= 1e-12:
        return np.ones_like(source_degree), np.ones_like(target_degree)

    source_mu = np.maximum(source_degree / mean_degree, 1e-6)
    target_mu = np.maximum(target_degree / mean_degree, 1e-6)
    return 1.0 / np.sqrt(source_mu), 1.0 / np.sqrt(target_mu)
