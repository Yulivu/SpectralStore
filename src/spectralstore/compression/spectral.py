"""Spectral compressors for temporal graph snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

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
    residual_threshold_scale: float = 1.0
    entrywise_bound_coverage: float = 1.0
    robust_iterations: int = 1
    tensor_iterations: int = 10
    tensor_ridge: float = 1e-6
    tensor_rank_energy: float = 1.0
    tensor_min_rank: int = 1
    rpca_iterations: int = 100
    rpca_tol: float = 1e-6
    rpca_reg_E: float = 1.0
    rpca_reg_J: float = 1.0
    rank_pruning_mode: str = "none"
    rank_pruning_threshold: float = 0.01
    rank_pruning_min_rank: int = 1
    rank_pruning_iterations: int = 1
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
            self.config,
        )
        return FactorizedTemporalStore(
            left=final_store.left,
            right=final_store.right,
            temporal=final_store.temporal,
            lambdas=final_store.lambdas,
            residuals=final_residuals,
            threshold_diagnostics={
                **diagnostics,
                **(final_store.threshold_diagnostics or {}),
                **bound_metadata["diagnostics"],
            },
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
        return _store_from_tensor_factors(
            left,
            right,
            temporal,
            self.config,
            method="tensor_unfolding_svd",
        )


class SparseUnfoldingAsymmetricCompressor:
    """Sparse temporal-unfolding SpectralStore compressor.

    This keeps the store format diagonal in components while extracting source
    and target spaces from sparse mode unfoldings instead of from the dense
    snapshot stack.
    """

    def __init__(self, config: SpectralCompressionConfig | None = None) -> None:
        self.config = config or SpectralCompressionConfig()

    def fit_transform(self, snapshots: list[ArrayLikeSnapshot]) -> FactorizedTemporalStore:
        if not snapshots:
            raise ValueError("snapshots cannot be empty")
        sparse_snapshots = [
            snapshot.tocsr() if sparse.issparse(snapshot) else sparse.csr_matrix(snapshot)
            for snapshot in snapshots
        ]
        first_shape = sparse_snapshots[0].shape
        if any(snapshot.shape != first_shape for snapshot in sparse_snapshots):
            raise ValueError("all snapshots must have the same shape")

        rank = min(self.config.rank, first_shape[0], first_shape[1])
        source_unfolding = sparse.hstack(sparse_snapshots, format="csr")
        target_unfolding = sparse.hstack(
            [snapshot.T.tocsr() for snapshot in sparse_snapshots],
            format="csr",
        )
        left = _truncated_sparse_left_singular_vectors(
            source_unfolding,
            rank,
            self.config.random_seed,
        )
        right = _truncated_sparse_left_singular_vectors(
            target_unfolding,
            rank,
            self.config.random_seed + 1,
        )
        temporal = _project_temporal_weights(sparse_snapshots, left, right)
        return _store_from_tensor_factors(
            left,
            right,
            temporal,
            self.config,
            method="sparse_unfolding_asym",
            extra_diagnostics={
                "source_unfolding_shape": list(source_unfolding.shape),
                "target_unfolding_shape": list(target_unfolding.shape),
                "source_unfolding_nnz": int(source_unfolding.nnz),
                "target_unfolding_nnz": int(target_unfolding.nnz),
            },
        )


class SplitAsymmetricUnfoldingCompressor:
    """Split-snapshot asymmetric compressor using independent triangular means."""

    def __init__(self, config: SpectralCompressionConfig | None = None) -> None:
        self.config = config or SpectralCompressionConfig()

    def fit_transform(self, snapshots: list[ArrayLikeSnapshot]) -> FactorizedTemporalStore:
        if len(snapshots) < 2:
            raise ValueError("at least two temporal snapshots are required")
        sparse_snapshots = [
            snapshot.tocsr() if sparse.issparse(snapshot) else sparse.csr_matrix(snapshot)
            for snapshot in snapshots
        ]
        first_shape = sparse_snapshots[0].shape
        if first_shape[0] != first_shape[1]:
            raise ValueError("split asymmetric unfolding requires square snapshots")
        if any(snapshot.shape != first_shape for snapshot in sparse_snapshots):
            raise ValueError("all snapshots must have the same shape")

        rng = np.random.default_rng(self.config.random_seed)
        order = rng.permutation(len(sparse_snapshots))
        split = max(1, len(sparse_snapshots) // 2)
        first_indices = order[:split]
        second_indices = order[split:]
        if second_indices.size == 0:
            second_indices = first_indices

        first_mean = _mean_sparse_snapshots(sparse_snapshots, first_indices)
        second_mean = _mean_sparse_snapshots(sparse_snapshots, second_indices)
        stitched = _split_triangular_sparse_matrix(first_mean, second_mean)

        rank = min(self.config.rank, min(stitched.shape))
        left, singular_values, right_t = _truncated_sparse_svd(
            stitched,
            rank,
            self.config.random_seed,
        )
        right = right_t.T
        safe_singular_values = np.where(np.abs(singular_values) > 1e-12, singular_values, 1.0)
        temporal = np.empty((len(sparse_snapshots), rank), dtype=float)
        for t, snapshot in enumerate(sparse_snapshots):
            temporal[t] = np.einsum("ij,ij->j", left, snapshot @ right) / safe_singular_values

        return FactorizedTemporalStore(
            left=left,
            right=right,
            temporal=temporal,
            lambdas=singular_values,
            threshold_diagnostics={
                "tensor_method": "split_asym_unfolding",
                "requested_rank": int(self.config.rank),
                "effective_rank": int(rank),
                "split_first_size": int(first_indices.size),
                "split_second_size": int(second_indices.size),
                "stitched_shape": list(stitched.shape),
                "stitched_nnz": int(stitched.nnz),
            },
        )


class CPALSCompressor:
    """TensorLy CP-ALS baseline for temporal adjacency tensors."""

    def __init__(self, config: SpectralCompressionConfig | None = None) -> None:
        self.config = config or SpectralCompressionConfig()

    def fit_transform(self, snapshots: list[ArrayLikeSnapshot]) -> FactorizedTemporalStore:
        dense = _as_dense_stack(snapshots)
        rank = min(self.config.rank, *dense.shape)
        try:
            from tensorly.decomposition import parafac
        except ImportError as exc:
            raise ImportError(
                "TensorLy is required for CPALSCompressor. "
                "Install experiment dependencies with `python -m pip install -e .[experiments]`."
            ) from exc

        cp_tensor = parafac(
            dense,
            rank=rank,
            n_iter_max=max(1, self.config.tensor_iterations),
            init="svd",
            tol=1e-8,
            random_state=self.config.random_seed,
            l2_reg=self.config.tensor_ridge,
            normalize_factors=False,
        )
        weights = np.asarray(cp_tensor.weights, dtype=float)
        temporal = np.asarray(cp_tensor.factors[0], dtype=float) * weights[None, :]
        left = np.asarray(cp_tensor.factors[1], dtype=float)
        right = np.asarray(cp_tensor.factors[2], dtype=float)

        return _store_from_tensor_factors(
            left,
            right,
            temporal,
            self.config,
            method="cp_als",
            extra_diagnostics={"tensorly_backend": "parafac"},
        )


class TuckerHOSVDCompressor:
    """TensorLy Tucker-ALS baseline projected back into the store format."""

    def __init__(self, config: SpectralCompressionConfig | None = None) -> None:
        self.config = config or SpectralCompressionConfig()

    def fit_transform(self, snapshots: list[ArrayLikeSnapshot]) -> FactorizedTemporalStore:
        dense = _as_dense_stack(snapshots)
        rank = min(self.config.rank, dense.shape[1], dense.shape[2])
        try:
            import tensorly as tl
            from tensorly.decomposition import tucker
        except ImportError as exc:
            raise ImportError(
                "TensorLy is required for TuckerHOSVDCompressor. "
                "Install experiment dependencies with `python -m pip install -e .[experiments]`."
            ) from exc

        tucker_rank = [min(rank, dense.shape[0]), rank, rank]
        tucker_tensor = tucker(
            dense,
            rank=tucker_rank,
            n_iter_max=max(1, self.config.tensor_iterations),
            init="svd",
            tol=1e-6,
            random_state=self.config.random_seed,
        )
        reconstructed = np.asarray(tl.tucker_to_tensor(tucker_tensor), dtype=float)
        left, right, temporal = _initialize_tensor_factors(reconstructed, rank)
        return _store_from_tensor_factors(
            left,
            right,
            temporal,
            self.config,
            method="tucker_als",
            extra_diagnostics={
                "tensorly_backend": "tucker",
                "tucker_core_shape": list(tucker_tensor.core.shape),
            },
        )


class RPCASVDCompressor:
    """Dense matrix PCP-RPCA on the temporal mean followed by SVD."""

    def __init__(self, config: SpectralCompressionConfig | None = None) -> None:
        self.config = config or SpectralCompressionConfig()

    def fit_transform(self, snapshots: list[ArrayLikeSnapshot]) -> FactorizedTemporalStore:
        dense = _as_dense_stack(snapshots)
        rank = min(self.config.rank, dense.shape[1], dense.shape[2])
        mean_snapshot = dense.mean(axis=0)
        mean_low_rank, sparse_error, diagnostics = _principal_component_pursuit(
            mean_snapshot,
            tol=self.config.rpca_tol,
            max_iterations=self.config.rpca_iterations,
        )
        sparse_nnz = int(np.count_nonzero(np.abs(sparse_error) > 1e-12))
        left_full, singular_values, right_t_full = np.linalg.svd(
            mean_low_rank,
            full_matrices=False,
        )
        left = left_full[:, :rank]
        right = right_t_full[:rank, :].T
        lambdas = singular_values[:rank].copy()
        safe_lambdas = np.where(np.abs(lambdas) > 1e-12, lambdas, 1.0)
        temporal = np.empty((dense.shape[0], rank), dtype=float)
        for t, snapshot in enumerate(dense):
            temporal[t] = np.einsum("ij,ij->j", left, snapshot @ right) / safe_lambdas

        return FactorizedTemporalStore(
            left=left,
            right=right,
            temporal=temporal,
            lambdas=lambdas,
            threshold_diagnostics={
                "tensor_method": "rpca_svd",
                "rpca_backend": "matrix_pcp_admm",
                "rpca_iterations": int(self.config.rpca_iterations),
                "rpca_tol": float(self.config.rpca_tol),
                "rpca_sparse_nnz": sparse_nnz,
                "rpca_iterations_mean": float(diagnostics["iterations"]),
                "rpca_iterations_max": int(diagnostics["iterations"]),
                "rpca_relative_residual_mean": float(diagnostics["relative_residual"]),
                "requested_rank": int(self.config.rank),
                "effective_rank": int(rank),
            },
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

    left, right, temporal, lambdas, pruning_diagnostics = _apply_rank_pruning(
        dense_snapshots,
        left,
        right,
        temporal,
        lambdas,
        config,
    )

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

    threshold_diagnostics = pruning_diagnostics if pruning_diagnostics else None
    return FactorizedTemporalStore(
        left=left,
        right=right,
        temporal=temporal,
        lambdas=lambdas,
        residuals=store_residuals,
        threshold_diagnostics=threshold_diagnostics,
    )


def _truncated_left_singular_vectors(matrix: np.ndarray, rank: int) -> np.ndarray:
    left_full, _singular_values, _right_t_full = np.linalg.svd(matrix, full_matrices=False)
    return left_full[:, :rank]


def _truncated_sparse_left_singular_vectors(
    matrix: sparse.spmatrix,
    rank: int,
    random_seed: int,
) -> np.ndarray:
    left, _singular_values, _right_t = _truncated_sparse_svd(matrix, rank, random_seed)
    return left


def _truncated_sparse_svd(
    matrix: sparse.spmatrix,
    rank: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rank = min(rank, min(matrix.shape))
    if rank <= 0:
        raise ValueError("rank must be positive")
    if rank >= min(matrix.shape):
        left_full, _singular_values, _right_t_full = np.linalg.svd(
            matrix.toarray(),
            full_matrices=False,
        )
        return left_full[:, :rank], _singular_values[:rank], _right_t_full[:rank]

    left, singular_values, _right_t = svds(
        matrix.astype(float),
        k=rank,
        which="LM",
        random_state=random_seed,
    )
    order = np.argsort(-singular_values)
    return left[:, order], singular_values[order], _right_t[order]


def _mean_sparse_snapshots(
    snapshots: list[sparse.csr_matrix],
    indices: np.ndarray,
) -> sparse.csr_matrix:
    if indices.size == 0:
        raise ValueError("cannot average an empty snapshot split")
    total = sparse.csr_matrix(snapshots[0].shape, dtype=float)
    for index in indices:
        total = total + snapshots[int(index)]
    return (total / float(indices.size)).tocsr()


def _split_triangular_sparse_matrix(
    first_mean: sparse.spmatrix,
    second_mean: sparse.spmatrix,
) -> sparse.csr_matrix:
    upper = sparse.triu(first_mean, k=1, format="csr")
    lower = sparse.tril(second_mean, k=-1, format="csr")
    diagonal = sparse.diags(
        0.5 * (first_mean.diagonal() + second_mean.diagonal()),
        offsets=0,
        shape=first_mean.shape,
        format="csr",
    )
    return (upper + lower + diagonal).tocsr()


def _principal_component_pursuit(
    matrix: np.ndarray,
    *,
    tol: float,
    max_iterations: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    data = np.asarray(matrix, dtype=float)
    rows, cols = data.shape
    regularization = 1.0 / np.sqrt(max(rows, cols))
    one_norm = max(float(np.sum(np.abs(data))), 1e-12)
    mu = rows * cols / (4.0 * one_norm)
    low_rank = np.zeros_like(data)
    sparse_part = np.zeros_like(data)
    dual = np.zeros_like(data)
    data_norm = max(float(np.linalg.norm(data, ord="fro")), 1e-12)
    relative_residual = float("inf")
    iterations = max(1, int(max_iterations))

    for iteration in range(iterations):
        low_rank = _svd_threshold(data - sparse_part + dual / mu, 1.0 / mu)
        sparse_part = _soft_threshold(data - low_rank + dual / mu, regularization / mu)
        residual = data - low_rank - sparse_part
        dual = dual + mu * residual
        relative_residual = float(np.linalg.norm(residual, ord="fro") / data_norm)
        if relative_residual < tol:
            iterations = iteration + 1
            break

    return low_rank, sparse_part, {
        "iterations": iterations,
        "relative_residual": relative_residual,
        "mu": mu,
        "lambda": regularization,
    }


def _svd_threshold(matrix: np.ndarray, threshold: float) -> np.ndarray:
    left, singular_values, right_t = np.linalg.svd(matrix, full_matrices=False)
    shrunk = np.maximum(singular_values - threshold, 0.0)
    keep = shrunk > 0.0
    if not np.any(keep):
        return np.zeros_like(matrix)
    return (left[:, keep] * shrunk[keep]) @ right_t[keep]


def _soft_threshold(matrix: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(matrix) * np.maximum(np.abs(matrix) - threshold, 0.0)


def _project_temporal_weights(
    dense_snapshots: list[ArrayLikeSnapshot] | np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    temporal = np.empty((len(dense_snapshots), left.shape[1]), dtype=float)
    for t, snapshot in enumerate(dense_snapshots):
        temporal[t] = np.einsum("ij,ij->j", left, snapshot @ right)
    return temporal


def _project_temporal_weights_with_lambdas(
    dense_snapshots: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    lambdas: np.ndarray,
) -> np.ndarray:
    projected = _project_temporal_weights(dense_snapshots, left, right)
    safe_lambdas = np.where(np.abs(lambdas) > 1e-12, lambdas, 1.0)
    return projected / safe_lambdas


def _apply_rank_pruning(
    dense_snapshots: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    temporal: np.ndarray,
    lambdas: np.ndarray,
    config: SpectralCompressionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    mode = config.rank_pruning_mode
    if mode == "none":
        return left, right, temporal, lambdas, {}
    if mode != "ard_like":
        raise ValueError(f"unsupported rank_pruning_mode: {mode}")
    if config.rank_pruning_threshold < 0.0:
        raise ValueError("rank_pruning_threshold must be non-negative")

    min_rank = max(1, min(int(config.rank_pruning_min_rank), lambdas.shape[0]))
    iterations = max(1, int(config.rank_pruning_iterations))
    original_rank = int(lambdas.shape[0])
    history: list[dict[str, Any]] = []
    kept = np.arange(lambdas.shape[0])

    for iteration in range(iterations):
        temporal = _project_temporal_weights_with_lambdas(
            dense_snapshots,
            left,
            right,
            lambdas,
        )
        strengths = np.abs(lambdas) * np.sqrt(np.mean(temporal**2, axis=0))
        max_strength = float(np.max(strengths)) if strengths.size else 0.0
        threshold = float(config.rank_pruning_threshold * max(max_strength, 1e-12))
        eligible = np.flatnonzero(strengths >= threshold)
        if eligible.size < min_rank:
            eligible = np.argsort(-strengths)[:min_rank]
        local_kept = np.sort(eligible)
        history.append(
            {
                "iteration": iteration,
                "input_rank": int(lambdas.shape[0]),
                "output_rank": int(local_kept.shape[0]),
                "absolute_threshold": threshold,
                "component_strengths": strengths.tolist(),
                "kept_local_indices": local_kept.tolist(),
                "kept_original_indices": kept[local_kept].tolist(),
            }
        )
        if local_kept.shape[0] == lambdas.shape[0]:
            break
        left = left[:, local_kept]
        right = right[:, local_kept]
        temporal = temporal[:, local_kept]
        lambdas = lambdas[local_kept]
        kept = kept[local_kept]

    temporal = _project_temporal_weights_with_lambdas(
        dense_snapshots,
        left,
        right,
        lambdas,
    )
    strengths = np.abs(lambdas) * np.sqrt(np.mean(temporal**2, axis=0))
    max_strength = float(np.max(strengths)) if strengths.size else 0.0
    threshold = float(config.rank_pruning_threshold * max(max_strength, 1e-12))
    diagnostics = {
        "rank_pruning_mode": mode,
        "requested_rank": int(config.rank),
        "initial_effective_rank": original_rank,
        "effective_rank": int(lambdas.shape[0]),
        "rank_pruning_threshold": float(config.rank_pruning_threshold),
        "rank_pruning_absolute_threshold": threshold,
        "rank_pruning_min_rank": int(config.rank_pruning_min_rank),
        "rank_pruning_iterations": iterations,
        "rank_pruning_refit": True,
        "component_strengths": strengths.tolist(),
        "kept_component_indices": kept.tolist(),
        "rank_pruning_history": history,
    }
    return left, right, temporal, lambdas, diagnostics


def _initialize_tensor_factors(
    dense_snapshots: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_unfolding = dense_snapshots.transpose(1, 0, 2).reshape(dense_snapshots.shape[1], -1)
    target_unfolding = dense_snapshots.transpose(2, 0, 1).reshape(dense_snapshots.shape[2], -1)
    left = _truncated_left_singular_vectors(source_unfolding, rank)
    right = _truncated_left_singular_vectors(target_unfolding, rank)
    temporal = _project_temporal_weights(dense_snapshots, left, right)
    return left, right, temporal


def _store_from_tensor_factors(
    left: np.ndarray,
    right: np.ndarray,
    temporal: np.ndarray,
    config: SpectralCompressionConfig,
    *,
    method: str,
    extra_diagnostics: dict[str, Any] | None = None,
) -> FactorizedTemporalStore:
    left, right, temporal, lambdas = _normalize_tensor_components(left, right, temporal)
    left, right, temporal, lambdas, kept = _prune_tensor_components(
        left,
        right,
        temporal,
        lambdas,
        config,
    )
    return FactorizedTemporalStore(
        left=left,
        right=right,
        temporal=temporal,
        lambdas=lambdas,
        threshold_diagnostics={
            "tensor_method": method,
            "requested_rank": int(config.rank),
            "effective_rank": int(lambdas.shape[0]),
            "tensor_rank_energy": float(config.tensor_rank_energy),
            "tensor_min_rank": int(config.tensor_min_rank),
            "kept_component_indices": kept.tolist(),
            **(extra_diagnostics or {}),
        },
    )


def _normalize_tensor_components(
    left: np.ndarray,
    right: np.ndarray,
    temporal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left = left.copy()
    right = right.copy()
    temporal = temporal.copy()
    lambdas = np.empty(left.shape[1], dtype=float)
    for component in range(left.shape[1]):
        left_norm = max(float(np.linalg.norm(left[:, component])), 1e-12)
        right_norm = max(float(np.linalg.norm(right[:, component])), 1e-12)
        temporal_norm = max(float(np.linalg.norm(temporal[:, component])), 1e-12)
        left[:, component] /= left_norm
        right[:, component] /= right_norm
        temporal[:, component] /= temporal_norm
        lambdas[component] = left_norm * right_norm * temporal_norm
    return left, right, temporal, lambdas


def _prune_tensor_components(
    left: np.ndarray,
    right: np.ndarray,
    temporal: np.ndarray,
    lambdas: np.ndarray,
    config: SpectralCompressionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    energy_target = float(config.tensor_rank_energy)
    if energy_target <= 0.0 or energy_target > 1.0:
        raise ValueError("tensor_rank_energy must be in the interval (0, 1]")

    order = np.argsort(-(lambdas**2))
    if energy_target >= 1.0:
        keep_count = lambdas.shape[0]
    else:
        ordered_energy = lambdas[order] ** 2
        total_energy = float(np.sum(ordered_energy))
        if total_energy <= 1e-12:
            keep_count = max(1, min(config.tensor_min_rank, lambdas.shape[0]))
        else:
            cumulative = np.cumsum(ordered_energy) / total_energy
            keep_count = int(np.searchsorted(cumulative, energy_target, side="left") + 1)
        keep_count = max(int(config.tensor_min_rank), keep_count)
        keep_count = min(keep_count, lambdas.shape[0])

    kept = np.sort(order[:keep_count])
    return (
        left[:, kept],
        right[:, kept],
        temporal[:, kept],
        lambdas[kept],
        kept,
    )


def _khatri_rao(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    columns = [
        np.kron(first[:, component], second[:, component])
        for component in range(first.shape[1])
    ]
    return np.column_stack(columns)


def _als_update(
    unfolding: np.ndarray,
    design: np.ndarray,
    first_factor: np.ndarray,
    second_factor: np.ndarray,
    ridge: float,
) -> np.ndarray:
    gram = (first_factor.T @ first_factor) * (second_factor.T @ second_factor)
    gram = gram + ridge * np.eye(gram.shape[0])
    return unfolding @ design @ np.linalg.pinv(gram)


def _normalize_cp_factors(
    left: np.ndarray,
    right: np.ndarray,
    temporal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for component in range(left.shape[1]):
        left_norm = max(float(np.linalg.norm(left[:, component])), 1e-12)
        right_norm = max(float(np.linalg.norm(right[:, component])), 1e-12)
        left[:, component] /= left_norm
        right[:, component] /= right_norm
        temporal[:, component] *= left_norm * right_norm
    return left, right, temporal


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

    base_threshold = threshold
    threshold = float(threshold * config.residual_threshold_scale)
    diagnostics = {
        **diagnostics,
        "base_threshold": base_threshold,
        "residual_threshold_scale": float(config.residual_threshold_scale),
        "estimated_threshold": threshold,
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
    config: SpectralCompressionConfig,
) -> dict[str, Any]:
    source_degree_scale, target_degree_scale = _degree_scales(dense_snapshots)
    edge_scale = 0.5 * (source_degree_scale[:, None] + target_degree_scale[None, :])
    residual_dense = np.stack([residual.toarray() for residual in residuals])
    omitted_residual = residual_stack - residual_dense
    per_entry_scale = np.abs(omitted_residual) / np.maximum(edge_scale[None, :, :], 1e-12)
    coverage = float(np.clip(config.entrywise_bound_coverage, 0.0, 1.0))
    entrywise_bound_scale = float(np.quantile(per_entry_scale, coverage))
    return {
        "source_degree_scale": source_degree_scale,
        "target_degree_scale": target_degree_scale,
        "entrywise_bound_scale": entrywise_bound_scale,
        "diagnostics": {
            "entrywise_bound_coverage_target": coverage,
            "degree_aware_bound_scale": entrywise_bound_scale,
            "degree_aware_bound_scale_max": float(np.max(per_entry_scale)),
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
