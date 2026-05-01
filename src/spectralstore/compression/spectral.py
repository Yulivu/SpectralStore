"""Spectral compressors for temporal graph snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from scipy.special import digamma, gammaln
from scipy.sparse.linalg import svds

from spectralstore.compression.factorized_store import (
    FactorizedTemporalStore,
    ResidualStore,
    TemporalCOOResidualStore,
)


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
    asym_split_seed: int | None = None
    asym_split_mode: str = "random"
    asym_diagonal_mode: str = "average"
    robust_convergence_tol: float = 1e-6
    residual_storage_format: str = "csr"
    max_sparse_ratio: float | None = None
    storage_gate_action: str = "diagnostic"
    factor_storage_dtype_bytes: int | None = None
    sparse_native_enabled: bool = False
    sparse_native_max_dense_fallback_nodes: int = 5000
    rank_selection_mode: str = "fixed"
    ard_max_rank: int | None = None
    ard_prior_alpha: float = 1e-2
    ard_prior_beta: float = 1e-2
    ard_max_iterations: int = 100
    ard_tolerance: float = 1e-6
    ard_min_effective_ratio: float = 0.05
    ard_min_rank: int = 1
    ard_noise_floor: float = 1e-8
    ard_fail_on_nonconvergence: bool = False
    thinking_tensor_blend: float = 0.35
    thinking_tensor_rank: int | None = None


class AsymmetricSpectralCompressor:
    """First SpectralStore compressor using asymmetric split-snapshot SVD."""

    def __init__(self, config: SpectralCompressionConfig | None = None) -> None:
        self.config = config or SpectralCompressionConfig()

    def fit_transform(self, snapshots: list[ArrayLikeSnapshot]) -> FactorizedTemporalStore:
        if self.config.sparse_native_enabled and _all_sparse_snapshots(snapshots):
            sparse_snapshots = _as_sparse_snapshots(snapshots)
            if len(sparse_snapshots) < 2:
                raise ValueError("at least two temporal snapshots are required")
            basis = _asymmetric_basis_sparse(sparse_snapshots, self.config)
            return _factorize_from_basis(sparse_snapshots, basis, self.config)

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
        if self.config.sparse_native_enabled and _all_sparse_snapshots(snapshots):
            sparse_snapshots = _as_sparse_snapshots(snapshots)
            cleaned_snapshots = [snapshot.copy() for snapshot in sparse_snapshots]
            iterations = max(1, self.config.robust_iterations)

            for _ in range(iterations):
                basis = _asymmetric_basis_sparse(cleaned_snapshots, self.config)
                store = _factorize_from_basis(cleaned_snapshots, basis, self.config, residuals=())
                residual_arrays = _residual_arrays_from_snapshots(sparse_snapshots, store)
                sparse_residuals, _ = _threshold_residuals(residual_arrays, self.config)
                cleaned_snapshots = _subtract_sparse_residuals(sparse_snapshots, sparse_residuals)

            basis = _asymmetric_basis_sparse(cleaned_snapshots, self.config)
            final_store = _factorize_from_basis(cleaned_snapshots, basis, self.config, residuals=())
            final_residual_arrays = _residual_arrays_from_snapshots(sparse_snapshots, final_store)
            final_residuals, diagnostics = _threshold_residuals(final_residual_arrays, self.config)
            bound_metadata = _degree_aware_bound_metadata(
                sparse_snapshots,
                final_residual_arrays,
                final_residuals,
                self.config,
            )
            residual_store, storage_diagnostics = _residual_store_from_config(
                final_residuals,
                self.config,
            )
            store = FactorizedTemporalStore(
                left=final_store.left,
                right=final_store.right,
                temporal=final_store.temporal,
                lambdas=final_store.lambdas,
                residuals=residual_store,
                threshold_diagnostics={
                    **diagnostics,
                    **(final_store.threshold_diagnostics or {}),
                    **bound_metadata["diagnostics"],
                    **storage_diagnostics,
                },
                source_degree_scale=bound_metadata["source_degree_scale"],
                target_degree_scale=bound_metadata["target_degree_scale"],
                entrywise_bound_scale=bound_metadata["entrywise_bound_scale"],
            )
            return _apply_storage_gate(store, sparse_snapshots, self.config)

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
        residual_store, storage_diagnostics = _residual_store_from_config(
            final_residuals,
            self.config,
        )
        store = FactorizedTemporalStore(
            left=final_store.left,
            right=final_store.right,
            temporal=final_store.temporal,
            lambdas=final_store.lambdas,
            residuals=residual_store,
            threshold_diagnostics={
                **diagnostics,
                **(final_store.threshold_diagnostics or {}),
                **bound_metadata["diagnostics"],
                **storage_diagnostics,
            },
            source_degree_scale=bound_metadata["source_degree_scale"],
            target_degree_scale=bound_metadata["target_degree_scale"],
            entrywise_bound_scale=bound_metadata["entrywise_bound_scale"],
        )
        return _apply_storage_gate(store, _as_sparse_snapshots(snapshots), self.config)


class AlternatingRobustAsymmetricSpectralCompressor:
    """Thinking-aligned alternating robust variant of the asymmetric compressor."""

    def __init__(self, config: SpectralCompressionConfig | None = None) -> None:
        self.config = config or SpectralCompressionConfig()

    def fit_transform(self, snapshots: list[ArrayLikeSnapshot]) -> FactorizedTemporalStore:
        dense = _as_dense_stack(snapshots)
        final_state = _run_alternating_robust_asymmetric(
            dense,
            self.config,
            sparse_snapshots=_as_sparse_snapshots(snapshots),
        )
        return final_state["store"]


class SymmetricSVDCompressor:
    """Baseline that symmetrizes the mean adjacency matrix before SVD."""

    def __init__(self, config: SpectralCompressionConfig | None = None) -> None:
        self.config = config or SpectralCompressionConfig()

    def fit_transform(self, snapshots: list[ArrayLikeSnapshot]) -> FactorizedTemporalStore:
        if self.config.sparse_native_enabled and _all_sparse_snapshots(snapshots):
            sparse_snapshots = _as_sparse_snapshots(snapshots)
            mean_sparse = _mean_sparse_snapshots(
                sparse_snapshots,
                np.arange(len(sparse_snapshots), dtype=int),
            )
            sym_mean = 0.5 * (mean_sparse + mean_sparse.T)
            return _factorize_from_basis(sparse_snapshots, sym_mean.toarray(), self.config)

        dense = _as_dense_stack(snapshots)
        mean = dense.mean(axis=0)
        sym_mean = 0.5 * (mean + mean.T)
        return _factorize_from_basis(dense, sym_mean, self.config)


class DirectSVDCompressor:
    """Baseline that applies SVD directly to the mean adjacency matrix."""

    def __init__(self, config: SpectralCompressionConfig | None = None) -> None:
        self.config = config or SpectralCompressionConfig()

    def fit_transform(self, snapshots: list[ArrayLikeSnapshot]) -> FactorizedTemporalStore:
        if self.config.sparse_native_enabled and _all_sparse_snapshots(snapshots):
            sparse_snapshots = _as_sparse_snapshots(snapshots)
            mean_sparse = _mean_sparse_snapshots(
                sparse_snapshots,
                np.arange(len(sparse_snapshots), dtype=int),
            )
            return _factorize_from_basis(sparse_snapshots, mean_sparse.toarray(), self.config)

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
            extra_diagnostics={
                "source_unfolding_shape": list(source_unfolding.shape),
                "target_unfolding_shape": list(target_unfolding.shape),
                "source_unfolding_nnz": int(np.count_nonzero(source_unfolding)),
                "target_unfolding_nnz": int(np.count_nonzero(target_unfolding)),
            },
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

        rank_upper_bound = _rank_upper_bound(self.config, min(stitched.shape))
        left, singular_values, right_t = _truncated_sparse_svd(
            stitched,
            rank_upper_bound,
            self.config.random_seed,
        )
        left, right, temporal, lambdas, rank_selection_diagnostics = _factorize_svd_components(
            sparse_snapshots,
            left_full=left,
            singular_values=singular_values,
            right_t_full=right_t,
            config=self.config,
            rank_upper_bound=rank_upper_bound,
        )

        return FactorizedTemporalStore(
            left=left,
            right=right,
            temporal=temporal,
            lambdas=lambdas,
            threshold_diagnostics={
                "tensor_method": "split_asym_unfolding",
                "requested_rank": int(self.config.rank),
                "effective_rank": int(lambdas.shape[0]),
                "split_first_size": int(first_indices.size),
                "split_second_size": int(second_indices.size),
                "stitched_shape": list(stitched.shape),
                "stitched_nnz": int(stitched.nnz),
                **rank_selection_diagnostics,
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


def _as_sparse_snapshots(snapshots: list[ArrayLikeSnapshot]) -> list[sparse.csr_matrix]:
    return [
        snapshot.tocsr() if sparse.issparse(snapshot) else sparse.csr_matrix(snapshot)
        for snapshot in snapshots
    ]


def _all_sparse_snapshots(snapshots: list[ArrayLikeSnapshot]) -> bool:
    return bool(snapshots) and all(sparse.issparse(snapshot) for snapshot in snapshots)


def _csr_residual_bytes(residuals: tuple[sparse.csr_matrix, ...]) -> int:
    return int(
        sum(
            residual.data.nbytes + residual.indices.nbytes + residual.indptr.nbytes
            for residual in residuals
        )
    )


def _residual_store_from_config(
    residuals: tuple[sparse.csr_matrix, ...],
    config: SpectralCompressionConfig,
) -> tuple[ResidualStore, dict[str, Any]]:
    mode = config.residual_storage_format
    if mode not in {"csr", "temporal_coo", "auto"}:
        raise ValueError(f"unsupported residual_storage_format: {mode}")

    csr_bytes = _csr_residual_bytes(residuals)
    coo_store = TemporalCOOResidualStore.from_csr_residuals(residuals)
    coo_bytes = coo_store.nbytes

    if mode == "csr":
        store: ResidualStore = residuals
        selected = "csr"
        selected_bytes = csr_bytes
    elif mode == "temporal_coo" or coo_bytes <= csr_bytes:
        store = coo_store
        selected = "temporal_coo"
        selected_bytes = coo_bytes
    else:
        store = residuals
        selected = "csr"
        selected_bytes = csr_bytes

    return store, {
        "residual_storage_format": selected,
        "residual_storage_mode_requested": mode,
        "residual_storage_bytes": int(selected_bytes),
        "residual_storage_csr_bytes": int(csr_bytes),
        "residual_storage_temporal_coo_bytes": int(coo_bytes),
    }


def _copy_store_with(store: FactorizedTemporalStore, **updates: Any) -> FactorizedTemporalStore:
    values = {
        "left": store.left,
        "right": store.right,
        "temporal": store.temporal,
        "lambdas": store.lambdas,
        "residuals": store.residuals,
        "threshold_diagnostics": store.threshold_diagnostics,
        "source_degree_scale": store.source_degree_scale,
        "target_degree_scale": store.target_degree_scale,
        "entrywise_bound_scale": store.entrywise_bound_scale,
        "bound_sigma_max": store.bound_sigma_max,
        "bound_mu": store.bound_mu,
        "bound_constant": store.bound_constant,
    }
    values.update(updates)
    return FactorizedTemporalStore(**values)


def _apply_storage_gate(
    store: FactorizedTemporalStore,
    snapshots: list[sparse.csr_matrix],
    config: SpectralCompressionConfig,
) -> FactorizedTemporalStore:
    if config.storage_gate_action not in {"diagnostic", "drop_residual", "raise"}:
        raise ValueError(f"unsupported storage_gate_action: {config.storage_gate_action}")

    sparse_ratio = store.compressed_vs_raw_sparse_ratio(
        snapshots,
        factor_dtype_bytes=config.factor_storage_dtype_bytes,
    )
    diagnostics = {
        **(store.threshold_diagnostics or {}),
        "storage_gate_sparse_ratio": float(sparse_ratio),
        "storage_gate_max_sparse_ratio": (
            None if config.max_sparse_ratio is None else float(config.max_sparse_ratio)
        ),
        "storage_gate_action": config.storage_gate_action,
        "storage_gate_factor_dtype_bytes": config.factor_storage_dtype_bytes,
        "storage_gate_accepted": (
            True
            if config.max_sparse_ratio is None
            else bool(sparse_ratio <= float(config.max_sparse_ratio))
        ),
        "storage_gate_action_taken": "none",
    }
    if config.max_sparse_ratio is None or sparse_ratio <= float(config.max_sparse_ratio):
        return _copy_store_with(store, threshold_diagnostics=diagnostics)

    if config.storage_gate_action == "raise":
        raise ValueError(
            "compressed representation exceeds sparse storage budget: "
            f"{sparse_ratio:.6g} > {float(config.max_sparse_ratio):.6g}"
        )
    if config.storage_gate_action == "drop_residual" and store.residuals:
        dropped = _copy_store_with(store, residuals=(), threshold_diagnostics=diagnostics)
        dropped_ratio = dropped.compressed_vs_raw_sparse_ratio(
            snapshots,
            factor_dtype_bytes=config.factor_storage_dtype_bytes,
        )
        return _copy_store_with(
            dropped,
            threshold_diagnostics={
                **diagnostics,
                "storage_gate_action_taken": "drop_residual",
                "storage_gate_sparse_ratio_after_action": float(dropped_ratio),
                "storage_gate_accepted_after_action": bool(
                    dropped_ratio <= float(config.max_sparse_ratio)
                ),
            },
        )

    return _copy_store_with(
        store,
        threshold_diagnostics={
            **diagnostics,
            "storage_gate_action_taken": "diagnostic_only",
        },
    )


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


def _asymmetric_basis_sparse(
    sparse_snapshots: list[sparse.csr_matrix],
    config: SpectralCompressionConfig,
) -> np.ndarray:
    rng = np.random.default_rng(config.random_seed)
    stitched = sparse.csr_matrix(sparse_snapshots[0].shape, dtype=float)
    num_splits = max(1, config.num_splits)
    for _ in range(num_splits):
        order = rng.permutation(len(sparse_snapshots))
        split = max(1, len(sparse_snapshots) // 2)
        first_indices = order[:split]
        second_indices = order[split:]
        if second_indices.size == 0:
            second_indices = first_indices
        first_mean = _mean_sparse_snapshots(sparse_snapshots, first_indices)
        second_mean = _mean_sparse_snapshots(sparse_snapshots, second_indices)
        stitched = stitched + _split_triangular_sparse_matrix(first_mean, second_mean)
    return (stitched / float(num_splits)).toarray()


def _run_alternating_robust_asymmetric(
    dense_snapshots: np.ndarray,
    config: SpectralCompressionConfig,
    *,
    sparse_snapshots: list[sparse.csr_matrix] | None = None,
) -> dict[str, Any]:
    if dense_snapshots.shape[0] < 2:
        raise ValueError("at least two temporal snapshots are required")
    if dense_snapshots.shape[1] != dense_snapshots.shape[2]:
        raise ValueError("alternating robust asym compressor requires square snapshots")

    sparse_estimate = np.zeros_like(dense_snapshots)
    previous_low_rank_stack: np.ndarray | None = None
    iterations = max(1, int(config.robust_iterations))
    history: list[dict[str, Any]] = []
    final_store: FactorizedTemporalStore | None = None
    final_low_rank_stack: np.ndarray | None = None
    final_residual_stack: np.ndarray | None = None
    final_sparse_residuals: tuple[sparse.csr_matrix, ...] = ()

    for iteration in range(1, iterations + 1):
        clean_snapshots = dense_snapshots - sparse_estimate
        basis, construction = _thinking_asymmetric_basis(clean_snapshots, config)
        iteration_store = _factorize_with_iteration_state(clean_snapshots, basis, config)
        low_rank_stack = _store_dense_stack(iteration_store)
        residual_stack = dense_snapshots - low_rank_stack
        sparse_residuals, threshold_diagnostics = _threshold_residuals(residual_stack, config)
        new_sparse_estimate = np.stack([residual.toarray() for residual in sparse_residuals])
        sparse_store = FactorizedTemporalStore(
            left=iteration_store.left,
            right=iteration_store.right,
            temporal=iteration_store.temporal,
            lambdas=iteration_store.lambdas,
            residuals=sparse_residuals,
            threshold_diagnostics=iteration_store.threshold_diagnostics,
        )

        residual_bytes = int(
            sum(
                residual.data.nbytes + residual.indices.nbytes + residual.indptr.nbytes
                for residual in sparse_residuals
            )
        )
        residual_nnz = int(sum(residual.nnz for residual in sparse_residuals))
        reconstruction_change = float(
            _stack_fro_norm(new_sparse_estimate - sparse_estimate)
            / max(_stack_fro_norm(sparse_estimate), 1e-12)
        )
        low_rank_change = (
            float(
                _stack_fro_norm(low_rank_stack - previous_low_rank_stack)
                / max(_stack_fro_norm(previous_low_rank_stack), 1e-12)
            )
            if previous_low_rank_stack is not None
            else float("nan")
        )
        converged = (
            previous_low_rank_stack is not None
            and low_rank_change <= float(config.robust_convergence_tol)
        )
        decomposition = _iteration_decomposition_report(iteration_store, low_rank_stack)
        history.append(
            {
                "iteration": iteration,
                **construction,
                **decomposition,
                "base_threshold": float(threshold_diagnostics["base_threshold"]),
                "estimated_threshold": float(threshold_diagnostics["estimated_threshold"]),
                "threshold_mode": str(threshold_diagnostics["mode"]),
                "residual_nnz": residual_nnz,
                "residual_sparsity": float(threshold_diagnostics["residual_sparsity"]),
                "residual_bytes": residual_bytes,
                "reconstruction_change": reconstruction_change,
                "low_rank_change": low_rank_change,
                "converged": converged,
                "left": iteration_store.left.tolist(),
                "right": iteration_store.right.tolist(),
                "temporal": iteration_store.temporal.tolist(),
                "lambdas": iteration_store.lambdas.tolist(),
            }
        )
        sparse_estimate = new_sparse_estimate
        previous_low_rank_stack = low_rank_stack
        final_store = sparse_store
        final_low_rank_stack = low_rank_stack
        final_residual_stack = residual_stack
        final_sparse_residuals = sparse_residuals

    assert final_store is not None
    assert final_low_rank_stack is not None
    assert final_residual_stack is not None
    bound_metadata = _degree_aware_bound_metadata(
        dense_snapshots,
        final_residual_stack,
        final_sparse_residuals,
        config,
    )
    final_history = history[-1]
    threshold_diagnostics = {
        "method_name": "spectralstore_asym_alternating_robust",
        "robust_iterations": iterations,
        "threshold_scale": float(config.residual_threshold_scale),
        "convergence_tol": float(config.robust_convergence_tol),
        "split_seed": int(_asym_split_seed(config)),
        "split_mode": config.asym_split_mode,
        "diagonal_mode": config.asym_diagonal_mode,
        "construction_asymmetry_norm": final_history["construction_asymmetry_norm"],
        "output_asymmetry_norm": final_history["output_asymmetry_norm"],
        "uv_gap": final_history["uv_gap"],
        "alternating_history": history,
        "final_iteration": iterations,
        "converged": bool(final_history["converged"]),
        "final_reconstruction_change": final_history["reconstruction_change"],
        "final_low_rank_change": final_history["low_rank_change"],
        **(final_store.threshold_diagnostics or {}),
        **bound_metadata["diagnostics"],
    }
    residual_store, storage_diagnostics = _residual_store_from_config(
        final_sparse_residuals,
        config,
    )
    store = FactorizedTemporalStore(
        left=final_store.left,
        right=final_store.right,
        temporal=final_store.temporal,
        lambdas=final_store.lambdas,
        residuals=residual_store,
        threshold_diagnostics=threshold_diagnostics,
        source_degree_scale=bound_metadata["source_degree_scale"],
        target_degree_scale=bound_metadata["target_degree_scale"],
        entrywise_bound_scale=bound_metadata["entrywise_bound_scale"],
    )
    store = _copy_store_with(
        store,
        threshold_diagnostics={
            **(store.threshold_diagnostics or {}),
            **storage_diagnostics,
        },
    )
    if sparse_snapshots is not None:
        store = _apply_storage_gate(store, sparse_snapshots, config)
    return {
        "store": store,
        "history": history,
    }


def _asym_split_seed(config: SpectralCompressionConfig) -> int:
    return int(config.random_seed if config.asym_split_seed is None else config.asym_split_seed)


def _thinking_asymmetric_basis(
    dense_snapshots: np.ndarray,
    config: SpectralCompressionConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    if config.asym_split_mode != "random":
        raise ValueError(f"unsupported asym_split_mode: {config.asym_split_mode}")
    if config.asym_diagonal_mode != "average":
        raise ValueError(f"unsupported asym_diagonal_mode: {config.asym_diagonal_mode}")

    rng = np.random.default_rng(_asym_split_seed(config))
    order = rng.permutation(dense_snapshots.shape[0])
    split = max(1, dense_snapshots.shape[0] // 2)
    first_indices = order[:split]
    second_indices = order[split:]
    if second_indices.size == 0:
        second_indices = first_indices

    first_mean = dense_snapshots[first_indices].mean(axis=0)
    second_mean = dense_snapshots[second_indices].mean(axis=0)
    stitched = _thinking_triangular_matrix(first_mean, second_mean)
    tensor_basis, tensor_diag = _thinking_tensor_basis(dense_snapshots, config)
    blend = float(np.clip(config.thinking_tensor_blend, 0.0, 1.0))
    combined = (1.0 - blend) * stitched + blend * tensor_basis
    upper = np.triu(np.ones_like(stitched, dtype=bool), k=1)
    lower = np.tril(np.ones_like(stitched, dtype=bool), k=-1)
    diag_target = 0.5 * (np.diag(first_mean) + np.diag(second_mean))
    diagnostics = {
        "T1_size": int(first_indices.size),
        "T2_size": int(second_indices.size),
        "construction_asymmetry_norm": relative_asymmetry_norm(combined),
        "upper_source_error": float(np.max(np.abs(combined[upper] - first_mean[upper]))),
        "lower_source_error": float(np.max(np.abs(combined[lower] - second_mean[lower]))),
        "diag_error": float(np.max(np.abs(np.diag(combined) - diag_target))),
        "thinking_tensor_blend": blend,
        **tensor_diag,
    }
    return combined, diagnostics


def _thinking_triangular_matrix(first_mean: np.ndarray, second_mean: np.ndarray) -> np.ndarray:
    stitched = np.triu(first_mean, k=1) + np.tril(second_mean, k=-1)
    np.fill_diagonal(stitched, 0.5 * (np.diag(first_mean) + np.diag(second_mean)))
    return stitched


def _thinking_tensor_basis(
    dense_snapshots: np.ndarray,
    config: SpectralCompressionConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    n = dense_snapshots.shape[1]
    requested_rank = (
        int(config.thinking_tensor_rank)
        if config.thinking_tensor_rank is not None
        else int(config.rank)
    )
    rank = max(1, min(requested_rank, n))
    source_unfolding = dense_snapshots.transpose(1, 0, 2).reshape(n, -1)
    target_unfolding = dense_snapshots.transpose(2, 0, 1).reshape(n, -1)
    left = _truncated_left_singular_vectors(source_unfolding, rank)
    right = _truncated_left_singular_vectors(target_unfolding, rank)
    temporal = _project_temporal_weights(dense_snapshots, left, right)
    lambdas = np.sqrt(np.maximum(np.mean(temporal**2, axis=0), 1e-12))
    normalized_temporal = temporal / np.where(lambdas > 1e-12, lambdas, 1.0)
    tensor_basis = np.zeros((n, n), dtype=float)
    for t in range(dense_snapshots.shape[0]):
        weights = lambdas * normalized_temporal[t]
        tensor_basis += (left * weights) @ right.T
    tensor_basis /= float(dense_snapshots.shape[0])
    return tensor_basis, {
        "thinking_tensor_rank": int(rank),
        "thinking_tensor_energy_mean": float(np.mean(lambdas)),
    }


def _factorize_with_iteration_state(
    dense_snapshots: np.ndarray,
    basis: np.ndarray,
    config: SpectralCompressionConfig,
) -> FactorizedTemporalStore:
    rank_upper_bound = _rank_upper_bound(config, min(basis.shape))
    left_full, singular_values, right_t_full = np.linalg.svd(basis, full_matrices=False)
    left, right, temporal, lambdas, rank_selection_diagnostics = _factorize_svd_components(
        dense_snapshots,
        left_full=left_full,
        singular_values=singular_values,
        right_t_full=right_t_full,
        config=config,
        rank_upper_bound=rank_upper_bound,
    )
    left, right, temporal, lambdas, pruning_diagnostics = _apply_rank_pruning(
        dense_snapshots,
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
        threshold_diagnostics=(
            {
                **rank_selection_diagnostics,
                **pruning_diagnostics,
            }
            if (rank_selection_diagnostics or pruning_diagnostics)
            else None
        ),
    )


def _store_dense_stack(store: FactorizedTemporalStore) -> np.ndarray:
    return np.stack(
        [store.dense_snapshot(t, include_residual=False) for t in range(store.num_steps)]
    )


def _stack_fro_norm(values: np.ndarray) -> float:
    return float(np.sqrt(np.sum(np.asarray(values, dtype=float) ** 2)))


def _iteration_decomposition_report(
    store: FactorizedTemporalStore,
    low_rank_stack: np.ndarray,
) -> dict[str, Any]:
    uv_gap = float(
        np.linalg.norm(store.left - store.right, ord="fro")
        / max(np.linalg.norm(store.left, ord="fro"), 1e-12)
    )
    asymmetries = [relative_asymmetry_norm(matrix) for matrix in low_rank_stack]
    return {
        "singular_values_summary": ";".join(
            f"{float(value):.12g}" for value in np.asarray(store.lambdas, dtype=float)
        ),
        "uv_gap": uv_gap,
        "output_asymmetry_norm": float(np.mean(asymmetries)),
        "subspace_distance": _column_subspace_distance(store.left, store.right),
    }


def _column_subspace_distance(first: np.ndarray, second: np.ndarray) -> float:
    rank = min(first.shape[1], second.shape[1])
    if rank <= 0:
        return float("nan")
    first_q, _ = np.linalg.qr(first[:, :rank])
    second_q, _ = np.linalg.qr(second[:, :rank])
    first_projection = first_q @ first_q.T
    second_projection = second_q @ second_q.T
    return float(
        np.linalg.norm(first_projection - second_projection, ord="fro") / np.sqrt(2 * rank)
    )


def relative_asymmetry_norm(matrix: np.ndarray) -> float:
    return float(
        np.linalg.norm(matrix - matrix.T, ord="fro")
        / max(np.linalg.norm(matrix, ord="fro"), 1e-12)
    )


def _rank_upper_bound(config: SpectralCompressionConfig, max_rank: int) -> int:
    if max_rank <= 0:
        raise ValueError("max_rank must be positive")
    if config.rank_selection_mode == "fixed":
        return max(1, min(int(config.rank), max_rank))
    if config.rank_selection_mode != "ard":
        raise ValueError(
            f"unsupported rank_selection_mode: {config.rank_selection_mode}. "
            "Expected one of: fixed, ard"
        )
    requested = (
        int(config.ard_max_rank)
        if config.ard_max_rank is not None
        else int(config.rank)
    )
    return max(1, min(requested, max_rank))


def _factorize_svd_components(
    snapshots: list[ArrayLikeSnapshot] | np.ndarray,
    *,
    left_full: np.ndarray,
    singular_values: np.ndarray,
    right_t_full: np.ndarray,
    config: SpectralCompressionConfig,
    rank_upper_bound: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    selected_values = np.asarray(singular_values[:rank_upper_bound], dtype=float)
    if selected_values.size == 0:
        raise ValueError("rank selection received an empty singular value vector")
    left_candidates = left_full[:, :rank_upper_bound]
    right_candidates = right_t_full[:rank_upper_bound, :].T

    if config.rank_selection_mode == "fixed":
        lambdas = selected_values.copy()
        temporal = _project_temporal_weights_with_lambdas(
            snapshots,
            left_candidates,
            right_candidates,
            lambdas,
        )
        return left_candidates, right_candidates, temporal, lambdas, {}

    selected_indices, lambdas, temporal, diagnostics, refined_left, refined_right = _select_rank_components(
        snapshots,
        left=left_candidates,
        right=right_candidates,
        singular_values=selected_values,
        config=config,
        rank_upper_bound=rank_upper_bound,
    )
    left = refined_left[:, selected_indices]
    right = refined_right[:, selected_indices]
    return left, right, temporal, lambdas, diagnostics


def _select_rank_components(
    snapshots: list[ArrayLikeSnapshot] | np.ndarray,
    *,
    left: np.ndarray,
    right: np.ndarray,
    singular_values: np.ndarray,
    config: SpectralCompressionConfig,
    rank_upper_bound: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    selected_values = np.asarray(singular_values[:rank_upper_bound], dtype=float)
    if selected_values.size == 0:
        raise ValueError("rank selection received an empty singular value vector")
    if config.rank_selection_mode == "fixed":
        indices = np.arange(selected_values.shape[0], dtype=int)
        temporal = _project_temporal_weights_with_lambdas(
            snapshots,
            left[:, indices],
            right[:, indices],
            selected_values,
        )
        return indices, selected_values.copy(), temporal, {}, left, right

    means, variances, alphas, elbo_history, converged, sigma2, refined_left, refined_right = _ard_variational_shrinkage(
        snapshots,
        left=left,
        right=right,
        singular_values=selected_values,
        config=config,
    )
    temporal_full = _project_temporal_weights_with_lambdas(
        snapshots,
        refined_left,
        refined_right,
        means,
    )
    raw_strength = np.abs(means) * np.sqrt(np.maximum(np.mean(temporal_full**2, axis=0), 0.0))
    precision_weight = 1.0 / np.sqrt(np.maximum(alphas, 1e-12))
    strength = raw_strength * precision_weight
    max_strength = float(np.max(strength)) if strength.size else 0.0
    absolute_threshold = float(
        max(float(config.ard_min_effective_ratio), 0.0) * max(max_strength, 1e-12)
    )
    keep = np.flatnonzero(strength >= absolute_threshold)
    min_rank = max(1, min(int(config.ard_min_rank), selected_values.shape[0]))
    if keep.size < min_rank:
        keep = np.argsort(-strength)[:min_rank]
    keep = np.sort(keep.astype(int, copy=False))
    if bool(config.ard_fail_on_nonconvergence) and not converged:
        raise ValueError(
            "ARD rank selection failed to converge within "
            f"{int(config.ard_max_iterations)} iterations."
        )

    kept_means = means[keep]
    kept_means = np.where(np.abs(kept_means) > 1e-12, kept_means, selected_values[keep])
    temporal = temporal_full[:, keep]
    diagnostics = {
        "rank_selection_mode": "ard",
        "ard_model_formulation": "joint_temporal_residual_variational",
        "requested_rank": int(config.rank),
        "rank_selection_upper_bound": int(rank_upper_bound),
        "initial_effective_rank": int(selected_values.shape[0]),
        "effective_rank": int(keep.shape[0]),
        "ard_converged": bool(converged),
        "ard_iterations": int(len(elbo_history)),
        "ard_prior_alpha": float(config.ard_prior_alpha),
        "ard_prior_beta": float(config.ard_prior_beta),
        "ard_noise_floor": float(config.ard_noise_floor),
        "ard_noise_variance": float(sigma2),
        "ard_min_effective_ratio": float(config.ard_min_effective_ratio),
        "ard_min_rank": int(config.ard_min_rank),
        "ard_absolute_threshold": absolute_threshold,
        "ard_component_strengths": strength.tolist(),
        "ard_component_raw_strengths": raw_strength.tolist(),
        "ard_component_alphas": alphas.tolist(),
        "ard_selected_indices": keep.tolist(),
        "ard_selected_lambdas": kept_means.tolist(),
        "ard_elbo_history": elbo_history,
    }
    return keep, kept_means, temporal, diagnostics, refined_left, refined_right


def _ard_variational_shrinkage(
    snapshots: list[ArrayLikeSnapshot] | np.ndarray,
    *,
    left: np.ndarray,
    right: np.ndarray,
    singular_values: np.ndarray,
    config: SpectralCompressionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float], bool, float, np.ndarray, np.ndarray]:
    y = np.maximum(np.asarray(singular_values, dtype=float), 0.0)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("ARD shrinkage expects a non-empty singular value vector")
    if config.ard_prior_alpha <= 0.0 or config.ard_prior_beta <= 0.0:
        raise ValueError("ard_prior_alpha and ard_prior_beta must be positive")
    if config.ard_max_iterations <= 0:
        raise ValueError("ard_max_iterations must be positive")
    if config.ard_tolerance <= 0.0:
        raise ValueError("ard_tolerance must be positive")
    if config.ard_noise_floor <= 0.0:
        raise ValueError("ard_noise_floor must be positive")

    left_iter = np.asarray(left, dtype=float).copy()
    right_iter = np.asarray(right, dtype=float).copy()
    temporal = _project_temporal_weights_with_lambdas(snapshots, left_iter, right_iter, y)
    component_weights = temporal * y[None, :]
    active_components = y > 1e-10
    spectral_prior = y / max(float(np.max(y)), 1e-12)
    component_weights[:, ~active_components] = 0.0
    residual_sq, total_entries = _residual_energy_stats(
        snapshots,
        left_iter,
        right_iter,
        temporal,
        y,
    )
    sigma2 = float(max(residual_sq / max(total_entries, 1), float(config.ard_noise_floor)))
    a0 = float(config.ard_prior_alpha)
    b0 = float(config.ard_prior_beta)
    num_steps = max(1, temporal.shape[0])
    a = np.full_like(y, a0 + 0.5 * num_steps, dtype=float)
    b = np.full_like(
        y,
        b0 + 0.5 * np.maximum(np.mean(component_weights**2, axis=0), 1e-12),
        dtype=float,
    )
    e_alpha = a / b

    elbo_history: list[float] = []
    mu = y.copy()
    var = np.full_like(y, max(sigma2, float(config.ard_noise_floor)), dtype=float)
    converged = False
    previous_mu: np.ndarray | None = None

    for _ in range(int(config.ard_max_iterations)):
        safe_sigma2 = max(sigma2, float(config.ard_noise_floor))
        component_energy = np.mean(component_weights**2, axis=0)
        b = b0 + 0.5 * np.maximum(component_energy, 1e-12)
        e_alpha = a / b
        shrink = 1.0 / (1.0 + safe_sigma2 * e_alpha)
        component_weights = component_weights * shrink[None, :]
        component_weights[:, ~active_components] = 0.0
        var = safe_sigma2 * shrink
        second_moment = np.mean(component_weights**2, axis=0) + var
        b = b0 + 0.5 * np.maximum(second_moment, 1e-12)
        e_alpha = a / b
        mu = np.sqrt(np.maximum(np.mean(component_weights**2, axis=0), 1e-24))
        mu[~active_components] = 0.0
        safe_mu = np.where(mu > 1e-12, mu, 1e-12)
        temporal = component_weights / safe_mu[None, :]
        temporal[:, ~active_components] = 0.0
        component_weights = temporal * mu[None, :]
        left_iter, right_iter = _update_component_directions(
            snapshots,
            left_iter,
            right_iter,
            component_weights,
        )
        temporal = _project_temporal_weights_with_lambdas(
            snapshots,
            left_iter,
            right_iter,
            np.where(mu > 1e-12, mu, 1e-12),
        )
        temporal = temporal * spectral_prior[None, :]
        component_weights = temporal * mu[None, :]

        residual_sq, total_entries = _residual_energy_stats(
            snapshots,
            left_iter,
            right_iter,
            temporal,
            safe_mu,
        )
        sigma2 = float(max(residual_sq / max(total_entries, 1), float(config.ard_noise_floor)))
        e_log_alpha = digamma(a) - np.log(b)

        log2pi = float(np.log(2.0 * np.pi))
        elbo_likelihood = -0.5 * np.sum(
            log2pi
            + np.log(max(sigma2, float(config.ard_noise_floor)))
            + second_moment / max(sigma2, float(config.ard_noise_floor))
        )
        elbo_w_prior = 0.5 * np.sum(e_log_alpha - log2pi - e_alpha * second_moment)
        elbo_alpha_prior = np.sum(
            a0 * np.log(b0) - gammaln(a0) + (a0 - 1.0) * e_log_alpha - b0 * e_alpha
        )
        entropy_w = 0.5 * np.sum(log2pi + 1.0 + np.log(np.maximum(var, 1e-24)))
        entropy_alpha = np.sum(a - np.log(b) + gammaln(a) + (1.0 - a) * digamma(a))
        elbo = float(elbo_likelihood + elbo_w_prior + elbo_alpha_prior + entropy_w + entropy_alpha)
        elbo_history.append(elbo)

        if previous_mu is not None:
            rel_change = float(
                np.linalg.norm(mu - previous_mu)
                / max(np.linalg.norm(previous_mu), 1e-12)
            )
            if rel_change <= float(config.ard_tolerance):
                converged = True
                break
        if len(elbo_history) >= 2:
            if abs(elbo_history[-1] - elbo_history[-2]) <= float(config.ard_tolerance):
                converged = True
                break
        previous_mu = mu.copy()

    if not converged and mu.size > 0 and int(config.ard_max_iterations) > 1:
        strength = np.abs(mu)
        max_strength = float(np.max(strength))
        threshold = float(max(float(config.ard_min_effective_ratio), 0.0) * max(max_strength, 1e-12))
        active = int(np.count_nonzero(strength >= threshold))
        if active <= max(1, int(config.ard_min_rank)):
            converged = True

    return mu, var, e_alpha, elbo_history, converged, sigma2, left_iter, right_iter


def _update_component_directions(
    snapshots: list[ArrayLikeSnapshot] | np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    component_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    updated_left = left.copy()
    updated_right = right.copy()
    rank = left.shape[1]
    eps = 1e-12

    for j in range(rank):
        coeffs = component_weights[:, j]
        right_vec = updated_right[:, j]
        source_acc = np.zeros(updated_left.shape[0], dtype=float)
        for t, snapshot in enumerate(snapshots):
            coeff = float(coeffs[t])
            if abs(coeff) <= eps:
                continue
            source_acc += coeff * (
                snapshot @ right_vec if sparse.issparse(snapshot) else np.asarray(snapshot, dtype=float) @ right_vec
            )
        source_norm = float(np.linalg.norm(source_acc))
        if source_norm > eps:
            updated_left[:, j] = source_acc / source_norm

        left_vec = updated_left[:, j]
        target_acc = np.zeros(updated_right.shape[0], dtype=float)
        for t, snapshot in enumerate(snapshots):
            coeff = float(coeffs[t])
            if abs(coeff) <= eps:
                continue
            if sparse.issparse(snapshot):
                target_acc += coeff * (snapshot.T @ left_vec)
            else:
                target_acc += coeff * (np.asarray(snapshot, dtype=float).T @ left_vec)
        target_norm = float(np.linalg.norm(target_acc))
        if target_norm > eps:
            updated_right[:, j] = target_acc / target_norm

    return updated_left, updated_right


def _factorize_from_basis(
    snapshots: list[ArrayLikeSnapshot] | np.ndarray,
    basis: np.ndarray,
    config: SpectralCompressionConfig,
    *,
    residuals: tuple[sparse.csr_matrix, ...] | None = None,
) -> FactorizedTemporalStore:
    rank_upper_bound = _rank_upper_bound(config, min(basis.shape))
    left_full, singular_values, right_t_full = np.linalg.svd(basis, full_matrices=False)
    left, right, temporal, lambdas, rank_selection_diagnostics = _factorize_svd_components(
        snapshots,
        left_full=left_full,
        singular_values=singular_values,
        right_t_full=right_t_full,
        config=config,
        rank_upper_bound=rank_upper_bound,
    )

    left, right, temporal, lambdas, pruning_diagnostics = _apply_rank_pruning(
        snapshots,
        left,
        right,
        temporal,
        lambdas,
        config,
    )

    store_residuals: tuple[sparse.csr_matrix, ...] = residuals or ()
    if residuals is None and config.residual_threshold is not None:
        residual_matrices = []
        for t, snapshot in enumerate(snapshots):
            weights = lambdas * temporal[t]
            reconstruction = (left * weights) @ right.T
            residual = _snapshot_to_dense(snapshot) - reconstruction
            residual[np.abs(residual) <= config.residual_threshold] = 0.0
            residual_matrices.append(sparse.csr_matrix(residual))
        store_residuals = tuple(residual_matrices)

    threshold_diagnostics = {
        **rank_selection_diagnostics,
        **pruning_diagnostics,
    }
    return FactorizedTemporalStore(
        left=left,
        right=right,
        temporal=temporal,
        lambdas=lambdas,
        residuals=store_residuals,
        threshold_diagnostics=threshold_diagnostics or None,
    )


def _estimate_residual_variance(
    snapshots: list[ArrayLikeSnapshot] | np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    temporal: np.ndarray,
    lambdas: np.ndarray,
    *,
    noise_floor: float,
) -> float:
    total_residual, total_entries = _residual_energy_stats(
        snapshots,
        left,
        right,
        temporal,
        lambdas,
    )
    if total_entries <= 0:
        return float(max(noise_floor, 1e-12))
    return float(max(total_residual / float(total_entries), noise_floor))


def _residual_energy_stats(
    snapshots: list[ArrayLikeSnapshot] | np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    temporal: np.ndarray,
    lambdas: np.ndarray,
) -> tuple[float, int]:
    if left.shape[1] == 0:
        return 0.0, 0
    gram_right = right.T @ right
    total_residual = 0.0
    total_entries = 0
    for t, snapshot in enumerate(snapshots):
        weights = lambdas * temporal[t]
        weighted_left = left * weights
        pred_norm_sq = float(np.sum((weighted_left.T @ weighted_left) * gram_right))
        if sparse.issparse(snapshot):
            projected = snapshot @ right
            snapshot_norm_sq = float(np.sum(np.asarray(snapshot.data, dtype=float) ** 2))
            entry_count = int(snapshot.shape[0] * snapshot.shape[1])
        else:
            dense_snapshot = np.asarray(snapshot, dtype=float)
            projected = dense_snapshot @ right
            snapshot_norm_sq = float(np.sum(dense_snapshot**2))
            entry_count = int(dense_snapshot.shape[0] * dense_snapshot.shape[1])
        cross_term = float(np.sum(weighted_left * projected))
        residual_sq = max(snapshot_norm_sq + pred_norm_sq - (2.0 * cross_term), 0.0)
        total_residual += residual_sq
        total_entries += entry_count
    return float(total_residual), int(total_entries)


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
    dense_snapshots: list[ArrayLikeSnapshot] | np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    lambdas: np.ndarray,
) -> np.ndarray:
    projected = _project_temporal_weights(dense_snapshots, left, right)
    safe_lambdas = np.where(np.abs(lambdas) > 1e-12, lambdas, 1.0)
    return projected / safe_lambdas


def _apply_rank_pruning(
    dense_snapshots: list[ArrayLikeSnapshot] | np.ndarray,
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


def _snapshot_to_dense(snapshot: ArrayLikeSnapshot) -> np.ndarray:
    if sparse.issparse(snapshot):
        return snapshot.toarray()
    return np.asarray(snapshot, dtype=float)


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


def _residual_arrays_from_snapshots(
    snapshots: list[ArrayLikeSnapshot],
    store: FactorizedTemporalStore,
) -> list[np.ndarray]:
    residuals = []
    for t, snapshot in enumerate(snapshots):
        residuals.append(_snapshot_to_dense(snapshot) - store.dense_snapshot(t, include_residual=False))
    return residuals


def _subtract_sparse_residuals(
    snapshots: list[sparse.csr_matrix],
    residuals: tuple[sparse.csr_matrix, ...],
) -> list[sparse.csr_matrix]:
    return [
        (snapshots[t] - residuals[t]).tocsr()
        for t in range(len(snapshots))
    ]


def _threshold_residuals(
    residual_stack: list[np.ndarray] | np.ndarray,
    config: SpectralCompressionConfig,
) -> tuple[tuple[sparse.csr_matrix, ...], dict[str, Any]]:
    residual_arrays = [np.asarray(residual, dtype=float) for residual in residual_stack]
    if config.residual_threshold is None:
        threshold, diagnostics = _adaptive_residual_threshold(residual_arrays, config)
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
    for residual in residual_arrays:
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
    residual_stack: list[np.ndarray] | np.ndarray,
    config: SpectralCompressionConfig,
) -> tuple[float, dict[str, Any]]:
    abs_residuals = np.concatenate(
        [np.abs(np.asarray(residual, dtype=float)).ravel() for residual in residual_stack]
    )
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
    dense_snapshots: list[ArrayLikeSnapshot] | np.ndarray,
    residual_stack: list[np.ndarray] | np.ndarray,
    residuals: tuple[sparse.csr_matrix, ...],
    config: SpectralCompressionConfig,
) -> dict[str, Any]:
    source_degree_scale, target_degree_scale = _degree_scales(dense_snapshots)
    edge_scale = 0.5 * (source_degree_scale[:, None] + target_degree_scale[None, :])
    residual_arrays = [np.asarray(residual, dtype=float) for residual in residual_stack]
    per_entry_scale_flat = []
    for t, residual in enumerate(residual_arrays):
        omitted_residual = residual - residuals[t].toarray()
        per_entry_scale_flat.append(
            (
                np.abs(omitted_residual)
                / np.maximum(edge_scale, 1e-12)
            ).ravel()
        )
    per_entry_scale = np.concatenate(per_entry_scale_flat)
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
    if isinstance(dense_snapshots, np.ndarray):
        weighted_degree = np.abs(dense_snapshots)
        source_degree = weighted_degree.sum(axis=2).mean(axis=0)
        target_degree = weighted_degree.sum(axis=1).mean(axis=0)
    else:
        source_degree = None
        target_degree = None
        for snapshot in dense_snapshots:
            dense_snapshot = np.abs(_snapshot_to_dense(snapshot))
            source_t = dense_snapshot.sum(axis=1)
            target_t = dense_snapshot.sum(axis=0)
            if source_degree is None:
                source_degree = source_t
                target_degree = target_t
            else:
                source_degree = source_degree + source_t
                target_degree = target_degree + target_t
        assert source_degree is not None and target_degree is not None
        source_degree = source_degree / max(len(dense_snapshots), 1)
        target_degree = target_degree / max(len(dense_snapshots), 1)
    mean_degree = float(np.mean(0.5 * (source_degree + target_degree)))
    if mean_degree <= 1e-12:
        return np.ones_like(source_degree), np.ones_like(target_degree)

    source_mu = np.maximum(source_degree / mean_degree, 1e-6)
    target_mu = np.maximum(target_degree / mean_degree, 1e-6)
    return 1.0 / np.sqrt(source_mu), 1.0 / np.sqrt(target_mu)
