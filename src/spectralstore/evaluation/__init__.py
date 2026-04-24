"""Evaluation metrics and experiment helpers."""

from spectralstore.evaluation.metrics import (
    anomaly_precision_recall,
    entrywise_bound_coverage,
    max_entrywise_error_bound,
    max_entrywise_error,
    mean_entrywise_error_bound,
    mean_entrywise_error,
    observed_edge_mae,
    observed_edge_rmse,
    residual_nnz,
    residual_sparsity,
    relative_frobenius_error,
    relative_frobenius_error_against_dense,
)

__all__ = [
    "anomaly_precision_recall",
    "entrywise_bound_coverage",
    "max_entrywise_error_bound",
    "max_entrywise_error",
    "mean_entrywise_error_bound",
    "mean_entrywise_error",
    "observed_edge_mae",
    "observed_edge_rmse",
    "residual_nnz",
    "residual_sparsity",
    "relative_frobenius_error",
    "relative_frobenius_error_against_dense",
]
