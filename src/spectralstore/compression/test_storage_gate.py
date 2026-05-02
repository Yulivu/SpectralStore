import numpy as np
from scipy import sparse

from spectralstore.compression.factorized_store import FactorizedTemporalStore
from spectralstore.compression.spectral import SpectralCompressionConfig, _apply_storage_gate


def test_drop_residual_reports_final_storage_gate_acceptance() -> None:
    n = 20
    steps = 2
    rank = 1
    snapshots = [sparse.csr_matrix(np.ones((n, n), dtype=float)) for _ in range(steps)]
    residuals = tuple(
        sparse.csr_matrix(np.full((n, n), 10.0, dtype=float)) for _ in range(steps)
    )
    store = FactorizedTemporalStore(
        left=np.ones((n, rank), dtype=float),
        right=np.ones((n, rank), dtype=float),
        temporal=np.ones((steps, rank), dtype=float),
        lambdas=np.ones(rank, dtype=float),
        residuals=residuals,
        threshold_diagnostics={"estimated_threshold": 1.0},
    )
    config = SpectralCompressionConfig(
        max_sparse_ratio=1.0,
        storage_gate_action="drop_residual",
    )

    gated = _apply_storage_gate(store, snapshots, config)
    diagnostics = gated.threshold_diagnostics or {}

    assert not gated.residuals
    assert diagnostics["storage_gate_action_taken"] == "drop_residual"
    assert diagnostics["storage_gate_accepted_before_action"] is False
    assert diagnostics["storage_gate_accepted_after_action"] is True
    assert diagnostics["storage_gate_accepted"] is True
    assert diagnostics["storage_gate_sparse_ratio_before_action"] > 1.0
    assert diagnostics["storage_gate_sparse_ratio_after_action"] <= 1.0
