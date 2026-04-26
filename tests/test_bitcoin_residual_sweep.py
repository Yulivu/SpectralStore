from scripts.preliminary.run_bitcoin_residual_sweep import acceptance_failures


def test_bitcoin_residual_sweep_acceptance_requires_sparse_ratio_and_error() -> None:
    baseline = {
        "held_out_observed_edge_rmse": 0.4,
        "held_out_observed_edge_mae": 0.2,
    }
    acceptance = {
        "max_sparse_ratio": 1.0,
        "max_heldout_rmse_regression": 0.05,
        "max_heldout_mae_regression": 0.05,
    }

    accepted = {
        "storage": {"compressed_vs_raw_sparse_ratio": 0.9},
        "held_out_observed_edge_rmse": 0.41,
        "held_out_observed_edge_mae": 0.205,
    }
    rejected = {
        "storage": {"compressed_vs_raw_sparse_ratio": 1.1},
        "held_out_observed_edge_rmse": 0.43,
        "held_out_observed_edge_mae": 0.22,
    }

    assert acceptance_failures(accepted, baseline, acceptance) == []
    assert acceptance_failures(rejected, baseline, acceptance) == [
        "sparse_ratio",
        "heldout_rmse",
        "heldout_mae",
    ]
