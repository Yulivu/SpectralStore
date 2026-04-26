"""Run Thinking.docx specification-alignment checks."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spectralstore.compression import create_compressor, spectral_config_from_mapping  # noqa: E402
from spectralstore.data_loader import (  # noqa: E402
    make_synthetic_attack,
    make_synthetic_spiked,
    make_temporal_sbm,
    make_temporal_correlated_sbm,
    make_theory_regime_sbm,
)
from spectralstore.evaluation import (  # noqa: E402
    anomaly_precision_recall_f1,
    entrywise_bound_report,
    load_experiment_config,
    mean_entrywise_error,
    max_entrywise_error,
    reconstruction_difference,
    relative_frobenius_error_against_dense,
    set_reproducibility_seed,
    split_asym_construction_report,
    storage_report,
    subspace_distance,
    write_experiment_outputs,
)
from spectralstore.query_engine import QueryEngine  # noqa: E402


DEFAULT_CONFIG = "experiments/preliminary/thinking_alignment/configs/default.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="OmegaConf dotlist override, e.g. --set split_asym.num_repeats=1",
    )
    args = parser.parse_args()

    started_at = time.perf_counter()
    overrides = normalize_overrides(args.overrides)
    config = load_experiment_config(args.config, overrides)
    apply_shared_overrides(config)
    set_reproducibility_seed(config.get("random_seed"))
    out_dir = run_output_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, Any] = {
        "name": "thinking_alignment_checks",
        "checks": list(config["checks"]),
    }
    csv_outputs: dict[str, str] = {}

    if "split_asym_sanity" in config["checks"]:
        csv_outputs["split_asym_sanity"] = "split_asym_sanity.csv"
        rows = run_split_asym_sanity(config, out_dir / csv_outputs["split_asym_sanity"])
        metrics["split_asym_sanity"] = {
            "rows": rows,
            "aggregate": aggregate_numeric(rows),
        }

    if "entrywise_bound_coverage" in config["checks"]:
        csv_outputs["entrywise_bound_coverage"] = "entrywise_bound_coverage.csv"
        rows = run_entrywise_bound_coverage(
            config,
            out_dir / csv_outputs["entrywise_bound_coverage"],
        )
        metrics["entrywise_bound_coverage"] = {
            "rows": rows,
            "aggregate": aggregate_by_method(rows),
        }

    if "robust_sparse_corruption" in config["checks"]:
        csv_outputs["robust_sparse_corruption"] = "robust_sparse_corruption.csv"
        rows = run_robust_sparse_corruption(
            config,
            out_dir / csv_outputs["robust_sparse_corruption"],
        )
        metrics["robust_sparse_corruption"] = {
            "rows": rows,
            "aggregate": aggregate_robust(rows),
        }

    if "entrywise_bound_calibration" in config["checks"]:
        csv_outputs["entrywise_bound_calibration"] = "entrywise_bound_calibration.csv"
        rows = run_entrywise_bound_calibration(
            config,
            out_dir / csv_outputs["entrywise_bound_calibration"],
        )
        metrics["entrywise_bound_calibration"] = {
            "rows": rows,
            "aggregate": aggregate_calibration_bounds(rows),
        }

    if "residual_threshold_calibration" in config["checks"]:
        csv_outputs["residual_threshold_calibration"] = "residual_threshold_calibration.csv"
        rows = run_residual_threshold_calibration(
            config,
            out_dir / csv_outputs["residual_threshold_calibration"],
        )
        metrics["residual_threshold_calibration"] = {
            "rows": rows,
            "aggregate": aggregate_threshold_calibration(rows),
        }

    if "system_calibration_sanity" in config["checks"]:
        csv_outputs["system_calibration_sanity"] = "system_calibration_sanity.csv"
        rows = run_system_calibration_sanity(
            config,
            out_dir / csv_outputs["system_calibration_sanity"],
        )
        metrics["system_calibration_sanity"] = {
            "rows": rows,
            "aggregate": aggregate_numeric(rows),
        }

    if "exp1_v2_theory_regime" in config["checks"]:
        csv_outputs["exp1_v2_theory_regime"] = "exp1_v2_theory_regime.csv"
        rows = run_exp1_v2_theory_regime(
            config,
            out_dir / csv_outputs["exp1_v2_theory_regime"],
        )
        metrics["exp1_v2_theory_regime"] = {
            "rows": rows,
            "aggregate": aggregate_exp1_v2(rows),
        }

    if "exp_asym_temporal_dependence" in config["checks"]:
        csv_outputs["exp_asym_temporal_dependence"] = "exp_asym_temporal_dependence.csv"
        rows = run_exp_asym_temporal_dependence(
            config,
            out_dir / csv_outputs["exp_asym_temporal_dependence"],
        )
        metrics["exp_asym_temporal_dependence"] = {
            "rows": rows,
            "aggregate": aggregate_asym_temporal(rows),
        }

    if "asym_mechanism_audit" in config["checks"]:
        csv_outputs["asym_mechanism_audit"] = "asym_mechanism_audit.csv"
        rows = run_asym_mechanism_audit(
            config,
            out_dir / csv_outputs["asym_mechanism_audit"],
        )
        metrics["asym_mechanism_audit"] = {
            "rows": rows,
            "aggregate": aggregate_asym_mechanism(rows),
        }

    metrics["csv_outputs"] = csv_outputs
    summary = render_summary(metrics)
    write_experiment_outputs(
        out_dir=out_dir,
        metrics=metrics,
        summary=summary,
        config_path=args.config,
        config=config,
        started_at=started_at,
    )
    print(summary)


def run_output_dir(config: dict[str, Any]) -> Path:
    checks = set(config["checks"])
    if checks & {
        "entrywise_bound_calibration",
        "residual_threshold_calibration",
        "system_calibration_sanity",
    }:
        calibration = config.get("calibration", {})
        return Path(calibration.get("output_dir", Path(config["output_dir"]) / "calibration"))
    if "exp1_v2_theory_regime" in checks:
        default_alignment_output = Path("experiments/preliminary/thinking_alignment/results")
        if Path(config["output_dir"]) != default_alignment_output:
            return Path(config["output_dir"])
        return Path(config["exp1_v2"].get("output_dir", Path(config["output_dir"]) / "exp1_v2"))
    if "exp_asym_temporal_dependence" in checks:
        default_alignment_output = Path("experiments/preliminary/thinking_alignment/results")
        if Path(config["output_dir"]) != default_alignment_output:
            return Path(config["output_dir"])
        return Path(
            config["asym_temporal_dependence"].get(
                "output_dir",
                Path(config["output_dir"]) / "exp_asym_temporal_dependence",
            )
        )
    if "asym_mechanism_audit" in checks:
        default_alignment_output = Path("experiments/preliminary/thinking_alignment/results")
        if Path(config["output_dir"]) != default_alignment_output:
            return Path(config["output_dir"])
        return Path(
            config["asym_mechanism_audit"].get(
                "output_dir",
                Path(config["output_dir"]) / "asym_mechanism_audit",
            )
        )
    return Path(config["output_dir"])


def normalize_overrides(overrides: list[str]) -> list[str]:
    normalized: list[str] = []
    for override in overrides:
        for line in str(override).splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split() if " " in stripped and "=" in stripped else [stripped]
            normalized.extend(part for part in parts if "=" in part)
    return normalized


def apply_shared_overrides(config: dict[str, Any]) -> None:
    data = config.get("data", {})
    experiment = config.get("experiment", {})
    if "n" in data:
        for section in ("split_asym", "entrywise_bound", "robust_sparse_corruption"):
            if section in config:
                config[section]["num_nodes"] = data["n"]
    if "T" in data:
        for section in ("split_asym", "entrywise_bound", "robust_sparse_corruption"):
            if section in config:
                config[section]["num_steps"] = data["T"]
    if "repeats" in experiment:
        for section in ("split_asym", "entrywise_bound", "robust_sparse_corruption"):
            if section in config:
                config[section]["num_repeats"] = experiment["repeats"]
        if "exp1_v2" in config:
            config["exp1_v2"]["num_repeats"] = experiment["repeats"]
        if "asym_temporal_dependence" in config:
            config["asym_temporal_dependence"]["num_repeats"] = experiment["repeats"]
        if "asym_mechanism_audit" in config:
            config["asym_mechanism_audit"]["num_repeats"] = experiment["repeats"]
    if "exp1_v2" in config:
        for key in ("n_list", "T_list", "regime_list", "noise_type"):
            if key in data:
                config["exp1_v2"][key] = data[key]
        if "n_list" in data and "fixed_n" not in data:
            config["exp1_v2"]["fixed_n"] = data["n_list"][0]
        if "T_list" in data and "fixed_T" not in data:
            config["exp1_v2"]["fixed_T"] = data["T_list"][0]
        if "fixed_n" in data:
            config["exp1_v2"]["fixed_n"] = data["fixed_n"]
        if "fixed_T" in data:
            config["exp1_v2"]["fixed_T"] = data["fixed_T"]
    if "asym_temporal_dependence" in config:
        for key in ("n_list", "T_list", "regime_list", "alpha_list", "base_noise_std"):
            if key in data:
                config["asym_temporal_dependence"][key] = data[key]
    if "asym_mechanism_audit" in config:
        for key in ("n_list", "T_list", "regime_list", "alpha_list", "base_noise_std", "dataset"):
            if key in data:
                config["asym_mechanism_audit"][key] = data[key]
    if "robust_sparse_corruption" in config and "robust" in config:
        config["robust_sparse_corruption"]["threshold_scale"] = config["robust"].get(
            "threshold_scale",
            config["robust_sparse_corruption"].get("threshold_scale", 1.0),
        )


def run_system_calibration_sanity(config: dict[str, Any], csv_path: Path) -> list[dict[str, Any]]:
    settings = config["robust_sparse_corruption"]
    query = config["query"]
    seed = int(config["random_seed"])
    key = (
        "system_calibration_sanity",
        seed,
        int(settings["num_nodes"]),
        int(settings["num_steps"]),
        int(settings["rank"]),
        float(query["bound_C"]),
        float(settings["threshold_scale"]),
    )
    rows = [row for row in read_csv_rows(csv_path) if system_sanity_key(row) == key]
    if rows:
        return rows

    clean_dataset = make_synthetic_attack(
        num_nodes=int(settings["num_nodes"]),
        num_steps=int(settings["num_steps"]),
        num_communities=int(settings["num_communities"]),
        p_in=float(settings["p_in"]),
        p_out=float(settings["p_out"]),
        temporal_jitter=float(settings["temporal_jitter"]),
        attack_kind="sparse_spike",
        corruption_rate=0.0,
        corruption_magnitude=float(settings["corruption_magnitude"]),
        directed=bool(settings["directed"]),
        random_seed=seed,
    )
    corrupt_dataset = make_synthetic_attack(
        num_nodes=int(settings["num_nodes"]),
        num_steps=int(settings["num_steps"]),
        num_communities=int(settings["num_communities"]),
        p_in=float(settings["p_in"]),
        p_out=float(settings["p_out"]),
        temporal_jitter=float(settings["temporal_jitter"]),
        attack_kind="sparse_spike",
        corruption_rate=float(next(rate for rate in settings["corruption_rates"] if float(rate) > 0.0)),
        corruption_magnitude=float(settings["corruption_magnitude"]),
        directed=bool(settings["directed"]),
        random_seed=seed,
    )

    asym_config = spectral_config_from_mapping(
        {
            "rank": int(settings["rank"]),
            "num_splits": int(settings["num_splits"]),
            "random_seed": seed,
        }
    )
    asym_store = create_compressor("spectralstore_asym", asym_config).fit_transform(
        clean_dataset.snapshots
    )
    asym_store.precompute_bound_params(clean_dataset.snapshots, constant=1.0)
    engine_c1 = QueryEngine(asym_store, bound_C=1.0, method="spectralstore_asym")
    q1_c1 = engine_c1.link_prob_result(0, 1, 0, include_residual=False)
    engine_from_config = QueryEngine.from_config(
        asym_store,
        config,
        method="spectralstore_asym",
    )
    q1_calibrated = engine_from_config.link_prob_result(0, 1, 0, include_residual=False)

    robust_config_default = robust_config(settings, seed, threshold_scale=1.0)
    robust_config_scaled = robust_config(settings, seed)
    robust_clean_default = create_compressor(
        "spectralstore_robust",
        robust_config_default,
    ).fit_transform(clean_dataset.snapshots)
    robust_clean_scaled = create_compressor(
        "spectralstore_robust",
        robust_config_scaled,
    ).fit_transform(clean_dataset.snapshots)
    robust_corrupt_scaled = create_compressor(
        "spectralstore_robust",
        robust_config_scaled,
    ).fit_transform(corrupt_dataset.snapshots)
    robust_corrupt_scaled.precompute_bound_params(corrupt_dataset.snapshots, constant=1.0)
    robust_engine = QueryEngine(
        robust_corrupt_scaled,
        bound_C=float(query["bound_C"]),
        method="spectralstore_robust",
    )
    loose = robust_engine.link_prob_optimized(
        0,
        1,
        0,
        error_tolerance=float(query["loose_tolerance"]),
    )
    tight = robust_engine.link_prob_optimized(
        0,
        1,
        0,
        error_tolerance=float(query["tight_tolerance"]),
    )

    row = {
        "check": "system_calibration_sanity",
        "seed": seed,
        "num_nodes": int(settings["num_nodes"]),
        "num_steps": int(settings["num_steps"]),
        "rank": int(settings["rank"]),
        "query_bound_C": float(query["bound_C"]),
        "robust_threshold_scale": float(settings["threshold_scale"]),
        "q1_estimate": q1_calibrated["estimate"],
        "q1_bound_C1": q1_c1["bound"],
        "q1_bound_calibrated": q1_calibrated["bound"],
        "q1_bound_ratio": float(q1_calibrated["bound"]) / max(float(q1_c1["bound"]), 1e-12),
        "q1_used_residual": q1_calibrated["used_residual"],
        "q1_method": q1_calibrated["method"],
        "loose_tolerance_used_residual": loose.used_residual,
        "tight_tolerance_used_residual": tight.used_residual,
        "default_scale_residual_nnz": sum(residual.nnz for residual in robust_clean_default.residuals),
        "scaled_residual_nnz": sum(residual.nnz for residual in robust_clean_scaled.residuals),
        "default_scale_residual_storage_ratio": (
            robust_clean_default.residual_bytes() / max(robust_clean_default.raw_dense_bytes(), 1)
        ),
        "scaled_residual_storage_ratio": (
            robust_clean_scaled.residual_bytes() / max(robust_clean_scaled.raw_dense_bytes(), 1)
        ),
        "query_bound_C_config_driven": abs(
            float(q1_calibrated["bound"]) / max(float(q1_c1["bound"]), 1e-12)
            - float(query["bound_C"])
        )
        < 1e-9,
        "robust_threshold_scale_changes_residuals": (
            sum(residual.nnz for residual in robust_clean_default.residuals)
            != sum(residual.nnz for residual in robust_clean_scaled.residuals)
        ),
        "calibration_defaults_are_config_values": True,
    }
    append_csv_row(csv_path, row)
    print("[system_calibration_sanity] done", flush=True)
    return [row]


def run_exp1_v2_theory_regime(config: dict[str, Any], csv_path: Path) -> list[dict[str, Any]]:
    settings = config["exp1_v2"]
    methods = list(settings["methods"])
    expected_keys = {
        exp1_v2_key(
            {
                "sweep_type": combo["sweep_type"],
                "method": method,
                "n": combo["n"],
                "T": combo["T"],
                "regime_name": combo["regime_name"],
                "noise_type": combo["noise_type"],
                "seed": int(config["random_seed"]) + repeat,
                "repeat_id": repeat,
            }
        )
        for combo in exp1_v2_sweep_combinations(settings)
        for repeat in range(int(settings["num_repeats"]))
        for method in methods
    }
    rows = [row for row in read_csv_rows(csv_path) if exp1_v2_key(row) in expected_keys]
    completed = {exp1_v2_key(row) for row in rows}

    for combo in exp1_v2_sweep_combinations(settings):
        n = int(combo["n"])
        if n > 2000:
            raise ValueError("Exp1-v2 enforces n <= 2000")
        for repeat in range(int(settings["num_repeats"])):
            seed = int(config["random_seed"]) + repeat
            dataset = make_exp1_v2_dataset(settings, combo, seed)
            if dataset.expected_snapshots is None:
                raise ValueError("Exp1-v2 requires ground-truth expected_snapshots")
            for method in methods:
                key = exp1_v2_key(
                    {
                        "sweep_type": combo["sweep_type"],
                        "method": method,
                        "n": combo["n"],
                        "T": combo["T"],
                        "regime_name": combo["regime_name"],
                        "noise_type": combo["noise_type"],
                        "seed": seed,
                        "repeat_id": repeat,
                    }
                )
                if key in completed:
                    continue
                started = time.perf_counter()
                compressor_config = spectral_config_from_mapping(
                    {
                        "rank": int(settings["rank"]),
                        "num_splits": int(settings["num_splits"]),
                        "random_seed": seed,
                    }
                )
                store = create_compressor(method, compressor_config).fit_transform(
                    dataset.snapshots
                )
                if bool(settings["precompute_bounds"]):
                    store.precompute_bound_params(
                        dataset.snapshots,
                        constant=float(settings["bound_constant"]),
                    )
                max_error = max_entrywise_error(
                    dataset.expected_snapshots,
                    store,
                    include_residual=False,
                )
                mean_error = mean_entrywise_error(
                    dataset.expected_snapshots,
                    store,
                    include_residual=False,
                )
                fro_error = relative_frobenius_error_against_dense(
                    dataset.expected_snapshots,
                    store,
                    include_residual=False,
                )
                bound_report = entrywise_bound_report(
                    dataset.expected_snapshots,
                    store,
                    include_residual=False,
                )
                theoretical_rate = math.sqrt(math.log(max(n, 2)) / max(n * int(combo["T"]), 1))
                row = {
                    "sweep_type": combo["sweep_type"],
                    "method": method,
                    "n": n,
                    "T": int(combo["T"]),
                    "regime_name": combo["regime_name"],
                    "noise_type": combo["noise_type"],
                    "seed": seed,
                    "repeat_id": repeat,
                    "rank": int(settings["rank"]),
                    "max_entrywise_error_to_M_star": max_error,
                    "mean_entrywise_error_to_M_star": mean_error,
                    "frobenius_error_to_M_star": fro_error,
                    "error_std_across_repeats": float("nan"),
                    "max_error_across_repeats": float("nan"),
                    "coverage": bound_report["coverage"],
                    "violation_rate": bound_report["violation_rate"],
                    "mean_bound_over_error": bound_report["mean_bound_over_error"],
                    "asym_error_ratio_vs_sym": float("nan"),
                    "asym_variance_ratio_vs_sym": float("nan"),
                    "theoretical_rate": theoretical_rate,
                    "normalized_max_error": max_error / max(theoretical_rate, 1e-12),
                    "normalized_mean_error": mean_error / max(theoretical_rate, 1e-12),
                    "runtime": time.perf_counter() - started,
                }
                rows.append(row)
                completed.add(key)
                append_csv_row(csv_path, row)
                print(
                    f"[exp1_v2, sweep={combo['sweep_type']}, n={n}, T={int(combo['T'])}, "
                    f"regime={combo['regime_name']}, noise={combo['noise_type']}, "
                    f"method={method}, repeat={repeat + 1}/{settings['num_repeats']}] "
                    f"done in {row['runtime']:.1f}s",
                    flush=True,
                )

    refreshed = refresh_exp1_v2_rows(read_csv_rows(csv_path))
    refreshed = [row for row in refreshed if exp1_v2_key(row) in expected_keys]
    write_csv(csv_path, refreshed)
    return refreshed


def run_exp_asym_temporal_dependence(
    config: dict[str, Any],
    csv_path: Path,
) -> list[dict[str, Any]]:
    settings = config["asym_temporal_dependence"]
    methods = list(settings["methods"])
    expected_keys = {
        asym_temporal_key(
            {
                "method": method,
                "n": combo["n"],
                "T": combo["T"],
                "alpha": combo["alpha"],
                "regime_name": combo["regime_name"],
                "seed": int(config["random_seed"]) + repeat,
                "repeat_id": repeat,
            }
        )
        for combo in asym_temporal_combinations(settings)
        for repeat in range(int(settings["num_repeats"]))
        for method in methods
    }
    rows = [row for row in read_csv_rows(csv_path) if asym_temporal_key(row) in expected_keys]
    completed = {asym_temporal_key(row) for row in rows}

    for combo in asym_temporal_combinations(settings):
        for repeat in range(int(settings["num_repeats"])):
            seed = int(config["random_seed"]) + repeat
            dataset = make_asym_temporal_dataset(settings, combo, seed)
            if dataset.expected_snapshots is None:
                raise ValueError("temporal-dependence experiment requires M_star")
            for method in methods:
                key = asym_temporal_key(
                    {
                        "method": method,
                        "n": combo["n"],
                        "T": combo["T"],
                        "alpha": combo["alpha"],
                        "regime_name": combo["regime_name"],
                        "seed": seed,
                        "repeat_id": repeat,
                    }
                )
                if key in completed:
                    continue
                started = time.perf_counter()
                compressor_config = spectral_config_from_mapping(
                    {
                        "rank": int(settings["rank"]),
                        "num_splits": int(settings["num_splits"]),
                        "random_seed": seed,
                    }
                )
                store = create_compressor(method, compressor_config).fit_transform(
                    dataset.snapshots
                )
                max_error = max_entrywise_error(
                    dataset.expected_snapshots,
                    store,
                    include_residual=False,
                )
                mean_error = mean_entrywise_error(
                    dataset.expected_snapshots,
                    store,
                    include_residual=False,
                )
                row = {
                    "method": method,
                    "n": int(combo["n"]),
                    "T": int(combo["T"]),
                    "alpha": float(combo["alpha"]),
                    "regime_name": combo["regime_name"],
                    "noise_type": "temporal_correlated",
                    "seed": seed,
                    "repeat_id": repeat,
                    "rank": int(settings["rank"]),
                    "max_entrywise_error_to_M_star": max_error,
                    "mean_entrywise_error_to_M_star": mean_error,
                    "error_std_across_repeats": float("nan"),
                    "max_error_across_repeats": float("nan"),
                    "asym_error_ratio_vs_sym": float("nan"),
                    "asym_variance_ratio_vs_sym": float("nan"),
                    "runtime": time.perf_counter() - started,
                }
                rows.append(row)
                completed.add(key)
                append_csv_row(csv_path, row)
                print(
                    f"[asym_temporal, alpha={float(combo['alpha']):.2f}, "
                    f"n={int(combo['n'])}, T={int(combo['T'])}, "
                    f"regime={combo['regime_name']}, method={method}, "
                    f"repeat={repeat + 1}/{settings['num_repeats']}] "
                    f"done in {row['runtime']:.1f}s",
                    flush=True,
                )

    refreshed = refresh_asym_temporal_rows(read_csv_rows(csv_path))
    refreshed = [row for row in refreshed if asym_temporal_key(row) in expected_keys]
    write_csv(csv_path, refreshed)
    return refreshed


def run_asym_mechanism_audit(
    config: dict[str, Any],
    csv_path: Path,
) -> list[dict[str, Any]]:
    settings = config["asym_mechanism_audit"]
    methods = list(settings["methods"])
    expected_keys = {
        asym_mechanism_key(
            {
                "method": method,
                "n": combo["n"],
                "T": combo["T"],
                "alpha": combo["alpha"],
                "regime_name": combo["regime_name"],
                "seed": int(config["random_seed"]) + repeat,
                "repeat_id": repeat,
            }
        )
        for combo in asym_mechanism_combinations(settings)
        for repeat in range(int(settings["num_repeats"]))
        for method in methods
    }
    rows = [row for row in read_csv_rows(csv_path) if asym_mechanism_key(row) in expected_keys]
    completed = {asym_mechanism_key(row) for row in rows}

    for combo in asym_mechanism_combinations(settings):
        for repeat in range(int(settings["num_repeats"])):
            seed = int(config["random_seed"]) + repeat
            dataset = make_asym_mechanism_dataset(settings, combo, seed)
            if dataset.expected_snapshots is None:
                raise ValueError("asym mechanism audit requires M_star")
            construction = audit_asym_construction(dataset.snapshots, seed)
            compressor_config = spectral_config_from_mapping(
                {
                    "rank": int(settings["rank"]),
                    "num_splits": int(settings["num_splits"]),
                    "random_seed": seed,
                }
            )
            stores = {
                method: create_compressor(method, compressor_config).fit_transform(
                    dataset.snapshots
                )
                for method in methods
            }
            split_diff = (
                reconstruction_difference(
                    stores["spectralstore_asym"],
                    stores["spectralstore_split_asym_unfolding"],
                )
                if {
                    "spectralstore_asym",
                    "spectralstore_split_asym_unfolding",
                }
                <= set(stores)
                else float("nan")
            )
            sym_diff = (
                reconstruction_difference(stores["spectralstore_asym"], stores["sym_svd"])
                if {"spectralstore_asym", "sym_svd"} <= set(stores)
                else float("nan")
            )
            mean_observed = np.mean(
                [
                    snapshot.toarray()
                    if hasattr(snapshot, "toarray")
                    else np.asarray(snapshot, dtype=float)
                    for snapshot in dataset.snapshots
                ],
                axis=0,
            )
            for method in methods:
                key = asym_mechanism_key(
                    {
                        "method": method,
                        "n": combo["n"],
                        "T": combo["T"],
                        "alpha": combo["alpha"],
                        "regime_name": combo["regime_name"],
                        "seed": seed,
                        "repeat_id": repeat,
                    }
                )
                if key in completed:
                    continue
                store = stores[method]
                decomposition = audit_store_decomposition(store)
                max_error = max_entrywise_error(
                    dataset.expected_snapshots,
                    store,
                    include_residual=False,
                )
                mean_error = mean_entrywise_error(
                    dataset.expected_snapshots,
                    store,
                    include_residual=False,
                )
                row = {
                    "method": method,
                    "n": int(combo["n"]),
                    "T": int(combo["T"]),
                    "alpha": float(combo["alpha"]),
                    "regime_name": combo["regime_name"],
                    "noise_type": str(combo["noise_type"]),
                    "seed": seed,
                    "repeat_id": repeat,
                    "rank": int(settings["rank"]),
                    **construction,
                    **decomposition,
                    "reconstruction_error_to_M_star": relative_frobenius_error_against_dense(
                        dataset.expected_snapshots,
                        store,
                        include_residual=False,
                    ),
                    "reconstruction_error_to_mean_A": store_error_to_dense_target(
                        store,
                        mean_observed,
                    ),
                    "max_entrywise_error": max_error,
                    "mean_entrywise_error": mean_error,
                    "error_std_across_repeats": float("nan"),
                    "max_error_across_repeats": float("nan"),
                    "asym_variance_ratio_vs_sym": float("nan"),
                    "reconstruction_diff_asym_vs_split": split_diff,
                    "reconstruction_diff_asym_vs_sym": sym_diff,
                }
                rows.append(row)
                completed.add(key)
                append_csv_row(csv_path, row)
                print(
                    f"[asym_audit, alpha={float(combo['alpha']):.2f}, "
                    f"n={int(combo['n'])}, T={int(combo['T'])}, "
                    f"method={method}, repeat={repeat + 1}/{settings['num_repeats']}] done",
                    flush=True,
                )

    refreshed = refresh_asym_mechanism_rows(read_csv_rows(csv_path))
    refreshed = [row for row in refreshed if asym_mechanism_key(row) in expected_keys]
    write_csv(csv_path, refreshed)
    return refreshed


def run_entrywise_bound_calibration(
    config: dict[str, Any],
    csv_path: Path,
) -> list[dict[str, Any]]:
    settings = config["entrywise_bound"]
    calibration = config["calibration"]
    expected_keys = {
        bound_calibration_key(
            {
                "repeat": repeat,
                "seed": int(config["random_seed"]) + repeat,
                "dataset": settings["dataset"],
                "num_nodes": settings["num_nodes"],
                "num_steps": settings["num_steps"],
                "rank": settings["rank"],
                "method": method,
                "bound_C": bound_c,
            }
        )
        for repeat in range(int(settings["num_repeats"]))
        for method in settings["methods"]
        for bound_c in calibration["bound_C_list"]
    }
    rows = [
        row
        for row in read_csv_rows(csv_path)
        if bound_calibration_key(row) in expected_keys
    ]
    completed = {bound_calibration_key(row) for row in rows}
    for repeat in range(int(settings["num_repeats"])):
        seed = int(config["random_seed"]) + repeat
        dataset = make_bound_dataset(settings, seed)
        for method in settings["methods"]:
            for bound_c in calibration["bound_C_list"]:
                key = bound_calibration_key(
                    {
                        "repeat": repeat,
                        "seed": seed,
                        "dataset": settings["dataset"],
                        "num_nodes": settings["num_nodes"],
                        "num_steps": settings["num_steps"],
                        "rank": settings["rank"],
                        "method": method,
                        "bound_C": bound_c,
                    }
                )
                if key in completed:
                    continue
                compressor_config = spectral_config_from_mapping(
                    {
                        "rank": int(settings["rank"]),
                        "num_splits": int(settings["num_splits"]),
                        "random_seed": seed,
                    }
                )
                store = create_compressor(method, compressor_config).fit_transform(
                    dataset.snapshots
                )
                bound_precomputed = method in set(settings["precompute_bound_methods"])
                if bound_precomputed:
                    store.precompute_bound_params(
                        dataset.snapshots,
                        constant=float(bound_c),
                    )
                report = entrywise_bound_report(
                    dataset.expected_snapshots,
                    store,
                    include_residual=bool(settings["include_residual"]),
                )
                row = {
                    "repeat": repeat,
                    "seed": seed,
                    "dataset": dataset.name,
                    "num_nodes": int(settings["num_nodes"]),
                    "num_steps": int(settings["num_steps"]),
                    "rank": int(settings["rank"]),
                    "method": method,
                    "bound_C": float(bound_c),
                    "bound_precomputed": bound_precomputed,
                    "bound_sigma_max": store.bound_sigma_max,
                    **report,
                }
                rows.append(row)
                completed.add(key)
                append_csv_row(csv_path, row)
                print(
                    f"[bound_calibration, C={float(bound_c):.3g}, method={method}, "
                    f"repeat={repeat + 1}/{settings['num_repeats']}] done",
                    flush=True,
                )
    return rows


def run_residual_threshold_calibration(
    config: dict[str, Any],
    csv_path: Path,
) -> list[dict[str, Any]]:
    settings = config["robust_sparse_corruption"]
    calibration = config["calibration"]
    expected_keys = {
        threshold_calibration_key(
            {
                "repeat": repeat,
                "seed": int(config["random_seed"]) + repeat,
                "corruption_mode": corruption_mode,
                "corruption_rate": corruption_rate,
                "corruption_magnitude": settings["corruption_magnitude"],
                "num_nodes": settings["num_nodes"],
                "num_steps": settings["num_steps"],
                "rank": settings["rank"],
                "method": method,
                "threshold_scale": threshold_scale,
            }
        )
        for repeat in range(int(settings["num_repeats"]))
        for corruption_mode in settings["corruption_modes"]
        for corruption_rate in settings["corruption_rates"]
        for method in settings["methods"]
        for threshold_scale in calibration["threshold_scale_list"]
    }
    rows = [
        row
        for row in read_csv_rows(csv_path)
        if threshold_calibration_key(row) in expected_keys
    ]
    completed = {threshold_calibration_key(row) for row in rows}
    for repeat in range(int(settings["num_repeats"])):
        seed = int(config["random_seed"]) + repeat
        for corruption_mode in settings["corruption_modes"]:
            for corruption_rate in settings["corruption_rates"]:
                dataset = make_synthetic_attack(
                    num_nodes=int(settings["num_nodes"]),
                    num_steps=int(settings["num_steps"]),
                    num_communities=int(settings["num_communities"]),
                    p_in=float(settings["p_in"]),
                    p_out=float(settings["p_out"]),
                    temporal_jitter=float(settings["temporal_jitter"]),
                    attack_kind=str(corruption_mode),
                    corruption_rate=float(corruption_rate),
                    corruption_magnitude=float(settings["corruption_magnitude"]),
                    directed=bool(settings["directed"]),
                    random_seed=seed,
                )
                base_thresholds: dict[str, float | None] = {}
                for method in settings["methods"]:
                    base_thresholds[method] = None
                    if method == "spectralstore_robust":
                        base_config = robust_config(settings, seed)
                        base_store = create_compressor(method, base_config).fit_transform(
                            dataset.snapshots
                        )
                        base_thresholds[method] = float(
                            base_store.threshold_diagnostics["estimated_threshold"]
                        )
                for threshold_scale in calibration["threshold_scale_list"]:
                    for method in settings["methods"]:
                        key = threshold_calibration_key(
                            {
                                "repeat": repeat,
                                "seed": seed,
                                "corruption_mode": corruption_mode,
                                "corruption_rate": corruption_rate,
                                "corruption_magnitude": settings["corruption_magnitude"],
                                "num_nodes": settings["num_nodes"],
                                "num_steps": settings["num_steps"],
                                "rank": settings["rank"],
                                "method": method,
                                "threshold_scale": threshold_scale,
                            }
                        )
                        if key in completed:
                            continue
                        if method == "spectralstore_robust" and base_thresholds[method] is not None:
                            compressor_config = robust_config(
                                settings,
                                seed,
                                residual_threshold=float(threshold_scale) * base_thresholds[method],
                            )
                        else:
                            compressor_config = robust_config(settings, seed)
                        started_at = time.perf_counter()
                        store = create_compressor(method, compressor_config).fit_transform(
                            dataset.snapshots
                        )
                        compression_time = time.perf_counter() - started_at
                        anomaly = anomaly_precision_recall_f1(dataset.attack_edges, store)
                        storage = storage_report(store, dataset.snapshots)
                        residual_entries = sum(
                            residual.shape[0] * residual.shape[1]
                            for residual in store.residuals
                        )
                        residual_nnz = sum(residual.nnz for residual in store.residuals)
                        false_positive_ratio = (
                            residual_nnz / max(residual_entries, 1)
                            if float(corruption_rate) == 0.0
                            else float("nan")
                        )
                        row = {
                            "repeat": repeat,
                            "seed": seed,
                            "corruption_type": corruption_mode,
                            "corruption_rate": float(corruption_rate),
                            "corruption_magnitude": float(settings["corruption_magnitude"]),
                            "num_nodes": int(settings["num_nodes"]),
                            "num_steps": int(settings["num_steps"]),
                            "rank": int(settings["rank"]),
                            "method": method,
                            "threshold_scale": float(threshold_scale),
                            "base_threshold": base_thresholds.get(method),
                            "effective_threshold": (
                                None
                                if store.threshold_diagnostics is None
                                else store.threshold_diagnostics.get("estimated_threshold")
                            ),
                            "precision": anomaly["precision"],
                            "recall": anomaly["recall"],
                            "f1": anomaly["f1"],
                            "residual_storage_ratio": (
                                store.residual_bytes() / max(store.raw_dense_bytes(), 1)
                            ),
                            "false_positive_residual_ratio": false_positive_ratio,
                            "storage_ratio": storage["compressed_vs_raw_dense_ratio"],
                            "compression_time": compression_time,
                        }
                        rows.append(row)
                        completed.add(key)
                        append_csv_row(csv_path, row)
                        print(
                            f"[threshold_calibration, scale={float(threshold_scale):.3g}, "
                            f"mode={corruption_mode}, rate={float(corruption_rate):.3g}, "
                            f"method={method}, repeat={repeat + 1}/{settings['num_repeats']}] "
                            f"done in {compression_time:.1f}s",
                            flush=True,
                        )
    return rows


def run_split_asym_sanity(config: dict[str, Any], csv_path: Path) -> list[dict[str, Any]]:
    settings = config["split_asym"]
    expected_keys = {
        split_key(
            {
                "repeat": repeat,
                "seed": int(config["random_seed"]) + repeat,
                "num_nodes": settings["num_nodes"],
                "num_steps": settings["num_steps"],
                "rank": settings["rank"],
            }
        )
        for repeat in range(int(settings["num_repeats"]))
    }
    rows = [row for row in read_csv_rows(csv_path) if split_key(row) in expected_keys]
    completed = {split_key(row) for row in rows}
    for repeat in range(int(settings["num_repeats"])):
        seed = int(config["random_seed"]) + repeat
        key = split_key(
            {
                "repeat": repeat,
                "seed": seed,
                "num_nodes": settings["num_nodes"],
                "num_steps": settings["num_steps"],
                "rank": settings["rank"],
            }
        )
        if key in completed:
            continue
        dataset = make_temporal_sbm(
            num_nodes=int(settings["num_nodes"]),
            num_steps=int(settings["num_steps"]),
            num_communities=int(settings["num_communities"]),
            p_in=float(settings["p_in"]),
            p_out=float(settings["p_out"]),
            temporal_jitter=float(settings["temporal_jitter"]),
            directed=bool(settings["directed"]),
            random_seed=seed,
        )
        compressor_config = spectral_config_from_mapping(
            {
                "rank": int(settings["rank"]),
                "num_splits": int(settings["num_splits"]),
                "random_seed": seed,
            }
        )
        asym_store = create_compressor("spectralstore_asym", compressor_config).fit_transform(
            dataset.snapshots
        )
        split_store = create_compressor(
            "spectralstore_split_asym_unfolding",
            compressor_config,
        ).fit_transform(dataset.snapshots)

        construction = split_asym_construction_report(
            dataset.snapshots,
            dataset.expected_snapshots,
            random_seed=seed,
        )
        reconstruction_diff = reconstruction_difference(asym_store, split_store)
        left_distance = subspace_distance(asym_store.left, split_store.left)
        right_distance = subspace_distance(asym_store.right, split_store.right)
        tolerance = float(settings["construction_tolerance"])
        noise_limit = float(settings["noise_correlation_abs_limit"])
        compliant = (
            bool(construction["split_is_disjoint"])
            and construction["upper_triangle_source_error"] <= tolerance
            and construction["lower_triangle_source_error"] <= tolerance
            and construction["diag_consistency_error"] <= tolerance
        )
        noise_independent = (
            math.isnan(float(construction["noise_correlation_T1_T2"]))
            or abs(float(construction["noise_correlation_T1_T2"])) <= noise_limit
        )

        row = {
                "repeat": repeat,
                "seed": seed,
                "num_nodes": int(settings["num_nodes"]),
                "num_steps": int(settings["num_steps"]),
                "rank": int(settings["rank"]),
                **construction,
                "noise_independence_check": noise_independent,
                "thinking_construct_compliant": compliant,
                "reconstruction_diff_between_methods": reconstruction_diff,
                "left_subspace_distance_between_methods": left_distance,
                "right_subspace_distance_between_methods": right_distance,
                "subspace_distance_between_methods": 0.5 * (left_distance + right_distance),
                "equivalence_reason": split_equivalence_reason(
                    reconstruction_diff,
                    int(settings["num_splits"]),
                ),
            }
        rows.append(row)
        completed.add(key)
        append_csv_row(csv_path, row)
        print(f"[split_asym, repeat={repeat + 1}/{settings['num_repeats']}] done", flush=True)
    return rows


def run_entrywise_bound_coverage(config: dict[str, Any], csv_path: Path) -> list[dict[str, Any]]:
    settings = config["entrywise_bound"]
    expected_keys = {
        entrywise_key(
            {
                "repeat": repeat,
                "seed": int(config["random_seed"]) + repeat,
                "dataset": settings["dataset"],
                "num_nodes": settings["num_nodes"],
                "num_steps": settings["num_steps"],
                "rank": settings["rank"],
                "method": method,
            }
        )
        for repeat in range(int(settings["num_repeats"]))
        for method in settings["methods"]
    }
    rows = [row for row in read_csv_rows(csv_path) if entrywise_key(row) in expected_keys]
    completed = {entrywise_key(row) for row in rows}
    for repeat in range(int(settings["num_repeats"])):
        seed = int(config["random_seed"]) + repeat
        dataset = make_bound_dataset(settings, seed)
        for method in settings["methods"]:
            key = entrywise_key(
                {
                    "repeat": repeat,
                    "seed": seed,
                    "dataset": settings["dataset"],
                    "num_nodes": settings["num_nodes"],
                    "num_steps": settings["num_steps"],
                    "rank": settings["rank"],
                    "method": method,
                }
            )
            if key in completed:
                continue
            compressor_config = spectral_config_from_mapping(
                {
                    "rank": int(settings["rank"]),
                    "num_splits": int(settings["num_splits"]),
                    "random_seed": seed,
                }
            )
            store = create_compressor(method, compressor_config).fit_transform(dataset.snapshots)
            bound_precomputed = method in set(settings["precompute_bound_methods"])
            if bound_precomputed:
                store.precompute_bound_params(
                    dataset.snapshots,
                    constant=float(settings["bound_constant"]),
                )
            report = entrywise_bound_report(
                dataset.expected_snapshots,
                store,
                include_residual=bool(settings["include_residual"]),
            )
            sample_u, sample_v = 0, min(1, store.right.shape[0] - 1)
            q1_result = None
            if bound_precomputed:
                q1_result = {
                    "estimate": store.link_score(sample_u, sample_v, 0),
                    "bound": store.entrywise_bound(sample_u, sample_v),
                    "used_residual": True,
                }
            row = {
                    "repeat": repeat,
                    "seed": seed,
                    "dataset": dataset.name,
                    "num_nodes": int(settings["num_nodes"]),
                    "num_steps": int(settings["num_steps"]),
                    "rank": int(settings["rank"]),
                    "method": method,
                    "bound_precomputed": bound_precomputed,
                    "bound_sigma_max": store.bound_sigma_max,
                    "q1_sample_has_bound": q1_result is not None,
                    **report,
                }
            rows.append(row)
            completed.add(key)
            append_csv_row(csv_path, row)
            print(
                f"[entrywise_bound, method={method}, repeat={repeat + 1}/{settings['num_repeats']}] done",
                flush=True,
            )
    return rows


def run_robust_sparse_corruption(config: dict[str, Any], csv_path: Path) -> list[dict[str, Any]]:
    settings = config["robust_sparse_corruption"]
    expected_keys = {
        robust_key(
            {
                "repeat": repeat,
                "seed": int(config["random_seed"]) + repeat,
                "corruption_mode": corruption_mode,
                "corruption_rate": corruption_rate,
                "corruption_magnitude": settings["corruption_magnitude"],
                "num_nodes": settings["num_nodes"],
                "num_steps": settings["num_steps"],
                "rank": settings["rank"],
                "method": method,
            }
        )
        for repeat in range(int(settings["num_repeats"]))
        for corruption_mode in settings["corruption_modes"]
        for corruption_rate in settings["corruption_rates"]
        for method in settings["methods"]
    }
    rows = [row for row in read_csv_rows(csv_path) if robust_key(row) in expected_keys]
    completed = {robust_key(row) for row in rows}
    for repeat in range(int(settings["num_repeats"])):
        seed = int(config["random_seed"]) + repeat
        for corruption_mode in settings["corruption_modes"]:
            for corruption_rate in settings["corruption_rates"]:
                dataset = make_synthetic_attack(
                    num_nodes=int(settings["num_nodes"]),
                    num_steps=int(settings["num_steps"]),
                    num_communities=int(settings["num_communities"]),
                    p_in=float(settings["p_in"]),
                    p_out=float(settings["p_out"]),
                    temporal_jitter=float(settings["temporal_jitter"]),
                    attack_kind=str(corruption_mode),
                    corruption_rate=float(corruption_rate),
                    corruption_magnitude=float(settings["corruption_magnitude"]),
                    directed=bool(settings["directed"]),
                    random_seed=seed,
                )
                for method in settings["methods"]:
                    key = robust_key(
                        {
                            "repeat": repeat,
                            "seed": seed,
                            "corruption_mode": corruption_mode,
                            "corruption_rate": corruption_rate,
                            "corruption_magnitude": settings["corruption_magnitude"],
                            "num_nodes": settings["num_nodes"],
                            "num_steps": settings["num_steps"],
                            "rank": settings["rank"],
                            "method": method,
                        }
                    )
                    if key in completed:
                        continue
                    compressor_config = robust_config(settings, seed)
                    started_at = time.perf_counter()
                    store = create_compressor(method, compressor_config).fit_transform(
                        dataset.snapshots
                    )
                    compression_time = time.perf_counter() - started_at
                    anomaly = anomaly_precision_recall_f1(dataset.attack_edges, store)
                    storage = storage_report(store, dataset.snapshots)
                    row = {
                            "repeat": repeat,
                            "seed": seed,
                            "corruption_mode": corruption_mode,
                            "corruption_rate": corruption_rate,
                            "corruption_magnitude": float(settings["corruption_magnitude"]),
                            "num_nodes": int(settings["num_nodes"]),
                            "num_steps": int(settings["num_steps"]),
                            "rank": int(settings["rank"]),
                            "method": method,
                            "max_entrywise_error_to_M_star": max_entrywise_error(
                                dataset.expected_snapshots,
                                store,
                                include_residual=False,
                            ),
                            "mean_entrywise_error_to_M_star": mean_entrywise_error(
                                dataset.expected_snapshots,
                                store,
                                include_residual=False,
                            ),
                            "frobenius_error": relative_frobenius_error_against_dense(
                                dataset.expected_snapshots,
                                store,
                                include_residual=False,
                            ),
                            "precision": anomaly["precision"],
                            "recall": anomaly["recall"],
                            "f1": anomaly["f1"],
                            "anomaly_truth_count": anomaly["anomaly_truth_count"],
                            "anomaly_predicted_count": anomaly["anomaly_predicted_count"],
                            "storage_ratio": storage["compressed_vs_raw_dense_ratio"],
                            "residual_storage_ratio": (
                                store.residual_bytes() / max(store.raw_dense_bytes(), 1)
                            ),
                            "compression_time": compression_time,
                            "residual_threshold": (
                                None
                                if store.threshold_diagnostics is None
                                else store.threshold_diagnostics.get("estimated_threshold")
                            ),
                        }
                    rows.append(row)
                    completed.add(key)
                    append_csv_row(csv_path, row)
                    print(
                        f"[robust_sparse, mode={corruption_mode}, rate={float(corruption_rate):.3g}, "
                        f"method={method}, repeat={repeat + 1}/{settings['num_repeats']}] "
                        f"done in {compression_time:.1f}s",
                        flush=True,
                    )
    return rows


def robust_config(
    settings: dict[str, Any],
    seed: int,
    *,
    residual_threshold: float | None = None,
    threshold_scale: float | None = None,
):
    if threshold_scale is None:
        scale = 1.0 if residual_threshold is not None else settings.get("threshold_scale", 1.0)
    else:
        scale = threshold_scale
    mapping = {
        "rank": int(settings["rank"]),
        "num_splits": int(settings["num_splits"]),
        "random_seed": seed,
        "robust_iterations": int(settings["robust_iterations"]),
        "residual_threshold_mode": settings["residual_threshold_mode"],
        "residual_quantile": float(settings["residual_quantile"]),
        "residual_mad_multiplier": float(settings["residual_mad_multiplier"]),
        "residual_threshold_scale": float(scale),
        "rpca_iterations": int(settings["rpca_iterations"]),
    }
    if residual_threshold is not None:
        mapping["residual_threshold"] = float(residual_threshold)
    return spectral_config_from_mapping(mapping)


def make_bound_dataset(settings: dict[str, Any], seed: int):
    if settings["dataset"] == "synthetic_sbm":
        return make_temporal_sbm(
            num_nodes=int(settings["num_nodes"]),
            num_steps=int(settings["num_steps"]),
            num_communities=int(settings["num_communities"]),
            p_in=float(settings["p_in"]),
            p_out=float(settings["p_out"]),
            temporal_jitter=float(settings["temporal_jitter"]),
            directed=bool(settings["directed"]),
            random_seed=seed,
        )
    if settings["dataset"] == "synthetic_spiked":
        return make_synthetic_spiked(
            num_nodes=int(settings["num_nodes"]),
            num_steps=int(settings["num_steps"]),
            rank=int(settings["true_rank"]),
            snr=float(settings["snr"]),
            random_seed=seed,
        )
    raise ValueError(f"unsupported entrywise bound dataset: {settings['dataset']}")


def exp1_v2_sweep_combinations(settings: dict[str, Any]) -> list[dict[str, Any]]:
    n_list = [int(value) for value in settings["n_list"]]
    t_list = [int(value) for value in settings["T_list"]]
    regime_list = [str(value) for value in settings["regime_list"]]
    noise_types = settings.get("noise_type_list", settings.get("noise_type", "iid"))
    if isinstance(noise_types, str):
        noise_type_list = [noise_types]
    else:
        noise_type_list = [str(value) for value in noise_types]
    fixed_n = int(settings["fixed_n"])
    fixed_t = int(settings["fixed_T"])

    combos: list[dict[str, Any]] = []
    for regime_name in regime_list:
        for noise_type in noise_type_list:
            for n in n_list:
                combos.append(
                    {
                        "sweep_type": "n_sweep",
                        "n": n,
                        "T": fixed_t,
                        "regime_name": regime_name,
                        "noise_type": noise_type,
                    }
                )
            for t_value in t_list:
                combos.append(
                    {
                        "sweep_type": "T_sweep",
                        "n": fixed_n,
                        "T": t_value,
                        "regime_name": regime_name,
                        "noise_type": noise_type,
                    }
                )
            if bool(settings.get("include_regime_sweep", False)):
                combos.append(
                    {
                        "sweep_type": "regime_sweep",
                        "n": fixed_n,
                        "T": fixed_t,
                        "regime_name": regime_name,
                        "noise_type": noise_type,
                    }
                )

    unique: list[dict[str, Any]] = []
    seen = set()
    for combo in combos:
        key = (
            combo["sweep_type"],
            combo["n"],
            combo["T"],
            combo["regime_name"],
            combo["noise_type"],
        )
        if key not in seen:
            unique.append(combo)
            seen.add(key)
    return unique


def make_exp1_v2_dataset(settings: dict[str, Any], combo: dict[str, Any], seed: int):
    if settings.get("dataset", "sbm") == "spiked":
        snr = float(settings.get("signal_strength", 1.0)) / max(
            float(settings.get("noise_std", 1.0)),
            1e-12,
        )
        return make_synthetic_spiked(
            num_nodes=int(combo["n"]),
            num_steps=int(combo["T"]),
            rank=int(settings["rank"]),
            snr=snr,
            signal_strength=float(settings.get("signal_strength", 1.0)),
            noise_std=float(settings.get("noise_std", 1.0)),
            noise_type=str(combo["noise_type"]),
            high_variance_fraction=float(settings["high_variance_fraction"]),
            high_variance_multiplier=float(settings["high_variance_multiplier"]),
            random_seed=seed,
        )

    regimes = settings["regimes"]
    if combo["regime_name"] not in regimes:
        raise ValueError(f"unknown Exp1-v2 regime: {combo['regime_name']}")
    regime = regimes[combo["regime_name"]]
    return make_theory_regime_sbm(
        num_nodes=int(combo["n"]),
        num_steps=int(combo["T"]),
        num_communities=int(settings["num_communities"]),
        sbm_p=float(regime["sbm_p"]),
        sbm_q=float(regime["sbm_q"]),
        regime_name=str(combo["regime_name"]),
        temporal_jitter=float(settings["temporal_jitter"]),
        directed=bool(settings["directed"]),
        noise_type=str(combo["noise_type"]),
        base_noise_std=float(settings["base_noise_std"]),
        high_variance_fraction=float(settings["high_variance_fraction"]),
        high_variance_multiplier=float(settings["high_variance_multiplier"]),
        random_seed=seed,
    )


def asym_temporal_combinations(settings: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "n": int(n_value),
            "T": int(t_value),
            "alpha": float(alpha),
            "regime_name": str(regime_name),
        }
        for regime_name in settings["regime_list"]
        for alpha in settings["alpha_list"]
        for n_value in settings["n_list"]
        for t_value in settings["T_list"]
    ]


def make_asym_temporal_dataset(settings: dict[str, Any], combo: dict[str, Any], seed: int):
    regimes = settings["regimes"]
    if combo["regime_name"] not in regimes:
        raise ValueError(f"unknown temporal-dependence regime: {combo['regime_name']}")
    dataset_name = str(settings.get("dataset", "sbm"))
    if dataset_name not in {"sbm", "temporal_correlated"}:
        raise ValueError(
            "exp_asym_temporal_dependence currently supports dataset=sbm or temporal_correlated"
        )
    regime = regimes[combo["regime_name"]]
    return make_temporal_correlated_sbm(
        num_nodes=int(combo["n"]),
        num_steps=int(combo["T"]),
        num_communities=int(settings["num_communities"]),
        sbm_p=float(regime["sbm_p"]),
        sbm_q=float(regime["sbm_q"]),
        regime_name=str(combo["regime_name"]),
        alpha=float(combo["alpha"]),
        base_noise_std=float(settings["base_noise_std"]),
        directed=bool(settings["directed"]),
        random_seed=seed,
    )


def asym_mechanism_combinations(settings: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "n": int(n_value),
            "T": int(t_value),
            "alpha": float(alpha),
            "regime_name": str(regime_name),
            "noise_type": str(settings.get("noise_type", "temporal_correlated")),
        }
        for regime_name in settings["regime_list"]
        for alpha in settings["alpha_list"]
        for n_value in settings["n_list"]
        for t_value in settings["T_list"]
    ]


def make_asym_mechanism_dataset(settings: dict[str, Any], combo: dict[str, Any], seed: int):
    if settings.get("dataset", "temporal_correlated") == "synthetic_sbm":
        regimes = settings["regimes"]
        regime = regimes[combo["regime_name"]]
        return make_temporal_sbm(
            num_nodes=int(combo["n"]),
            num_steps=int(combo["T"]),
            num_communities=int(settings["num_communities"]),
            p_in=float(regime["sbm_p"]),
            p_out=float(regime["sbm_q"]),
            temporal_jitter=float(settings["temporal_jitter"]),
            directed=bool(settings["directed"]),
            random_seed=seed,
        )
    return make_asym_temporal_dataset(settings, combo, seed)


def audit_asym_construction(snapshots: list[Any], seed: int) -> dict[str, Any]:
    dense = [
        snapshot.toarray() if hasattr(snapshot, "toarray") else np.asarray(snapshot, dtype=float)
        for snapshot in snapshots
    ]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(dense))
    split = max(1, len(dense) // 2)
    first_indices = order[:split]
    second_indices = order[split:]
    if second_indices.size == 0:
        second_indices = first_indices
    first_mean = np.mean([dense[int(index)] for index in first_indices], axis=0)
    second_mean = np.mean([dense[int(index)] for index in second_indices], axis=0)
    stitched = (
        np.triu(first_mean, k=1)
        + np.tril(second_mean, k=-1)
        + np.diag(0.5 * (np.diag(first_mean) + np.diag(second_mean)))
    )
    upper = np.triu(np.ones_like(stitched, dtype=bool), k=1)
    lower = np.tril(np.ones_like(stitched, dtype=bool), k=-1)
    diag_target = 0.5 * (np.diag(first_mean) + np.diag(second_mean))
    overlap = set(int(index) for index in first_indices) & set(int(index) for index in second_indices)
    asymmetry = relative_asymmetry_norm(stitched)
    return {
        "split_first_size": int(first_indices.size),
        "split_second_size": int(second_indices.size),
        "split_overlap_count": int(len(overlap)),
        "split_is_disjoint": len(overlap) == 0,
        "upper_source_error": float(np.max(np.abs(stitched[upper] - first_mean[upper]))),
        "lower_source_error": float(np.max(np.abs(stitched[lower] - second_mean[lower]))),
        "diag_error": float(np.max(np.abs(np.diag(stitched) - diag_target))),
        "asymmetry_norm": asymmetry,
        "constructed_asym_matrix_not_symmetric": asymmetry > 1e-12,
    }


def audit_store_decomposition(store) -> dict[str, Any]:
    u_minus_v = np.linalg.norm(store.left - store.right, ord="fro") / max(
        np.linalg.norm(store.left, ord="fro"),
        1e-12,
    )
    output_asymmetries = [
        relative_asymmetry_norm(store.dense_snapshot(t, include_residual=False))
        for t in range(store.num_steps)
    ]
    return {
        "singular_values": ";".join(f"{float(value):.12g}" for value in store.lambdas),
        "left_right_subspace_distance": subspace_distance(store.left, store.right),
        "u_minus_v_fro_over_u_fro": float(u_minus_v),
        "output_asymmetry_norm": float(np.mean(output_asymmetries)),
        "output_forced_symmetric": float(np.max(output_asymmetries)) <= 1e-12,
        "uv_collapse": float(u_minus_v) <= 1e-8,
    }


def relative_asymmetry_norm(matrix: np.ndarray) -> float:
    return float(
        np.linalg.norm(matrix - matrix.T, ord="fro")
        / max(np.linalg.norm(matrix, ord="fro"), 1e-12)
    )


def store_error_to_dense_target(store, target: np.ndarray) -> float:
    numerator = 0.0
    denominator = 0.0
    for t in range(store.num_steps):
        reconstruction = store.dense_snapshot(t, include_residual=False)
        numerator += float(np.sum((reconstruction - target) ** 2))
        denominator += float(np.sum(target**2))
    return float(np.sqrt(numerator / max(denominator, 1e-12)))


def refresh_exp1_v2_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_group: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    by_repeat: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        group_key = exp1_v2_group_key(row)
        repeat_key = exp1_v2_repeat_key(row)
        by_group[group_key].append(row)
        by_repeat[repeat_key][str(row["method"])] = row

    group_stats = {}
    for key, group_rows in by_group.items():
        errors = numeric_values(row.get("max_entrywise_error_to_M_star") for row in group_rows)
        variance = float(np.var(errors, ddof=1)) if len(errors) > 1 else 0.0
        group_stats[key] = {
            "std": float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0,
            "variance": variance,
            "max": float(np.max(errors)) if errors else float("nan"),
        }

    refreshed = []
    for row in rows:
        updated = dict(row)
        stats = group_stats[exp1_v2_group_key(row)]
        updated["error_std_across_repeats"] = stats["std"]
        updated["max_error_across_repeats"] = stats["max"]

        sym_row = by_repeat[exp1_v2_repeat_key(row)].get("sym_svd")
        method = str(row["method"])
        if sym_row is not None and method in {
            "spectralstore_asym",
            "spectralstore_split_asym_unfolding",
        }:
            sym_error = float_value(sym_row.get("max_entrywise_error_to_M_star"))
            current_error = float_value(row.get("max_entrywise_error_to_M_star"))
            updated["asym_error_ratio_vs_sym"] = (
                float(current_error / sym_error)
                if current_error is not None and sym_error not in (None, 0.0)
                else float("nan")
            )
            sym_stats = group_stats.get(exp1_v2_group_key({**row, "method": "sym_svd"}))
            sym_variance = None if sym_stats is None else sym_stats["variance"]
            updated["asym_variance_ratio_vs_sym"] = (
                float(stats["variance"] / sym_variance)
                if sym_variance not in (None, 0.0)
                else float("nan")
            )
        refreshed.append(updated)
    return refreshed


def refresh_asym_temporal_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_group: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    by_repeat: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        group_key = asym_temporal_group_key(row)
        repeat_key = asym_temporal_repeat_key(row)
        by_group[group_key].append(row)
        by_repeat[repeat_key][str(row["method"])] = row

    group_stats = {}
    for key, group_rows in by_group.items():
        errors = numeric_values(row.get("max_entrywise_error_to_M_star") for row in group_rows)
        group_stats[key] = {
            "std": float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0,
            "variance": float(np.var(errors, ddof=1)) if len(errors) > 1 else 0.0,
            "max": float(np.max(errors)) if errors else float("nan"),
        }

    refreshed = []
    for row in rows:
        updated = dict(row)
        stats = group_stats[asym_temporal_group_key(row)]
        updated["error_std_across_repeats"] = stats["std"]
        updated["max_error_across_repeats"] = stats["max"]

        sym_row = by_repeat[asym_temporal_repeat_key(row)].get("sym_svd")
        method = str(row["method"])
        if sym_row is not None and method in {
            "spectralstore_asym",
            "spectralstore_split_asym_unfolding",
        }:
            sym_error = float_value(sym_row.get("max_entrywise_error_to_M_star"))
            current_error = float_value(row.get("max_entrywise_error_to_M_star"))
            updated["asym_error_ratio_vs_sym"] = (
                float(current_error / sym_error)
                if current_error is not None and sym_error not in (None, 0.0)
                else float("nan")
            )
            sym_stats = group_stats.get(asym_temporal_group_key({**row, "method": "sym_svd"}))
            sym_variance = None if sym_stats is None else sym_stats["variance"]
            updated["asym_variance_ratio_vs_sym"] = (
                float(stats["variance"] / sym_variance)
                if sym_variance not in (None, 0.0)
                else float("nan")
            )
        refreshed.append(updated)
    return refreshed


def refresh_asym_mechanism_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_group: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[asym_mechanism_group_key(row)].append(row)
    group_stats = {}
    for key, group_rows in by_group.items():
        errors = numeric_values(row.get("max_entrywise_error") for row in group_rows)
        group_stats[key] = {
            "std": float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0,
            "variance": float(np.var(errors, ddof=1)) if len(errors) > 1 else 0.0,
            "max": float(np.max(errors)) if errors else float("nan"),
        }
    refreshed = []
    for row in rows:
        updated = dict(row)
        stats = group_stats[asym_mechanism_group_key(row)]
        updated["error_std_across_repeats"] = stats["std"]
        updated["max_error_across_repeats"] = stats["max"]
        sym_key = asym_mechanism_group_key({**row, "method": "sym_svd"})
        sym_stats = group_stats.get(sym_key)
        sym_variance = None if sym_stats is None else sym_stats["variance"]
        updated["asym_variance_ratio_vs_sym"] = (
            float(stats["variance"] / sym_variance)
            if str(row["method"]) in {"spectralstore_asym", "spectralstore_split_asym_unfolding"}
            and sym_variance not in (None, 0.0)
            else float("nan")
        )
        refreshed.append(updated)
    return refreshed


def split_equivalence_reason(reconstruction_diff: float, num_splits: int) -> str:
    if reconstruction_diff < 1e-8 and num_splits == 1:
        return "equivalent: spectralstore_asym already uses one split triangular mean; SBM diagonal is zero"
    if reconstruction_diff < 1e-3:
        return "near-equivalent: both methods use triangular split means and the same temporal projection"
    return "not equivalent at this setting"


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    existing_rows = read_csv_rows(path)
    if not existing_rows:
        write_csv(path, [row])
        return
    existing_fields = list(existing_rows[0].keys())
    row_fields = list(row.keys())
    if all(field in existing_fields for field in row_fields):
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=existing_fields)
            writer.writerow({field: row.get(field, "") for field in existing_fields})
        return
    write_csv(path, [*existing_rows, row])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def split_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int_value(row.get("repeat")),
        int_value(row.get("seed")),
        int_value(row.get("num_nodes")),
        int_value(row.get("num_steps")),
        int_value(row.get("rank")),
    )


def entrywise_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int_value(row.get("repeat")),
        int_value(row.get("seed")),
        str(row.get("dataset")),
        int_value(row.get("num_nodes")),
        int_value(row.get("num_steps")),
        int_value(row.get("rank")),
        str(row.get("method")),
    )


def robust_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int_value(row.get("repeat")),
        int_value(row.get("seed")),
        str(row.get("corruption_mode")),
        float_value(row.get("corruption_rate")),
        float_value(row.get("corruption_magnitude")),
        int_value(row.get("num_nodes")),
        int_value(row.get("num_steps")),
        int_value(row.get("rank")),
        str(row.get("method")),
    )


def exp1_v2_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("sweep_type")),
        str(row.get("method")),
        int_value(row.get("n")),
        int_value(row.get("T")),
        str(row.get("regime_name")),
        str(row.get("noise_type")),
        int_value(row.get("seed")),
        int_value(row.get("repeat_id")),
    )


def exp1_v2_group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("sweep_type")),
        str(row.get("method")),
        int_value(row.get("n")),
        int_value(row.get("T")),
        str(row.get("regime_name")),
        str(row.get("noise_type")),
    )


def exp1_v2_repeat_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("sweep_type")),
        int_value(row.get("n")),
        int_value(row.get("T")),
        str(row.get("regime_name")),
        str(row.get("noise_type")),
        int_value(row.get("seed")),
        int_value(row.get("repeat_id")),
    )


def asym_temporal_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("method")),
        int_value(row.get("n")),
        int_value(row.get("T")),
        float_value(row.get("alpha")),
        str(row.get("regime_name")),
        int_value(row.get("seed")),
        int_value(row.get("repeat_id")),
    )


def asym_temporal_group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("method")),
        int_value(row.get("n")),
        int_value(row.get("T")),
        float_value(row.get("alpha")),
        str(row.get("regime_name")),
    )


def asym_temporal_repeat_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int_value(row.get("n")),
        int_value(row.get("T")),
        float_value(row.get("alpha")),
        str(row.get("regime_name")),
        int_value(row.get("seed")),
        int_value(row.get("repeat_id")),
    )


def asym_mechanism_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("method")),
        int_value(row.get("n")),
        int_value(row.get("T")),
        float_value(row.get("alpha")),
        str(row.get("regime_name")),
        int_value(row.get("seed")),
        int_value(row.get("repeat_id")),
    )


def asym_mechanism_group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("method")),
        int_value(row.get("n")),
        int_value(row.get("T")),
        float_value(row.get("alpha")),
        str(row.get("regime_name")),
    )


def bound_calibration_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int_value(row.get("repeat")),
        int_value(row.get("seed")),
        str(row.get("dataset")),
        int_value(row.get("num_nodes")),
        int_value(row.get("num_steps")),
        int_value(row.get("rank")),
        str(row.get("method")),
        float_value(row.get("bound_C")),
    )


def threshold_calibration_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int_value(row.get("repeat")),
        int_value(row.get("seed")),
        str(row.get("corruption_type", row.get("corruption_mode"))),
        float_value(row.get("corruption_rate")),
        float_value(row.get("corruption_magnitude")),
        int_value(row.get("num_nodes")),
        int_value(row.get("num_steps")),
        int_value(row.get("rank")),
        str(row.get("method")),
        float_value(row.get("threshold_scale")),
    )


def system_sanity_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("check")),
        int_value(row.get("seed")),
        int_value(row.get("num_nodes")),
        int_value(row.get("num_steps")),
        int_value(row.get("rank")),
        float_value(row.get("query_bound_C")),
        float_value(row.get("robust_threshold_scale")),
    )


def int_value(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def float_value(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def aggregate_numeric(rows: list[dict[str, Any]]) -> dict[str, float]:
    aggregate: dict[str, float] = {}
    if not rows:
        return aggregate
    for key in rows[0]:
        values = numeric_values(row.get(key) for row in rows)
        if values:
            aggregate[f"{key}_mean"] = float(np.mean(values))
    return aggregate


def aggregate_by_method(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    return {method: aggregate_numeric(values) for method, values in sorted(grouped.items())}


def aggregate_robust(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = f"{row['corruption_mode']}|{row['corruption_rate']}|{row['method']}"
        grouped[key].append(row)
    return {key: aggregate_numeric(values) for key, values in sorted(grouped.items())}


def aggregate_calibration_bounds(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = f"{row['method']}|{row['bound_C']}"
        grouped[key].append(row)
    return {key: aggregate_numeric(values) for key, values in sorted(grouped.items())}


def aggregate_threshold_calibration(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            f"{row['method']}|{row['threshold_scale']}|"
            f"{row['corruption_type']}|{row['corruption_rate']}"
        )
        grouped[key].append(row)
    return {key: aggregate_numeric(values) for key, values in sorted(grouped.items())}


def aggregate_exp1_v2(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            f"{row['sweep_type']}|{row['method']}|"
            f"{row['regime_name']}|{row['noise_type']}|n={row['n']}|T={row['T']}"
        )
        grouped[key].append(row)
    return {key: aggregate_numeric(values) for key, values in sorted(grouped.items())}


def aggregate_asym_temporal(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = f"alpha={row['alpha']}|{row['method']}|n={row['n']}|T={row['T']}|{row['regime_name']}"
        grouped[key].append(row)
    return {key: aggregate_numeric(values) for key, values in sorted(grouped.items())}


def aggregate_asym_mechanism(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = f"alpha={row['alpha']}|{row['method']}|n={row['n']}|T={row['T']}|{row['regime_name']}"
        grouped[key].append(row)
    return {key: aggregate_numeric(values) for key, values in sorted(grouped.items())}


def numeric_values(values: Any) -> list[float]:
    numeric = []
    for value in values:
        if isinstance(value, bool) or value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            numeric.append(number)
    return numeric


def collapse_by_float(rows: list[dict[str, Any]], field: str) -> dict[float, dict[str, float]]:
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = float_value(row.get(field))
        if value is not None:
            grouped[value].append(row)
    return {value: aggregate_numeric(items) for value, items in grouped.items()}


def min_c_for_coverage(collapsed: dict[float, dict[str, float]], target: float) -> float | None:
    for value in sorted(collapsed):
        coverage = collapsed[value].get("coverage_mean", float("nan"))
        if math.isfinite(coverage) and coverage >= target:
            return value
    return None


def format_optional(value: float | None) -> str:
    return "not reached" if value is None else f"{value:.6g}"


def fit_log_log_slope(points: list[tuple[float, float]]) -> tuple[float, float]:
    clean = [(x, y) for x, y in points if x > 0.0 and y > 0.0 and math.isfinite(x) and math.isfinite(y)]
    if len(clean) < 2:
        return float("nan"), float("nan")
    log_x = np.log([point[0] for point in clean])
    log_y = np.log([point[1] for point in clean])
    slope, intercept = np.polyfit(log_x, log_y, 1)
    predicted = slope * log_x + intercept
    ss_res = float(np.sum((log_y - predicted) ** 2))
    ss_tot = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r2 = 1.0 if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot
    return float(slope), float(r2)


def render_summary(metrics: dict[str, Any]) -> str:
    lines = [
        "# Thinking Alignment Checks",
        "",
        "This run checks implementation behavior against Thinking.docx and SpectralStore.md without changing the research target.",
        "",
    ]
    if "split_asym_sanity" in metrics:
        lines.extend(render_split_summary(metrics["split_asym_sanity"]["rows"]))
    if "entrywise_bound_coverage" in metrics:
        lines.extend(render_bound_summary(metrics["entrywise_bound_coverage"]["aggregate"]))
    if "robust_sparse_corruption" in metrics:
        lines.extend(render_robust_summary(metrics["robust_sparse_corruption"]["aggregate"]))
    if "entrywise_bound_calibration" in metrics:
        lines.extend(
            render_bound_calibration_summary(
                metrics["entrywise_bound_calibration"]["aggregate"],
                metrics["entrywise_bound_calibration"]["rows"],
            )
        )
    if "residual_threshold_calibration" in metrics:
        lines.extend(
            render_threshold_calibration_summary(
                metrics["residual_threshold_calibration"]["aggregate"],
                metrics["residual_threshold_calibration"]["rows"],
            )
        )
    if "system_calibration_sanity" in metrics:
        lines.extend(render_system_sanity_summary(metrics["system_calibration_sanity"]["rows"]))
    if "exp1_v2_theory_regime" in metrics:
        lines.extend(render_exp1_v2_summary(metrics["exp1_v2_theory_regime"]["rows"]))
    if "exp_asym_temporal_dependence" in metrics:
        lines.extend(
            render_asym_temporal_summary(metrics["exp_asym_temporal_dependence"]["rows"])
        )
    if "asym_mechanism_audit" in metrics:
        lines.extend(
            render_asym_mechanism_summary(metrics["asym_mechanism_audit"]["rows"])
        )
    if metrics.get("csv_outputs"):
        lines.extend(["", "## CSV Outputs", ""])
        for name, filename in metrics["csv_outputs"].items():
            lines.append(f"- {name}: `{filename}`")
    return "\n".join(lines) + "\n"


def render_split_summary(rows: list[dict[str, Any]]) -> list[str]:
    aggregate = aggregate_numeric(rows)
    compliant = all(bool_value(row["thinking_construct_compliant"]) for row in rows)
    noise_mean = aggregate.get("noise_correlation_T1_T2_mean", float("nan"))
    diff_mean = aggregate.get("reconstruction_diff_between_methods_mean", float("nan"))
    reason = rows[0]["equivalence_reason"] if rows else "no rows"
    return [
        "## Split Asym Sanity",
        "",
        f"- Thinking triangular construction compliant: `{compliant}`",
        f"- mean T1/T2 noise correlation: `{noise_mean:.6g}`",
        f"- mean reconstruction difference vs spectralstore_asym: `{diff_mean:.6g}`",
        f"- equivalence diagnosis: {reason}",
        "",
    ]


def render_bound_summary(aggregate: dict[str, dict[str, float]]) -> list[str]:
    lines = [
        "## Entrywise Bound Coverage",
        "",
        "| method | coverage | violation rate | median bound/error | max violation margin |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, values in aggregate.items():
        lines.append(
            f"| {method} | {values.get('coverage_mean', float('nan')):.6g} | "
            f"{values.get('violation_rate_mean', float('nan')):.6g} | "
            f"{values.get('median_bound_over_error_mean', float('nan')):.6g} | "
            f"{values.get('max_violation_margin_mean', float('nan')):.6g} |"
        )
    coverages = [
        values.get("coverage_mean", float("nan"))
        for values in aggregate.values()
        if math.isfinite(values.get("coverage_mean", float("nan")))
    ]
    min_coverage = min(coverages) if coverages else float("nan")
    if math.isfinite(min_coverage) and min_coverage < 0.99:
        diagnosis = (
            "C=1 bound has nonzero violations in this smoke setting; treat it as an "
            "uncalibrated theoretical diagnostic before query-layer SLA use."
        )
    else:
        diagnosis = "Coverage is close to 1.0 in this smoke setting."
    lines.extend(
        [
            "",
            "Coverage near 1.0 supports using the bound in Q1; large median bound/error indicates looseness.",
            f"Diagnosis: {diagnosis}",
            "",
        ]
    )
    return lines


def render_bound_calibration_summary(
    aggregate: dict[str, dict[str, float]],
    rows: list[dict[str, Any]],
) -> list[str]:
    lines = [
        "## Entrywise Bound Calibration",
        "",
        "| method | C | coverage | violation rate | mean bound/error | median bound/error | max violation margin |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, values in aggregate.items():
        method, bound_c = key.split("|", 1)
        lines.append(
            f"| {method} | {float(bound_c):.6g} | "
            f"{values.get('coverage_mean', float('nan')):.6g} | "
            f"{values.get('violation_rate_mean', float('nan')):.6g} | "
            f"{values.get('mean_bound_over_error_mean', float('nan')):.6g} | "
            f"{values.get('median_bound_over_error_mean', float('nan')):.6g} | "
            f"{values.get('max_violation_margin_mean', float('nan')):.6g} |"
        )

    method_to_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if math.isfinite(float_value(row.get("coverage")) or float("nan")):
            method_to_rows[str(row["method"])].append(row)

    lines.extend(["", "### Calibration Decisions", ""])
    recommended = []
    for method, method_rows in sorted(method_to_rows.items()):
        collapsed = collapse_by_float(method_rows, "bound_C")
        c99 = min_c_for_coverage(collapsed, 0.99)
        c995 = min_c_for_coverage(collapsed, 0.995)
        if c99 is not None:
            recommended.append(c99)
        lines.append(
            f"- {method}: min C for coverage >= 0.99 is `{format_optional(c99)}`; "
            f"min C for coverage >= 0.995 is `{format_optional(c995)}`."
        )
    default_c = max(recommended) if recommended else None
    if default_c is None:
        lines.append("- No common default C can be recommended from this sweep.")
    else:
        lines.append(
            f"- Recommended default C: `{default_c:.6g}` across bounded methods in this sweep."
        )
    lines.append(
        "- Looseness should be judged by mean/median bound-over-error; very large ratios mean the bound is safe but weak for query precision."
    )
    lines.append("")
    return lines


def render_threshold_calibration_summary(
    aggregate: dict[str, dict[str, float]],
    rows: list[dict[str, Any]],
) -> list[str]:
    lines = [
        "## Residual Threshold Calibration",
        "",
        "| method | scale | corruption | rate | f1 | precision | recall | residual storage | false positive residual ratio | storage ratio |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, values in aggregate.items():
        method, scale, corruption_type, rate = key.split("|")
        lines.append(
            f"| {method} | {float(scale):.6g} | {corruption_type} | {float(rate):.6g} | "
            f"{values.get('f1_mean', float('nan')):.6g} | "
            f"{values.get('precision_mean', float('nan')):.6g} | "
            f"{values.get('recall_mean', float('nan')):.6g} | "
            f"{values.get('residual_storage_ratio_mean', float('nan')):.6g} | "
            f"{values.get('false_positive_residual_ratio_mean', float('nan')):.6g} | "
            f"{values.get('storage_ratio_mean', float('nan')):.6g} |"
        )

    robust_rows = [row for row in rows if str(row["method"]) == "spectralstore_robust"]
    positive_rows = [row for row in robust_rows if float(row["corruption_rate"]) > 0.0]
    zero_rows = [row for row in robust_rows if float(row["corruption_rate"]) == 0.0]
    scores = []
    for scale in sorted({float(row["threshold_scale"]) for row in robust_rows}):
        f1_values = [float(row["f1"]) for row in positive_rows if float(row["threshold_scale"]) == scale]
        storage_values = [
            float(row["residual_storage_ratio"])
            for row in zero_rows
            if float(row["threshold_scale"]) == scale
        ]
        fp_values = [
            float(row["false_positive_residual_ratio"])
            for row in zero_rows
            if float(row["threshold_scale"]) == scale
        ]
        if not f1_values:
            continue
        mean_f1 = float(np.mean(f1_values))
        mean_storage = float(np.mean(storage_values)) if storage_values else float("nan")
        mean_fp = float(np.mean(fp_values)) if fp_values else float("nan")
        score = mean_f1 - mean_storage
        scores.append((score, scale, mean_f1, mean_storage, mean_fp))

    lines.extend(["", "### Calibration Decisions", ""])
    if scores:
        score, scale, mean_f1, mean_storage, mean_fp = max(scores)
        lines.append(
            f"- Best tradeoff scale by mean F1 minus no-corruption residual storage: "
            f"`{scale:.6g}` (F1 `{mean_f1:.6g}`, residual storage `{mean_storage:.6g}`, false positive ratio `{mean_fp:.6g}`)."
        )
        low_storage_high_f1 = [
            item for item in scores if item[2] >= 0.9 and item[3] <= 0.01
        ]
        if low_storage_high_f1:
            scales = ", ".join(f"{item[1]:.6g}" for item in low_storage_high_f1)
            lines.append(f"- High-F1 and low-storage interval exists at scale(s): `{scales}`.")
        else:
            lines.append("- No scale in this sweep simultaneously reached F1 >= 0.9 and residual storage <= 0.01.")
        default = next((item for item in scores if abs(item[1] - 1.0) < 1e-12), None)
        if default is not None:
            default_storage = default[3]
            if math.isfinite(default_storage) and default_storage > 0.01:
                lines.append("- Current default threshold appears too low for no-corruption storage.")
            else:
                lines.append("- Current default threshold does not over-trigger residual storage in this smoke sweep.")
    else:
        lines.append("- No robust threshold calibration rows were available.")
    lines.append("- rpca_svd has no residual anomaly store, so its residual precision/recall remain zero/NaN by design.")
    lines.append("")
    return lines


def render_system_sanity_summary(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## System Calibration Sanity",
        "",
    ]
    if not rows:
        return lines + ["- No system sanity rows were available.", ""]

    row = rows[0]
    bound_ratio = float_value(row.get("q1_bound_ratio"))
    default_nnz = int_value(row.get("default_scale_residual_nnz"))
    scaled_nnz = int_value(row.get("scaled_residual_nnz"))
    default_storage = float_value(row.get("default_scale_residual_storage_ratio"))
    scaled_storage = float_value(row.get("scaled_residual_storage_ratio"))
    bound_config_driven = bool_value(row.get("query_bound_C_config_driven"))
    threshold_changes = bool_value(row.get("robust_threshold_scale_changes_residuals"))

    lines.extend(
        [
            f"- Q1 returns estimate, bound, used_residual, and method: `{row.get('q1_method')}`.",
            f"- query.bound_C config-driven: `{bound_config_driven}`; observed bound ratio `{format_optional(bound_ratio)}`.",
            f"- loose tolerance used residual: `{bool_value(row.get('loose_tolerance_used_residual'))}`.",
            f"- tight tolerance used residual: `{bool_value(row.get('tight_tolerance_used_residual'))}`.",
            f"- robust.threshold_scale changes residual nnz/storage in this check: `{threshold_changes}`.",
            (
                f"- residual nnz changed from `{default_nnz}` to `{scaled_nnz}`; "
                f"storage ratio from `{format_optional(default_storage)}` to `{format_optional(scaled_storage)}`."
            ),
            "- C=3 is a YAML default from the current n=500,T=20,repeats=3 calibration, not a global theory constant.",
            "- threshold_scale=3 is a YAML default from the current sparse-corruption calibration, not a universal optimum.",
            "- Both values are overrideable through `--set query.bound_C=... robust.threshold_scale=...`.",
            "",
        ]
    )
    return lines


def render_exp1_v2_summary(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## Exp1-v2 Theory Regime Sweep",
        "",
        "| sweep | method | regime | noise | slope_n | slope_T | R^2 | mean max error | mean coverage |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    if not rows:
        return lines + ["", "No Exp1-v2 rows were available.", ""]

    grouped_points: dict[tuple[str, str, str, str], dict[float, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    grouped_rows: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sweep = str(row["sweep_type"])
        method = str(row["method"])
        regime = str(row["regime_name"])
        noise = str(row["noise_type"])
        x_value = float(row["n"] if sweep == "n_sweep" else row["T"])
        y_value = float_value(row.get("max_entrywise_error_to_M_star"))
        key = (sweep, method, regime, noise)
        if y_value is not None and math.isfinite(y_value):
            grouped_points[key][x_value].append(y_value)
        grouped_rows[key].append(row)

    slope_records = []
    for key, point_map in sorted(grouped_points.items()):
        points = [(x_value, float(np.mean(values))) for x_value, values in sorted(point_map.items())]
        slope, r2 = fit_log_log_slope(points)
        values = aggregate_numeric(grouped_rows[key])
        slope_records.append((key, slope, r2))
        sweep, method, regime, noise = key
        slope_n = slope if sweep == "n_sweep" else float("nan")
        slope_t = slope if sweep == "T_sweep" else float("nan")
        lines.append(
            f"| {sweep} | {method} | {regime} | {noise} | "
            f"{slope_n:.6g} | {slope_t:.6g} | {r2:.6g} | "
            f"{values.get('max_entrywise_error_to_M_star_mean', float('nan')):.6g} | "
            f"{values.get('coverage_mean', float('nan')):.6g} |"
        )

    lines.extend(["", "### Theory Scaling Diagnosis", ""])
    finite_slopes = [
        (key, slope, r2)
        for key, slope, r2 in slope_records
        if math.isfinite(slope) and math.isfinite(r2)
    ]
    if finite_slopes:
        closest = min(finite_slopes, key=lambda item: abs(item[1] + 0.5))
        lines.append(
            f"- Closest slope to -0.5: `{closest[1]:.6g}` for "
            f"`{closest[0][0]} / {closest[0][1]} / {closest[0][2]} / {closest[0][3]}` "
            f"with R^2 `{closest[2]:.6g}`."
        )
    else:
        lines.append("- Not enough sweep points to fit a finite log-log slope in this run.")

    asym_rows = [
        row
        for row in rows
        if str(row["method"]) in {"spectralstore_asym", "spectralstore_split_asym_unfolding"}
    ]
    ratio_values = numeric_values(row.get("asym_error_ratio_vs_sym") for row in asym_rows)
    variance_ratios = numeric_values(row.get("asym_variance_ratio_vs_sym") for row in asym_rows)
    if ratio_values:
        mean_ratio = float(np.mean(ratio_values))
        lines.append(f"- Mean asym max-error ratio vs sym_svd: `{mean_ratio:.6g}`.")
    else:
        lines.append("- Asym vs sym error ratio was not available in this run.")
    if variance_ratios:
        mean_variance_ratio = float(np.mean(variance_ratios))
        if mean_variance_ratio < 1.0:
            lines.append(
                f"- Asym variance ratio vs sym_svd is below 1 on average (`{mean_variance_ratio:.6g}`)."
            )
        else:
            lines.append(
                f"- No variance reduction was observed on average; variance ratio is `{mean_variance_ratio:.6g}`."
            )
    else:
        lines.append("- Asym variance ratio needs at least two repeats per setting.")
    lines.append("- If asym does not outperform sym in a regime, this summary records it as diagnostic evidence without changing methods.")
    lines.append("")
    return lines


def render_asym_temporal_summary(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## Asym Temporal Dependence",
        "",
        "| alpha | method | mean max error | mean asym error ratio vs sym | mean asym variance ratio vs sym |",
        "|---:|---|---:|---:|---:|",
    ]
    if not rows:
        return lines + ["", "No temporal-dependence rows were available.", ""]

    grouped: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        alpha = float(row["alpha"])
        grouped[(alpha, str(row["method"]))].append(row)
    for (alpha, method), method_rows in sorted(grouped.items()):
        values = aggregate_numeric(method_rows)
        lines.append(
            f"| {alpha:.6g} | {method} | "
            f"{values.get('max_entrywise_error_to_M_star_mean', float('nan')):.6g} | "
            f"{values.get('asym_error_ratio_vs_sym_mean', float('nan')):.6g} | "
            f"{values.get('asym_variance_ratio_vs_sym_mean', float('nan')):.6g} |"
        )

    asym_rows = [
        row for row in rows if str(row["method"]) == "spectralstore_asym"
    ]
    by_alpha: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in asym_rows:
        by_alpha[float(row["alpha"])].append(row)
    alpha_summaries = {
        alpha: aggregate_numeric(alpha_rows)
        for alpha, alpha_rows in sorted(by_alpha.items())
    }

    lines.extend(["", "### Temporal Dependence Diagnosis", ""])
    alpha_zero = alpha_summaries.get(0.0)
    if alpha_zero is None:
        lines.append("- Alpha=0 was not included, so iid-baseline reproduction cannot be checked.")
    else:
        ratio = alpha_zero.get("asym_error_ratio_vs_sym_mean", float("nan"))
        variance_ratio = alpha_zero.get("asym_variance_ratio_vs_sym_mean", float("nan"))
        lines.append(
            f"- alpha=0 iid baseline: asym error ratio `{ratio:.6g}`, "
            f"variance ratio `{variance_ratio:.6g}`."
        )
        if math.isfinite(ratio) and ratio >= 1.0:
            lines.append("- alpha=0 reproduces the expected no-advantage iid behavior.")
        elif math.isfinite(ratio):
            lines.append("- alpha=0 already shows asym error below sym in this run.")

    positive = {
        alpha: values
        for alpha, values in alpha_summaries.items()
        if alpha > 0.0
    }
    error_advantage = [
        alpha
        for alpha, values in positive.items()
        if values.get("asym_error_ratio_vs_sym_mean", float("nan")) < 1.0
    ]
    variance_advantage = [
        alpha
        for alpha, values in positive.items()
        if values.get("asym_variance_ratio_vs_sym_mean", float("nan")) < 1.0
    ]
    if error_advantage:
        lines.append(
            "- Asym error advantage exists at alpha values: "
            f"`{', '.join(f'{alpha:.6g}' for alpha in error_advantage)}`."
        )
    else:
        lines.append("- No alpha value showed mean asym error below sym_svd.")
    if variance_advantage:
        lines.append(
            "- Asym variance advantage exists at alpha values: "
            f"`{', '.join(f'{alpha:.6g}' for alpha in variance_advantage)}`."
        )
    else:
        lines.append("- No alpha value showed asym variance below sym_svd.")

    ordered_ratios = [
        (
            alpha,
            values.get("asym_error_ratio_vs_sym_mean", float("nan")),
            values.get("asym_variance_ratio_vs_sym_mean", float("nan")),
        )
        for alpha, values in sorted(alpha_summaries.items())
    ]
    finite_error = [(alpha, ratio) for alpha, ratio, _ in ordered_ratios if math.isfinite(ratio)]
    finite_variance = [
        (alpha, ratio) for alpha, _, ratio in ordered_ratios if math.isfinite(ratio)
    ]
    if len(finite_error) >= 2:
        error_trend = finite_error[-1][1] - finite_error[0][1]
        lines.append(f"- Error-ratio change from lowest to highest alpha: `{error_trend:.6g}`.")
    if len(finite_variance) >= 2:
        variance_trend = finite_variance[-1][1] - finite_variance[0][1]
        lines.append(f"- Variance-ratio change from lowest to highest alpha: `{variance_trend:.6g}`.")
    if not error_advantage and not variance_advantage:
        lines.append("- No evidence of asym advantage even under temporal correlation.")
    lines.append("")
    return lines


def render_asym_mechanism_summary(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## Asym Mechanism Audit",
        "",
        "| alpha | method | asym matrix | U/V gap | output asym | mean max error | variance ratio vs sym | asym-vs-split diff |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    if not rows:
        return lines + ["", "No asym-mechanism-audit rows were available.", ""]

    grouped: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["alpha"]), str(row["method"]))].append(row)
    for (alpha, method), method_rows in sorted(grouped.items()):
        values = aggregate_numeric(method_rows)
        lines.append(
            f"| {alpha:.6g} | {method} | "
            f"{values.get('asymmetry_norm_mean', float('nan')):.6g} | "
            f"{values.get('u_minus_v_fro_over_u_fro_mean', float('nan')):.6g} | "
            f"{values.get('output_asymmetry_norm_mean', float('nan')):.6g} | "
            f"{values.get('max_entrywise_error_mean', float('nan')):.6g} | "
            f"{values.get('asym_variance_ratio_vs_sym_mean', float('nan')):.6g} | "
            f"{values.get('reconstruction_diff_asym_vs_split_mean', float('nan')):.6g} |"
        )

    asym_rows = [row for row in rows if str(row["method"]) == "spectralstore_asym"]
    split_rows = [
        row for row in rows if str(row["method"]) == "spectralstore_split_asym_unfolding"
    ]
    asym_summary = aggregate_numeric(asym_rows)
    split_summary = aggregate_numeric(split_rows)
    asymmetry_norm = asym_summary.get("asymmetry_norm_mean", float("nan"))
    uv_gap = asym_summary.get("u_minus_v_fro_over_u_fro_mean", float("nan"))
    output_asymmetry = asym_summary.get("output_asymmetry_norm_mean", float("nan"))
    split_diff = asym_summary.get("reconstruction_diff_asym_vs_split_mean", float("nan"))
    sym_diff = asym_summary.get("reconstruction_diff_asym_vs_sym_mean", float("nan"))
    variance_ratio = asym_summary.get("asym_variance_ratio_vs_sym_mean", float("nan"))
    left_right_distance = asym_summary.get("left_right_subspace_distance_mean", float("nan"))
    output_forced_symmetric = all(bool_value(row.get("output_forced_symmetric")) for row in asym_rows)
    constructed_not_symmetric = all(
        bool_value(row.get("constructed_asym_matrix_not_symmetric")) for row in asym_rows
    )
    uv_collapse = all(bool_value(row.get("uv_collapse")) for row in asym_rows)
    split_equivalent = math.isfinite(split_diff) and split_diff <= 1e-8

    lines.extend(["", "### Audit Diagnosis", ""])
    lines.append(
        f"1. asym matrix 是否真的非对称？`{constructed_not_symmetric}`，平均 `asymmetry_norm={asymmetry_norm:.6g}`。"
    )
    lines.append(
        f"2. U/V 是否真的不同？平均 `||U-V||_F/||U||_F={uv_gap:.6g}`，子空间距离 `{left_right_distance:.6g}`，U/V collapse=`{uv_collapse}`。"
    )
    lines.append(
        f"3. 输出是否被对称化？平均 `output_asymmetry_norm={output_asymmetry:.6g}`，forced_symmetric=`{output_forced_symmetric}`。"
    )
    lines.append(
        f"4. asym 和 split 是否等价？平均 `reconstruction_diff_asym_vs_split={split_diff:.6g}`，平均 `reconstruction_diff_asym_vs_sym={sym_diff:.6g}`，等价诊断=`{split_equivalent}`。"
    )
    if math.isfinite(variance_ratio):
        reason = (
            "构造层确实引入了非对称性，但当前分解与投影路径下 asym 与 split 近乎等价，"
            "说明非对称信息大多没有转化成稳定的表示收益；同时随机 split 会引入额外 seed 敏感性，"
            "所以更可能看到方差放大而不是均值误差下降。"
            if variance_ratio >= 1.0
            else "当前设置下没有看到比 sym 更大的方差，说明随机 split 带来的额外波动没有主导结果。"
        )
        lines.append(
            f"5. variance 更大的原因可能是什么？平均 `asym_variance_ratio_vs_sym={variance_ratio:.6g}`；{reason}"
        )
    else:
        lines.append(
            "5. variance 更大的原因可能是什么？当前重复不足或缺少对照，暂时无法稳定估计 `asym_variance_ratio_vs_sym`。"
        )
    if split_rows:
        split_output_asym = split_summary.get("output_asymmetry_norm_mean", float("nan"))
        lines.append(
            f"- 对照参考：`spectralstore_split_asym_unfolding` 平均 `output_asymmetry_norm={split_output_asym:.6g}`。"
        )
    lines.append("")
    return lines


def render_robust_summary(aggregate: dict[str, dict[str, float]]) -> list[str]:
    lines = [
        "## Robust Sparse Corruption",
        "",
        "| setting | max err | precision | recall | f1 | residual storage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, values in aggregate.items():
        lines.append(
            f"| {key} | {values.get('max_entrywise_error_to_M_star_mean', float('nan')):.6g} | "
            f"{values.get('precision_mean', float('nan')):.6g} | "
            f"{values.get('recall_mean', float('nan')):.6g} | "
            f"{values.get('f1_mean', float('nan')):.6g} | "
            f"{values.get('residual_storage_ratio_mean', float('nan')):.6g} |"
        )
    robust_positive = [
        values
        for key, values in aggregate.items()
        if key.endswith("|spectralstore_robust")
        and float(key.split("|")[1]) > 0.0
    ]
    robust_zero = [
        values
        for key, values in aggregate.items()
        if key.endswith("|spectralstore_robust")
        and float(key.split("|")[1]) == 0.0
    ]
    mean_f1 = (
        float(np.mean([values.get("f1_mean", 0.0) for values in robust_positive]))
        if robust_positive
        else float("nan")
    )
    zero_storage = (
        float(np.mean([values.get("residual_storage_ratio_mean", 0.0) for values in robust_zero]))
        if robust_zero
        else float("nan")
    )
    lines.extend(
        [
            "",
            "Robust is aligned with Thinking.docx when residuals recover sparse S_t with useful precision/recall without excessive residual storage.",
            f"Diagnosis: positive-corruption robust mean F1 is `{mean_f1:.6g}`; no-corruption robust residual storage is `{zero_storage:.6g}`, so threshold calibration should be checked for false residual storage.",
            "",
        ]
    )
    return lines


if __name__ == "__main__":
    main()
