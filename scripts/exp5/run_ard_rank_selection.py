"""Run Experiment 5 — ARD rank selection vs fixed and cross-validation."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spectralstore.compression import SpectralCompressionConfig, create_compressor  # noqa: E402
from spectralstore.data_loader import make_temporal_sbm  # noqa: E402
from spectralstore.evaluation import (  # noqa: E402
    load_experiment_config,
    relative_frobenius_error_against_dense,
    set_reproducibility_seed,
)

METHOD = "spectralstore_thinking"
RAW_FIELDNAMES = [
    "true_rank",
    "n",
    "T",
    "repeat",
    "rank_mode",
    "requested_rank",
    "effective_rank",
    "ard_converged",
    "ard_iterations",
    "frobenius_error",
    "runtime",
]

SUMMARY_FIELDNAMES = [
    "true_rank",
    "n",
    "T",
    "rank_mode",
    "rank_accuracy_mean",
    "rank_accuracy_std",
    "frobenius_error_mean",
    "frobenius_error_std",
    "runtime_mean",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()
    config = load_experiment_config(args.config, args.overrides)
    set_reproducibility_seed(config.get("random_seed"))

    out_dir = Path(args.out_dir or config.get("output_dir", "experiments/results/exp5"))
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "ard_rank_raw.csv"
    raw_rows = run_sweep(config, raw_path)
    summary_rows = summarize(raw_rows)

    summary_path = out_dir / "ard_rank_summary.csv"
    write_csv(summary_path, summary_rows, SUMMARY_FIELDNAMES)
    plot_results(summary_rows, out_dir)

    print(f"wrote {raw_path}")
    print(f"wrote {summary_path}")


def run_sweep(run_cfg: dict, raw_path: Path) -> list[dict]:
    rows = _read_csv_rows(raw_path, RAW_FIELDNAMES)
    completed = {
        (int(r["true_rank"]), int(r["n"]), int(r["T"]), int(r["repeat"]), str(r["rank_mode"]), int(r["requested_rank"]))
        for r in rows
    }
    base_seed = int(run_cfg.get("random_seed", 0))
    total_repeats = int(run_cfg["num_repeats"])
    ard_cfg = run_cfg.get("ard", {})
    ard_max_rank = int(ard_cfg.get("max_rank", 30))

    for true_rank in run_cfg["sbm"]["true_rank_values"]:
        for n in run_cfg["sbm"]["n_values"]:
            for T in run_cfg["sbm"]["t_values"]:
                for repeat in range(total_repeats):
                    seed = base_seed + repeat
                    dataset = make_temporal_sbm(
                        num_nodes=n,
                        num_steps=T,
                        num_communities=run_cfg["sbm"]["num_communities"],
                        p_in=run_cfg["sbm"]["p_in"],
                        p_out=run_cfg["sbm"]["p_out"],
                        directed=True,
                        random_seed=seed,
                    )
                    expected = [
                        snap.toarray() if hasattr(snap, "toarray") else np.asarray(snap)
                        for snap in dataset.expected_snapshots
                    ]

                    for rank_mode, requested in _rank_candidates(true_rank, ard_max_rank):
                        key = (int(true_rank), int(n), int(T), int(repeat), str(rank_mode), int(requested))
                        if key in completed:
                            continue
                        started = time.perf_counter()
                        compressor_config = _build_config(run_cfg, rank_mode, requested, seed)
                        store = create_compressor(METHOD, compressor_config).fit_transform(
                            dataset.snapshots
                        )
                        runtime = time.perf_counter() - started
                        diag = store.threshold_diagnostics or {}
                        row = {
                            "true_rank": true_rank,
                            "n": n,
                            "T": T,
                            "repeat": repeat,
                            "rank_mode": rank_mode,
                            "requested_rank": requested,
                            "effective_rank": int(diag.get("effective_rank", store.rank)),
                            "ard_converged": bool(diag.get("ard_converged", False)),
                            "ard_iterations": int(diag.get("ard_iterations", 0)),
                            "frobenius_error": relative_frobenius_error_against_dense(
                                expected, store, include_residual=False
                            ),
                            "runtime": runtime,
                        }
                        rows.append(row)
                        completed.add(key)
                        _append_csv_row(raw_path, row, RAW_FIELDNAMES)
                        print(
                            f"[r*={true_rank} n={n} T={T} repeat={repeat+1}/{total_repeats} "
                            f"mode={rank_mode} req={requested} eff={row['effective_rank']}] "
                            f"done in {runtime:.1f}s",
                            flush=True,
                        )
    return rows


def _rank_candidates(true_rank: int, max_rank: int) -> list[tuple[str, int]]:
    return [
        ("ard", max_rank),
        ("fixed_over", max_rank),
        ("fixed_exact", true_rank),
    ]


def _build_config(run_cfg: dict, rank_mode: str, rank: int, seed: int) -> SpectralCompressionConfig:
    from spectralstore.compression import SpectralCompressionConfig as SCC

    if rank_mode == "ard":
        ard = run_cfg.get("ard", {})
        return SCC(
            rank=rank,
            random_seed=seed,
            rank_selection_mode="ard",
            ard_max_rank=rank,
            ard_prior_alpha=float(ard.get("prior_alpha", 1e-2)),
            ard_prior_beta=float(ard.get("prior_beta", 1e-2)),
            ard_max_iterations=int(ard.get("max_iterations", 100)),
            ard_tolerance=float(ard.get("tolerance", 1e-6)),
            ard_min_effective_ratio=float(ard.get("min_effective_ratio", 0.05)),
        )
    return SCC(rank=rank, random_seed=seed, rank_selection_mode="fixed")


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[(int(r["true_rank"]), int(r["n"]), int(r["T"]), str(r["rank_mode"]))].append(r)

    summary = []
    for (true_rank, n, T, rank_mode), vals in sorted(grouped.items()):
        eff = np.asarray([float(v["effective_rank"]) for v in vals], dtype=float)
        frob = np.asarray([float(v["frobenius_error"]) for v in vals], dtype=float)
        rt = np.asarray([float(v["runtime"]) for v in vals], dtype=float)
        rank_acc = np.mean(eff == true_rank)
        summary.append({
            "true_rank": true_rank,
            "n": n,
            "T": T,
            "rank_mode": rank_mode,
            "rank_accuracy_mean": float(rank_acc),
            "rank_accuracy_std": 0.0,
            "frobenius_error_mean": float(np.mean(frob)),
            "frobenius_error_std": float(np.std(frob, ddof=1)) if len(frob) > 1 else 0.0,
            "runtime_mean": float(np.mean(rt)),
        })
    return summary


def plot_results(summary_rows: list[dict], out_dir: Path) -> None:
    modes = sorted(set(r["rank_mode"] for r in summary_rows))
    for n in sorted(set(r["n"] for r in summary_rows)):
        for T in sorted(set(r["T"] for r in summary_rows)):
            subset = [
                r for r in summary_rows
                if int(r["n"]) == int(n) and int(r["T"]) == int(T)
            ]
            if not subset:
                continue
            fig, ax = plt.subplots(figsize=(7.2, 4.8))
            x_labels = sorted(set(r["true_rank"] for r in subset))
            x = np.arange(len(x_labels))
            width = 0.25
            for i, mode in enumerate(modes):
                mode_rows = [r for r in subset if r["rank_mode"] == mode]
                values = [float(r["rank_accuracy_mean"]) for r in sorted(mode_rows, key=lambda r: r["true_rank"])]
                if values:
                    ax.bar(x + i * width, values, width, label=mode)
            ax.set_xticks(x + width)
            ax.set_xticklabels(x_labels)
            ax.set_xlabel("true rank")
            ax.set_ylabel("rank recovery accuracy")
            ax.set_title(f"ARD rank selection n={n} T={T}")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / f"ard_rank_n{n}_T{T}.png", dpi=180)
            plt.close(fig)


def _read_csv_rows(path: Path, fieldnames: list[str]) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _append_csv_row(path: Path, row: dict, fieldnames: list[str]) -> None:
    import pandas as pd
    pd.DataFrame([row], columns=fieldnames).to_csv(
        path, mode="a", header=not os.path.exists(path), index=False,
    )


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
