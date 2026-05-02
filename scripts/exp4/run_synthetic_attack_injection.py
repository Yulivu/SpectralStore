"""Run Experiment 4 injection-attack robustness sweep."""

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
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spectralstore.compression import create_compressor, spectral_config_from_mapping  # noqa: E402
from spectralstore.data_loader import make_temporal_sbm  # noqa: E402
from spectralstore.evaluation import (  # noqa: E402
    load_experiment_config,
    max_entrywise_error,
    set_reproducibility_seed,
)

DEFAULT_METHODS = ["spectralstore_thinking", "sym_svd", "direct_svd", "rpca_svd"]
DELTAS = [0.01, 0.05, 0.10]
RAW_FIELDNAMES = [
    "delta",
    "method",
    "repeat",
    "max_entrywise",
    "nmi",
    "runtime",
]
SUMMARY_FIELDNAMES = [
    "delta",
    "method",
    "max_entrywise_mean",
    "max_entrywise_std",
    "nmi_mean",
    "nmi_std",
    "runtime_mean",
    "runtime_std",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()
    config = load_experiment_config(args.config, args.overrides)
    set_reproducibility_seed(config.get("random_seed"))

    out_dir = Path(args.out_dir or config.get("output_dir", "experiments/results/exp4/injection_attack"))
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "raw_records.csv"
    methods = list(config.get("methods", DEFAULT_METHODS))
    raw_rows = run_sweep(config, raw_path, methods)
    summary_rows = summarize(raw_rows)

    summary_path = out_dir / "summary.csv"
    write_csv(summary_path, summary_rows, SUMMARY_FIELDNAMES)
    plot_metric(
        summary_rows,
        metric="max_entrywise",
        ylabel="max entrywise error vs M_star",
        output_path=out_dir / "injection_max_entrywise_vs_delta.png",
        methods=methods,
    )
    plot_metric(
        summary_rows,
        metric="nmi",
        ylabel="NMI",
        output_path=out_dir / "injection_nmi_vs_delta.png",
        methods=methods,
    )

    print(f"wrote {raw_path}")
    print(f"wrote {summary_path}")
    print_endpoint_summary(summary_rows, methods)


def make_injection_attack_data(
    run_cfg: dict,
    delta: float,
    seed: int,
):
    clean = make_temporal_sbm(
        num_nodes=run_cfg["num_nodes"],
        num_steps=run_cfg["num_steps"],
        num_communities=run_cfg["num_communities"],
        p_in=run_cfg["p_in"],
        p_out=run_cfg["p_out"],
        directed=True,
        random_seed=seed,
    )
    rng = np.random.default_rng(seed + 20_000)
    num_nodes = clean.num_nodes
    num_fake = max(1, int(round(delta * num_nodes)))
    communities = clean.communities
    num_communities = run_cfg["num_communities"]

    fake_communities = rng.integers(0, num_communities, size=num_fake)
    fake_snaps: list[np.ndarray] = []
    expected_snaps_fake: list[np.ndarray] = []
    for t, snap in enumerate(clean.snapshots):
        dense = snap.toarray().astype(float, copy=True)
        extra_rows = np.zeros((num_fake, num_nodes), dtype=float)
        extra_cols = np.zeros((num_nodes, num_fake), dtype=float)
        for i in range(num_fake):
            comm = fake_communities[i]
            candidates = np.flatnonzero(communities == comm)
            if candidates.size == 0:
                continue
            targets = rng.choice(candidates, size=min(5, candidates.size), replace=False)
            for v in targets:
                extra_rows[i, v] = 1.0
                extra_cols[v, i] = 1.0
        augmented = np.block([[dense, extra_cols], [extra_rows, np.zeros((num_fake, num_fake))]])
        fake_snaps.append(augmented)

        if clean.expected_snapshots:
            exp = clean.expected_snapshots[t].copy()
            exp_aug = np.block([
                [exp, np.zeros((num_nodes, num_fake))],
                [np.zeros((num_fake, num_nodes)), np.zeros((num_fake, num_fake))],
            ])
            expected_snaps_fake.append(exp_aug)

    from scipy import sparse
    return (
        [sparse.csr_matrix(s) for s in fake_snaps],
        expected_snaps_fake or None,
        np.concatenate([communities, fake_communities]),
    )


def run_sweep(run_cfg: dict, raw_path: Path, methods: list[str]) -> list[dict]:
    rows = _read_csv_rows(raw_path, RAW_FIELDNAMES)
    completed = {
        (round(float(r["delta"]), 10), str(r["method"]), int(r["repeat"]))
        for r in rows
    }
    total_repeats = int(run_cfg["num_repeats"])
    base_seed = int(run_cfg.get("random_seed", 0))
    deltas = run_cfg.get("injection_deltas", DELTAS)

    for delta in deltas:
        for repeat in range(total_repeats):
            seed = base_seed + repeat
            snaps, expected, communities = make_injection_attack_data(run_cfg, delta, seed)
            for method in methods:
                key = (round(float(delta), 10), method, repeat)
                if key in completed:
                    continue
                started = time.perf_counter()
                compressor_config = spectral_config_from_mapping({
                    "rank": run_cfg["rank"],
                    "random_seed": seed,
                    "rpca_iterations": run_cfg["rpca_iterations"],
                })
                store = create_compressor(method, compressor_config).fit_transform(snaps)
                runtime = time.perf_counter() - started
                row = {
                    "delta": delta,
                    "method": method,
                    "repeat": repeat,
                    "max_entrywise": (
                        max_entrywise_error(expected, store, include_residual=False)
                        if expected else 0.0
                    ),
                    "nmi": community_nmi_from_left(
                        store.left, communities,
                        num_communities=run_cfg["num_communities"],
                        random_seed=seed,
                    ),
                    "runtime": runtime,
                }
                rows.append(row)
                completed.add(key)
                _append_csv_row(raw_path, row, RAW_FIELDNAMES)
                print(
                    f"[injection, delta={delta:.2f}, method={method}, "
                    f"repeat={repeat + 1}/{total_repeats}] done in {runtime:.1f}s",
                    flush=True,
                )
    return rows


def community_nmi_from_left(left, true_labels, *, num_communities, random_seed):
    labels = KMeans(
        n_clusters=num_communities,
        random_state=random_seed,
        n_init=10,
    ).fit_predict(left)
    return float(normalized_mutual_info_score(true_labels, labels))


def summarize(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[(float(r["delta"]), str(r["method"]))].append(r)
    summary = []
    for (delta, method) in sorted(grouped):
        vals = grouped[(delta, method)]
        maxv = np.asarray([float(v["max_entrywise"]) for v in vals], dtype=float)
        nmi = np.asarray([float(v["nmi"]) for v in vals], dtype=float)
        rt = np.asarray([float(v["runtime"]) for v in vals], dtype=float)
        summary.append({
            "delta": delta, "method": method,
            "max_entrywise_mean": float(np.mean(maxv)),
            "max_entrywise_std": float(np.std(maxv, ddof=1)) if len(maxv) > 1 else 0.0,
            "nmi_mean": float(np.mean(nmi)),
            "nmi_std": float(np.std(nmi, ddof=1)) if len(nmi) > 1 else 0.0,
            "runtime_mean": float(np.mean(rt)),
            "runtime_std": float(np.std(rt, ddof=1)) if len(rt) > 1 else 0.0,
        })
    return summary


def plot_metric(summary_rows, *, metric, ylabel, output_path, methods):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for method in methods:
        rows = sorted(
            [r for r in summary_rows if r["method"] == method],
            key=lambda r: float(r["delta"]),
        )
        x = np.asarray([float(r["delta"]) for r in rows], dtype=float)
        mean = np.asarray([float(r[f"{metric}_mean"]) for r in rows], dtype=float)
        std = np.asarray([float(r[f"{metric}_std"]) for r in rows], dtype=float)
        ax.plot(x, mean, marker="o", linewidth=1.6, label=method)
        ax.fill_between(x, mean - std, mean + std, alpha=0.16)
    ax.set_xlabel("injection delta")
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def print_endpoint_summary(summary_rows, methods):
    for delta in (0.01, 0.10):
        print(f"delta={delta:.2f}")
        rows = [r for r in summary_rows if abs(float(r["delta"]) - delta) < 1e-9]
        if not rows:
            continue
        for method in methods:
            row = next((r for r in rows if r["method"] == method), None)
            if row:
                print(f"  {method}: max_entrywise={float(row['max_entrywise_mean']):.6f}, nmi={float(row['nmi_mean']):.6f}")


def _read_csv_rows(path, fieldnames):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _append_csv_row(path, row, fieldnames):
    pd.DataFrame([row], columns=fieldnames).to_csv(
        path, mode="a", header=not os.path.exists(path), index=False,
    )


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


if __name__ == "__main__":
    main()
