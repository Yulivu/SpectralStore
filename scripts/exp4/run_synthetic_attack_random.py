"""Run Experiment 4 random-attack robustness sweep."""

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
from spectralstore.data_loader import make_synthetic_attack  # noqa: E402
from spectralstore.evaluation import max_entrywise_error  # noqa: E402


METHODS = [
    "spectralstore_robust",
    "spectralstore_asym",
    "sym_svd",
    "direct_svd",
    "rpca_svd",
]
EPSILONS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
RAW_FIELDNAMES = [
    "epsilon",
    "method",
    "repeat",
    "max_entrywise",
    "nmi",
    "runtime",
]
SUMMARY_FIELDNAMES = [
    "epsilon",
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
    parser.add_argument("--out-dir", default="experiments/results/exp4/random_attack")
    parser.add_argument("--num-nodes", type=int, default=1000)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--num-communities", type=int, default=3)
    parser.add_argument("--p-in", type=float, default=0.30)
    parser.add_argument("--p-out", type=float, default=0.05)
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--num-repeats", type=int, default=5)
    parser.add_argument("--rpca-iterations", type=int, default=100)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "raw_records.csv"
    raw_rows = run_sweep(args, raw_path)
    summary_rows = summarize(raw_rows)

    summary_path = out_dir / "summary.csv"
    write_csv(summary_path, summary_rows, SUMMARY_FIELDNAMES)
    plot_metric(
        summary_rows,
        metric="max_entrywise",
        ylabel="max entrywise error vs M_star",
        output_path=out_dir / "max_entrywise_vs_epsilon.png",
    )
    plot_metric(
        summary_rows,
        metric="nmi",
        ylabel="NMI",
        output_path=out_dir / "nmi_vs_epsilon.png",
    )

    print(f"wrote {raw_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {out_dir / 'max_entrywise_vs_epsilon.png'}")
    print(f"wrote {out_dir / 'nmi_vs_epsilon.png'}")
    print_endpoint_summary(summary_rows)


def run_sweep(
    args: argparse.Namespace,
    raw_path: Path,
) -> list[dict[str, float | int | str]]:
    rows = read_raw_rows(raw_path)
    completed = {
        (round(float(row["epsilon"]), 10), str(row["method"]), int(row["repeat"]))
        for row in rows
    }
    total_repeats = int(args.num_repeats)
    for epsilon in EPSILONS:
        for repeat in range(total_repeats):
            dataset = make_synthetic_attack(
                num_nodes=args.num_nodes,
                num_steps=args.num_steps,
                num_communities=args.num_communities,
                p_in=args.p_in,
                p_out=args.p_out,
                attack_kind="random_flip",
                attack_fraction=epsilon,
                directed=True,
                random_seed=repeat,
            )
            for method in METHODS:
                key = (round(float(epsilon), 10), method, repeat)
                if key in completed:
                    continue
                started_at = time.perf_counter()
                config = spectral_config_from_mapping(
                    {
                        "rank": args.rank,
                        "random_seed": repeat,
                        "rpca_iterations": args.rpca_iterations,
                    }
                )
                store = create_compressor(method, config).fit_transform(dataset.snapshots)
                runtime = time.perf_counter() - started_at
                row = {
                    "epsilon": epsilon,
                    "method": method,
                    "repeat": repeat,
                    "max_entrywise": max_entrywise_error(
                        dataset.expected_snapshots,
                        store,
                        include_residual=False,
                    ),
                    "nmi": community_nmi_from_left(
                        store.left,
                        dataset.communities,
                        num_communities=args.num_communities,
                        random_seed=repeat,
                    ),
                    "runtime": runtime,
                }
                rows.append(row)
                completed.add(key)
                append_raw_row(raw_path, row)
                print(
                    f"[eps={epsilon:.2f}, method={method}, "
                    f"repeat={repeat + 1}/{total_repeats}] done in {runtime:.1f}s",
                    flush=True,
                )
    return rows


def read_raw_rows(path: Path) -> list[dict[str, float | int | str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_raw_row(path: Path, row: dict[str, float | int | str]) -> None:
    frame = pd.DataFrame([row], columns=RAW_FIELDNAMES)
    frame.to_csv(
        path,
        mode="a",
        header=not os.path.exists(path),
        index=False,
    )


def community_nmi_from_left(
    left: np.ndarray,
    true_labels: np.ndarray,
    *,
    num_communities: int,
    random_seed: int,
) -> float:
    labels = KMeans(
        n_clusters=num_communities,
        random_state=random_seed,
        n_init=10,
    ).fit_predict(left)
    return float(normalized_mutual_info_score(true_labels, labels))


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | str]]:
    grouped: dict[tuple[float, str], list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["epsilon"]), str(row["method"]))].append(row)

    summary_rows = []
    for epsilon, method in sorted(grouped):
        values = grouped[(epsilon, method)]
        max_values = np.asarray([float(row["max_entrywise"]) for row in values], dtype=float)
        nmi_values = np.asarray([float(row["nmi"]) for row in values], dtype=float)
        runtime_values = np.asarray([float(row["runtime"]) for row in values], dtype=float)
        summary_rows.append(
            {
                "epsilon": epsilon,
                "method": method,
                "max_entrywise_mean": float(np.mean(max_values)),
                "max_entrywise_std": float(np.std(max_values, ddof=1)) if len(max_values) > 1 else 0.0,
                "nmi_mean": float(np.mean(nmi_values)),
                "nmi_std": float(np.std(nmi_values, ddof=1)) if len(nmi_values) > 1 else 0.0,
                "runtime_mean": float(np.mean(runtime_values)),
                "runtime_std": (
                    float(np.std(runtime_values, ddof=1)) if len(runtime_values) > 1 else 0.0
                ),
            }
        )
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_metric(
    summary_rows: list[dict[str, float | str]],
    *,
    metric: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for method in METHODS:
        rows = sorted(
            [row for row in summary_rows if row["method"] == method],
            key=lambda row: float(row["epsilon"]),
        )
        x = np.asarray([float(row["epsilon"]) for row in rows], dtype=float)
        mean = np.asarray([float(row[f"{metric}_mean"]) for row in rows], dtype=float)
        std = np.asarray([float(row[f"{metric}_std"]) for row in rows], dtype=float)
        ax.plot(x, mean, marker="o", linewidth=1.6, label=method)
        ax.fill_between(x, mean - std, mean + std, alpha=0.16)

    ax.set_xlabel("random attack epsilon")
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def print_endpoint_summary(summary_rows: list[dict[str, float | str]]) -> None:
    for epsilon in (0.0, 0.30):
        print(f"epsilon={epsilon:.2f}")
        rows = [row for row in summary_rows if float(row["epsilon"]) == epsilon]
        for method in METHODS:
            row = next(item for item in rows if item["method"] == method)
            print(
                "  "
                f"{method}: "
                f"max_entrywise={float(row['max_entrywise_mean']):.6f}, "
                f"nmi={float(row['nmi_mean']):.6f}"
            )


if __name__ == "__main__":
    main()
