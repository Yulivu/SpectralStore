"""Plot Experiment 1 theory-reference figures from an existing results CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_LABELS = {
    "spectralstore_asym": "asym",
    "sym_svd": "sym_svd",
    "direct_svd": "direct_svd",
    "tensor_unfolding_svd": "tensor_unfolding_svd",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-csv",
        default="experiments/results/exp1/results.csv",
        help="Existing Experiment 1 results.csv to read.",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/results/exp1",
        help="Directory where theory plots will be written.",
    )
    args = parser.parse_args()

    rows = read_results(Path(args.results_csv))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_sweep_with_theory(
        rows,
        sweep="sbm_n",
        x_field="n",
        output_path=out_dir / "sbm_n_sweep_with_theory.png",
        theory_kind="n",
    )
    plot_sweep_with_theory(
        rows,
        sweep="sbm_t",
        x_field="T",
        output_path=out_dir / "sbm_t_sweep_with_theory.png",
        theory_kind="t",
    )

    print(f"wrote {out_dir / 'sbm_n_sweep_with_theory.png'}")
    print(f"wrote {out_dir / 'sbm_t_sweep_with_theory.png'}")


def read_results(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def plot_sweep_with_theory(
    rows: list[dict[str, str]],
    *,
    sweep: str,
    x_field: str,
    output_path: Path,
    theory_kind: str,
) -> None:
    sweep_rows = [row for row in rows if row["sweep"] == sweep]
    if not sweep_rows:
        raise ValueError(f"no rows found for sweep={sweep}")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    methods = sorted({row["method"] for row in sweep_rows})
    for method in methods:
        method_rows = sorted(
            [row for row in sweep_rows if row["method"] == method],
            key=lambda row: float(row[x_field]),
        )
        x = np.asarray([float(row[x_field]) for row in method_rows], dtype=float)
        y = np.asarray([float(row["max_entrywise_mean"]) for row in method_rows], dtype=float)
        yerr = np.asarray([float(row["max_entrywise_std"]) for row in method_rows], dtype=float)
        slope = loglog_slope(x, y)
        label = f"{METHOD_LABELS.get(method, method)} (slope={slope:.2f})"
        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, linewidth=1.6, label=label)

    theory_x, theory_y = theory_curve(sweep_rows, x_field=x_field, theory_kind=theory_kind)
    theory_slope = loglog_slope(theory_x, theory_y)
    theory_label = theory_label_for(theory_kind, theory_slope)
    ax.plot(theory_x, theory_y, linestyle="--", linewidth=1.8, color="black", label=theory_label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_field)
    ax.set_ylabel("max entrywise error")
    ax.grid(True, which="both", linewidth=0.4, alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def theory_curve(
    rows: list[dict[str, str]],
    *,
    x_field: str,
    theory_kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    x_values = sorted({float(row[x_field]) for row in rows})
    x = np.asarray(x_values, dtype=float)

    if theory_kind == "n":
        t_values = {float(row["T"]) for row in rows}
        if len(t_values) != 1:
            raise ValueError("sbm_n theory curve expects one fixed T value")
        t_value = next(iter(t_values))
        raw_theory = np.sqrt(np.log(x) / (x * t_value))
    elif theory_kind == "t":
        raw_theory = 1.0 / np.sqrt(x)
    else:
        raise ValueError(f"unsupported theory_kind: {theory_kind}")

    data_rows = min(
        rows,
        key=lambda row: float(row["max_entrywise_mean"]),
    )
    anchor_x = float(data_rows[x_field])
    anchor_y = float(data_rows["max_entrywise_mean"])
    anchor_index = int(np.where(x == anchor_x)[0][0])
    scale = anchor_y / raw_theory[anchor_index]
    return x, scale * raw_theory


def loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
    keep = (x > 0.0) & (y > 0.0)
    if int(np.count_nonzero(keep)) < 2:
        return float("nan")
    return float(np.polyfit(np.log(x[keep]), np.log(y[keep]), deg=1)[0])


def theory_label_for(theory_kind: str, slope: float) -> str:
    if theory_kind == "n":
        return f"Theory O(√(log n/(nT))) (slope={slope:.2f})"
    return f"Theory O(1/√T) (slope={slope:.2f})"


if __name__ == "__main__":
    main()
