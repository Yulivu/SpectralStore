"""Run Experiment 1 theory-validation sweeps."""

from __future__ import annotations

import argparse
import csv
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
from spectralstore.data_loader import make_synthetic_spiked, make_temporal_sbm  # noqa: E402
from spectralstore.evaluation import (  # noqa: E402
    load_experiment_config,
    max_entrywise_error,
    mean_entrywise_error,
    relative_frobenius_error_against_dense,
    set_reproducibility_seed,
    write_experiment_outputs,
)


CSV_FIELDS = [
    "experiment",
    "sweep",
    "n",
    "T",
    "rank",
    "snr",
    "p_in",
    "p_out",
    "method",
    "repeats",
    "seconds_mean",
    "seconds_std",
    "max_entrywise_mean",
    "max_entrywise_std",
    "mean_entrywise_mean",
    "mean_entrywise_std",
    "relative_frobenius_mean",
    "relative_frobenius_std",
    "max_entrywise_power_law_slope",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML config.",
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument(
        "--refresh-from-results",
        action="store_true",
        help="Read existing results.csv, update slopes, and regenerate plots only.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        help="OmegaConf dotlist override, e.g. --set num_repeats=1",
    )
    args = parser.parse_args()
    started_at = time.perf_counter()

    config = load_experiment_config(args.config, args.overrides)
    set_reproducibility_seed(config.get("random_seed"))
    out_dir = Path(args.out_dir or config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.refresh_from_results:
        rows = read_csv(out_dir / "results.csv")
        annotate_power_law_slopes(rows)
        write_csv(out_dir / "results.csv", rows)
        plot_sweep(
            rows,
            sweep="sbm_n",
            x_field="n",
            output_path=out_dir / "sbm_n_sweep.png",
            theory="sbm_n",
        )
        plot_sweep(
            rows,
            sweep="sbm_t",
            x_field="T",
            output_path=out_dir / "sbm_t_sweep.png",
            theory="sbm_t",
        )
        print(render_summary(rows, pilot=False))
        return

    rows = run_pilot(config) if args.pilot else run_full(config)
    annotate_power_law_slopes(rows)
    write_csv(out_dir / "results.csv", rows)
    if not args.pilot:
        plot_sweep(
            rows,
            sweep="sbm_n",
            x_field="n",
            output_path=out_dir / "sbm_n_sweep.png",
            theory="sbm_n",
        )
        plot_sweep(
            rows,
            sweep="sbm_t",
            x_field="T",
            output_path=out_dir / "sbm_t_sweep.png",
            theory="sbm_t",
        )
        plot_sweep(
            rows,
            sweep="spiked_snr",
            x_field="snr",
            output_path=out_dir / "spiked_snr_sweep.png",
            log_x=False,
        )

    metrics = {"rows": rows, "pilot": args.pilot}
    summary = render_summary(rows, pilot=args.pilot)
    write_experiment_outputs(
        out_dir=out_dir,
        metrics=metrics,
        summary=summary,
        config_path=args.config,
        config=config,
        started_at=started_at,
    )
    print(summary)


def run_pilot(config: dict) -> list[dict]:
    return flatten_settings([
        run_setting(
            config,
            experiment="synthetic_sbm",
            sweep="pilot_n500",
            n=500,
            num_steps=20,
            snr=None,
            repeats=1,
        )
    ])


def run_full(config: dict) -> list[dict]:
    sbm = config["sbm"]
    spiked = config["spiked"]
    rows = []
    for n in sbm["n_values"]:
        rows.append(
            run_setting(
                config,
                experiment="synthetic_sbm",
                sweep="sbm_n",
                n=n,
                num_steps=sbm["fixed_t"],
                snr=None,
                repeats=config["num_repeats"],
                p_in=sbm["p_in"],
                p_out=sbm["p_out"],
            )
        )
    for num_steps in sbm["t_values"]:
        rows.append(
            run_setting(
                config,
                experiment="synthetic_sbm",
                sweep="sbm_t",
                n=sbm["fixed_n"],
                num_steps=num_steps,
                snr=None,
                repeats=config["num_repeats"],
                p_in=sbm["p_in"],
                p_out=sbm["p_out"],
            )
        )
    rows.append(
        run_setting(
            config,
            experiment="synthetic_sbm",
            sweep="sbm_low_snr",
            n=sbm["fixed_n"],
            num_steps=sbm["fixed_t"],
            snr=None,
            repeats=config["num_repeats"],
            p_in=sbm["low_snr"]["p_in"],
            p_out=sbm["low_snr"]["p_out"],
        )
    )
    for snr in spiked["snr_values"]:
        rows.append(
            run_setting(
                config,
                experiment="synthetic_spiked",
                sweep="spiked_snr",
                n=spiked["fixed_n"],
                num_steps=spiked["fixed_t"],
                snr=snr,
                repeats=config["num_repeats"],
            )
        )
    return flatten_settings(rows)


def run_setting(
    config: dict,
    *,
    experiment: str,
    sweep: str,
    n: int,
    num_steps: int,
    snr: float | None,
    repeats: int,
    p_in: float | None = None,
    p_out: float | None = None,
) -> dict:
    estimates: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for repeat in range(repeats):
        seed = config["random_seed"] + repeat
        dataset = make_dataset(config, experiment, n, num_steps, snr, seed, p_in, p_out)
        compressor_config = SpectralCompressionConfig(
            rank=config["rank"],
            random_seed=seed,
            num_splits=config["num_splits"],
        )
        for method in config["methods"]:
            started = time.perf_counter()
            store = create_compressor(method, compressor_config).fit_transform(dataset.snapshots)
            estimates[method]["seconds"].append(time.perf_counter() - started)
            estimates[method]["max_entrywise"].append(
                max_entrywise_error(dataset.expected_snapshots, store)
            )
            estimates[method]["mean_entrywise"].append(
                mean_entrywise_error(dataset.expected_snapshots, store)
            )
            estimates[method]["relative_frobenius"].append(
                relative_frobenius_error_against_dense(dataset.expected_snapshots, store)
            )
    return {
        "experiment": experiment,
        "sweep": sweep,
        "n": n,
        "T": num_steps,
        "rank": config["rank"],
        "snr": snr,
        "p_in": p_in,
        "p_out": p_out,
        "repeats": repeats,
        "methods": estimates,
    }


def make_dataset(
    config: dict,
    experiment: str,
    n: int,
    num_steps: int,
    snr: float | None,
    seed: int,
    p_in: float | None,
    p_out: float | None,
):
    if experiment == "synthetic_sbm":
        sbm = config["sbm"]
        return make_temporal_sbm(
            num_nodes=n,
            num_steps=num_steps,
            num_communities=sbm["num_communities"],
            p_in=sbm["p_in"] if p_in is None else p_in,
            p_out=sbm["p_out"] if p_out is None else p_out,
            temporal_jitter=sbm["temporal_jitter"],
            directed=sbm["directed"],
            random_seed=seed,
        )
    if experiment == "synthetic_spiked":
        return make_synthetic_spiked(
            num_nodes=n,
            num_steps=num_steps,
            rank=config["rank"],
            snr=float(snr),
            random_seed=seed,
        )
    raise ValueError(f"unsupported experiment: {experiment}")


def flatten_settings(settings: list[dict]) -> list[dict]:
    rows = []
    for setting in settings:
        for method, metrics in setting["methods"].items():
            rows.append(
                {
                    "experiment": setting["experiment"],
                    "sweep": setting["sweep"],
                    "n": setting["n"],
                    "T": setting["T"],
                    "rank": setting["rank"],
                    "snr": setting["snr"],
                    "p_in": setting["p_in"],
                    "p_out": setting["p_out"],
                    "method": method,
                    "repeats": setting["repeats"],
                    "seconds_mean": mean(metrics["seconds"]),
                    "seconds_std": std(metrics["seconds"]),
                    "max_entrywise_mean": mean(metrics["max_entrywise"]),
                    "max_entrywise_std": std(metrics["max_entrywise"]),
                    "mean_entrywise_mean": mean(metrics["mean_entrywise"]),
                    "mean_entrywise_std": std(metrics["mean_entrywise"]),
                    "relative_frobenius_mean": mean(metrics["relative_frobenius"]),
                    "relative_frobenius_std": std(metrics["relative_frobenius"]),
                }
            )
    return rows


def mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def std(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=float), ddof=0))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def annotate_power_law_slopes(rows: list[dict]) -> None:
    for row in rows:
        row["max_entrywise_power_law_slope"] = ""
    for sweep, x_field in (("sbm_n", "n"), ("sbm_t", "T")):
        sweep_rows = [row for row in rows if row["sweep"] == sweep]
        for method in sorted({row["method"] for row in sweep_rows}):
            method_rows = sorted(
                [row for row in sweep_rows if row["method"] == method],
                key=lambda row: float(row[x_field]),
            )
            x = np.asarray([float(row[x_field]) for row in method_rows], dtype=float)
            y = np.asarray(
                [float(row["max_entrywise_mean"]) for row in method_rows],
                dtype=float,
            )
            keep = (x > 0.0) & (y > 0.0)
            if int(np.count_nonzero(keep)) < 2:
                continue
            slope = float(np.polyfit(np.log(x[keep]), np.log(y[keep]), deg=1)[0])
            for row in method_rows:
                row["max_entrywise_power_law_slope"] = slope


def plot_sweep(
    rows: list[dict],
    *,
    sweep: str,
    x_field: str,
    output_path: Path,
    log_x: bool = True,
    theory: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sweep_rows = [row for row in rows if row["sweep"] == sweep]
    methods = sorted({row["method"] for row in sweep_rows})
    for method in methods:
        method_rows = sorted(
            [row for row in sweep_rows if row["method"] == method],
            key=lambda row: float(row[x_field]),
        )
        x = [float(row[x_field]) for row in method_rows]
        y = [float(row["max_entrywise_mean"]) for row in method_rows]
        yerr = [float(row["max_entrywise_std"]) for row in method_rows]
        label = method
        slope = method_rows[0].get("max_entrywise_power_law_slope", "")
        if slope != "":
            label = f"{method} (slope={float(slope):.2f})"
        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=label)
    if theory is not None:
        add_theory_reference(ax, sweep_rows, x_field, theory)
    if log_x:
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_field)
    ax.set_ylabel("max entrywise error")
    ax.grid(True, which="both", linewidth=0.4, alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def add_theory_reference(
    ax: plt.Axes,
    sweep_rows: list[dict],
    x_field: str,
    theory: str,
) -> None:
    anchor_rows = sorted(
        [row for row in sweep_rows if row["method"] == "spectralstore_thinking"],
        key=lambda row: float(row[x_field]),
    )
    if not anchor_rows:
        return

    x = np.asarray([float(row[x_field]) for row in anchor_rows], dtype=float)
    anchor_y = float(anchor_rows[0]["max_entrywise_mean"])
    if theory == "sbm_n":
        t_value = float(anchor_rows[0]["T"])
        reference = np.sqrt(np.log(x) / (x * t_value))
        label = "theory O(√(logn/nT))"
    elif theory == "sbm_t":
        reference = 1.0 / np.sqrt(x)
        label = "theory O(1/√T)"
    else:
        raise ValueError(f"unsupported theory reference: {theory}")

    scale = anchor_y / reference[0]
    ax.plot(x, scale * reference, linestyle="--", linewidth=1.6, color="black", label=label)


def render_summary(rows: list[dict], *, pilot: bool) -> str:
    title = "# Experiment 1 Pilot Timing" if pilot else "# Experiment 1 Theory Validation"
    lines = [
        title,
        "",
        "| sweep | n | T | SNR | p_in | p_out | method | seconds | max entrywise | mean entrywise | rel. Frobenius |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['sweep']} | {row['n']} | {row['T']} | {row['snr']} | "
            f"{row.get('p_in', '')} | {row.get('p_out', '')} | "
            f"{row['method']} | "
            f"{float(row['seconds_mean']):.3f} +/- {float(row['seconds_std']):.3f} | "
            f"{float(row['max_entrywise_mean']):.6g} +/- {float(row['max_entrywise_std']):.3g} | "
            f"{float(row['mean_entrywise_mean']):.6g} +/- {float(row['mean_entrywise_std']):.3g} | "
            f"{float(row['relative_frobenius_mean']):.6g} +/- {float(row['relative_frobenius_std']):.3g} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
