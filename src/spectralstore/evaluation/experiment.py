"""Experiment configuration and reproducibility helpers."""

from __future__ import annotations

import json
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf


def load_experiment_config(
    path: str | Path,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Load a YAML experiment config through OmegaConf/Hydra's config stack."""
    config_path = Path(path)
    config = OmegaConf.load(config_path)
    if overrides:
        _validate_override_keys(config, overrides)
        for override in overrides:
            key, value = _split_override(override)
            OmegaConf.update(config, key, _parse_override_value(value), merge=True)
    return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]


def _validate_override_keys(config: DictConfig | ListConfig, overrides: list[str]) -> None:
    unknown_keys = [
        key
        for key in (_override_key(override) for override in overrides)
        if key and not _config_key_exists(config, key)
    ]
    if unknown_keys:
        joined = ", ".join(sorted(unknown_keys))
        raise KeyError(f"unknown experiment config override key(s): {joined}")


def _override_key(override: str) -> str:
    return _split_override(override)[0]


def _split_override(override: str) -> tuple[str, str]:
    if "=" not in override:
        raise ValueError(f"override must use key=value syntax: {override}")
    key, value = override.split("=", 1)
    return key.strip(), value


def _parse_override_value(value: str) -> Any:
    parsed = OmegaConf.from_dotlist([f"value={value}"])
    return parsed["value"]


def _config_key_exists(config: DictConfig | ListConfig, dotted_key: str) -> bool:
    current: Any = config
    for part in dotted_key.split("."):
        if isinstance(current, DictConfig):
            if part not in current:
                return False
            current = current[part]
        elif isinstance(current, ListConfig):
            if not part.isdigit():
                return False
            index = int(part)
            if index < 0 or index >= len(current):
                return False
            current = current[index]
        else:
            return False
    return True


def set_reproducibility_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def validate_dense_stack_memory_budget(
    num_nodes: int,
    num_steps: int,
    *,
    limit_gb: float,
    label: str = "run",
) -> None:
    estimated_gb = num_steps * num_nodes * num_nodes * 8 / (1024**3)
    if estimated_gb > limit_gb:
        raise MemoryError(
            f"{label} exceeds the configured dense compressor memory budget: "
            f"{num_steps} snapshots x {num_nodes} nodes requires at least "
            f"{estimated_gb:.2f} GiB just for the dense input stack, above "
            f"dense_memory_limit_gb={limit_gb}. Use a smaller max_nodes value or "
            "implement the sparse compressor path before running this scale."
        )


def write_experiment_outputs(
    *,
    out_dir: str | Path,
    metrics: dict[str, Any],
    summary: str,
    config_path: str | Path,
    config: dict[str, Any],
    started_at: float | None = None,
    metrics_filename: str = "metrics.json",
    summary_filename: str = "summary.md",
) -> None:
    ended_at = time.perf_counter()
    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / metrics_filename).write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_path / summary_filename).write_text(summary, encoding="utf-8")
    resolved_config = OmegaConf.create(config)
    (output_path / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(resolved_config, resolve=True),
        encoding="utf-8",
    )
    (output_path / "run_metadata.json").write_text(
        json.dumps(
            run_metadata(
                config_path=config_path,
                config=config,
                out_dir=output_path,
                metrics_filename=metrics_filename,
                summary_filename=summary_filename,
                started_at=started_at,
                ended_at=ended_at,
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def run_metadata(
    *,
    config_path: str | Path,
    config: dict[str, Any],
    out_dir: str | Path | None = None,
    metrics_filename: str = "metrics.json",
    summary_filename: str = "summary.md",
    started_at: float | None = None,
    ended_at: float | None = None,
) -> dict[str, Any]:
    duration = (
        float(ended_at - started_at)
        if started_at is not None and ended_at is not None
        else None
    )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "cwd": str(Path.cwd()),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "config": config,
        "config_hash": object_sha256(config),
        "outputs": {
            "out_dir": str(out_dir) if out_dir is not None else None,
            "metrics": metrics_filename,
            "summary": summary_filename,
            "resolved_config": "resolved_config.yaml",
            "metadata": "run_metadata.json",
        },
        "timing": {
            "duration_seconds": duration,
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": package_versions(
            [
                "numpy",
                "scipy",
                "scikit-learn",
                "hydra-core",
                "omegaconf",
                "tensorly",
                "ogb",
                "spectralstore",
            ]
        ),
        "git": git_metadata(),
    }


def object_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str | None:
    try:
        return sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def package_versions(names: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_metadata() -> dict[str, str | bool | None]:
    return {
        "branch": _git_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _git_output(["git", "rev-parse", "HEAD"]),
        "dirty": _git_dirty(),
    }


def _git_dirty() -> bool | None:
    output = _git_output(["git", "status", "--porcelain"])
    if output is None:
        return None
    return bool(output)


def _git_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()
