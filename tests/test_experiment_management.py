import json
import shutil
from pathlib import Path

import pytest

from spectralstore.evaluation import (
    load_experiment_config,
    object_sha256,
    write_experiment_outputs,
)


def test_experiment_config_and_metadata_outputs() -> None:
    config_path = Path("tests") / "_experiment_config_test.yaml"
    out_dir = Path("tests") / "_experiment_output_test"
    config_path.write_text(
        "random_seed: 123\nrank: 4\nnested:\n  value: false\n",
        encoding="utf-8",
    )

    try:
        config = load_experiment_config(config_path, ["rank=6", "nested.value=true"])
        write_experiment_outputs(
            out_dir=out_dir,
            metrics={"value": 1.0},
            summary="# Summary\n",
            config_path=config_path,
            config=config,
            started_at=1.0,
        )

        metadata = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
        resolved = (out_dir / "resolved_config.yaml").read_text(encoding="utf-8")
        assert config == {"random_seed": 123, "rank": 6, "nested": {"value": True}}
        assert (out_dir / "metrics.json").exists()
        assert (out_dir / "summary.md").exists()
        assert "rank: 6" in resolved
        assert metadata["config"] == config
        assert metadata["config_hash"] == object_sha256(config)
        assert metadata["outputs"]["resolved_config"] == "resolved_config.yaml"
        assert metadata["timing"]["duration_seconds"] is not None
        assert metadata["packages"]["hydra-core"] is not None
        assert metadata["packages"]["omegaconf"] is not None
        assert "git" in metadata
    finally:
        config_path.unlink(missing_ok=True)
        shutil.rmtree(out_dir, ignore_errors=True)


def test_experiment_config_rejects_unknown_override_keys() -> None:
    config_path = Path("tests") / "_experiment_config_strict_test.yaml"
    config_path.write_text("rank: 4\nnested:\n  value: false\n", encoding="utf-8")

    try:
        assert load_experiment_config(config_path, ["nested.value=true"])["nested"]["value"] is True
        with pytest.raises(KeyError, match="rank_typo"):
            load_experiment_config(config_path, ["rank_typo=6"])
        with pytest.raises(KeyError, match="nested.missing"):
            load_experiment_config(config_path, ["nested.missing=true"])
    finally:
        config_path.unlink(missing_ok=True)


def test_experiment_config_validates_list_override_keys() -> None:
    config_path = Path("tests") / "_experiment_config_list_test.yaml"
    config_path.write_text(
        "candidates:\n"
        "  - name: first\n"
        "    residual_quantile: 0.99\n",
        encoding="utf-8",
    )

    try:
        config = load_experiment_config(
            config_path,
            ["candidates.0.residual_quantile=0.999"],
        )
        assert config["candidates"][0]["residual_quantile"] == 0.999
        with pytest.raises(KeyError, match="candidates.0.missing"):
            load_experiment_config(config_path, ["candidates.0.missing=1"])
    finally:
        config_path.unlink(missing_ok=True)
