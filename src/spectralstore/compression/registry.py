"""Compressor registry for experiment scripts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
from typing import Any, Mapping

from spectralstore.compression.spectral import (
    CPALSCompressor,
    DirectSVDCompressor,
    NMFCompressor,
    RPCASVDCompressor,
    SpectralCompressionConfig,
    SymmetricSVDCompressor,
    TensorUnfoldingSVDCompressor,
    TuckerHOSVDCompressor,
    UnifiedThinkingSpectralCompressor,
)


CompressorFactory = Callable[[SpectralCompressionConfig], object]


COMPRESSOR_REGISTRY: dict[str, CompressorFactory] = {
    "spectralstore_thinking": UnifiedThinkingSpectralCompressor,
    "tensor_unfolding_svd": TensorUnfoldingSVDCompressor,
    "sym_svd": SymmetricSVDCompressor,
    "direct_svd": DirectSVDCompressor,
    "rpca_svd": RPCASVDCompressor,
    "cp_als": CPALSCompressor,
    "tucker_als": TuckerHOSVDCompressor,
    "tucker_hosvd": TuckerHOSVDCompressor,
    "nmf": NMFCompressor,
}

PROTOTYPE_COMPRESSORS = frozenset()


def create_compressor(name: str, config: SpectralCompressionConfig) -> object:
    try:
        factory = COMPRESSOR_REGISTRY[name]
    except KeyError as exc:
        supported = ", ".join(sorted(COMPRESSOR_REGISTRY))
        raise ValueError(f"unsupported compressor '{name}'. Supported: {supported}") from exc
    return factory(config)


def spectral_config_from_mapping(
    values: Mapping[str, Any],
    /,
    **overrides: Any,
) -> SpectralCompressionConfig:
    field_names = {field.name for field in fields(SpectralCompressionConfig)}
    config_values = {
        key: value
        for key, value in values.items()
        if key in field_names
    }
    unknown_overrides = sorted(set(overrides) - field_names)
    if unknown_overrides:
        joined = ", ".join(unknown_overrides)
        raise KeyError(f"unknown spectral compression config field(s): {joined}")
    config_values.update(overrides)
    return SpectralCompressionConfig(**config_values)


def available_compressors(*, include_prototypes: bool = True) -> tuple[str, ...]:
    names = sorted(COMPRESSOR_REGISTRY)
    if not include_prototypes:
        names = [name for name in names if name not in PROTOTYPE_COMPRESSORS]
    return tuple(names)
