"""Compression primitives for SpectralStore."""

from spectralstore.compression.factorized_store import (
    FactorizedTemporalStore,
    TemporalCOOResidualStore,
)
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
from spectralstore.compression.registry import (
    COMPRESSOR_REGISTRY,
    PROTOTYPE_COMPRESSORS,
    available_compressors,
    create_compressor,
    spectral_config_from_mapping,
)

__all__ = [
    "COMPRESSOR_REGISTRY",
    "CPALSCompressor",
    "DirectSVDCompressor",
    "FactorizedTemporalStore",
    "NMFCompressor",
    "PROTOTYPE_COMPRESSORS",
    "RPCASVDCompressor",
    "SpectralCompressionConfig",
    "SymmetricSVDCompressor",
    "TemporalCOOResidualStore",
    "TensorUnfoldingSVDCompressor",
    "TuckerHOSVDCompressor",
    "UnifiedThinkingSpectralCompressor",
    "available_compressors",
    "create_compressor",
    "spectral_config_from_mapping",
]
