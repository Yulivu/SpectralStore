"""Compression primitives for SpectralStore."""

from spectralstore.compression.factorized_store import FactorizedTemporalStore
from spectralstore.compression.spectral import (
    AsymmetricSpectralCompressor,
    CPALSCompressor,
    DirectSVDCompressor,
    RPCASVDCompressor,
    RobustAsymmetricSpectralCompressor,
    SplitAsymmetricUnfoldingCompressor,
    SparseUnfoldingAsymmetricCompressor,
    SpectralCompressionConfig,
    SymmetricSVDCompressor,
    TensorUnfoldingSVDCompressor,
    TuckerHOSVDCompressor,
)
from spectralstore.compression.registry import (
    COMPRESSOR_REGISTRY,
    PROTOTYPE_COMPRESSORS,
    available_compressors,
    create_compressor,
    spectral_config_from_mapping,
)

__all__ = [
    "AsymmetricSpectralCompressor",
    "COMPRESSOR_REGISTRY",
    "CPALSCompressor",
    "DirectSVDCompressor",
    "FactorizedTemporalStore",
    "PROTOTYPE_COMPRESSORS",
    "RPCASVDCompressor",
    "RobustAsymmetricSpectralCompressor",
    "SplitAsymmetricUnfoldingCompressor",
    "SparseUnfoldingAsymmetricCompressor",
    "SpectralCompressionConfig",
    "SymmetricSVDCompressor",
    "TensorUnfoldingSVDCompressor",
    "TuckerHOSVDCompressor",
    "available_compressors",
    "create_compressor",
    "spectral_config_from_mapping",
]
