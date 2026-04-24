"""Dataset loading utilities."""

from spectralstore.data_loader.bitcoin import TemporalGraphDataset, load_bitcoin_otc
from spectralstore.data_loader.synthetic import (
    SyntheticTemporalGraph,
    make_low_rank_temporal_graph,
    make_synthetic_attack,
    make_temporal_sbm,
)

__all__ = [
    "SyntheticTemporalGraph",
    "TemporalGraphDataset",
    "load_bitcoin_otc",
    "make_low_rank_temporal_graph",
    "make_synthetic_attack",
    "make_temporal_sbm",
]
