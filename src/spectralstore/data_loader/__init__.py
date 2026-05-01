"""Dataset loading utilities."""

from spectralstore.data_loader.bitcoin import (
    TemporalGraphDataset,
    load_bitcoin_alpha,
    load_bitcoin_otc,
)
from spectralstore.data_loader.ogb import (
    load_ogbl_collab,
    temporal_graph_from_ogbl_collab_graph,
)
from spectralstore.data_loader.synthetic import (
    inject_sparse_corruption,
    SyntheticTemporalGraph,
    make_low_rank_temporal_graph,
    make_synthetic_spiked,
    make_synthetic_attack,
    make_temporal_sbm,
    make_temporal_correlated_sbm,
    make_theory_regime_sbm,
)

__all__ = [
    "SyntheticTemporalGraph",
    "TemporalGraphDataset",
    "load_bitcoin_alpha",
    "load_bitcoin_otc",
    "load_ogbl_collab",
    "inject_sparse_corruption",
    "make_low_rank_temporal_graph",
    "make_synthetic_spiked",
    "make_synthetic_attack",
    "make_temporal_sbm",
    "make_temporal_correlated_sbm",
    "make_theory_regime_sbm",
    "temporal_graph_from_ogbl_collab_graph",
]
