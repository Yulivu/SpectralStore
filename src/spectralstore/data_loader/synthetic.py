"""Synthetic temporal graph generators."""

from __future__ import annotations

import numpy as np
from scipy import sparse


def make_low_rank_temporal_graph(
    *,
    num_nodes: int = 24,
    num_steps: int = 4,
    rank: int = 3,
    random_seed: int = 0,
    noise_scale: float = 0.02,
) -> list[sparse.csr_matrix]:
    rng = np.random.default_rng(random_seed)
    left = rng.normal(size=(num_nodes, rank))
    right = rng.normal(size=(num_nodes, rank))
    temporal = rng.normal(loc=1.0, scale=0.15, size=(num_steps, rank))
    lambdas = np.linspace(1.0, 0.35, rank)

    snapshots = []
    for t in range(num_steps):
        matrix = (left * (lambdas * temporal[t])) @ right.T
        matrix += rng.normal(scale=noise_scale, size=matrix.shape)
        snapshots.append(sparse.csr_matrix(matrix))
    return snapshots
