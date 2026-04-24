"""Synthetic temporal graph generators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class SyntheticTemporalGraph:
    snapshots: list[sparse.csr_matrix]
    expected_snapshots: list[np.ndarray]
    communities: np.ndarray


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


def make_temporal_sbm(
    *,
    num_nodes: int = 300,
    num_steps: int = 20,
    num_communities: int = 5,
    p_in: float = 0.30,
    p_out: float = 0.05,
    temporal_jitter: float = 0.08,
    directed: bool = True,
    random_seed: int = 0,
) -> SyntheticTemporalGraph:
    """Generate a temporal stochastic block model.

    The latent community assignment is fixed across time. Each snapshot gets a
    small multiplicative temporal perturbation per community pair, producing
    independent noisy observations of a shared low-rank block structure.
    """

    rng = np.random.default_rng(random_seed)
    communities = np.arange(num_nodes) % num_communities
    rng.shuffle(communities)

    base_block = np.full((num_communities, num_communities), p_out, dtype=float)
    np.fill_diagonal(base_block, p_in)

    snapshots: list[sparse.csr_matrix] = []
    expected_snapshots: list[np.ndarray] = []
    for _t in range(num_steps):
        block_noise = rng.normal(loc=1.0, scale=temporal_jitter, size=base_block.shape)
        if not directed:
            block_noise = 0.5 * (block_noise + block_noise.T)
        block_probs = np.clip(base_block * block_noise, 0.0, 1.0)
        probability = block_probs[communities[:, None], communities[None, :]]
        np.fill_diagonal(probability, 0.0)
        expected_snapshots.append(probability.copy())

        sampled = rng.binomial(1, probability).astype(float)
        if not directed:
            sampled = np.triu(sampled, k=1)
            sampled = sampled + sampled.T
        snapshots.append(sparse.csr_matrix(sampled))

    return SyntheticTemporalGraph(
        snapshots=snapshots,
        expected_snapshots=expected_snapshots,
        communities=communities,
    )
