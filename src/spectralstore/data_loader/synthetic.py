"""Synthetic temporal graph generators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class SyntheticTemporalGraph:
    name: str
    snapshots: list[sparse.csr_matrix]
    expected_snapshots: list[np.ndarray] | None = None
    communities: np.ndarray | None = None
    node_ids: list[int] | None = None
    time_bins: list[str] | None = None
    attack_edges: tuple[tuple[int, int, int], ...] = ()
    attack_kind: str | None = None
    held_out_edges: tuple[tuple[int, int, int, float], ...] = ()
    corruption_masks: tuple[sparse.csr_matrix, ...] = ()

    @property
    def num_nodes(self) -> int:
        return int(self.snapshots[0].shape[0]) if self.snapshots else 0

    @property
    def num_steps(self) -> int:
        return len(self.snapshots)


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
        name="synthetic_sbm",
        snapshots=snapshots,
        expected_snapshots=expected_snapshots,
        communities=communities,
    )


def make_theory_regime_sbm(
    *,
    num_nodes: int = 300,
    num_steps: int = 20,
    num_communities: int = 5,
    sbm_p: float = 0.12,
    sbm_q: float = 0.08,
    regime_name: str = "medium_snr",
    temporal_jitter: float = 0.0,
    directed: bool = True,
    noise_type: str = "iid",
    base_noise_std: float = 0.0,
    high_variance_fraction: float = 0.05,
    high_variance_multiplier: float = 5.0,
    random_seed: int = 0,
) -> SyntheticTemporalGraph:
    """Generate a theory-regime SBM with optional heteroskedastic observation noise.

    The returned ``expected_snapshots`` are the latent SBM probabilities M_star.
    Extra Gaussian noise, when enabled, is added only to observations.
    """

    dataset = make_temporal_sbm(
        num_nodes=num_nodes,
        num_steps=num_steps,
        num_communities=num_communities,
        p_in=sbm_p,
        p_out=sbm_q,
        temporal_jitter=temporal_jitter,
        directed=directed,
        random_seed=random_seed,
    )
    if noise_type not in {"iid", "heteroskedastic_entry", "heteroskedastic_node"}:
        raise ValueError(f"unsupported noise_type: {noise_type}")
    if base_noise_std <= 0.0:
        return SyntheticTemporalGraph(
            name=f"theory_regime_sbm_{regime_name}",
            snapshots=dataset.snapshots,
            expected_snapshots=dataset.expected_snapshots,
            communities=dataset.communities,
        )

    rng = np.random.default_rng(random_seed + 20_000)
    std_matrix = np.full((num_nodes, num_nodes), float(base_noise_std), dtype=float)
    if noise_type == "heteroskedastic_entry":
        candidate_mask = ~np.eye(num_nodes, dtype=bool)
        selected = rng.random((num_nodes, num_nodes)) < float(high_variance_fraction)
        selected &= candidate_mask
        if not directed:
            selected = np.triu(selected, k=1)
            selected = selected | selected.T
        std_matrix[selected] *= float(high_variance_multiplier)
    elif noise_type == "heteroskedastic_node":
        node_count = max(1, int(round(float(high_variance_fraction) * num_nodes)))
        high_nodes = rng.choice(num_nodes, size=node_count, replace=False)
        std_matrix[high_nodes, :] *= float(high_variance_multiplier)
        std_matrix[:, high_nodes] *= float(high_variance_multiplier)
    np.fill_diagonal(std_matrix, 0.0)

    noisy_snapshots: list[sparse.csr_matrix] = []
    for snapshot in dataset.snapshots:
        observed = snapshot.toarray().astype(float, copy=True)
        observed += rng.normal(scale=std_matrix, size=observed.shape)
        np.fill_diagonal(observed, 0.0)
        if not directed:
            observed = 0.5 * (observed + observed.T)
        noisy_snapshots.append(sparse.csr_matrix(observed))

    return SyntheticTemporalGraph(
        name=f"theory_regime_sbm_{regime_name}",
        snapshots=noisy_snapshots,
        expected_snapshots=dataset.expected_snapshots,
        communities=dataset.communities,
    )


def make_temporal_correlated_sbm(
    *,
    num_nodes: int = 300,
    num_steps: int = 20,
    num_communities: int = 5,
    sbm_p: float = 0.12,
    sbm_q: float = 0.08,
    regime_name: str = "medium_snr",
    alpha: float = 0.0,
    base_noise_std: float = 0.05,
    directed: bool = True,
    random_seed: int = 0,
) -> SyntheticTemporalGraph:
    """Generate a fixed-M* SBM with AR(1) temporally correlated noise.

    Observations follow A_t = M* + H_t where H_t = alpha H_{t-1} + epsilon_t.
    The latent ``expected_snapshots`` contain copies of M* for evaluation.
    """

    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must be in [0, 1)")

    rng = np.random.default_rng(random_seed)
    communities = np.arange(num_nodes) % num_communities
    rng.shuffle(communities)
    base_block = np.full((num_communities, num_communities), sbm_q, dtype=float)
    np.fill_diagonal(base_block, sbm_p)
    expected = base_block[communities[:, None], communities[None, :]].astype(float)
    np.fill_diagonal(expected, 0.0)
    if not directed:
        expected = 0.5 * (expected + expected.T)

    noise = rng.normal(scale=float(base_noise_std), size=expected.shape)
    np.fill_diagonal(noise, 0.0)
    if not directed:
        noise = 0.5 * (noise + noise.T)

    snapshots: list[sparse.csr_matrix] = []
    expected_snapshots: list[np.ndarray] = []
    for t in range(num_steps):
        if t > 0:
            innovation = rng.normal(scale=float(base_noise_std), size=expected.shape)
            np.fill_diagonal(innovation, 0.0)
            if not directed:
                innovation = 0.5 * (innovation + innovation.T)
            noise = float(alpha) * noise + innovation
        observed = expected + noise
        np.fill_diagonal(observed, 0.0)
        snapshots.append(sparse.csr_matrix(observed))
        expected_snapshots.append(expected.copy())

    return SyntheticTemporalGraph(
        name=f"temporal_correlated_sbm_{regime_name}",
        snapshots=snapshots,
        expected_snapshots=expected_snapshots,
        communities=communities,
    )


def make_synthetic_spiked(
    *,
    num_nodes: int = 200,
    num_steps: int = 10,
    rank: int = 3,
    snr: float = 2.0,
    signal_strength: float | None = None,
    noise_std: float = 1.0,
    noise_type: str = "iid",
    high_variance_fraction: float = 0.05,
    high_variance_multiplier: float = 5.0,
    random_seed: int = 0,
) -> SyntheticTemporalGraph:
    """Generate a temporal spiked matrix model with Gaussian observations."""

    rng = np.random.default_rng(random_seed)
    basis, _ = np.linalg.qr(rng.normal(size=(num_nodes, rank)))
    leading_strength = (
        float(signal_strength)
        if signal_strength is not None
        else float(snr * noise_std * np.sqrt(num_nodes))
    )
    lambdas = np.geomspace(leading_strength, max(leading_strength * 0.25, 1e-3), rank)
    expected = (basis * lambdas) @ basis.T
    np.fill_diagonal(expected, 0.0)
    if noise_type not in {"iid", "heteroskedastic_entry", "heteroskedastic_node"}:
        raise ValueError(f"unsupported noise_type: {noise_type}")
    noise_scale = np.full(expected.shape, float(noise_std), dtype=float)
    if noise_type == "heteroskedastic_entry":
        selected = rng.random(expected.shape) < float(high_variance_fraction)
        selected &= ~np.eye(num_nodes, dtype=bool)
        noise_scale[selected] *= float(high_variance_multiplier)
    elif noise_type == "heteroskedastic_node":
        node_count = max(1, int(round(float(high_variance_fraction) * num_nodes)))
        high_nodes = rng.choice(num_nodes, size=node_count, replace=False)
        noise_scale[high_nodes, :] *= float(high_variance_multiplier)
        noise_scale[:, high_nodes] *= float(high_variance_multiplier)
    np.fill_diagonal(noise_scale, 0.0)

    snapshots = []
    expected_snapshots = []
    for _t in range(num_steps):
        noise = rng.normal(scale=noise_scale, size=expected.shape)
        observed = expected + noise
        np.fill_diagonal(observed, 0.0)
        snapshots.append(sparse.csr_matrix(observed))
        expected_snapshots.append(expected.copy())

    return SyntheticTemporalGraph(
        name="synthetic_spiked",
        snapshots=snapshots,
        expected_snapshots=expected_snapshots,
        communities=np.zeros(num_nodes, dtype=int),
    )


def make_synthetic_attack(
    *,
    num_nodes: int = 300,
    num_steps: int = 20,
    num_communities: int = 5,
    p_in: float = 0.30,
    p_out: float = 0.05,
    temporal_jitter: float = 0.08,
    attack_kind: str = "sparse_outlier_edges",
    attack_fraction: float = 0.02,
    outlier_weight: float = 3.0,
    corruption_rate: float | None = None,
    corruption_magnitude: float | None = None,
    directed: bool = True,
    random_seed: int = 0,
) -> SyntheticTemporalGraph:
    """Generate a temporal SBM and inject sparse attacks into observed snapshots."""

    clean = make_temporal_sbm(
        num_nodes=num_nodes,
        num_steps=num_steps,
        num_communities=num_communities,
        p_in=p_in,
        p_out=p_out,
        temporal_jitter=temporal_jitter,
        directed=directed,
        random_seed=random_seed,
    )
    rng = np.random.default_rng(random_seed + 10_000)
    rate = attack_fraction if corruption_rate is None else corruption_rate
    magnitude = outlier_weight if corruption_magnitude is None else corruption_magnitude
    possible_pairs = _candidate_pairs(clean.communities, attack_kind, directed=directed)
    attacks_per_step = int(round(rate * len(possible_pairs)))
    attacked_snapshots: list[sparse.csr_matrix] = []
    attack_edges: list[tuple[int, int, int]] = []
    corruption_masks: list[sparse.csr_matrix] = []

    for t, snapshot in enumerate(clean.snapshots):
        dense = snapshot.toarray().astype(float, copy=True)
        chosen = _sample_attack_pairs(
            rng,
            possible_pairs,
            attacks_per_step,
            attack_kind=attack_kind,
            communities=clean.communities,
        )
        mask = np.zeros(dense.shape, dtype=bool)
        for u, v in chosen:
            if attack_kind == "random_flip":
                dense[u, v] = 1.0 - dense[u, v]
            elif attack_kind == "targeted_cross_community":
                dense[u, v] = 1.0
            elif attack_kind == "sparse_outlier_edges":
                dense[u, v] = outlier_weight
            elif attack_kind == "sparse_spike":
                dense[u, v] += magnitude
            elif attack_kind == "signed_spike":
                dense[u, v] += rng.choice([-1.0, 1.0]) * magnitude
            elif attack_kind == "block_sparse_spike":
                dense[u, v] += magnitude
            else:
                raise ValueError(f"unsupported attack_kind: {attack_kind}")
            mask[u, v] = True
            attack_edges.append((t, int(u), int(v)))

            if not directed:
                dense[v, u] = dense[u, v]
                mask[v, u] = True
                attack_edges.append((t, int(v), int(u)))

        attacked_snapshots.append(sparse.csr_matrix(dense))
        corruption_masks.append(sparse.csr_matrix(mask.astype(float)))

    return SyntheticTemporalGraph(
        name="synthetic_attack",
        snapshots=attacked_snapshots,
        expected_snapshots=clean.expected_snapshots,
        communities=clean.communities,
        attack_edges=tuple(attack_edges),
        attack_kind=attack_kind,
        corruption_masks=tuple(corruption_masks),
    )


def _candidate_pairs(
    communities: np.ndarray,
    attack_kind: str,
    *,
    directed: bool,
) -> np.ndarray:
    num_nodes = communities.shape[0]
    row, col = np.where(~np.eye(num_nodes, dtype=bool))
    if not directed:
        keep = row < col
        row = row[keep]
        col = col[keep]

    if attack_kind == "targeted_cross_community":
        keep = communities[row] != communities[col]
        row = row[keep]
        col = col[keep]
    elif attack_kind not in {
        "random_flip",
        "sparse_outlier_edges",
        "sparse_spike",
        "signed_spike",
        "block_sparse_spike",
    }:
        raise ValueError(f"unsupported attack_kind: {attack_kind}")

    return np.column_stack([row, col])


def _sample_attack_pairs(
    rng: np.random.Generator,
    pairs: np.ndarray,
    count: int,
    *,
    attack_kind: str,
    communities: np.ndarray,
) -> np.ndarray:
    if attack_kind != "block_sparse_spike":
        return _sample_pairs(rng, pairs, count)
    if count <= 0:
        return np.empty((0, 2), dtype=int)

    unique_communities = np.unique(communities)
    block_source = rng.choice(unique_communities)
    block_target = rng.choice(unique_communities)
    block_mask = (
        (communities[pairs[:, 0]] == block_source)
        & (communities[pairs[:, 1]] == block_target)
    )
    block_pairs = pairs[block_mask]
    if block_pairs.shape[0] < count:
        return _sample_pairs(rng, pairs, count)
    return _sample_pairs(rng, block_pairs, count)


def _sample_pairs(
    rng: np.random.Generator,
    pairs: np.ndarray,
    count: int,
) -> np.ndarray:
    if count <= 0:
        return np.empty((0, 2), dtype=int)
    if count >= len(pairs):
        return pairs.copy()
    indices = rng.choice(len(pairs), size=count, replace=False)
    return pairs[indices]
