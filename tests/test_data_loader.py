from pathlib import Path

import pytest

from spectralstore.data_loader import load_bitcoin_otc, make_synthetic_attack
from spectralstore.data_loader import temporal_graph_from_ogbl_collab_graph


def test_synthetic_attack_uses_unified_dataset_fields() -> None:
    dataset = make_synthetic_attack(
        num_nodes=16,
        num_steps=3,
        num_communities=2,
        attack_fraction=0.01,
        random_seed=21,
    )

    assert dataset.name == "synthetic_attack"
    assert dataset.num_nodes == 16
    assert dataset.num_steps == 3
    assert dataset.communities is not None
    assert dataset.attack_kind == "sparse_outlier_edges"
    assert dataset.expected_snapshots is not None


def test_bitcoin_loader_uses_unified_dataset_fields() -> None:
    path = Path("data/raw/soc-sign-bitcoinotc.csv.gz")
    if not path.exists():
        pytest.skip("Bitcoin-OTC raw data is not present locally")

    dataset = load_bitcoin_otc(path, max_nodes=20)

    assert dataset.name == "bitcoin_otc"
    assert dataset.num_nodes == len(dataset.node_ids)
    assert dataset.num_steps == len(dataset.snapshots)
    assert dataset.num_nodes <= 20
    assert dataset.num_steps > 0
    assert dataset.snapshots[0].shape == (dataset.num_nodes, dataset.num_nodes)


def test_ogbl_collab_graph_adapter_builds_yearly_snapshots() -> None:
    graph = {
        "num_nodes": 5,
        "edge_index": [[0, 1, 2, 3], [1, 2, 3, 4]],
        "edge_year": [[2018], [2018], [2019], [2020]],
        "edge_weight": [[2.0], [1.0], [3.0], [4.0]],
    }

    dataset = temporal_graph_from_ogbl_collab_graph(graph, max_nodes=None)

    assert dataset.name == "ogbl_collab"
    assert dataset.num_nodes == 5
    assert dataset.time_bins == ["2018", "2019", "2020"]
    assert dataset.num_steps == 3
    assert dataset.snapshots[0].shape == (5, 5)
