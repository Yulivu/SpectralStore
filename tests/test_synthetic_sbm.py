from spectralstore.data_loader import make_temporal_sbm


def test_temporal_sbm_returns_snapshots_and_ground_truth() -> None:
    dataset = make_temporal_sbm(
        num_nodes=30,
        num_steps=4,
        num_communities=3,
        random_seed=9,
    )

    assert len(dataset.snapshots) == 4
    assert len(dataset.expected_snapshots) == 4
    assert dataset.snapshots[0].shape == (30, 30)
    assert dataset.expected_snapshots[0].shape == (30, 30)
    assert dataset.communities.shape == (30,)
