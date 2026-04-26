"""Load OGB link-prediction datasets through the official OGB package."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from spectralstore.data_loader.bitcoin import TemporalGraphDataset


def load_ogbl_collab(
    *,
    root: str | Path = "data/ogb",
    max_nodes: int | None = 1000,
    min_year: int | None = None,
    max_year: int | None = None,
) -> TemporalGraphDataset:
    """Load ogbl-collab as yearly weighted adjacency snapshots.

    This uses OGB's official ``LinkPropPredDataset``. If OGB is not installed,
    the ImportError is intentionally explicit; we do not provide a lightweight
    replacement loader.
    """

    try:
        from ogb.linkproppred import LinkPropPredDataset
    except ImportError as exc:
        raise ImportError(
            "OGB is required for load_ogbl_collab. Install it with `python -m pip install ogb`."
        ) from exc

    with _ogb_torch_load_compat():
        dataset = LinkPropPredDataset(name="ogbl-collab", root=str(root))
    graph = dataset[0]
    return temporal_graph_from_ogbl_collab_graph(
        graph,
        max_nodes=max_nodes,
        min_year=min_year,
        max_year=max_year,
    )


def temporal_graph_from_ogbl_collab_graph(
    graph: dict[str, Any],
    *,
    max_nodes: int | None = 1000,
    min_year: int | None = None,
    max_year: int | None = None,
) -> TemporalGraphDataset:
    edge_index = _as_numpy(graph["edge_index"]).astype(int, copy=False)
    edge_year = _as_numpy(graph["edge_year"]).reshape(-1).astype(int, copy=False)
    edge_weight = _as_numpy(graph.get("edge_weight", np.ones(edge_year.shape[0]))).reshape(-1)
    num_nodes = int(graph.get("num_nodes", int(edge_index.max()) + 1))

    keep = np.ones(edge_year.shape[0], dtype=bool)
    if min_year is not None:
        keep &= edge_year >= min_year
    if max_year is not None:
        keep &= edge_year <= max_year

    edge_index = edge_index[:, keep]
    edge_year = edge_year[keep]
    edge_weight = edge_weight[keep].astype(float, copy=False)

    node_ids = _select_nodes(edge_index, num_nodes, max_nodes)
    node_to_idx = {node: idx for idx, node in enumerate(node_ids)}
    edge_keep = np.array(
        [
            int(source) in node_to_idx and int(target) in node_to_idx
            for source, target in edge_index.T
        ],
        dtype=bool,
    )
    edge_index = edge_index[:, edge_keep]
    edge_year = edge_year[edge_keep]
    edge_weight = edge_weight[edge_keep]

    time_bins = [str(year) for year in sorted(np.unique(edge_year))]
    shape = (len(node_ids), len(node_ids))
    snapshots: list[sparse.csr_matrix] = []
    for year_text in time_bins:
        year = int(year_text)
        year_keep = edge_year == year
        rows = np.fromiter(
            (node_to_idx[int(node)] for node in edge_index[0, year_keep]),
            dtype=int,
        )
        cols = np.fromiter(
            (node_to_idx[int(node)] for node in edge_index[1, year_keep]),
            dtype=int,
        )
        data = edge_weight[year_keep]
        snapshots.append(sparse.coo_matrix((data, (rows, cols)), shape=shape).tocsr())

    return TemporalGraphDataset(
        name="ogbl_collab",
        snapshots=snapshots,
        node_ids=node_ids,
        time_bins=time_bins,
    )


def _select_nodes(
    edge_index: np.ndarray,
    num_nodes: int,
    max_nodes: int | None,
) -> list[int]:
    if max_nodes is None or max_nodes >= num_nodes:
        return list(range(num_nodes))
    degree = Counter(int(node) for node in edge_index.reshape(-1))
    return [node for node, _count in degree.most_common(max_nodes)]


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


@contextmanager
def _ogb_torch_load_compat():
    """Make OGB 1.3.x processed-cache loading work with newer PyTorch.

    OGB 1.3.6 calls torch.load without weights_only. PyTorch 2.6+ defaults that
    argument to True, which cannot load OGB's graph dictionaries. This shim is
    scoped to the official OGB loader call and keeps the dataset path usable.
    """

    try:
        import torch
    except ImportError:
        yield
        return

    original_load = torch.load

    def load_with_ogb_default(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = load_with_ogb_default
    try:
        yield
    finally:
        torch.load = original_load
