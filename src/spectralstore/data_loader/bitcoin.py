"""Load SNAP Bitcoin signed trust datasets."""

from __future__ import annotations

import csv
import gzip
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class TemporalGraphDataset:
    name: str
    snapshots: list[sparse.csr_matrix]
    node_ids: list[int]
    time_bins: list[str]


def load_bitcoin_otc(
    path: str | Path,
    *,
    max_nodes: int = 300,
    normalize_ratings: bool = True,
) -> TemporalGraphDataset:
    """Load Bitcoin-OTC as monthly signed weighted snapshots.

    SNAP rows are `source,target,rating,time`. Ratings are in `[-10, 10]`.
    """

    rows = _read_rows(Path(path))
    degree = Counter[int]()
    for source, target, _rating, _timestamp in rows:
        degree[source] += 1
        degree[target] += 1

    node_ids = [node for node, _count in degree.most_common(max_nodes)]
    node_to_idx = {node: idx for idx, node in enumerate(node_ids)}
    monthly_edges: dict[str, list[tuple[int, int, float]]] = {}

    for source, target, rating, timestamp in rows:
        if source not in node_to_idx or target not in node_to_idx:
            continue
        month = datetime.fromtimestamp(timestamp, UTC).strftime("%Y-%m")
        weight = rating / 10.0 if normalize_ratings else rating
        monthly_edges.setdefault(month, []).append((node_to_idx[source], node_to_idx[target], weight))

    time_bins = sorted(monthly_edges)
    snapshots = []
    shape = (len(node_ids), len(node_ids))
    for month in time_bins:
        entries = monthly_edges[month]
        row = np.fromiter((edge[0] for edge in entries), dtype=int)
        col = np.fromiter((edge[1] for edge in entries), dtype=int)
        data = np.fromiter((edge[2] for edge in entries), dtype=float)
        matrix = sparse.coo_matrix((data, (row, col)), shape=shape).tocsr()
        snapshots.append(matrix)

    return TemporalGraphDataset(
        name="bitcoin_otc",
        snapshots=snapshots,
        node_ids=node_ids,
        time_bins=time_bins,
    )


def _read_rows(path: Path) -> list[tuple[int, int, float, int]]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows: list[tuple[int, int, float, int]] = []
    with opener(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for source, target, rating, timestamp in reader:
            rows.append((int(source), int(target), float(rating), int(float(timestamp))))
    return rows
