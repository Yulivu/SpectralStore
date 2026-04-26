# Preliminary ogbl-collab Experiment

Goal: load OGB's `ogbl-collab` dataset through the official OGB package and
run the SpectralStore real-data path on yearly collaboration snapshots.

The default config caps the first run to the highest-degree nodes so the dense
prototype compressors remain tractable. This is a data-integration smoke path,
not the final full-scale OGB benchmark.

The script enforces a dense-memory budget before compression. With the current
dense compressor path, `max_nodes=5000` requires at least 8 GiB just for the
dense input tensor, so larger OGB runs need the sparse compressor path first.

Outputs:

- `results/metrics.json`: storage and held-out observed-edge metrics
- `results/summary.md`: human-readable comparison table
