# Data Directory

This directory keeps datasets out of source code and separates raw downloads
from derived artifacts.

```text
data/
  raw/        Original downloaded files
  interim/    Temporary converted files
  processed/  Stable experiment-ready artifacts
```

Large data files are ignored by Git. Keep only small metadata or instructions in
the repository.

## Available Download Scripts

| Dataset | Command | Source | Local path |
| --- | --- | --- | --- |
| Bitcoin-OTC | `python scripts/download_dataset.py bitcoin_otc` | SNAP: `https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz` | `data/raw/soc-sign-bitcoinotc.csv.gz` |
