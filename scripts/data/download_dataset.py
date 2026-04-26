"""Download datasets used by SpectralStore experiments."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


DATASETS = {
    "bitcoin_otc": {
        "url": "https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz",
        "path": Path("data/raw/soc-sign-bitcoinotc.csv.gz"),
    }
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(DATASETS))
    args = parser.parse_args()

    spec = DATASETS[args.dataset]
    output_path = spec["path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"Already downloaded: {output_path}")
        return

    print(f"Downloading {spec['url']} -> {output_path}")
    urllib.request.urlretrieve(spec["url"], output_path)
    print(f"Done: {output_path}")


if __name__ == "__main__":
    main()

