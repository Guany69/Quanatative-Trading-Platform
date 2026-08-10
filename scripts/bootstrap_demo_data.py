"""Generate the deterministic fixture dataset as Parquet files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/fixtures"))
    parser.add_argument("--securities", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from quant_platform.data.adapters.fixture import FixtureDataProvider, FixtureSpec
    from quant_platform.utilities.reproducibility import configure_logging

    configure_logging("INFO")
    args.out.mkdir(parents=True, exist_ok=True)
    dataset = FixtureDataProvider(
        FixtureSpec(n_securities=args.securities, seed=args.seed)
    ).generate()

    for name in (
        "securities",
        "identifiers",
        "prices",
        "benchmark",
        "membership",
        "corporate_actions",
        "fundamentals",
        "macro",
    ):
        frame = getattr(dataset, name)
        frame.write_parquet(args.out / f"{name}.parquet")
        print(f"  {name:18s} {frame.shape}")

    print("\nSYNTHETIC DATA: randomly generated. No investment meaning.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
