"""Run the full synthetic demonstration (mirrors `quant-platform run-demo`)."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/research_charter.yaml"))
    parser.add_argument("--out", type=Path, default=Path("reports/demo"))
    parser.add_argument("--securities", type=int, default=400)
    parser.add_argument("--test-start", type=str, default="2021-01-04")
    args = parser.parse_args()

    from quant_platform.demo import run_full_demo
    from quant_platform.utilities.reproducibility import configure_logging

    configure_logging("INFO")
    paths = run_full_demo(
        config_path=args.config,
        out_dir=args.out,
        n_securities=args.securities,
        test_start=date.fromisoformat(args.test_start),
    )
    print(f"\nReports written to {paths['dir']}")
    print("Results are SYNTHETIC and carry no investment meaning.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
